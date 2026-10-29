"""Independent numerical differentiation, legacy identity, and admission tests."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from wiwo_ma.platform.config import validate, fingerprint, ROOT
from wiwo_ma.platform.motion import bounded_pose, trajectory_diagnostics
from wiwo_ma.platform.physics import pose_at, PhysicalWorld
from wiwo_ma.platform.examples import reflector_example, physical_munich_example
from wiwo_ma.platform.scenario_checks import path_quality, check_motion_contract, StaticGeometry


def entity():
    c=reflector_example();u=c['users'][0];u.update(motion=dict(model='bounded_quintic',max_speed_m_s=2.,max_acceleration_m_s2=4.,max_angular_speed_rad_s=2.,max_angular_acceleration_rad_s2=4.))
    u['trajectory']=[dict(time_s=0.,position_m=u['position_m'],orientation_deg=[0,0,0]),dict(time_s=1.,position_m=[4.5,1,1.5],orientation_deg=[20,0,0]),dict(time_s=2.,position_m=u['position_m'],orientation_deg=[0,0,0])]
    return validate(c)['users'][0]


def test_C2_reversal_derivatives_and_limits_without_finite_difference_velocity_jump():
    u=entity();eps=1e-5
    for t in [0.,1.,2.]:
        _,_,v,w,a,alpha=bounded_pose(u,t,derivatives=True)
        assert max(np.linalg.norm(x) for x in (v,w,a,alpha))<1e-12
    for t in np.linspace(.02,1.98,87):
        p,r,v,w,a,alpha=bounded_pose(u,t,derivatives=True)
        left=bounded_pose(u,t-eps);right=bounded_pose(u,t+eps)
        np.testing.assert_allclose((right[0]-left[0])/(2*eps),v,atol=1e-8)
        if abs(t-1.)>eps:
            np.testing.assert_allclose((right[2]-left[2])/(2*eps),a,atol=3e-8)
        assert np.linalg.norm(v)<=u['motion']['max_speed_m_s'] and np.linalg.norm(a)<=u['motion']['max_acceleration_m_s2']
        assert np.linalg.norm(w)<=u['motion']['max_angular_speed_rad_s'] and np.linalg.norm(alpha)<=u['motion']['max_angular_acceleration_rad_s2']
    assert not trajectory_diagnostics(u)['velocity_discontinuities']
    # Acceleration is continuous at the reversal, jerk need not be. A centred
    # difference across that join has O(eps), rather than O(eps²), truncation.
    for h in (1e-3,1e-4,1e-5):
        assert np.linalg.norm(bounded_pose(u,1-h,derivatives=True)[4])<31*h
        assert np.linalg.norm(bounded_pose(u,1+h,derivatives=True)[4])<31*h
    assert trajectory_diagnostics(u)['repeated_waypoint_positions'] # Smooth motion does not prevent split leakage.


def test_analytic_speed_and_acceleration_admission_rejects_impossible_route():
    c=reflector_example();c['users'][0]=entity();c['users'][0]['motion']['max_acceleration_m_s2']=.1
    with pytest.raises(ValueError,match='max_acceleration'):validate(c)


def test_frozen_30s_configuration_identity_and_legacy_reversal_are_preserved():
    c=json.loads((ROOT/'tests/fixtures/legacy30s_config.json').read_text())
    assert fingerprint(c)=='d19a9c42cd186ee797b90fdfb4d9284f316e06c4d519789470f253e40d81c08d'
    u=c['users'][0];jump=pose_at(u,15.)[2]-pose_at(u,15.-1e-6)[2]
    assert np.linalg.norm(jump)==pytest.approx(.3605551275463989)
    assert check_motion_contract(c)['status']=='Failed'


def test_real_city_scenario_is_bounded_and_not_a_mirrored_return():
    c=physical_munich_example();assert check_motion_contract(c)['status']=='Passed'
    assert len(c['users'])==40 and len(c['base_stations'][0]['panels'])==16
    assert all(not trajectory_diagnostics(u)['repeated_waypoint_positions'] for u in c['users'])
    assert c['physics_validation']['require_LOS_cycle']


def test_quality_flags_do_not_change_sdk_geometry():
    p=np.array([[0.,0,32.],[0,0,.01],[.001,0,.01],[5.,5.,1.5]])
    before=p.copy();q=path_quality(p)
    assert q['short_segment_offset_sensitive'] and q['min_segment_m']==pytest.approx(.001)
    np.testing.assert_array_equal(p,before)


def test_triangle_OBB_SAT_rejects_only_true_interior_surface_intersections():
    # Independent triangles: a crossing, a distant parallel face, ground contact.
    g=StaticGeometry.__new__(StaticGeometry)
    g.triangles=np.array([[[0,-2,-2],[0,2,-2],[0,0,2]],[[3,-2,-2],[3,2,-2],[3,0,2]],[[-4,-4,-1],[4,-4,-1],[0,4,-1]]],float)
    g.lo=g.triangles.min(1);g.hi=g.triangles.max(1)
    assert g.box_surface_intersections(np.zeros(3),np.eye(3),[2,2,2],-1)==1
    assert g.box_surface_intersections(np.array([1.8,0,0]),np.eye(3),[1,1,1])==0


def test_new_job_rejects_unacknowledged_legacy_motion_before_any_GPU(tmp_path):
    from wiwo_ma.platform.jobs import prepare_job
    with pytest.raises(ValueError,match='Discontinuous external motion'):
        prepare_job(reflector_example(),ROOT/'artifacts/platform/SHOULD_NOT_EXIST_motion_test',dict(max_wall_s=60,max_disk_bytes=10**8,max_allocator_bytes=10**8))
    assert not (ROOT/'artifacts/platform/SHOULD_NOT_EXIST_motion_test').exists()
