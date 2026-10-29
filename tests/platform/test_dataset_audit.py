"""Numerical audit checks with independent analytic expectations."""
import numpy as np
import pytest
from wiwo_ma.platform.dataset_audit import (nmse_all_users, temporal_correlation,
    correlation_matrix, segment_box, local_array, prescribed_pose, decode_plain)


def test_nmse_includes_weak_and_zero_users():
    b=np.array([[1+1j,2],[1e-12,2e-12],[0,0]],complex)
    a=b.copy();a[1]*=2;a[2,0]=1
    err,_,_=nmse_all_users(a,b)
    np.testing.assert_allclose(err[:2],[0,1]);assert np.isinf(err[2])
    a[:]=b;assert nmse_all_users(a,b)[0][2]==0


def test_nonfinite_never_passes():
    a=np.ones((2,3),complex);a[1,0]=complex(np.nan,0)
    assert np.isinf(nmse_all_users(a,np.ones_like(a))[0][1])


def test_acf_matches_direct_overlap_not_circular():
    rng=np.random.default_rng(42);x=rng.normal(size=(31,3))+1j*rng.normal(size=(31,3))
    direct=[]
    for k in range(len(x)):
        q=x[k:];p=x[:len(x)-k]
        direct.append(np.sum(q*p.conj())/np.sqrt(np.sum(abs(p)**2)*np.sum(abs(q)**2)))
    np.testing.assert_allclose(temporal_correlation(x),direct,atol=1e-14)


def test_rotating_tone_has_unit_correlation_and_correct_sign():
    t=np.arange(121)/100;x=np.exp(2j*np.pi*7*t)[:,None]
    np.testing.assert_allclose(temporal_correlation(x)[:50],np.exp(2j*np.pi*7*t[:50]),atol=1e-13)
    assert np.isnan(temporal_correlation(np.zeros((10,3)))).all()


def test_spatial_correlation_zeros_are_unknown():
    x=np.column_stack((np.arange(1,10),2j*np.arange(1,10),np.zeros(9)))
    m=correlation_matrix(x);np.testing.assert_allclose(abs(m[:2,:2]),1)
    assert np.isnan(m[2]).all()


def test_box_visibility_parallel_edges_and_rotated():
    a=np.array([[-2,0,0],[-2,2,0],[-2,1,0],[0,0,0]])
    b=np.array([[2,0,0],[2,2,0],[2,1,0],[.1,0,0]])
    np.testing.assert_array_equal(segment_box(a,b,np.zeros(3),np.eye(3),[2,2,2]),[True,False,True,True])
    from scipy.spatial.transform import Rotation
    r=Rotation.from_euler('z',37,degrees=True).as_matrix()
    np.testing.assert_array_equal(segment_box(a@r.T,b@r.T,np.zeros(3),r,[2,2,2]),[True,False,True,True])


def test_array_order_and_route_reversal():
    np.testing.assert_allclose(local_array(dict(rows=2,cols=2,spacing_m=2)),[[0,-1,1],[0,-1,-1],[0,1,1],[0,1,-1]])
    e=dict(trajectory=[dict(time_s=0,position_m=[0,0,0],orientation_deg=[0,0,0]),dict(time_s=1,position_m=[1,0,0],orientation_deg=[90,0,0]),dict(time_s=2,position_m=[0,0,0],orientation_deg=[0,0,0])])
    p,r,v=prescribed_pose(e,1);np.testing.assert_allclose(p,[1,0,0]);np.testing.assert_allclose(v,[-1,0,0]);np.testing.assert_allclose(r@np.array([1,0,0]),[0,1,0],atol=1e-15)


def test_unknown_snapshot_tag_rejected():
    with pytest.raises(ValueError):decode_plain({'@':'untrusted','value':[]})


def load_audit_script(name):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[2]/'scripts'/name
    spec=importlib.util.spec_from_file_location('audit_test_'+name.replace('.','_'),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_independent_bounded_profile_triangular_and_cruise():
    profile=load_audit_script('diagnose_platform_csi30s.py').scalar_profile
    x,d=profile(1,1,1,np.array([-1,0,.5,1,1.5,2,3]))
    np.testing.assert_allclose(x,[0,0,.125,.5,.875,1,1]);assert d==2
    x,d=profile(3,1,1,np.array([0,1,2,3,4,5]))
    np.testing.assert_allclose(x,[0,.5,1.5,2.5,3,3]);assert d==4


def test_los_physical_amplitude_broadside_and_inverse_range():
    model=load_audit_script('enrich_platform_csi30s_audit.py').los_amplitude
    v=np.array([[1.,0,0],[1.,0,0]]);r=np.broadcast_to(np.eye(3),(2,3,3))
    got=model(v,r,np.array([10.,20.]),.01)
    np.testing.assert_allclose(got,.01/(4*np.pi*np.array([10.,20.]))*10**(8/20),rtol=1e-14)


def test_actual_snapshot_schema_and_actions_are_integer_count():
    from wiwo_ma.platform.examples import munich_example
    from wiwo_ma.platform.physics import PhysicalWorld
    from wiwo_ma.platform.storage import encode
    from wiwo_ma.platform.dataset_audit import independent_geometry
    config=munich_example();world=PhysicalWorld(config);world.advance_to(1.02)
    s=world.state();snapshot=decode_plain(encode(world.snapshot()))
    assert isinstance(s['actions_executed'],int)
    checks,measurements=independent_geometry(s,config,snapshot)
    assert all(checks.values());assert measurements['tx_port_max_error_m']==0
    s['mechanics_tick']+=1
    assert not independent_geometry(s,config,snapshot)[0]['clock']


def test_archive_absolute_source_stays_inside_snapshot(tmp_path):
    archive=load_audit_script('close_platform_csi30s_delivery.py').archive_target
    project=tmp_path/'project';source=project/'scripts/close.py';source.parent.mkdir(parents=True);source.write_text('pass\n')
    snapshot=project/'artifacts/snapshot'
    dest=archive(snapshot,source,root=project)
    assert dest==snapshot/'scripts/close.py';assert dest!=source
    with pytest.raises(ValueError):archive(snapshot,tmp_path/'external.py',root=project)
