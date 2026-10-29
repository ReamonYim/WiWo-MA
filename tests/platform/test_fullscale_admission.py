from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from wiwo_ma.platform.config import validate,mechanical_axes,fingerprint,load,ROOT
from wiwo_ma.platform.examples import continuous_fullscale_example,continuous_munich_example
from wiwo_ma.platform.physics import PhysicalWorld
from wiwo_ma.platform.resource_estimate import estimate,sampling_assessment


def fixture():
    c=continuous_munich_example();c['actions']=[];return c


def test_optional_axes_preserve_historical_fingerprints():
    assert fingerprint(continuous_fullscale_example())=='02c1d340e2cdc30254d22869ba97e3690658c01103d8817b8cd03a098e908460'
    c=load(ROOT/'tests/fixtures/legacy30s_config.json')
    assert fingerprint(c)=='d19a9c42cd186ee797b90fdfb4d9284f316e06c4d519789470f253e40d81c08d'
    assert 'translation_axes' not in validate(fixture())['base_stations'][0]['panels'][0]['mechanics']


def test_hinge_and_translation_restrictions_execute_and_reject_other_axes():
    c=fixture();p=c['base_stations'][0]['panels'][0];p['mechanics'].update(translation_axes=[0,2],rotation_axes=[2])
    w=PhysicalWorld(c);initial=w.state();key=dict(bs_id='bs0',panel_id=p['id'],type='target_pose')
    R=Rotation.from_euler('z',.2,degrees=True)*Rotation.from_euler('ZYX',p['orientation_deg'],degrees=True)
    a={**key,'position_m':(np.array(p['position_m'])+[.002,0,.001]).tolist(),'orientation_deg':R.as_euler('ZYX',degrees=True).tolist()}
    w.command(a);w.advance_to(.5);s=w.state()
    np.testing.assert_allclose(w.controllers[0].state.position,a['position_m'],atol=1e-10)
    assert np.linalg.norm(s['tx_port_positions'][:2]-initial['tx_port_positions'][:2])>0
    before=w.snapshot()
    with pytest.raises(ValueError,match='degrees of freedom'):w.command({**a,'position_m':(np.array(p['position_m'])+[0,.001,0]).tolist()})
    assert w.snapshot()['tick']==before['tick']
    with pytest.raises(ValueError,match='physical limits'):w.command(dict(bs_id='bs0',panel_id=p['id'],type='velocity',velocity_m_s=[0,.01,0],duration_s=.1))


@pytest.mark.parametrize('bad',[dict(rotation_axes=[0,1]),dict(translation_axes=[0,0]),dict(rotation_axes=[True]),dict(translation_axes=[3])])
def test_unrealizable_axes_rejected(bad):
    c=fixture();c['base_stations'][0]['panels'][0]['mechanics'].update(bad)
    with pytest.raises(ValueError):validate(c)


def test_sampling_and_cost_do_not_invent_measurements_or_authorize():
    c=continuous_fullscale_example();q=estimate(c)
    assert q['dimensions']['tx_ports']==64 and q['dimensions']['frames']==401 and q['estimated_SDK_calls']==6560
    assert q['dimensions']['CSI_uncompressed_bytes']==525598720
    assert not q['GPU_authorized'] and q['new_GPU_seconds']==0
    assert q['sampling']['persistent_specular_phase_bound_Hz']<50
    c['timing']['csi_dt_s']=.02
    c['dataset']['independent_reference_frames']=[]
    assert sampling_assessment(c)['status'].startswith('Failed')
    c['rt']['diffraction']=True
    assert estimate(c)['estimated_total_wall_s'] is None
    assert sampling_assessment(c)['status']=='Unknown'


def test_nonstationary_BS_not_given_false_sampling_certificate():
    c=fixture();bs=c['base_stations'][0];bs['trajectory']=[dict(time_s=0,position_m=bs['position_m']),dict(time_s=4,position_m=(np.array(bs['position_m'])+[.1,0,0]).tolist())]
    assert sampling_assessment(c)['status']=='Unknown'


def test_adverse_reference_selection_is_deterministic_and_preserves_zero_and_weak_users():
    from wiwo_ma.platform.continuous_validation import select_reference_indices
    class Reader:
        config={'dataset':{'independent_reference_frames':[0],'independent_reference_extremes':2}}
        count=4
        def frame(self,i):
            return dict(H=np.array([[1,1],[.1,.1]] if i==0 else [[1,1],[0,0]] if i==1 else [[1,1],[.2,.2]] if i==2 else [[1,1],[.3,.3]],dtype=complex))
    indices,q=select_reference_indices(Reader())
    assert indices==[0,1,2]
    assert q['additional'][0]['user']==1 and q['additional'][0]['index']==1
    assert q['additional'][1]['score_infinite'] and q['additional'][1]['index']==2


def test_panel_penetrating_actual_city_surface_is_rejected_before_CUDA():
    from wiwo_ma.platform.examples import reflector_example
    from wiwo_ma.platform.scenario_checks import StaticGeometry,preflight
    c=reflector_example();c.update(actions=[],actors=[]);c['users'][0]['trajectory']=[]
    c['timing'].update(duration_s=.002,csi_dt_s=.002,control_dt_s=.002)
    g=StaticGeometry(c);c['base_stations'][0]['position_m']=g.triangles[0].mean(axis=0).tolist()
    result=preflight(c)
    assert result['status']=='Failed'
    assert any(v['reason']=='panel_static_surface_penetration' for v in result['violations'])


def test_constrained_relative_UI_target_preserves_unrounded_mount_and_axis():
    from wiwo_ma.platform.physics import constrained_target
    c=continuous_fullscale_example();c['actions']=[];p=c['base_stations'][0]['panels'][15];p['mechanics'].update(translation_axes=[0,2],rotation_axes=[2])
    a=constrained_target(c,'bs0',p['id'],[.003,0,.001],[0,0,.001745])
    assert a['position_m'][1]==p['position_m'][1]
    w=PhysicalWorld(c);w.command(a);w.advance_to(.5);np.testing.assert_allclose(w.controllers[15].state.position,a['position_m'],atol=1e-12,rtol=0)
    with pytest.raises(ValueError,match='locked'):constrained_target(c,'bs0',p['id'],[0,1e-12,0],[0,0,0])
