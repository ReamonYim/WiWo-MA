from copy import deepcopy
import numpy as np
import pytest
from wiwo_ma.platform.config import SCHEMA,validate,estimated_storage
from wiwo_ma.platform.physics import PhysicalWorld


def config(n=1):
    return validate(dict(schema=SCHEMA,scene=dict(name='reflector'),timing=dict(duration_s=2.,csi_dt_s=.01,mechanics_dt_s=.002,control_dt_s=.1),
        base_stations=[dict(id='bs0',position_m=[0,0,3],panels=[dict(id=str(i),position_m=[0,i*.5,0],array=dict(kind='UPA',rows=2,cols=2,spacing_m=.04)) for i in range(n)])],
        users=[dict(id='ue0',position_m=[5,0,1.5])]))


def target(x=.02):return dict(bs_id='bs0',panel_id='0',type='target_pose',position_m=[x,0,0],orientation_deg=[2,0,0])


def test_arbitrary_scale_rigid_offsets_and_display_RT_port_identity():
    c=config(3);w=PhysicalWorld(c);assert estimated_storage(c)['tx_ports']==12
    initial=w.state();w.command(target());w.advance_to(.4);s=w.state()
    assert s['positions'][0,0]>initial['positions'][0,0]
    assert s['tx_port_positions'].dtype==np.float32 and s['tx_port_map'].shape==(12,4)
    before=np.linalg.norm(initial['tx_port_positions'][:4,None]-initial['tx_port_positions'][None,:4],axis=-1)
    after=np.linalg.norm(s['tx_port_positions'][:4,None]-s['tx_port_positions'][None,:4],axis=-1)
    np.testing.assert_allclose(before,after,atol=3e-7,rtol=0)


def test_joint_mechanical_limits_reversal_and_ancestral_pending_command():
    w=PhysicalWorld(config());w.command(target(.04));w.advance_to(.06);snapshot=w.snapshot()
    a=[]
    for t in np.arange(.062,.202,.002):
        s=w.advance_to(round(float(t),6));a.append(s['velocities'][0])
    v=np.asarray(a);assert np.max(np.linalg.norm(v,axis=1))<=.15+1e-10
    assert np.max(np.linalg.norm(np.diff(v,axis=0)/.002,axis=1))<=.4+1e-8
    final=w.state()['tx_port_positions'];w.restore(snapshot);w.advance_to(.2)
    assert np.array_equal(w.state()['tx_port_positions'],final)
    w.command(target(-.01));before=w.state()['velocities'].copy();w.advance_to(.202)
    assert np.linalg.norm(w.state()['velocities']-before)<=.4*.002+1e-10


def test_invalid_command_is_transactional_no_clipping():
    w=PhysicalWorld(config());before=w.state()['tx_port_positions'].copy()
    for bad in [target(2.),dict(bs_id='bs0',panel_id='0',type='velocity',velocity_m_s=[1,0,0],duration_s=.1)]:
        with pytest.raises(ValueError):w.command(bad)
        assert w.time==0 and not w.executed and np.array_equal(w.state()['tx_port_positions'],before)


def test_hardware_DOF_fixed_rail_and_collision_rejection():
    for mechanism in ['fixed','rail_y']:
        c=config();c['base_stations'][0]['panels'][0]['mechanics']['mechanism']=mechanism;w=PhysicalWorld(c)
        with pytest.raises(ValueError):w.command(target())
    c=config(2);c['base_stations'][0]['panels'][1]['position_m']=[0,0,0]
    c['base_stations'][0]['panels'][1]['mechanics'].update(workspace_min_m=[-.25]*3,workspace_max_m=[.25]*3)
    with pytest.raises(ValueError,match='collision'):PhysicalWorld(c)


def test_exact_target_duration_and_velocity_expiration():
    w=PhysicalWorld(config());w.command({**target(), 'duration_s':1.})
    w.advance_to(.5);assert not w.controllers[0].state.completed
    w.advance_to(1.);assert w.controllers[0].state.completed
    np.testing.assert_allclose(w.controllers[0].state.position,[.02,0,0],atol=1e-10)
    w.command(dict(bs_id='bs0',panel_id='0',type='velocity',velocity_m_s=[.05,0,0],duration_s=.1));w.advance_to(1.4)
    assert np.linalg.norm(w.controllers[0].state.velocity)==0 and not w.auto_stops


def test_actor_absolute_geometry_and_unknown_configuration_restore():
    c=config();c['actors']=[dict(id='car',position_m=[3,0,1],size_m=[1,2,2],trajectory=[dict(time_s=0,position_m=[3,0,1]),dict(time_s=2,position_m=[3,1,1])])]
    w=PhysicalWorld(c);s=w.snapshot();w.advance_to(.4);first=w.state()['actors'][0]['vertices'];w.advance_to(1.);w.restore(s);w.advance_to(.4)
    assert np.array_equal(w.state()['actors'][0]['vertices'],first)
    other=PhysicalWorld(config(2))
    with pytest.raises(ValueError):other.restore(s)


def test_radio_grid_and_approximate_mode_rejected():
    c=config();c['radio']['num_subcarriers']=10000
    with pytest.raises(ValueError):validate(c)
    c=config();c['rt']['synthetic_array']=True
    with pytest.raises(ValueError):validate(c)
