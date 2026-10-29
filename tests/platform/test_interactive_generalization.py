"""CPU provenance and workflow tests; fixture arrays are never native CSI."""
from copy import deepcopy
import fcntl
import json
import numpy as np
import pytest
from wiwo_ma.platform.controller import PlatformController
from wiwo_ma.platform.examples import reflector_example
from wiwo_ma.platform.config import fingerprint
from wiwo_ma.platform.state_identity import geometry_digest


def simple():
    c=reflector_example();c.update(actions=[],actors=[])
    for u in c['users']:u['trajectory']=[]
    return c


@pytest.mark.parametrize('operand',['tx_port_positions','rx_port_positions','rx_port_orientations_rad','tx_port_map','actors'])
def test_CSI_identity_tracks_every_SDK_operand(operand):
    x=PlatformController(simple())
    try:
        s=x.state();x.result=dict(physical_timestamp=s['timestamp'],solver_wall_seconds=0,
            _geometry_sha256=geometry_digest(s))
        assert x.snapshot()[2]['current_CSI']=='Computed_at_current_pose'
        changed=deepcopy(s)
        if operand=='actors':changed['actors']=[dict(id='new',material='metal',vertices=[[0,0,0]])]
        else:changed[operand].flat[0]+=.001 if 'map' not in operand else 1
        x.world.state=lambda:changed
        assert x.snapshot()[2]['current_CSI']=='Unknown_current_pose'
    finally:x.close()


def test_same_time_UE_edit_clears_CSI_and_old_authorization():
    x=PlatformController(simple(),authorization='experiments/nonexistent_not_used.json')
    try:
        x.result=dict(physical_timestamp=0.,_geometry_sha256=geometry_digest(x.state()))
        c=deepcopy(x.config);c['users'][0]['position_m'][0]+=.1;x.configure(c)
        assert x.result is None and x.authorization is None and x.last_action is None
        assert x.snapshot()[2]['current_CSI']=='Unknown_current_pose'
    finally:x.close()


def test_current_pose_is_frozen_in_async_request_and_new_preparation_cannot_duplicate(tmp_path,monkeypatch):
    import wiwo_ma.platform.workflow as flow
    import wiwo_ma.platform.controller as controller
    monkeypatch.setattr(controller,'ROOT',tmp_path)
    seen={}
    def prepare(c,folder,limits,**kw):seen.update(config=c,**kw);return tmp_path
    monkeypatch.setattr(flow,'prepare_background',prepare)
    x=PlatformController(simple())
    try:
        x.action(dict(bs_id='bs0',panel_id='panel0',type='target_pose',position_m=[.01,0,0],orientation_deg=[.2,0,0]))
        x.command('step',dict(seconds=.4));state=x.state()
        x.command('prepare_current',dict(path='irrelevant',max_wall_s=60,max_disk_bytes=1000,max_allocator_bytes=1000))
        from wiwo_ma.platform.physics import PhysicalWorld
        w=PhysicalWorld(seen['config']);w.restore(seen['initial_snapshot'])
        np.testing.assert_array_equal(w.state()['tx_port_positions'],state['tx_port_positions'])
        assert w.time==.4 and x.job_kind=='current_state'
        with pytest.raises(ValueError,match='detach'):x.command('prepare_current',dict())
    finally:x.close()


def test_current_pose_requires_snapshot_specific_grant_and_detects_tampering(tmp_path,monkeypatch):
    import wiwo_ma.platform.jobs as jobs
    import wiwo_ma.platform.workflow as flow
    from wiwo_ma.platform.physics import PhysicalWorld
    c=simple();w=PhysicalWorld(c);w.advance_to(.2)
    folder=tmp_path/'job';grant=tmp_path/'grant.json'
    monkeypatch.setattr(jobs,'inside',lambda _:folder)
    jobs.prepare_job(c,folder,dict(max_wall_s=60,max_disk_bytes=1000,max_allocator_bytes=1000),initial_snapshot=w.snapshot())
    plan=json.loads((folder/'plan.json').read_text())
    assert plan['query_mode']=='current_state' and plan['physical_time_s']==.2
    q=dict(approved=True,scope='native_platform',configuration_sha256=[fingerprint(c)],total_wall_s=60,gpu=3,max_disk_bytes=1000,max_allocator_bytes=1000)
    monkeypatch.setattr(flow,'inside',lambda p:grant if p=='grant' else folder)
    grant.write_text(json.dumps(q))
    with pytest.raises(ValueError,match='explicitly bind'):flow.check_authorization('job','grant')
    q['initial_snapshot_sha256']=plan['initial_snapshot_sha256'];grant.write_text(json.dumps(q))
    assert flow.check_authorization('job','grant')['approved']
    with (folder/'initial_snapshot.json').open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='snapshot changed'):flow.check_authorization('job','grant')


@pytest.mark.parametrize('reference_failed',[False,True])
def test_single_pose_generator_uses_nonzero_saved_pose_once_and_keeps_Unknown(tmp_path,monkeypatch,reference_failed):
    import wiwo_ma.platform.jobs as jobs
    import wiwo_ma.platform.workflow as flow
    import wiwo_ma.platform.native_rt as native
    from wiwo_ma.platform.physics import PhysicalWorld
    from wiwo_ma.platform.storage import CSIReader
    folder=tmp_path/'job';monkeypatch.setattr(jobs,'inside',lambda _:folder);monkeypatch.setattr(flow,'inside',lambda _:folder)
    monkeypatch.setattr(jobs.signal,'signal',lambda *_:None);monkeypatch.setattr(jobs.Budget,'check',lambda _:None)
    calls=[]
    class Fixture:
        def __init__(self,*a,**k):pass
        def solve(self,s):
            calls.append(s['timestamp'])
            return dict(H=np.ones((1,1,4,64),np.complex64),physical_timestamp=s['timestamp'],solver_wall_seconds=0,allocator_peak_bytes=0,label_source='test_fixture',independent_validation_status='Unknown')
        def close(self):pass
    monkeypatch.setattr(native,'NativeRT',Fixture)
    c=simple();w=PhysicalWorld(c);w.advance_to(.4)
    jobs.prepare_job(c,folder,dict(max_wall_s=60,max_disk_bytes=10*1024**2,max_allocator_bytes=1000),initial_snapshot=w.snapshot(),current_reference=reference_failed)
    if reference_failed:
        import wiwo_ma.platform.continuous_validation as validation
        def failed(*args,**kwargs):return dict(counts=dict(Passed=0,Failed=1,Unknown=0),path_counts=dict(Unknown=1))
        monkeypatch.setattr(validation,'validate_selected',failed)
        with pytest.raises(ValueError,match='gate failed'):jobs.generate(folder)
        assert json.loads((folder/'state.json').read_text())['status']=='failed_preserved'
    else:jobs.generate(folder)
    r=CSIReader(folder/'csi.h5');assert r.complete and r.count==1 and calls==[.4]
    assert r.frame(0)['result']['label_source']=='test_fixture'
    if not reference_failed:assert json.loads((folder/'last_CPU_audit.json').read_text())['recorded_labels']=={'Unknown':1}


def test_launch_rejects_a_grant_still_owned_by_supervisor(tmp_path,monkeypatch):
    import wiwo_ma.platform.jobs as jobs
    import wiwo_ma.platform.workflow as flow
    folder=tmp_path/'job';folder.mkdir();grant=tmp_path/'grant.json';grant.write_text('{}')
    monkeypatch.setattr(jobs,'inside',lambda p:grant if p=='grant' else folder)
    monkeypatch.setattr(flow,'check_authorization',lambda *a:True)
    with grant.with_suffix('.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(jobs.ResourceStop,match='still supervising'):jobs.launch_job('job','grant')


def test_running_task_rejects_config_edit_and_action_without_mutation(tmp_path):
    x=PlatformController(simple())
    try:
        x.job=tmp_path;x.job_kind='generate';(tmp_path/'state.json').write_text(json.dumps(dict(status='solving_native_frame')))
        before=geometry_digest(x.state());c=deepcopy(x.config);c['users'][0]['position_m'][0]+=.1
        with pytest.raises(ValueError,match='Pause'):x.configure(c)
        with pytest.raises(ValueError,match='frozen'):x.action(dict())
        assert geometry_digest(x.state())==before
    finally:x.close()


def test_timing_risks_are_reported_without_promoting_legacy_sampling():
    from wiwo_ma.platform.resource_estimate import sampling_assessment
    c=reflector_example();q=sampling_assessment(c)
    assert q['status']=='Unknown' and q['timing_diagnostics']['legacy_velocity_jumps']
    assert q['timing_diagnostics']['every_control_boundary_on_CSI_grid']
    c=simple();c['timing'].update(csi_dt_s=.02,control_dt_s=.03)
    q=sampling_assessment(c);assert not q['timing_diagnostics']['every_control_boundary_on_CSI_grid']
    assert q['timing_diagnostics']['integration_accuracy'].startswith('Unknown')


def test_dual_polarization_reference_does_not_call_single_polarization_path_export(tmp_path,monkeypatch):
    import wiwo_ma.platform.continuous_validation as validation
    from wiwo_ma.platform.physics import PhysicalWorld
    from wiwo_ma.platform.storage import CSIWriter
    c=simple();c['radio'].update(tx_polarization='VH',rx_polarization='cross');w=PhysicalWorld(c);s=w.state()
    H=np.ones((1,2,8,64),np.complex64);writer=CSIWriter(tmp_path,c)
    writer.stage(dict(H=H,label_source='test_fixture',independent_validation_status='Unknown',physical_timestamp=0.),s,w.snapshot());writer.commit_pending();writer.complete();writer.close()
    class Budget:
        def check(self):pass
    def reference(config,state,dest,check,complete_path_sink):
        assert complete_path_sink is None
        return np.ones((1,2,8,64),np.complex64),dict(scope='CPU_fixture_not_SDK')
    monkeypatch.setattr(validation,'ROOT',tmp_path);monkeypatch.setattr(validation,'reference',reference)
    result=validation.validate_selected(tmp_path,[0],Budget())
    assert result['counts']['Passed']==1 and result['path_counts']['Unknown']==1
