"""Mechanical transactions and durable workflow tests; never native CSI claims."""
import json
from copy import deepcopy
import numpy as np
import pytest
from test_physics import config
from wiwo_ma.platform.physics import PhysicalWorld
from wiwo_ma.platform.storage import encode,CSIWriter
from wiwo_ma.platform.workflow import audit_closed,check_authorization


def batch(c):
    return [dict(bs_id='bs0',panel_id=p['id'],type='target_pose',position_m=(np.array(p['position_m'])+[.012 if i%2 else -.012,0,.004]).tolist(),orientation_deg=[.5*(-1 if i%2 else 1),.2,0]) for i,p in enumerate(c['base_stations'][0]['panels'])]


def test_all16_joint_motion_rigidity_speed_acceleration_and_branch():
    c=config(16);w=PhysicalWorld(c);initial=w.state();w.advance_to(.02)
    receipt=w.submit_many(batch(c));assert receipt['effective_time_s']==.1 and len(w.scheduled)==16
    ancestor=w.snapshot();v=[];omega=[]
    for tick in range(11,251):
        s=w.advance_to(tick*.002);v.append(s['velocities']);omega.append(s['angular_velocities'])
    expected=s['tx_port_positions'].copy();assert np.all(np.linalg.norm(s['positions']-initial['positions'],axis=1)>0)
    assert np.max(np.linalg.norm(np.diff(v,axis=0)/.002,axis=-1))<=.4+1e-8
    assert np.max(np.linalg.norm(np.diff(omega,axis=0)/.002,axis=-1))<=.6+1e-8
    for j in range(16):
        a=initial['tx_port_positions'][j*4:j*4+4];b=s['tx_port_positions'][j*4:j*4+4]
        np.testing.assert_allclose(np.linalg.norm(a[:,None]-a[None],axis=-1),np.linalg.norm(b[:,None]-b[None],axis=-1),atol=1.5e-6,rtol=0)
    w.restore(ancestor);w.advance_to(.5);np.testing.assert_array_equal(expected,w.state()['tx_port_positions'])


def test_joint_invalid_member_rolls_back_commands_and_pending_queue():
    c=config(3);w=PhysicalWorld(c);old=json.dumps(encode(w.snapshot()),sort_keys=True);q=batch(c);q[-1]['position_m']=[999,0,0]
    with pytest.raises(ValueError):w.submit_many(q)
    assert json.dumps(encode(w.snapshot()),sort_keys=True)==old
    with pytest.raises(ValueError,match='one command'):w.submit_many([q[0],q[0]])


def test_combined_crossing_paths_are_rejected_before_execution():
    c=config(2)
    for p in c['base_stations'][0]['panels']:p['mechanics'].update(workspace_min_m=[-1,-1,-1],workspace_max_m=[1,1,1])
    w=PhysicalWorld(c);q=batch(c);q[0]['position_m']=[0,.5,0];q[1]['position_m']=[0,0,0]
    with pytest.raises(ValueError,match='collision|clearance'):w.submit_many(q)
    assert not w.executed and w.time==0


def test_actor_is_a_mechanical_obstacle_not_only_a_display_mesh():
    c=config();c['actors']=[dict(id='obstacle',position_m=[0,0,3],size_m=[.5,.5,.5])]
    with pytest.raises(ValueError,match='Panel/actor collision'):PhysicalWorld(c)


def test_closed_CPU_audit_every_row_and_no_numeric_promotion(tmp_path,monkeypatch):
    import wiwo_ma.platform.workflow as module
    monkeypatch.setattr(module,'inside',lambda x:tmp_path)
    c=config(2);c['timing']['duration_s']=.04;c['actions']=[];w=PhysicalWorld(c);w.submit_many(batch(c));writer=CSIWriter(tmp_path,c)
    for i in range(3):
        w.advance_to(i*.01);s=w.state();r=dict(H=np.full((1,1,8,64),i+1j,np.complex64),physical_timestamp=s['timestamp'],label_source='test_fixture',independent_validation_status='Unknown',paths=[])
        writer.stage(r,s,w.snapshot());writer.commit_pending()
    writer.close();writer=CSIWriter(tmp_path,c,resume=True);w.restore(writer.latest_snapshot())
    for i in range(3,5):
        w.advance_to(i*.01);s=w.state();r.update(H=np.full((1,1,8,64),i+1j,np.complex64),physical_timestamp=s['timestamp']);writer.stage(r,s,w.snapshot());writer.commit_pending()
    writer.complete();writer.close();report=audit_closed(tmp_path)
    assert report['status']=='Passed' and report['frames']==5 and report['sources']=={'test_fixture':5} and report['recorded_labels']=={'Unknown':5}


def test_grant_matches_frozen_job_and_does_not_merge_ledgers(tmp_path,monkeypatch):
    import wiwo_ma.platform.workflow as module
    from wiwo_ma.platform.jobs import prepare_job
    import wiwo_ma.platform.jobs as jobs
    monkeypatch.setattr(jobs,'inside',lambda p:tmp_path/'job')
    c=config();prepare_job(c,'job',dict(max_wall_s=40,max_disk_bytes=100,max_allocator_bytes=100))
    grant=tmp_path/'grant.json';grant.write_text(json.dumps(dict(approved=True,scope='native_platform',configuration_sha256=[module.fingerprint(c)],total_wall_s=50,gpu=3,max_disk_bytes=100,max_allocator_bytes=100)))
    grant.with_suffix('.ledger.json').write_text(json.dumps(dict(spent_wall_s=15)))
    monkeypatch.setattr(module,'inside',lambda p:tmp_path/'job' if str(p)=='job' else grant)
    assert check_authorization('job','grant')['remaining_wall_s']==35
    q=json.loads(grant.read_text());q['configuration_sha256']=[];grant.write_text(json.dumps(q))
    with pytest.raises(ValueError,match='authorization'):check_authorization('job','grant')


def test_generator_frame_boundary_restart_preserves_history_and_labels(tmp_path,monkeypatch):
    """Exercise the real writer/job loop with an explicitly named CPU fixture."""
    import wiwo_ma.platform.jobs as jobs
    import wiwo_ma.platform.workflow as flow
    import wiwo_ma.platform.native_rt as native
    monkeypatch.setattr(jobs,'inside',lambda _:tmp_path/'job');monkeypatch.setattr(flow,'inside',lambda _:tmp_path/'job')
    monkeypatch.setattr(jobs.signal,'signal',lambda *_:None)
    monkeypatch.setattr(jobs.Budget,'check',lambda _:None)
    original_audit=flow.audit_closed
    def audit_after_closure(folder):
        # Terminal completion must not precede the closed-file audit.
        assert json.load(open(folder/'state.json'))['status']=='complete_native_generation_pending_CPU_closure'
        return original_audit(folder)
    monkeypatch.setattr(flow,'audit_closed',audit_after_closure)
    calls=[]
    class FixtureEngine:
        def __init__(self,*args,**kwargs):self.dr=None
        def solve(self,s):
            calls.append(s['timestamp'])
            return dict(H=np.full((1,1,4,64),s['timestamp']+1j,np.complex64),physical_timestamp=s['timestamp'],label_source='test_fixture',independent_validation_status='Unknown',paths=[],solver_wall_seconds=.01,allocator_peak_bytes=0)
        def close(self):pass
    monkeypatch.setattr(native,'NativeRT',FixtureEngine)
    c=config();c['timing']['duration_s']=.04
    folder=jobs.prepare_job(c,'job',dict(max_wall_s=40,max_disk_bytes=10*1024**2,max_allocator_bytes=100))
    jobs.generate('job',frame_limit=2)
    assert json.load(open(folder/'state.json'))['status']=='partial_requested_frame_limit'
    original=(folder/'source_versions.json').read_bytes();jobs.generate('job',resume=True)
    assert calls==[0,.01,.02,.03,.04]
    assert len(list((folder/'attempts').glob('*/source_versions.json')))==2 and (folder/'source_versions.json').read_bytes()==original
    audit=json.load(open(folder/'last_CPU_audit.json'))
    assert audit['status']=='Passed' and audit['recorded_labels']=={'Unknown':5} and audit['sources']=={'test_fixture':5}


@pytest.mark.parametrize('reference_value',[1e-12,0.])
def test_independent_comparator_keeps_weak_zero_failures_and_unknown_rows(tmp_path,monkeypatch,reference_value):
    import wiwo_ma.platform.continuous_validation as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    c=config();c['timing']['duration_s']=.01;w=PhysicalWorld(c);folder=tmp_path/'job';writer=CSIWriter(folder,c)
    for t in (0,.01):
        w.advance_to(t);s=w.state();r=dict(H=np.full((1,1,4,64),2e-12,np.complex64),physical_timestamp=t,label_source='test_fixture',independent_validation_status='Unknown',paths=[])
        writer.stage(r,s,w.snapshot());writer.commit_pending()
    writer.complete();writer.close()
    class Budget:
        def check(self):pass
    def independent(c,s,folder,check,complete_path_sink):
        assert 'H' not in s and 'paths' not in s
        return np.full((1,1,4,64),reference_value,np.complex64),dict(scope='test_fixture_only')
    monkeypatch.setattr(module,'reference',independent)
    ledger=module.validate_selected(folder,[0],Budget())
    assert ledger['counts']==dict(Passed=0,Failed=1,Unknown=1)
    assert ledger['rows'][0]['status']=='Failed'
    if reference_value==0:assert ledger['rows'][0]['per_user_NMSE']==[None] and ledger['rows'][0]['nonfinite_NMSE_users']==[0]
    else:assert ledger['rows'][0]['per_user_NMSE'][0]>.001
    from wiwo_ma.platform.storage import CSIReader
    assert CSIReader(folder/'csi.h5').frame(0)['result']['independent_validation_status']=='Unknown'
