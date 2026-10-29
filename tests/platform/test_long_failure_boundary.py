"""CPU orchestration fixture, never native CSI/physics acceptance evidence."""
import json
import numpy as np
import pytest
from wiwo_ma.platform.config import estimated_storage
from wiwo_ma.platform.examples import reflector_example
from wiwo_ma.platform.physics import PhysicalWorld
from wiwo_ma.platform.storage import atomic_json,CSIReader


def test_independent_failure_halts_long_job_preserves_frame_and_unknown_future(tmp_path,monkeypatch):
    import wiwo_ma.platform.longstudy as module
    import wiwo_ma.platform.native_rt as native
    c=reflector_example();c['actions']=[];c['timing']['duration_s']=.02;w=PhysicalWorld(c);s=w.state();shape=estimated_storage(c)
    folder=tmp_path/'job';folder.mkdir();prior=tmp_path/'CPU_only';prior.mkdir()
    np.savez(prior/'mechanics.npz',tx_port_positions=np.tile(s['tx_port_positions'],(3,1,1)))
    atomic_json(folder/'config.json',c);atomic_json(folder/'effective_limits.json',dict(max_wall_s=60,max_disk_bytes=1024**3,max_allocator_bytes=12*1024**3))
    atomic_json(folder/'proposal.json',dict(inputs={},source_sha256={},CPU_mechanics=str(prior),estimate=shape,reference_checks=[0],original_NMSE_threshold=.001))
    target=folder/'references/0';target.mkdir(parents=True);atomic_json(target/'state.json',s)
    calls=[];H=np.ones((1,1,4,64),np.complex64)
    class FixtureRT:
        def __init__(self,*args,**kwargs):pass
        def solve(self,state):
            calls.append(state['timestamp']);return dict(H=H,physical_timestamp=state['timestamp'],label_source='test_fixture',independent_validation_status='Unknown',solver_wall_seconds=0.,allocator_peak_bytes=0)
        def close(self):pass
    monkeypatch.setattr(native,'NativeRT',FixtureRT);monkeypatch.setattr(module,'inside',lambda _:folder);monkeypatch.setattr(module.Budget,'check',lambda _:None)
    monkeypatch.setattr(module,'CHECKS',[0]);monkeypatch.setattr(module,'reference',lambda *_:(np.zeros_like(H),dict(explicit_scope='CPU_fault_injection_not_RT')))
    with pytest.raises(ValueError,match='accuracy Failed'):module.worker(folder)
    assert calls==[0.] and CSIReader(folder/'dataset/csi.h5').count==1
    failed=json.loads((target/'report.json').read_text());assert failed['status']=='Failed' and failed['nonfinite_error_users']==[0]
    assert failed['per_user_NMSE']==[None]
    module.close_cpu(folder);ledger=json.loads((folder/'label_ledger.json').read_text())
    assert ledger['counts']['Failed']==1 and ledger['uncomputed_frame_count']==2 and ledger['uncomputed_status']=='Unknown'
    assert json.loads((folder/'closure.json').read_text())['status']=='Failed_selected_SDK_checks'
