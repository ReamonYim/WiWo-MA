"""CPU fault injection, real process exit and durable writes; not RT evidence."""
from concurrent.futures import ThreadPoolExecutor
import errno,json,os,subprocess,sys
from copy import deepcopy
import numpy as np
import pytest
from wiwo_ma.platform.session_control import ControlLease
from wiwo_ma.platform.storage import CSIWriter,CSIReader
from wiwo_ma.platform.controller import PlatformController
from wiwo_ma.platform.physics import PhysicalWorld
from wiwo_ma.platform.examples import reflector_example


def config():
    c=reflector_example();c.update(actions=[],actors=[]);c['timing']['duration_s']=.02
    for u in c['users']:u['trajectory']=[]
    return c


def frame(w):
    s=w.state();return dict(H=np.ones((1,1,4,64),np.complex64),physical_timestamp=s['timestamp'],label_source='test_fixture',independent_validation_status='Unknown'),s,w.snapshot()


def test_two_operators_serialize_claim_stale_revision_and_disconnect_never_touches_job(tmp_path):
    lease=ControlLease(tmp_path/'events.jsonl');lease.connect(1);lease.connect(2)
    with pytest.raises(PermissionError):lease.require(2)
    old=lease.revision;lease.mutation(1,'edit')
    with pytest.raises(ValueError,match='Stale'):lease.require(1,old)
    with pytest.raises(PermissionError):lease.claim(2)
    lease.disconnect(1);assert lease.owner is None and lease.view()['connected_clients']==[2]
    lease.claim(2);lease.release(2)
    def claim(i):
        try:lease.claim(i);return True
        except PermissionError:return False
    with ThreadPoolExecutor(2) as pool:assert sum(pool.map(claim,[3,4]))==1
    assert all('api:' not in line for line in (tmp_path/'events.jsonl').read_text().splitlines())


def test_watch_only_can_neither_pause_nor_stop_running_frozen_worker(tmp_path,monkeypatch):
    import wiwo_ma.platform.controller as module
    c=config();(tmp_path/'config.json').write_text(json.dumps(c));(tmp_path/'state.json').write_text(json.dumps(dict(status='solving_native_frame')))
    monkeypatch.setattr(module,'inside',lambda _:tmp_path);monkeypatch.setattr(module,'ROOT',tmp_path.parent)
    x=PlatformController(c)
    try:
        x.command('attach_job',dict(path=tmp_path.name))
        for command in ('pause','stop','start','resume'):
            assert x.command(command)['watching_native_job']
            assert not (tmp_path/'control.json').exists()
        with pytest.raises(ValueError):x.configure(deepcopy(c))
    finally:x.close()


def test_disk_full_during_pending_write_preserves_committed_HDF_and_recovers(tmp_path,monkeypatch):
    from pathlib import Path
    c=config();w=PhysicalWorld(c);writer=CSIWriter(tmp_path,c);writer.stage(*frame(w));writer.commit_pending();w.advance_to(.01)
    old_sha=writer.file['frame_sha256'].asstr()[0];real=Path.open
    def disk_full(path,*args,**kwargs):
        if path.name=='pending_frame.tmp':raise OSError(errno.ENOSPC,'Injected disk full')
        return real(path,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'open',disk_full)
        with pytest.raises(OSError) as e:writer.stage(*frame(w))
        assert e.value.errno==errno.ENOSPC and writer.count==1
    writer.close();writer=CSIWriter(tmp_path,c,resume=True);writer.stage(*frame(w));writer.commit_pending();writer.complete();writer.close()
    r=CSIReader(tmp_path/'csi.h5');assert r.complete and r.count==2 and r.frame(0)['frame_sha256']==old_sha


def test_worker_actual_abnormal_exit_is_billed_and_becomes_failed_not_stale_running(tmp_path,monkeypatch):
    import wiwo_ma.platform.jobs as jobs
    import wiwo_ma.platform.workflow as flow
    folder=tmp_path/'job';folder.mkdir();grant=tmp_path/'grant.json'
    c=config();(folder/'config.json').write_text(json.dumps(c))
    (folder/'limits.json').write_text(json.dumps(dict(max_wall_s=20,max_disk_bytes=1000,max_allocator_bytes=1000)))
    (folder/'state.json').write_text(json.dumps(dict(status='solving_native_frame',committed_frames=1)))
    grant.write_text(json.dumps(dict(approved=True,scope='native_platform',configuration_sha256=[jobs.fingerprint(c)],gpu=3,total_wall_s=20,max_allocator_bytes=1000,max_disk_bytes=1000)))
    monkeypatch.setattr(jobs,'inside',lambda p:grant if 'grant' in str(p) else folder);monkeypatch.setattr(jobs,'ROOT',tmp_path)
    monkeypatch.setattr(flow,'check_authorization',lambda *_:True)
    real=subprocess.Popen
    def child(*args,**kwargs):return real([sys.executable,'-c','import os; os._exit(7)'],**kwargs)
    monkeypatch.setattr(jobs.subprocess,'Popen',child)
    assert jobs.supervise(folder,grant)==7
    state=json.loads((folder/'state.json').read_text());ledger=json.loads(grant.with_suffix('.ledger.json').read_text())
    assert state['status']=='failed_worker_exit_preserved' and state['committed_frames']==1 and state['no_automatic_retry']
    assert len(ledger['runs'])==1 and ledger['runs'][0]['returncode']==7 and ledger['spent_wall_s']>0


def test_committed_pending_duplicate_recovery_never_appends_another_row(tmp_path):
    c=config();w=PhysicalWorld(c);writer=CSIWriter(tmp_path,c);writer.stage(*frame(w));saved=writer.pending.read_bytes();writer.commit_pending();writer.pending.write_bytes(saved)
    assert writer.commit_pending() and writer.count==1
    writer.close();assert CSIReader(tmp_path/'csi.h5').count==1


def test_filesystem_low_space_guard_stops_before_SDK_initialization(tmp_path,monkeypatch):
    import shutil
    from types import SimpleNamespace
    from wiwo_ma.platform.jobs import Budget,ResourceStop
    b=Budget(tmp_path,dict(max_wall_s=60,max_disk_bytes=10000,max_allocator_bytes=10000))
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    with pytest.raises(ResourceStop,match='Filesystem free space'):b.check()
