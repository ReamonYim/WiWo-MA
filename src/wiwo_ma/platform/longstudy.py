"""Budgeted long native stream with early independent gates and automatic audit.

Planning reuses completed CPU mechanics, never repeats801 SDK RT. Fresh native
frames do not read reference H. Only the validator compares separate answers.
"""
import json,os,time,subprocess,signal,fcntl,hashlib
from pathlib import Path
import numpy as np
from .config import ROOT,load,fingerprint,estimated_storage
from .physics import PhysicalWorld
from .storage import CSIWriter,CSIReader,atomic_json,encode,decode,plain
from .jobs import inside,Budget,ResourceStop,folder_bytes
from .validation import sha,bake_actors,reference

CHECKS=[0,1,2,102,1500,1502,3000]


def prepare(folder):
    out=inside(folder)
    if out.exists():raise FileExistsError(out)
    c=load(ROOT/'configs/platform/munich64_k40_28ghz.json');old=ROOT/'artifacts/platform/mechanics30s_attempt01'
    report=json.loads((old/'report.json').read_text())
    if report['configuration_sha256']!=fingerprint(c) or sha(old/'mechanics.npz')!=report['raw_sha256']:raise ValueError('Closed mechanics preflight changed')
    out.mkdir(parents=True);atomic_json(out/'config.json',c)
    available=[(int(p.stem.split('_')[-1]),p) for p in old.glob('snapshot_*.json')]
    for i in CHECKS:
        _,path=max((x for x in available if x[0]<=i),key=lambda x:x[0]);w=PhysicalWorld(c);w.restore(decode(json.loads(path.read_text())));w.advance_to(i*c['timing']['csi_dt_s'])
        s=w.state();target=out/'references'/str(i);target.mkdir(parents=True)
        atomic_json(target/'state.json',s);bake_actors(s,target)
    atomic_json(out/'limits.json',dict(max_wall_s=36*3600.,max_allocator_bytes=12*1024**3,max_disk_bytes=8*1024**3))
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    source={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/wiwo_ma/platform').glob('*.py') if p.stem not in ['web','controller','delivery']}
    for name in ['scene.py','mechanics.py','v4/collision.py','v5/actor_mesh.py']:source['src/wiwo_ma/'+name]=sha(ROOT/'src/wiwo_ma'/name)
    p=dict(schema='native-long-platform-v1',status='prepared_not_started',configuration_sha256=fingerprint(c),gpu=3,
        limits=json.loads((out/'limits.json').read_text()),estimate=estimated_storage(c),expected_wall_hours=[18,30],
        timing_basis='One native64 query34.1805s, SDK-only reference21.1084s; first native query has cold JIT. No stable repeated-frame rate yet. Hard cap36h includes references/write/init; stop preserved if slower.',
        native_main_queries=3001,independent_reference_queries=len(CHECKS),branch_restoration_queries=2,total_Full64_queries=3010,
        original801_queries=0,reference_checks=CHECKS,original_NMSE_threshold=.001,
        validation='First3 consecutive same-engine frames independently checked; restore/query1→restore/query2 before continuing; each later checked mismatch stops. Unchecked labels remainUnknown.',
        inputs=files,source_sha256=source,CPU_mechanics=str(old.relative_to(ROOT)),CPU_mechanics_raw_sha256=report['raw_sha256'])
    atomic_json(out/'proposal.json',p);atomic_json(out/'state.json',dict(status='prepared_not_started',GPU_started=False));return p


def verify_inputs(folder,p):
    for name,h in p['inputs'].items():
        if sha(folder/name)!=h:raise ValueError('Frozen long input changed: '+name)
    for name,h in p['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen long source changed: '+name)


def nmse(a,b):
    x=np.asarray(a,np.complex128);y=np.asarray(b,np.complex128);num=np.sum(abs(x-y)**2,axis=(1,2,3));den=np.sum(abs(y)**2,axis=(1,2,3))
    return np.divide(num,den,out=np.where(num==0,0.,np.inf),where=den>0)


def error_fields(error):
    return dict(per_user_NMSE=[float(x) if np.isfinite(x) else None for x in error],
        nonfinite_error_users=np.flatnonzero(~np.isfinite(error)).tolist(),
        max_NMSE=float(max(error)) if np.isfinite(error).all() else None)


def worker(folder):
    from .native_rt import NativeRT
    folder=inside(folder);p=json.loads((folder/'proposal.json').read_text());verify_inputs(folder,p)
    c=load(folder/'config.json');limits=json.loads((folder/'effective_limits.json').read_text());budget=Budget(folder,limits);world=PhysicalWorld(c)
    original=np.load(ROOT/p['CPU_mechanics']/'mechanics.npz',allow_pickle=False)['tx_port_positions'];expected=p['estimate']['frames']
    writer=CSIWriter(folder/'dataset',c,resume=(folder/'dataset/csi.h5').exists());writer.commit_pending();snap=writer.latest_snapshot()
    if snap is not None:world.restore(snap)
    started=time.monotonic();engine=None;stop={'requested':False};times=[]
    signal.signal(signal.SIGTERM,lambda *_:stop.update(requested=True));signal.signal(signal.SIGINT,lambda *_:stop.update(requested=True))
    def state(status,**kw):
        value=dict(status=status,committed_frames=writer.count,expected_frames=expected,physical_time_s=world.time,elapsed_wall_s=time.monotonic()-started,**kw)
        atomic_json(folder/'state.json',value)
        with (folder/'events.jsonl').open('a') as f:f.write(json.dumps(plain(value))+'\n');f.flush()
    def check_reference(i):
        target=folder/'references'/str(i);report=target/'report.json'
        if report.exists():
            if json.loads(report.read_text())['status']!='Passed':raise ValueError('Preserved independent failure requires diagnosis, not automatic retry')
            return
        q=CSIReader(folder/'dataset/csi.h5').frame(i);s=json.loads((target/'state.json').read_text())
        for key in ['tx_port_positions','rx_port_positions','tx_port_orientations_rad','rx_port_orientations_rad']:
            if not np.array_equal(np.asarray(s[key],np.float32),np.asarray(q['state'][key],np.float32)):raise ValueError('Frozen reference/physical state disagreement')
        state('independent_reference',reference_frame=i)
        if not (target/'H.npz').exists():
            h,meta=reference(c,s,target,budget.check);np.savez_compressed(target/'H.npz',H=h);atomic_json(target/'SDK_metadata.json',meta)
        else:h=np.load(target/'H.npz',allow_pickle=False)['H']
        err=nmse(q['H'],h);r=dict(index=i,status='Passed' if np.all(err<=p['original_NMSE_threshold']) else 'Failed',**error_fields(err),reference_sha256=sha(target/'H.npz'))
        atomic_json(report,r)
        if r['status']!='Passed':raise ValueError('Original SDK independent accuracy Failed at'+str(i))
    def branch_check():
        if (folder/'branch_report.json').exists():
            if json.loads((folder/'branch_report.json').read_text())['status']!='Passed':raise ValueError('Preserved branch failure')
            return
        saved=world.snapshot();rows=[]
        for i in (1,2):
            q=CSIReader(folder/'dataset/csi.h5').frame(i);world.restore(decode(q['snapshot']));r=engine.solve(world.state())
            target=folder/('branch_'+str(i)+'.npz');np.savez_compressed(target,H=r['H']);h=np.load(folder/'references'/str(i)/'H.npz',allow_pickle=False)['H'];err=nmse(r['H'],h)
            rows.append(dict(frame=i,status='Passed' if np.all(err<=.001) else 'Failed',**error_fields(err),native_sha256=sha(target)))
        world.restore(saved);r=dict(status='Passed' if all(x['status']=='Passed' for x in rows) else 'Failed',rows=rows);atomic_json(folder/'branch_report.json',r)
        if r['status']!='Passed':raise ValueError('Restored same-engine CSI failed')
    try:
        if writer.count>=expected and (folder/'closure.json').exists():return
        state('initializing_native_SDK');engine=NativeRT(c,check_budget=budget.check);budget.engine=engine
        for i in p['reference_checks']:
            if i<writer.count:check_reference(i)
        if writer.count>=3:branch_check()
        while writer.count<expected:
            control=json.loads((folder/'control.json').read_text()) if (folder/'control.json').exists() else {}
            if stop['requested'] or control.get('kind') in ('pause','stop'):
                state('paused' if control.get('kind')=='pause' else 'stopped_partial');return
            budget.check();i=writer.count;world.advance_to(i*c['timing']['csi_dt_s']);s=world.state()
            if not np.array_equal(np.asarray(s['tx_port_positions'],np.float32).view(np.uint32),original[i].view(np.uint32)):raise ValueError('Previously frozen CPU mechanics differs at'+str(i))
            state('solving_native_frame',frame=i);r=engine.solve(s)
            if folder_bytes(folder)+r['H'].nbytes*4+2*1024**2>limits['max_disk_bytes']:raise ResourceStop('Space reserve for next frame reached')
            writer.stage(r,s,world.snapshot());writer.commit_pending();times.append(r['solver_wall_seconds'])
            atomic_json(folder/'progress.json',dict(committed_frames=writer.count,expected_frames=expected,physical_time_s=world.time,
                solver_last_wall_s=times[-1],solver_mean_this_process_wall_s=float(np.mean(times)),estimated_remaining_wall_s=float(np.mean(times))*(expected-writer.count),
                allocator_peak_bytes=r['allocator_peak_bytes'],disk_bytes=folder_bytes(folder),independent_validated_frames=[j for j in CHECKS if (folder/'references'/str(j)/'report.json').exists()],
                unverified_frames='Unknown',physical_clock='Fixed10ms CSI;2ms mechanics,100ms controls; no wall-time warping'))
            state('frame_committed')
            if i in CHECKS:check_reference(i)
            if i==2:branch_check()
        writer.complete();state('complete_native_generation_pending_CPU_closure')
    except BaseException as exc:state('failed_or_resource_preserved_no_automatic_retry',error=repr(exc));raise
    finally:
        writer.close()
        if engine is not None:engine.close()


def close_cpu(folder):
    """Always runs after worker exits; incomplete data stay Partial/Unknown."""
    folder=inside(folder);data=folder/'dataset/csi.h5'
    if not data.exists():atomic_json(folder/'closure.json',dict(status='Partial_no_dataset',overall='Partial'));return
    reader=CSIReader(data);p=json.loads((folder/'proposal.json').read_text());counts=dict(Passed=0,Partial=0,Failed=0,Unknown=0);ledger=[]
    for i in range(reader.count):
        q=reader.frame(i,verify=True);status='Unknown';err=None;target=folder/'references'/str(i)/'H.npz'
        if target.exists():
            h=np.load(target,allow_pickle=False)['H'];err=nmse(q['H'],h);status='Passed' if np.all(err<=.001) else 'Failed'
        counts[status]+=1;ledger.append(dict(index=i,physical_time_s=q['state']['timestamp'],status=status,**({} if err is None else error_fields(err))))
    atomic_json(folder/'label_ledger.json',dict(counts=counts,frames=ledger,uncomputed_frame_count=p['estimate']['frames']-reader.count,uncomputed_status='Unknown'))
    result=dict(status='Failed_selected_SDK_checks' if counts['Failed'] else 'Passed_storage_and_selected_SDK_checks' if reader.complete and counts['Passed']==len(CHECKS) else 'Partial',
        committed_frames=reader.count,expected_frames=p['estimate']['frames'],complete_HDF=reader.complete,counts=counts,overall='Partial',
        all_unchecked_labels='Unknown_not_promoted',new_GPU_seconds_by_CPU_closure=0)
    atomic_json(folder/'closure.json',result)
    state=json.loads((folder/'state.json').read_text());state['CPU_closure']=result
    if reader.complete:state['status']='complete_native_dataset_with_selected_audit' if counts['Failed']==0 else 'failed_original_SDK_numerical_preserved'
    atomic_json(folder/'state.json',state)


def supervise(folder,grant):
    folder=inside(folder);grant=inside(grant);p=json.loads((folder/'proposal.json').read_text());verify_inputs(folder,p);a=json.loads(grant.read_text())
    if a.get('approved') is not True or a.get('scope')!='native_platform_long30s' or a.get('proposal_sha256')!=sha(folder/'proposal.json'):raise ValueError('Explicit independent long30s authorization required')
    if a['gpu']!=p['gpu'] or any(a[k]!=p['limits'][k] for k in p['limits']):raise ValueError('Resource caps outside proposed scope')
    lock=grant.with_suffix('.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ledger=grant.with_suffix('.ledger.json')
    billed=json.loads(ledger.read_text()) if ledger.exists() else dict(spent_wall_s=0.,runs=[]);remaining=a['max_wall_s']-billed['spent_wall_s']
    if remaining<30:raise ResourceStop('Long grant exhausted')
    limits={**p['limits'],'max_wall_s':remaining};atomic_json(folder/'effective_limits.json',limits)
    guard=(ROOT/'.state/platform_gpu3.lock').open('a+b');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy();env.update(WIWO_GPU=str(a['gpu']),CUDA_VISIBLE_DEVICES=str(a['gpu']),WIWO_LONG_GATE_TOKEN=sha(folder/'proposal.json'))
    log=(folder/'worker.log').open('ab');start=time.monotonic();proc=subprocess.Popen([str(ROOT/'scripts/platform/run'),'long-worker','--job',str(folder)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(folder/'worker_pid.json',dict(pid=proc.pid,gpu=a['gpu'],hard_remaining_wall_s=remaining));forced=False
    try:
        while proc.poll() is None:
            if time.monotonic()-start>=remaining or folder_bytes(folder)>a['max_disk_bytes']:
                os.killpg(proc.pid,signal.SIGKILL);forced=True;break
            time.sleep(1)
        code=proc.wait()
    finally:
        elapsed=time.monotonic()-start;billed['spent_wall_s']+=elapsed;billed['runs'].append(dict(elapsed_wall_s=elapsed,returncode=proc.returncode,forced_kill=forced));atomic_json(ledger,billed);log.close();lock.close();guard.close()
    if forced:atomic_json(folder/'state.json',dict(status='interrupted_resource_limit_preserved',no_automatic_retry=True))
    # Separate CPU process: no CUDA/RT initialization and no new labels.
    result=subprocess.run([str(ROOT/'scripts/platform/run'),'long-close','--job',str(folder)],cwd=ROOT,env={**os.environ,'WIWO_GPU':'-1'},stdout=(folder/'CPU_closure.log').open('ab'),stderr=subprocess.STDOUT)
    atomic_json(folder/'supervisor_result.json',dict(worker_returncode=code,CPU_closure_returncode=result.returncode,overall='Partial'))
    return code
