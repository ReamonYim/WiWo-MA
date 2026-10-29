"""CPU-only job admission and closed-file checks, shared by Web and CLI.

Preparing a job never grants permission to run CUDA. The preparation process
survives the browser, and failures and frozen requests remain on disk.
"""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
import numpy as np
from .config import ROOT,validate,fingerprint,estimated_storage
from .storage import atomic_json,CSIReader,encode,decode
from .jobs import inside


def prepare_background(config,folder,limits,*,initial_snapshot=None,current_reference=False):
    c=validate(config);folder=inside(folder)
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True)
    request=dict(config=c,limits=limits,configuration_sha256=fingerprint(c))
    if initial_snapshot is not None:request.update(initial_snapshot=encode(initial_snapshot),current_reference=bool(current_reference))
    atomic_json(folder/'request.json',request)
    atomic_json(folder/'state.json',dict(status='preparing_CPU_admission',GPU_started=False,configuration_sha256=fingerprint(c)))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',WIWO_GPU='-1',PYTHONPATH=str(ROOT/'src'),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    with (folder/'preparation.log').open('ab') as log:
        proc=subprocess.Popen([sys.executable,'-m','wiwo_ma.platform.cli','prepare-worker','--job',str(folder)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(folder/'preparation_pid.json',dict(pid=proc.pid,CPU_only=True))
    return folder


def prepare_worker(folder):
    from .jobs import prepare_job
    folder=inside(folder);request=json.loads((folder/'request.json').read_text())
    try:
        if fingerprint(request['config'])!=request['configuration_sha256']:raise ValueError('Preparation request fingerprint mismatch')
        prepare_job(request['config'],folder,request['limits'],reserved=True,
                    initial_snapshot=decode(request['initial_snapshot']) if 'initial_snapshot' in request else None,current_reference=request.get('current_reference',False))
    except BaseException as exc:
        atomic_json(folder/'state.json',dict(status='failed_CPU_preparation_preserved',error=repr(exc),GPU_started=False))
        raise


def check_authorization(folder,grant):
    """Validate exact frozen job without launching or charging any device."""
    folder=inside(folder);grant=inside(grant);q=json.loads(grant.read_text())
    config=json.loads((folder/'config.json').read_text());plan=json.loads((folder/'plan.json').read_text());limits=json.loads((folder/'limits.json').read_text())
    frozen=folder/'frozen_sources.json'
    if frozen.exists():
        for name,expected in json.loads(frozen.read_text()).items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen execution source changed; prepare a new reviewed proposal: '+name)
    if plan['configuration_sha256']!=fingerprint(config):raise ValueError('Prepared configuration changed')
    if plan.get('query_mode')=='current_state':
        initial=folder/'initial_snapshot.json'
        if not initial.exists() or hashlib.sha256(initial.read_bytes()).hexdigest()!=plan['initial_snapshot_sha256']:raise ValueError('Frozen current-state snapshot changed')
        allowed=q.get('initial_snapshot_sha256',[])
        if isinstance(allowed,str):allowed=[allowed]
        if plan['initial_snapshot_sha256'] not in allowed:raise ValueError('Current-state authorization must explicitly bind the initial snapshot SHA')
    inputs=folder/'frozen_inputs.json'
    if inputs.exists():
        for name,expected in json.loads(inputs.read_text()).items():
            file=(folder/name).resolve()
            if not file.is_relative_to(folder) or hashlib.sha256(file.read_bytes()).hexdigest()!=expected:raise ValueError('Frozen prepared input changed: '+name)
    if q.get('approved') is not True or q.get('scope')!='native_platform' or fingerprint(config) not in q.get('configuration_sha256',[]):raise ValueError('No explicit authorization for this frozen native-platform configuration')
    if q.get('job_path') and inside(q['job_path'])!=folder:raise ValueError('Authorization belongs to another job')
    if q.get('job_paths') and folder not in [inside(p) for p in q['job_paths']]:raise ValueError('Authorization belongs to other explicitly reviewed jobs')
    for key in ('max_disk_bytes','max_allocator_bytes'):
        if limits[key]>q[key]:raise ValueError('Prepared resource cap exceeds authorization')
    ledger=grant.with_suffix('.ledger.json');spent=json.loads(ledger.read_text()).get('spent_wall_s',0) if ledger.exists() else 0
    remaining=float(q['total_wall_s'])-spent
    if remaining<=10:raise ValueError('Independent authorization exhausted')
    return dict(approved=True,gpu=q['gpu'],remaining_wall_s=remaining,effective_wall_cap_s=min(limits['max_wall_s'],remaining),configuration_sha256=fingerprint(config))


def audit_closed(folder):
    """Traverse every committed row. This is NOT an SDK numerical reference."""
    folder=inside(folder);reader=CSIReader(folder/'csi.h5');start=time.perf_counter();bad=[];sources={};labels={};actions=set();moved=set();first=None;last=None
    c=reader.config;dt=c['timing']['csi_dt_s'];t=reader.timestamps
    if len(t)>1 and not np.allclose(np.diff(t),dt,rtol=0,atol=1e-8):bad.append(dict(type='time_grid'))
    for i in range(reader.count):
        try:
            q=reader.frame(i);s=q['state'];h=q['H'];last=s
            if first is None:first=s
            if not np.isfinite(h).all():raise ValueError('Nonfinite CSI')
            if abs(s['timestamp']-t[i])>1e-9 or s['mechanics_tick']*c['timing']['mechanics_dt_s']!=s['timestamp']:raise ValueError('State/clock mismatch')
            if s['configuration_sha256']!=fingerprint(c):raise ValueError('State configuration mismatch')
            from .physics import PhysicalWorld
            from .storage import decode
            w=PhysicalWorld(c);w.restore(decode(q['snapshot']));actual=w.state()
            for key in ('tx_port_positions','tx_port_orientations_rad','rx_port_positions','positions','rotations','user_positions'):
                if not np.array_equal(np.asarray(actual[key]),np.asarray(s[key])):raise ValueError('Snapshot mismatch: '+key)
            if len(actual['actors'])!=len(s['actors']):raise ValueError('Actor count mismatch')
            for a,b in zip(actual['actors'],s['actors']):
                if a['id']!=b['id'] or not np.array_equal(a['vertices'],b['vertices']):raise ValueError('Actor geometry mismatch')
            moved.update(j for j,(a,b) in enumerate(zip(first['positions'],s['positions'])) if np.linalg.norm(np.asarray(a)-b)>1e-7 or np.linalg.norm(np.asarray(first['rotations'][j])-s['rotations'][j])>1e-7)
            for a in decode(q['snapshot'])['executed']:actions.add(json.dumps(a,sort_keys=True))
            for key,d in [('label_source',sources),('independent_validation_status',labels)]:
                value=q['result'][key];d[value]=d.get(value,0)+1
            for chunk in q['result'].get('complete_path_files',[]):
                p=folder/chunk['path']
                if not p.resolve().is_relative_to(folder) or hashlib.sha256(p.read_bytes()).hexdigest()!=chunk['sha256']:raise ValueError('Complete path chunk binding mismatch')
        except Exception as exc:bad.append(dict(frame=i,error=repr(exc)))
    plan=json.loads((folder/'plan.json').read_text()) if (folder/'plan.json').exists() else {}
    expected=1 if plan.get('query_mode')=='current_state' else estimated_storage(c)['frames']
    status='Failed' if bad else 'Passed' if reader.complete and reader.count==expected else 'Partial'
    result=dict(status=status,scope='CPU storage, state reconstruction and clock checks only; no fresh SDK validation',frames=reader.count,expected_frames=expected,
        complete=reader.complete,first_time_s=None if not len(t) else float(t[0]),last_time_s=None if not len(t) else float(t[-1]),issues=bad,
        sources=sources,recorded_labels=labels,panels_observed_moving=sorted(moved),distinct_executed_actions=len(actions),elapsed_wall_s=time.perf_counter()-start,
        independent_SDK_numerical_status='Unknown_unless_separately_recomputed',original_HDF_unchanged=True)
    out=folder/'audits'/('closed_CPU_'+str(time.time_ns())+'.json');out.parent.mkdir(exist_ok=True);atomic_json(out,result)
    return result


def freeze_closed_delivery(folder,budget):
    """Small-job immutable copy after writer/audits close; active logs excluded."""
    import shutil
    from .jobs import folder_bytes,ResourceStop
    folder=Path(folder);files=[p for p in folder.rglob('*') if p.is_file() and 'deliveries' not in p.relative_to(folder).parts and p.suffix not in ('.log','.lock','.tmp') and p.name not in ('events.jsonl',)]
    if folder_bytes(folder)+sum(p.stat().st_size for p in files)>budget.limits['max_disk_bytes']:raise ResourceStop('Reserve for closed delivery copy exceeds storage budget')
    out=folder/'deliveries'/str(time.time_ns());out.mkdir(parents=True)
    for file in files:
        dest=out/'frozen'/file.relative_to(folder);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
    hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'frozen').rglob('*') if p.is_file()}
    atomic_json(out/'manifest.json',dict(status='Partial',scope='Closed job storage and checked SDK rows only; unchecked labels remain Unknown',files=hashes))
    changed=[n for n,h in hashes.items() if hashlib.sha256((out/n).read_bytes()).hexdigest()!=h]
    atomic_json(out/'verification.json',dict(status='Passed' if not changed else 'Failed',changed_files=changed,files=len(hashes)))
    return out
