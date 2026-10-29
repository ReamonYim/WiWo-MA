"""Durable native jobs, independent resource grants, bounded background owner.

Long generation is explicit. No old V5 research grant is read or borrowed.
Workers run independently of the UI/Codex session. Pause/stop occurs at a
committed frame; an exhausted/killed solve leaves the preceding state intact.
"""
from copy import deepcopy
import fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
from .config import ROOT,validate,fingerprint,estimated_storage
from .physics import PhysicalWorld
from .storage import CSIWriter,atomic_json,encode,decode


class ResourceStop(RuntimeError):pass


def inside(value):
    p=(ROOT/value).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Job files must remain inside project')
    return p


def folder_bytes(folder):return sum(p.stat().st_size for p in Path(folder).rglob('*') if p.is_file() and not p.is_symlink())


class Budget:
    def __init__(self,folder,limits):self.folder=Path(folder);self.limits=limits;self.started=time.monotonic();self.engine=None
    def check(self):
        if time.monotonic()-self.started>self.limits['max_wall_s']-min(5.,self.limits['max_wall_s']/10):raise ResourceStop('Wall budget reserve reached')
        if folder_bytes(self.folder)>self.limits['max_disk_bytes']:raise ResourceStop('Job storage budget reached')
        import shutil
        if shutil.disk_usage(self.folder).free<2*1024**2:raise ResourceStop('Filesystem free space below frame-commit safety reserve')
        if self.engine is not None:
            dr=self.engine.dr
            if dr.detail.malloc_watermark(dr.JitBackend.CUDA)>self.limits['max_allocator_bytes']:raise ResourceStop('GPU allocator budget reached')


def generate(folder,*,resume=False,frame_limit=None):
    folder=inside(folder);config=validate(json.loads((folder/'config.json').read_text()));limits=json.loads((folder/('effective_limits.json' if (folder/'effective_limits.json').exists() else 'limits.json')).read_text())
    plan=json.loads((folder/'plan.json').read_text());single=plan.get('query_mode')=='current_state';single_ref=single and plan.get('current_reference',False)
    budget=Budget(folder,limits);writer=None;engine=None;requested={'stop':False};start=time.monotonic();events=folder/'events.jsonl'
    def state(status,**kw):
        row=dict(status=status,pid=os.getpid(),configuration_sha256=fingerprint(config),elapsed_wall_s=time.monotonic()-start,**kw)
        atomic_json(folder/'state.json',row)
        with events.open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
    def signal_stop(*_):requested['stop']=True
    def should_stop():
        control=json.loads((folder/'control.json').read_text()) if (folder/'control.json').exists() else {}
        return requested['stop'] or control.get('kind') in ('pause','stop')
    def finish():
        try:
            if single_ref or not single and (config['dataset'].get('independent_reference_frames') or config['dataset'].get('independent_reference_extremes')):
                state('auditing_independent_SDK',committed_frames=expected,expected_frames=expected)
                from .continuous_validation import validate_selected,select_reference_indices
                from .storage import CSIReader
                indices,selection=([0],dict(indices=[0],scope='Only frozen current pose')) if single else select_reference_indices(CSIReader(folder/'csi.h5'));atomic_json(folder/'independent_sampling_selection.json',selection)
                ledger=validate_selected(folder,indices,budget,should_stop=should_stop)
                atomic_json(folder/'independent_audit_summary.json',ledger)
                if ledger['counts'].get('Failed',0) or ledger.get('path_counts',{}).get('Failed',0):raise ValueError('Independent SDK numerical gate failed; raw data and failed references preserved')
            from .workflow import audit_closed
            result=audit_closed(folder);atomic_json(folder/'last_CPU_audit.json',result)
            if result['status']!='Passed':raise ValueError('Closed CPU dataset audit failed; preserve frames and inspect issues')
            if single_ref or config['dataset'].get('independent_reference_frames') or config['dataset'].get('independent_reference_extremes'):
                state('complete_native_generation_pending_CPU_closure',committed_frames=expected,phase='freezing_closed_delivery')
                from .workflow import freeze_closed_delivery
                freeze_closed_delivery(folder,budget)
            state('complete_native_dataset',committed_frames=expected,expected_frames=expected,CPU_audit=result['status'],independent_numerical_validation='See_independent_sidecar_checked_rows_only')
        except BaseException as exc:
            state('paused' if isinstance(exc,InterruptedError) else 'interrupted_resource_preserved' if isinstance(exc,ResourceStop) else 'failed_preserved',error=repr(exc),committed_frames=expected,phase='post_writer_audit',no_automatic_retry=True)
            raise
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    try:
        world=PhysicalWorld(config);writer=CSIWriter(folder,config,resume=resume)
        writer.commit_pending()
        snapshot=writer.latest_snapshot()
        if snapshot is None and (folder/'initial_snapshot.json').exists():snapshot=decode(json.loads((folder/'initial_snapshot.json').read_text()))
        if snapshot is not None:world.restore(snapshot)
        origin=float(writer.file['timestamps'][0]) if writer.count else world.time
        expected=1 if single else int(round((config['timing']['duration_s']-origin)/config['timing']['csi_dt_s']))+1
        if single:writer.file.attrs.update(query_mode='current_state',expected_frames=1,initial_snapshot_sha256=json.loads((folder/'plan.json').read_text())['initial_snapshot_sha256'])
        source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/wiwo_ma/platform').glob('*.py')}
        if resume and (folder/'source_versions.json').exists():
            original=json.loads((folder/'source_versions.json').read_text())['platform_sources']
            core=('physics.py','motion.py','native_rt.py','path_capture.py','config.py','storage.py','jobs.py')
            if any(original.get('src/wiwo_ma/platform/'+n)!=source['src/wiwo_ma/platform/'+n] for n in core):raise ValueError('Core source changed across resume; preserve this episode and prepare a new job')
        attempt=folder/'attempts'/str(time.time_ns());attempt.mkdir(parents=True)
        version=dict(platform_sources=source,scope='Original SDK solver, no propagation cache or approximate labels',resumed=resume,starting_frame=writer.count)
        atomic_json(attempt/'source_versions.json',version)
        if not (folder/'source_versions.json').exists():atomic_json(folder/'source_versions.json',version)
        for name in source:
            saved=attempt/'executed_sources'/name;saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes((ROOT/name).read_bytes())
        if writer.count>=expected:
            writer.complete();writer.close();writer=None;finish();return
        # All mechanical/shape/storage checks precede CUDA initialization.
        state('initializing_native_SDK',committed_frames=writer.count,expected_frames=expected)
        from .native_rt import NativeRT
        # Complete path files are immutable attempt-scoped outputs. Their SHA
        # receipts are committed with the HDF row, never used as solver inputs.
        chunks=[];direct={}
        def capture(first,arrays):
            if folder_bytes(folder)+sum(x.nbytes for x in arrays.values())+2*1024**2>limits['max_disk_bytes']:raise ResourceStop('Complete path export reserve exceeds disk cap')
            path=attempt/('frame_%06d_src_%04d.npz'%(writer.count,first))
            with path.open('xb') as f:np.savez_compressed(f,**arrays)
            chunks.append(dict(path=str(path.relative_to(folder)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),paths=len(arrays['user'])))
            for user,port in zip(arrays['user'][~np.any(arrays['interaction_types'],axis=1)],arrays['tx_port'][~np.any(arrays['interaction_types'],axis=1)]):direct.setdefault(int(user),set()).add(int(port))
            budget.check()
        engine=NativeRT(config,check_budget=budget.check);budget.engine=engine
        initial_count=writer.count;times=[]
        while writer.count<expected:
            control=json.loads((folder/'control.json').read_text()) if (folder/'control.json').exists() else {}
            if requested['stop'] or control.get('kind') in ('pause','stop'):
                state('paused' if control.get('kind')=='pause' else 'stopped_partial',committed_frames=writer.count,expected_frames=expected);return
            if frame_limit is not None and writer.count-initial_count>=frame_limit:
                state('partial_requested_frame_limit',committed_frames=writer.count,expected_frames=expected);return
            budget.check()
            if writer.count:world.advance_to(origin+writer.count*config['timing']['csi_dt_s'])
            physical=world.state();state('solving_native_frame',frame=writer.count,physical_time_s=world.time,committed_frames=writer.count,expected_frames=expected)
            chunks.clear();direct.clear();engine.complete_path_sink=capture if config['dataset'].get('capture_complete_paths',False) else None
            result=engine.solve(physical)
            if chunks:result.update(complete_path_files=deepcopy(chunks),complete_paths_count=sum(x['paths'] for x in chunks),direct_LOS_port_counts=[len(direct.get(u,())) for u in range(len(config['users']))],complete_path_scope='All SDK discovered paths; not an exhaustive physical ray set')
            if folder_bytes(folder)+result['H'].nbytes*4+2*1024**2>limits['max_disk_bytes']:raise ResourceStop('Reserve for pending CSI/HDF frame exceeds storage cap')
            writer.stage(result,physical,world.snapshot());writer.commit_pending();times.append(result['solver_wall_seconds'])
            progress=dict(committed_frames=writer.count,expected_frames=expected,physical_time_s=world.time,total_physical_duration_s=config['timing']['duration_s'],
                solver_last_wall_s=times[-1],solver_mean_wall_s=float(np.mean(times)),estimated_remaining_wall_s=float(np.mean(times))*(expected-writer.count),
                physical_clock='Fixed configured mechanics/CSI ticks; never advanced from solver wall time',render_clock='Independent browser RAF',
                label_source=result['label_source'],independent_numerical_validation='Unknown',allocator_peak_bytes=result['allocator_peak_bytes'],disk_bytes=folder_bytes(folder))
            progress['worker_elapsed_wall_s']=time.monotonic()-start
            atomic_json(folder/'progress.json',progress);state('frame_committed',**progress)
        writer.complete();state('complete_native_generation_pending_CPU_closure',committed_frames=writer.count,expected_frames=expected,independent_numerical_validation='Unknown_pending_audit')
    except BaseException as exc:
        resource=isinstance(exc,ResourceStop) or any(x in str(exc).lower() for x in ['out of memory','bad_alloc','cuda_error_out_of_memory'])
        count=writer.count if writer is not None else expected if 'expected' in locals() else 0
        state('paused' if isinstance(exc,InterruptedError) else 'interrupted_resource_preserved' if resource else 'failed_preserved',error=repr(exc),committed_frames=count,no_automatic_retry=True)
        raise
    finally:
        if writer is not None:writer.close()
        if engine is not None:engine.close()
    # The writer is closed before a separate all-row CPU audit reads HDF5.
    # Partial/failed runs retain their records and can be audited explicitly.
    if writer is not None and writer.count>=expected:
        finish()


def prepare_job(config,folder,limits,*,initial_snapshot=None,reserved=False,current_reference=False):
    config=validate(config);folder=inside(folder)
    if folder.exists() and not reserved:raise FileExistsError(folder)
    if reserved and (folder/'plan.json').exists():raise FileExistsError('Job already prepared')
    if initial_snapshot is not None and initial_snapshot['configuration']!=fingerprint(config):raise ValueError('Initial branch snapshot has a different configuration')
    for key in ['max_wall_s','max_disk_bytes','max_allocator_bytes']:
        if not np.isfinite(limits[key]) or limits[key]<=0:raise ValueError('Explicit positive resource cap required')
    estimate=estimated_storage(config)
    from .scenario_checks import check_motion_contract
    motion=check_motion_contract(config)
    if motion['status']!='Passed' and not config.get('physics_validation',{}).get('allow_legacy_discontinuous_motion',False):
        raise ValueError('Discontinuous external motion: use bounded_quintic or explicitly acknowledge legacy motion in physics_validation.allow_legacy_discontinuous_motion')
    # Explicitly requested full CPU preflight runs BEFORE any CUDA job exists.
    if config.get('physics_validation',{}).get('require_preflight'):
        from .scenario_checks import preflight,current_state_preflight
        admission=current_state_preflight(config,initial_snapshot) if initial_snapshot is not None else preflight(config)
        if admission['status']!='Passed':raise ValueError('CPU physical preflight failed: '+str(admission['failure_types']))
    else:admission=dict(status='Unknown',scope='Only analytic motion checked; city/actor geometry not screened')
    folder.mkdir(parents=True,exist_ok=reserved);atomic_json(folder/'config.json',config);atomic_json(folder/'limits.json',limits)
    atomic_json(folder/'physical_admission.json',admission)
    if initial_snapshot is not None:
        atomic_json(folder/'initial_snapshot.json',encode(initial_snapshot))
    atomic_json(folder/'plan.json',dict(status='prepared_not_started',configuration_sha256=fingerprint(config),estimate=estimate,limits=limits,
        physical_reference='Original Sionna2.2.0 each spatial port with configured polarization, synthetic_array=False, fresh scene state each sample',
        propagation_speedup_research=False,high_cost_run_requires_separate_explicit_authorization=True))
    atomic_json(folder/'state.json',dict(status='prepared_not_started',configuration_sha256=fingerprint(config)))
    if initial_snapshot is not None:
        from .physics import PhysicalWorld
        from .state_identity import geometry_digest
        world=PhysicalWorld(config);world.restore(initial_snapshot)
        plan=json.loads((folder/'plan.json').read_text());plan.update(query_mode='current_state',expected_frames=1,
            current_reference=bool(current_reference),
            physical_time_s=world.time,initial_geometry_sha256=geometry_digest(world.state()),
            initial_snapshot_sha256=hashlib.sha256((folder/'initial_snapshot.json').read_bytes()).hexdigest())
        atomic_json(folder/'plan.json',plan)
    sources=list((ROOT/'src/wiwo_ma/platform').glob('*.py'))+[ROOT/'src/wiwo_ma/mechanics.py',ROOT/'src/wiwo_ma/v5/actor_mesh.py',ROOT/'scripts/platform/run']
    atomic_json(folder/'frozen_sources.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    names=['config.json','limits.json','plan.json','physical_admission.json','frozen_sources.json']+(['initial_snapshot.json'] if initial_snapshot is not None else [])
    atomic_json(folder/'frozen_inputs.json',{name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names})
    return folder


def launch_job(folder,authorization,*,resume=False,frame_limit=None):
    folder=inside(folder);grant=inside(authorization)
    from .workflow import check_authorization
    check_authorization(folder,grant)
    reservation=(folder/'launch.lock').open('a+b')
    try:fcntl.flock(reservation,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:reservation.close();raise ResourceStop('Another launch request is being processed')
    # A long-lived supervisor holds the grant lock, bills failures as well as
    # successes, and kills a worker which exceeds the independent wall budget.
    args=[str(ROOT/'scripts/platform/run'),'supervise','--job',str(folder),'--authorization',str(grant)]
    if resume:args+=['--resume']
    if frame_limit is not None:args+=['--frame-limit',str(int(frame_limit))]
    try:
        if (folder/'launcher.json').exists():
            old=json.loads((folder/'launcher.json').read_text());procdir=Path('/proc',str(old['pid']))
            try:
                cmd=(procdir/'cmdline').read_bytes().replace(b'\0',b' ').decode();status=(procdir/'stat').read_text().split(') ',1)[1][0]
                if status!='Z' and 'supervise --job '+str(folder) in cmd:raise ResourceStop('Previous job supervisor is still active; duplicate launch rejected')
            except FileNotFoundError:pass
        with grant.with_suffix('.lock').open('a+b') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ResourceStop('Independent grant is still supervising a worker; wait for closure')
        with (folder/'supervisor.log').open('ab') as log:proc=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(folder/'launcher.json',dict(pid=proc.pid,command=args,authorization=str(grant.relative_to(ROOT)),independent_background=True))
        return proc.pid
    finally:reservation.close()


def supervise(folder,authorization,*,resume=False,frame_limit=None):
    folder=inside(folder);grant=inside(authorization);approved=json.loads(grant.read_text());cfg=json.loads((folder/'config.json').read_text())
    from .workflow import check_authorization
    check_authorization(folder,grant)
    if approved.get('approved') is not True or approved.get('scope')!='native_platform':raise ValueError('Explicit native-platform authorization required; old research grants cannot be used')
    if fingerprint(cfg) not in approved['configuration_sha256']:raise ValueError('Configuration outside authorized scope')
    ledger=grant.with_suffix('.ledger.json');lock=grant.with_suffix('.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    billed=json.loads(ledger.read_text()) if ledger.exists() else dict(spent_wall_s=0.,runs=[])
    limits=json.loads((folder/'limits.json').read_text());remaining=approved['total_wall_s']-billed['spent_wall_s']
    if remaining<=10:raise ResourceStop('Independent native grant exhausted')
    effective={**limits,'max_wall_s':min(limits['max_wall_s'],remaining)}
    atomic_json(folder/'effective_limits.json',effective)
    if limits['max_allocator_bytes']>approved['max_allocator_bytes'] or limits['max_disk_bytes']>approved['max_disk_bytes']:raise ResourceStop('Job resources exceed authorization')
    env=os.environ.copy();env['WIWO_GPU']=str(approved['gpu']);env['CUDA_VISIBLE_DEVICES']=str(approved['gpu'])
    env['WIWO_PLATFORM_WORKER_TOKEN']=hashlib.sha256((folder/'config.json').read_bytes()+(folder/'effective_limits.json').read_bytes()).hexdigest()
    command=[str(ROOT/'scripts/platform/run'),'generate','--job',str(folder),'--authorized-worker']
    if resume:command+=['--resume']
    if frame_limit is not None:command+=['--frame-limit',str(frame_limit)]
    gpu_lock=ROOT/('.state/platform_gpu'+str(approved['gpu'])+'.lock');gpu_lock.parent.mkdir(exist_ok=True,parents=True)
    device_lock=gpu_lock.open('a+b');fcntl.flock(device_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(folder/'control.json',dict(kind='run'));started=time.monotonic();log=(folder/'worker.log').open('ab')
    worker=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(folder/'worker_pid.json',dict(pid=worker.pid,gpu=approved['gpu'],wall_cap_s=effective['max_wall_s']))
    terminated=False
    try:
        while worker.poll() is None:
            if time.monotonic()-started>=effective['max_wall_s']:
                os.killpg(worker.pid,signal.SIGKILL);terminated=True;break
            time.sleep(.1)
        code=worker.wait()
    finally:
        elapsed=time.monotonic()-started;log.close();billed['spent_wall_s']+=elapsed
        billed['runs'].append(dict(job=str(folder.relative_to(ROOT)),wall_seconds=elapsed,forced_kill=terminated,returncode=worker.returncode))
        atomic_json(ledger,billed);lock.close();device_lock.close()
    if terminated:atomic_json(folder/'state.json',dict(status='interrupted_hard_wall_limit_preserved',no_automatic_retry=True,resource_gate='Failed',elapsed_wall_s=elapsed))
    elif code:
        previous=json.loads((folder/'state.json').read_text()) if (folder/'state.json').exists() else {}
        if previous.get('status') not in ('failed_preserved','paused','interrupted_resource_preserved','stopped_partial'):
            atomic_json(folder/'state.json',dict(status='failed_worker_exit_preserved',returncode=code,previous_worker_state=previous,
                no_automatic_retry=True,committed_frames=previous.get('committed_frames',0),elapsed_wall_s=elapsed))
    return code
