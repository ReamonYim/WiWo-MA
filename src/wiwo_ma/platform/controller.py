"""Shared physical controller for Web/API, native jobs and stored-data replay."""
from copy import deepcopy
from pathlib import Path
import json,threading,time
import hashlib
import h5py,numpy as np
from .config import ROOT,validate,fingerprint,estimated_storage
from .physics import PhysicalWorld
from .storage import CSIReader,decode,atomic_json,plain,time_index
from .jobs import inside,prepare_job,launch_job
from .state_identity import geometry_digest


class Replay:
    def __init__(self,path):
        self.path=inside(path)
        with h5py.File(self.path,'r') as f:self.native=f.attrs.get('schema')=='wiwo-native-csi-v1'
        if self.native:
            self.reader=CSIReader(self.path);self.timestamps=self.reader.timestamps;self.count=self.reader.count;self.config=self.reader.config
            self.verified_rows={};self.checked_rows={};self.cached_row=None;sidecar=self.path.with_name('independent_validation.json')
            if sidecar.exists():
                audit=json.loads(sidecar.read_text())
                if audit.get('schema')=='native-frame-validation-sidecar-v1':
                    for row in audit['rows']:
                        ref=inside(row['reference_array_path']);h=hashlib.sha256(ref.read_bytes()).hexdigest()
                        errors=np.asarray(row['per_user_NMSE'],float)
                        if row['status']=='Passed' and np.isfinite(errors).all() and np.all(errors<=.001) and h==row['reference_array_file_sha256']:
                            self.verified_rows[(float(row['physical_timestamp']),row['frame_sha256'])]=row
        else:
            with h5py.File(self.path,'r') as f:
                if 'episodes' not in f:raise ValueError('Unsupported stored dataset')
                self.group='episodes/'+next(iter(f['episodes']));g=f[self.group]
                self.timestamps=g['timestamps'][:];self.count=len(self.timestamps);self.legacy_config=json.loads(g.attrs['config_json'])
            self.config=None

    def frame(self,index):
        if self.native:
            stat=self.path.stat();key=(index,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
            if self.cached_row is None or self.cached_row[0]!=key:self.cached_row=(key,self.reader.frame(index))
            q=self.cached_row[1];result={**q['result'],'H':q['H']};rowhash=q['frame_sha256']
            row=self.verified_rows.get((float(q['state']['timestamp']),rowhash))
            if row and row['configuration_sha256']==q['state']['configuration_sha256']:
                # A JSON error value and reference hash alone cannot establish
                # numeric acceptance. Recompute from the actual complex arrays.
                ref=inside(row['reference_array_path']);stat=ref.stat();key=(rowhash,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
                if key not in self.checked_rows:
                    valid=False
                    if hashlib.sha256(ref.read_bytes()).hexdigest()==row['reference_array_file_sha256']:
                        from .dataset_audit import nmse_all_users
                        with np.load(ref,allow_pickle=False) as f:reference=f['H']
                        if reference.shape==result['H'].shape:
                            errors,_,_=nmse_all_users(result['H'],reference);valid=bool(np.isfinite(errors).all() and np.all(errors<=.001))
                    self.checked_rows[key]=valid
                if self.checked_rows[key]:
                    result.update(recorded_independent_validation_status=q['result']['independent_validation_status'],independent_validation_status='Passed_original_SDK_checked_frame_sidecar',independent_validation_report=str(self.path.with_name('independent_validation.json').relative_to(ROOT)))
                else:result['independent_validation_issue']='Failed_sidecar_raw_array_check; original stored label unchanged'
            return q['state'],result,q['snapshot']
        with h5py.File(self.path,'r') as f:
            g=f[self.group];h=g['cfr'][index];positions=g['positions'][index];rotations=g['rotations'][index];elements=g['element_positions'][index]
            users=g['users'][index];n=len(positions);k=len(users);count=elements.shape[1];t=float(g['timestamps'][index]);freq=g['frequency_offsets_hz'][:] if 'frequency_offsets_hz' in g else np.asarray(self.legacy_config['radio'].get('subcarrier_indices',range(h.shape[-1])))
            state=dict(timestamp=t,mechanics_tick=index,configuration_sha256='historical_V4',panel_ids=[('legacyBS',str(i)) for i in range(n)],
                positions=positions,rotations=rotations,target_positions=positions,target_rotations=rotations,mount_positions=positions,
                velocities=g['velocities'][index] if 'velocities' in g else np.zeros_like(positions),angular_velocities=np.zeros_like(positions),
                tx_port_positions=elements.reshape(-1,3).astype(np.float32),tx_port_map=np.array([(0,p,e,0) for p in range(n) for e in range(count)]),
                rx_port_positions=users.astype(np.float32),rx_port_map=np.array([(u,0,0) for u in range(k)]),user_positions=users,user_rotations=np.tile(np.eye(3),(k,1,1)),
                bs_positions=np.mean(positions,axis=0)[None],bs_rotations=np.eye(3)[None],actors=[],mechanics=[dict(mechanism='stored_V4',movement_state='replay') for _ in range(n)])
            return state,dict(H=h[:,None],physical_timestamp=t,label_source='Stored_historical_V4_incremental_Partial',independent_validation_status='Partial_historical',
                paths=[],frequency_offsets_hz=freq,solver_wall_seconds=None,read_only=True,path_capture_scope='Legacy paths not loaded by this generic viewer; retained original dataset remains readable'),None


class PlatformController:
    def __init__(self,config,*,authorization=None):
        self.lock=threading.RLock();self.config=validate(config);self.world=PhysicalWorld(self.config);self.playing=False;self.closed=threading.Event()
        self.result=None;self.last_action=None;self.error=None;self.saved=None;self.replay=None;self.replay_index=0;self.authorization=None if authorization is None else inside(authorization)
        self.job=None;self.job_kind=None;self.attached_query_mode=None;self.last_job_frame=None;self.braking=False;self.wall_started=time.monotonic();self.render_events=0;self.worker=threading.Thread(target=self._run,daemon=True);self.worker.start()

    def state(self):
        with self.lock:
            if self.replay is not None:return self.replay.frame(self.replay_index)[0]
            return self.world.state()

    def _job_running(self):
        if self.job is None:return False
        p=self.job/'state.json'
        if not p.exists():return False
        status=json.loads(p.read_text())['status']
        return status in ('preparing_CPU_admission','initializing_native_SDK','solving_native_frame','frame_committed','launching_native_worker','auditing_independent_SDK','independent_reference','complete_native_generation_pending_CPU_closure')

    def snapshot(self):
        with self.lock:
            s=self.state();h=self.result
            matching=(h is not None and abs(float(h['physical_timestamp'])-s['timestamp'])<1e-9)
            if matching and h.get('_tx_coordinates') is not None:matching=np.array_equal(np.asarray(h['_tx_coordinates'],np.float32),np.asarray(s['tx_port_positions'],np.float32))
            if matching:matching=h.get('_geometry_sha256')==geometry_digest(s)
            current_query=self.job_kind=='current_state' or (self.job_kind=='watch_only' and self.attached_query_mode=='current_state')
            mode='stored_replay' if self.replay else 'native_current_state' if current_query else 'native_background_frames' if self.job is not None else 'physical_preview_no_synthetic_CSI'
            stats=dict(playing=self.playing,mode=mode,
                task_control_role='read_only_observer' if self.job_kind=='watch_only' else 'stored_replay' if self.replay else 'authorized_executor' if self.job is not None and self.authorization is not None else 'prepared_without_authorization' if self.job is not None else 'live_mechanical_preview',
                physical_time_s=s['timestamp'],solver_last_wall_s=None if h is None else h.get('solver_wall_seconds'),
                current_CSI=('Stored_at_recorded_pose' if self.replay else 'Computed_at_current_pose') if matching else 'Unknown_current_pose',CSI_timestamp=None if h is None else h['physical_timestamp'],
                CSI_age_physical_s=None if h is None else s['timestamp']-h['physical_timestamp'],last_action=self.last_action,error=self.error,
                native_job=None if self.job is None else str(self.job.relative_to(ROOT)),GPU_enabled=self.authorization is not None,
                physical_dt_s=self.config['timing']['mechanics_dt_s'],requested_CSI_dt_s=self.config['timing']['csi_dt_s'],
                browser_draw_FPS='Unknown_not_inferred_from_server_updates',replay_index=self.replay_index if self.replay else None,replay_frames=self.replay.count if self.replay else None)
            if self.job is not None and (self.job/'progress.json').exists():stats['job_progress']=json.loads((self.job/'progress.json').read_text())
            if self.job is not None and (self.job/'state.json').exists():stats['job_state']=json.loads((self.job/'state.json').read_text())
            return deepcopy(s),deepcopy(h),stats

    def action(self,payload):
        with self.lock:
            if self.replay is not None:raise ValueError('Stored replay is read-only; load a live preset to apply actions')
            if self.job_kind=='watch_only':raise ValueError('Attached native job has a frozen action script; load a live preset for new actions')
            if self._job_running():raise ValueError('Native job uses a frozen trajectory; pause or stop at a frame boundary first')
            if self.job is not None:raise ValueError('Prepared job is immutable; load a live configuration to submit new actions')
            try:
                self.last_action=self.world.submit_many(payload['actions']) if 'actions' in payload else self.world.submit(payload)
                self.last_action['submitted_action']=deepcopy(payload);self.error=None;return deepcopy(self.last_action)
            except Exception as exc:self.last_action=dict(accepted=False,error=str(exc));self.error=str(exc);raise

    def configure(self,config):
        c=validate(config);world=PhysicalWorld(c)
        with self.lock:
            if self._job_running():raise ValueError('Pause the active native job before changing physical configuration')
            self.playing=False;self.braking=False;self.config=c;self.world=world;self.result=None;self.replay=None;self.replay_index=0;self.saved=None;self.job=None;self.job_kind=None;self.last_job_frame=None;self.error=None;self.authorization=None;self.last_action=None
        return dict(accepted=True,configuration_sha256=fingerprint(c),estimate=estimated_storage(c))

    def export_actions(self):
        c=deepcopy(self.config)
        done=[dict(time_s=x['time_s'],**x['action']) for x in self.world.executed]
        c['actions']=done+[dict(time_s=t,**a) for t,a in self.world.scheduled]+[x for x in c['actions'] if x['time_s']>self.world.time+1e-10];c['actions'].sort(key=lambda x:x['time_s'])
        return validate(c)

    def command(self,kind,payload=None):
        payload=payload or {}
        with self.lock:
            if kind in ('prepare_generation','prepare_current'):
                if self._job_running() or self.replay is not None or self.job is not None:raise ValueError('Load a live configuration and stop/detach any job before preparation')
                from .workflow import prepare_background
                current=kind=='prepare_current';c=self.config if current else self.export_actions();folder=payload.get('path','artifacts/platform/jobs/web_'+str(time.time_ns()))
                limits=dict(max_wall_s=float(payload['max_wall_s']),max_disk_bytes=int(payload['max_disk_bytes']),max_allocator_bytes=int(payload['max_allocator_bytes']))
                for value in limits.values():
                    if not np.isfinite(value) or value<=0:raise ValueError('Explicit finite positive resource limits required')
                self.playing=False;self.job=prepare_background(c,folder,limits,initial_snapshot=self.world.snapshot() if current else None,current_reference=bool(payload.get('current_reference',False)));self.job_kind='current_state' if current else 'prepared';self.last_job_frame=None
                self.authorization=None
                return dict(preparing=True,job=str(self.job.relative_to(ROOT)),estimate=estimated_storage(c),GPU_started=False)
            if kind=='return_live':
                if self._job_running():raise ValueError('Wait for the native task to close before returning to live controls')
                if self.replay is not None:raise ValueError('Stored replay remains read-only; load a live preset to edit')
                self.job=None;self.job_kind=None;self.last_job_frame=None;self.authorization=None;self.playing=False
                return dict(detached=True,CSI='Retained only at exactly matching SDK geometry; new movement invalidates it')
            if kind=='bind_authorization':
                if self.job is None or self._job_running():raise ValueError('Prepare an idle frozen job first')
                from .workflow import check_authorization
                grant=inside(payload['path']);report=check_authorization(self.job,grant);self.authorization=grant
                if self.job_kind=='watch_only':self.job_kind='current_state' if json.loads((self.job/'plan.json').read_text()).get('query_mode')=='current_state' else 'generate'
                return report
            if kind=='launch_prepared':
                if self.job is None or self.authorization is None:raise ValueError('Prepare a frozen job and bind its explicit authorization first')
                status=json.loads((self.job/'state.json').read_text())['status']
                if status!='prepared_not_started':raise ValueError('Prepared job already attempted; use explicit resume for a preserved partial job')
                limit=payload.get('frame_limit')
                if limit is not None and (not isinstance(limit,int) or limit<=0):raise ValueError('Positive first-stage frame limit required')
                self.playing=False;pid=launch_job(self.job,self.authorization,frame_limit=limit)
                atomic_json(self.job/'state.json',dict(status='launching_native_worker',pid=pid))
                if self.job_kind!='current_state':self.job_kind='generate'
                return dict(started=True,pid=pid,job=str(self.job.relative_to(ROOT)))
            if kind in ('pause','stop'):
                self.playing=False
                if self.job_kind=='watch_only':return dict(watching_native_job=True,physical_state_not_advanced=True,task_control='Observer has no worker control; bind matching explicit authorization first')
                if self.job is not None and self._job_running():atomic_json(self.job/'control.json',dict(kind=kind));return dict(requested=kind,effective='After_current_frame_is_committed')
                if self.job_kind=='watch_only':return dict(watching_native_job=True,physical_state_not_advanced=True)
                if kind=='stop' and self.replay is None:
                    # Batch braking commands. Re-simulating the entire future
                    # sixteen times stalls Web/API callbacks. Each executed
                    # integration interval still checks workspace/collisions.
                    self.world.scheduled=[]
                    for bs,p in self.world.keys:self.world.command(dict(bs_id=bs,panel_id=p,type='stop'),preflight=False)
                    self.braking=True;self.playing=True
                    return dict(braking=True,effective='Bounded deceleration; no future scripted actions during braking')
                return dict(playing=False)
            if kind in ('start','resume'):
                if self.job_kind=='watch_only':return dict(watching_native_job=True,scope='Actual committed frames; no preview time advances or new GPU launch')
                if self.job is not None and (self.job/'state.json').exists() and json.loads((self.job/'state.json').read_text())['status'] in ('paused','stopped_partial','interrupted_resource_preserved','interrupted_hard_wall_limit_preserved','partial_requested_frame_limit','failed_preserved'):
                    if self.authorization is None:raise ValueError('Native job resume needs remaining explicit authorization')
                    pid=launch_job(self.job,self.authorization,resume=True);atomic_json(self.job/'state.json',dict(status='launching_native_worker',pid=pid,resumed=True));return dict(resumed_job=True,pid=pid)
                if self._job_running():return dict(native_job_running=True)
                if self.job is not None:raise ValueError('A prepared/completed native job cannot advance a geometry preview clock; launch it or load its HDF5')
                self.playing=True;return dict(playing=True,scope='Stored replay' if self.replay else 'Physical preview only; CSI remains Unknown until solved')
            if kind=='reset':
                if self._job_running():raise ValueError('Stop the native job before reset')
                self.playing=False
                if self.replay is not None:self.replay_index=0;self._read_replay();return dict(reset=True,read_only=True)
                self.world=PhysicalWorld(self.config);self.result=None;self.error=None;self.job=None;self.job_kind=None;self.braking=False;return dict(reset=True)
            if kind=='step':
                if self.job is not None:raise ValueError('Attached/prepared task has a frozen physical timeline; return to live controls before preview stepping')
                if self._job_running():raise ValueError('Native job advances its own fixed physical timeline')
                self.playing=False
                if self.replay is not None:self.replay_index=min(self.replay_index+1,self.replay.count-1);self._read_replay();return dict(timestamp=self.state()['timestamp'],stored=True)
                seconds=float(payload.get('seconds',self.config['timing']['csi_dt_s']));self.world.advance_to(min(self.world.time+seconds,self.config['timing']['duration_s']))
                return dict(timestamp=self.world.time,CSI='Unknown_not_computed_by_geometry_preview')
            if kind=='save':self.saved=(self.world.snapshot(),deepcopy(self.result));return dict(saved=True,includes_pending_commands=True)
            if kind=='restore':
                if self.saved is None or self.replay is not None or self.job is not None:raise ValueError('No restorable live mechanical snapshot; return to live controls before restoring')
                self.playing=False;self.world.restore(self.saved[0]);self.result=deepcopy(self.saved[1]);return dict(restored=True,exogenous_trajectories_unchanged=True)
            if kind=='load_dataset':
                if self._job_running():raise ValueError('Pause native generation before loading a stored HDF5')
                self.replay=Replay(payload['path']);self.replay_index=0;self.playing=False
                self.job=None;self.job_kind=None
                if self.replay.config is not None:self.config=self.replay.config;self.world=PhysicalWorld(self.config)
                else:
                    self.config=deepcopy(self.config);self.config['scene']=deepcopy(self.replay.legacy_config['scene'])
                self._read_replay();return dict(loaded=True,frames=self.replay.count,source='native' if self.replay.native else 'historical_V4_Partial')
            if kind=='seek':
                if self.replay is None:raise ValueError('Timeline seeking uses a saved dataset; use explicit snapshot restore for live mechanics')
                t=float(payload['time_s']);self.replay_index=time_index(self.replay.timestamps,t);self._read_replay();return dict(timestamp=self.state()['timestamp'])
            if kind=='attach_job':
                if self._job_running():raise ValueError('Detach or stop the current native job first')
                job=inside(payload['path']);c=validate(json.loads((job/'config.json').read_text()))
                # Restore the frozen input immediately, even while the first
                # native frame is still computing. Validate before adopting
                # anything so a corrupt attachment leaves live state intact.
                world=PhysicalWorld(c);planfile=job/'plan.json'
                plan=json.loads(planfile.read_text()) if planfile.exists() else {}
                if plan.get('query_mode')=='current_state':
                    file=job/'initial_snapshot.json'
                    if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest()!=plan.get('initial_snapshot_sha256'):
                        raise ValueError('Attached current-state snapshot is missing or its hash changed')
                    world.restore(decode(json.loads(file.read_text())))
                    if geometry_digest(world.state())!=plan.get('initial_geometry_sha256'):
                        raise ValueError('Attached current-state geometry differs from the frozen plan')
                self.config=c;self.world=world;self.playing=False;self.replay=None;self.result=None;self.job=job;self.job_kind='watch_only';self.attached_query_mode=plan.get('query_mode');self.last_job_frame=None;self.error=None
                self.authorization=None;self.saved=None;self.last_action=None;self.braking=False
                self._poll_job();return dict(attached=True,job=str(job.relative_to(ROOT)),scope='Atomic actual committed native frame stream; no RT launched')
            if kind in ('solve_current','generate'):
                raise ValueError('Direct native solving is disabled: freeze the current pose or trajectory first, then bind an explicit job-specific authorization and launch the prepared task')
            raise ValueError('Unknown platform command')

    def _read_replay(self):
        s,r,_=self.replay.frame(self.replay_index);r['_tx_coordinates']=s['tx_port_positions'];r['_geometry_sha256']=geometry_digest(s);self.result=r

    def _poll_job(self):
        if self.job is None:return
        file=self.job/'latest_frame.npz'
        if not file.exists():file=self.job/'dataset/latest_frame.npz'
        if not file.exists():return
        signature=file.stat().st_mtime_ns
        if signature==self.last_job_frame:return
        with np.load(file,allow_pickle=False) as f:H=f['H'];q=json.loads(str(f['frame_json']))
        snapshot=decode(q['snapshot'])
        if snapshot['configuration']!=self.world.configuration:
            # A frozen generated action script may have a new configuration
            # hash. Rebuild from that exact job config, then restore its state.
            self.config=validate(json.loads((self.job/'config.json').read_text()));self.world=PhysicalWorld(self.config)
        self.world.restore(snapshot);self.result={**q['result'],'H':H,'_tx_coordinates':q['state']['tx_port_positions'],'_geometry_sha256':geometry_digest(q['state'])};self.last_job_frame=signature

    def _run(self):
        while not self.closed.wait(.02):
            try:
                with self.lock:
                    self._poll_job()
                    if not self.playing:continue
                    if self.replay is not None:
                        self.replay_index=min(self.replay_index+1,self.replay.count-1);self._read_replay()
                        if self.replay_index==self.replay.count-1:self.playing=False
                    else:
                        t=min(self.world.time+self.config['timing']['csi_dt_s'],self.config['timing']['duration_s']);self.world.advance_to(t,script=not self.braking)
                        if self.braking and all(c.state.completed for c in self.world.controllers):self.braking=False;self.playing=False
                        if t>=self.config['timing']['duration_s']:self.playing=False
            except Exception as exc:
                with self.lock:self.error=str(exc);self.playing=False

    def close(self):self.closed.set();self.worker.join(timeout=5)
