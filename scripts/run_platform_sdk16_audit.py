#!/usr/bin/env python3
"""Scoped, resumable fresh-SDK references; never reads the audited CSI.

Each of16 states uses a fresh scene and solver. Four-source chunks have atomic
receipts, so interrupted work can resume without repeating committed SDK calls.
The supervisor bills every GPU worker run, including errors, to a NEW grant.
Only the separate CPU postprocessor opens the original dataset.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name=='scripts' else Path(os.environ['WIWO_PROJECT_ROOT'])
sys.path.insert(0,str(ROOT/'src'))
from wiwo_ma.platform.dataset_audit import save_json,sha_file


class BudgetStop(RuntimeError):pass


def folder_bytes(path):return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file() and not p.is_symlink())


def atomic_npz(path,**arrays):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def verify_proposal(path,expected):
    if sha_file(path/'proposal.json')!=expected:raise ValueError('Approved proposal changed')
    p=json.loads((path/'proposal.json').read_text())
    for name,h in p['frozen_inputs'].items():
        if sha_file(path/name)!=h:raise ValueError('Frozen input changed: '+name)
    return p


def assemble_chunk(raw,rx_map):
    # Independent explicit axis enumeration; no NativeRT import/mapping.
    raw=np.asarray(raw)
    if raw.shape[:4]!=(40,1,4,1) or raw.shape[4:]!=(1,64):raise ValueError('Unexpected SDK axes '+repr(raw.shape))
    h=np.zeros((40,1,4,64),np.complex64)
    for ri,m in enumerate(rx_map):
        for ti in range(4):h[int(m[0]),int(m[1]),ti,:]=raw[ri,0,ti,0,0,:]
    return h


def committed_chunk(path):
    receipt=path.with_suffix('.receipt.json')
    if not receipt.exists():return False
    data=json.loads(receipt.read_text())
    if not path.exists() or sha_file(path)!=data['sha256']:raise ValueError('Committed reference chunk changed')
    return True


def worker(out,grant):
    a=json.loads(grant.read_text());plan=Path(a['proposal_folder']);p=verify_proposal(plan,a['proposal_sha256'])
    if os.environ.get('WIWO_SDK16_TOKEN')!=sha_file(grant):raise ValueError('Only bounded authorized supervisor may invoke CUDA worker')
    remaining=float(os.environ['WIWO_SDK16_REMAINING_S']);started=time.monotonic();stop={'requested':False}
    def handle(*_):stop['requested']=True
    signal.signal(signal.SIGTERM,handle);signal.signal(signal.SIGINT,handle)
    dr=None
    def check(extra_bytes=0):
        if stop['requested']:raise BudgetStop('Signal stop requested; committed chunks preserved')
        if time.monotonic()-started>remaining-8:raise BudgetStop('Independent900s budget reserve reached')
        if folder_bytes(out)+extra_bytes>a['max_new_storage_bytes']:raise BudgetStop('Storage cap')
        if dr is not None and int(dr.detail.malloc_watermark(dr.JitBackend.CUDA))>a['max_VRAM_bytes']:raise BudgetStop('GPU allocator cap')
    def event(status,**kw):
        row=dict(status=status,worker_pid=os.getpid(),elapsed_worker_wall_s=time.monotonic()-started,**kw)
        save_json(out/'state.json',row)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
    try:
        check();event('initializing_original_SDK')
        import sionna.rt as rt
        import mitsuba as mi
        import drjit as dr
        from wiwo_ma.platform.scene_geometry import scene_xml
        if rt.__version__!='2.2.0' or not mi.variant().startswith('cuda'):raise RuntimeError('Pinned native SDK CUDA required')
        dr.set_flag(dr.JitFlag.SymbolicCalls,False);dr.set_flag(dr.JitFlag.SymbolicLoops,False)
        config=json.loads((plan/'config.json').read_text());r=config['radio'];rtcfg=config['rt']
        freq=(np.arange(r['num_subcarriers'])-(r['num_subcarriers']-1)/2)*r['subcarrier_spacing_hz']
        settings={k:rtcfg[k] for k in ['max_depth','max_num_paths_per_src','samples_per_src','synthetic_array','los','specular_reflection','diffuse_reflection','refraction','diffraction','edge_diffraction','diffraction_lit_region','seed']}
        if rtcfg['tx_device_chunk']!=4 or settings['synthetic_array'] is not False:raise ValueError('Approved physical reference definition changed')
        committed_states=[]
        for case in p['cases']:
            i=case['index'];target=out/'references'/str(i);target.mkdir(parents=True,exist_ok=True)
            if (target/'complete.json').exists():
                complete=json.loads((target/'complete.json').read_text())
                for name,digest in complete['files'].items():
                    if sha_file(target/name)!=digest:raise ValueError('Committed state changed')
                committed_states.append(i);continue
            check();case_start=time.monotonic();event('fresh_scene',index=i,committed_states=committed_states)
            state=json.loads((plan/str(i)/'state.json').read_text());scene=rt.load_scene(str(scene_xml(config['scene'])),merge_shapes=False)
            scene.frequency=r['carrier_hz']
            scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern=r['tx_pattern'],polarization=r['tx_polarization'])
            scene.rx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern=r['rx_pattern'],polarization=r['rx_polarization'])
            if len(scene.tx_array.antenna_pattern.patterns)!=1 or len(scene.rx_array.antenna_pattern.patterns)!=1:raise ValueError('Actual reference must remain V-only')
            if config['scene'].get('material_override'):
                mat=rt.ITURadioMaterial(name='independent_static',itu_type=config['scene']['material_override'],thickness=.1);scene.add(mat)
                for obj in scene.objects.values():obj.radio_material=mat
            for actor in state['actors']:
                mat=rt.ITURadioMaterial(name='independent_material_'+actor['id'],itu_type=actor['material'],thickness=.01)
                obj=rt.SceneObject(name='reference_'+actor['id'],fname=str(plan/str(i)/('reference_actor_'+actor['id']+'.ply')),radio_material=mat);scene.add(obj)
                actual=np.asarray(obj.mi_mesh.vertex_positions_buffer()).reshape(-1,3)
                if not np.array_equal(actual.view(np.uint32),np.array(actor['vertices'],np.float32).view(np.uint32)):raise ValueError('Baked actor readback mismatch')
            object_map={str(int(o.object_id)):dict(name=name,material=o.radio_material.name) for name,o in scene.objects.items()}
            save_json(target/'object_map.json',object_map)
            tx=np.array(state['tx_port_positions'],np.float32);rx=np.array(state['rx_port_positions'],np.float32)
            ta=np.array(state['tx_port_orientations_rad'],np.float32);ra=np.array(state['rx_port_orientations_rad'],np.float32);rm=np.array(state['rx_port_map'],int)
            for ri,xyz in enumerate(rx):
                obj=rt.Receiver(name='independent_rx'+str(ri),position=mi.Point3f(xyz.tolist()),orientation=mi.Point3f(ra[ri].tolist()));scene.add(obj)
                if not np.array_equal(np.asarray(obj.orientation).reshape(3).view(np.uint32),ra[ri].view(np.uint32)):raise ValueError('RX orientation mismatch')
            solver=rt.PathSolver(deterministic=True);solver.loop_mode='evaluated';names=[];chunks=[]
            # Object pointer IDs are process-local. Each chunk saves its own
            # map; resume into a fresh scene must never reinterpret old IDs.
            for first in range(0,64,4):
                path=target/f'chunk_{first:02d}.npz'
                if committed_chunk(path):chunks.append(path);continue
                if path.exists():raise ValueError('Uncommitted chunk preserved; requires explicit inspection before retry')
                check();event('reference_chunk',index=i,first_tx=first,committed_states=committed_states)
                begin=time.monotonic()
                for name in names:scene.remove(name)
                names=[]
                for ti in range(first,first+4):
                    name='independent_tx'+str(ti);obj=rt.Transmitter(name=name,position=mi.Point3f(tx[ti].tolist()),orientation=mi.Point3f(ta[ti].tolist()));scene.add(obj);names.append(name)
                    if not np.array_equal(np.asarray(obj.orientation).reshape(3).view(np.uint32),ta[ti].view(np.uint32)):raise ValueError('TX orientation mismatch')
                paths=solver(scene=scene,**settings);dr.eval();dr.sync_thread();solve_s=time.monotonic()-begin
                if not np.array_equal(np.asarray(paths.sources).T.view(np.uint32),tx[first:first+4].view(np.uint32)) or not np.array_equal(np.asarray(paths.targets).T.view(np.uint32),rx.view(np.uint32)):raise ValueError('SDK endpoint mismatch')
                raw=np.asarray(paths.cfr(frequencies=mi.Float(freq),num_time_steps=1,normalize_delays=False,normalize=False,out_type='numpy'))
                h=assemble_chunk(raw,rm);cir,tau=paths.cir(normalize_delays=False,num_time_steps=1,out_type='numpy')
                b=paths._paths_buffer;count=b.buffer_size;si=np.array(b.source_indices,np.int32);ri=np.array(b.target_indices,np.int32);pid=np.array(paths._path_ind,np.int32)
                if not np.asarray(b.valid).all():raise ValueError('Unexpected invalid rows in compacted SDK path buffer')
                coeff=np.asarray(cir[ri,0,si,0,pid,0],np.complex64);delay=np.asarray(tau[ri,0,si,0,pid],np.float32)
                vertices=np.zeros((count,3,3),np.float32);types=np.zeros((count,3),np.uint32);objects=np.zeros((count,3),np.uint32);primitives=objects.copy()
                triangles=np.zeros((count,3,3,3),np.float32);normals=vertices.copy()
                active=dr.full(mi.Bool,True,count)
                for d in range(3):
                    kind=b.get_interaction_type(mi.UInt(d+1),active);mask=kind!=0
                    types[:,d]=np.asarray(kind);vertices[:,d]=np.asarray(b.get_vertex(mi.UInt(d+1),mask)).T
                    objects[:,d]=np.asarray(b.get_shape(mi.UInt(d+1),mask));primitives[:,d]=np.asarray(b.get_primitive(mi.UInt(d+1),mask))
                    normal,v0,v1,v2=b.get_primitive_props(mi.UInt(d+1),True,True,mask)
                    normals[:,d]=np.asarray(normal).T
                    triangles[:,d]=np.stack([np.asarray(v).T for v in [v0,v1,v2]],axis=1)
                field=np.asarray(dr.real(b.a[0][0]),np.float32)+1j*np.asarray(dr.imag(b.a[0][0]),np.float32)
                arrays=dict(H=h,user=rm[ri,0].astype(np.int16),tx_port=(si+first).astype(np.int16),rx_port=rm[ri,1].astype(np.int16),
                    coefficient=coeff,field_coefficient=field.astype(np.complex64),delay_s=delay,vertices=vertices,interaction_types=types,
                    object_ids=objects,primitive_ids=primitives,primitive_triangles=triangles,normals=normals,
                    k_tx=np.asarray(b.k_tx).T,k_rx=np.asarray(b.k_rx).T,source_positions=tx[si+first],target_positions=rx[ri],
                    object_map_json=np.array(json.dumps(object_map,sort_keys=True)))
                if not np.isfinite(h).all() or not np.isfinite(coeff).all() or not np.isfinite(delay).all():raise ValueError('Nonfinite independent SDK result')
                check(sum(v.nbytes for v in arrays.values())+1024**2)
                atomic_npz(path,**arrays)
                save_json(path.with_suffix('.receipt.json'),dict(sha256=sha_file(path),index=i,first_tx=first,paths=count,SDK_s=solve_s,
                    total_chunk_s=time.monotonic()-begin,allocator_peak_bytes=int(dr.detail.malloc_watermark(dr.JitBackend.CUDA))))
                chunks.append(path)
                save_json(out/'progress.json',dict(index=i,first_tx_completed=first,committed_chunks_this_state=len(chunks),committed_states=committed_states,
                    elapsed_worker_wall_s=time.monotonic()-started,disk_bytes=folder_bytes(out),independent_validation='Unknown_pending_CPU_array_comparison'))
                del paths,b,cir,tau,raw,arrays
            h=np.concatenate([np.load(path,allow_pickle=False)['H'] for path in chunks],axis=2)
            check(h.nbytes*2+1024**2);atomic_npz(target/'H.npz',H=h)
            files={str(path.relative_to(target)):sha_file(path) for path in target.iterdir() if path.is_file() and path.name!='complete.json'}
            save_json(target/'complete.json',dict(index=i,files=files,total_state_wall_s=time.monotonic()-case_start,
                SDK_chunks=16,path_capture='All compacted paths, all CIR/Field coefficients, delays, vertices, primitive triangles/IDs; no strongest-path filtering'))
            committed_states.append(i);event('state_reference_committed',index=i,committed_states=committed_states)
            del scene,solver
        event('all_reference_computation_finished_pending_independent_CPU_audit',committed_states=committed_states)
    except BaseException as exc:
        event('resource_stop_preserved' if isinstance(exc,BudgetStop) else 'failed_preserved_no_automatic_retry',error=repr(exc));raise


def supervise(out,grant):
    a=json.loads(grant.read_text())
    if a.get('approved') is not True or a.get('scope')!='CSI30s_SDK16_independent_reaudit':raise ValueError('New matching explicit grant required')
    p=verify_proposal(Path(a['proposal_folder']),a['proposal_sha256'])
    if a['gpu']!=p['gpu'] or a['hard_wall_s']!=p['requested_hard_wall_s'] or a['max_VRAM_bytes']!=p['max_VRAM_bytes'] or a['max_new_storage_bytes']!=p['max_new_storage_bytes']:raise ValueError('Grant exceeds frozen proposal')
    with grant.with_suffix('.lock').open('a+b') as lock,(ROOT/'.state/platform_gpu3.lock').open('a+b') as device:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(device,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=grant.with_suffix('.ledger.json');bill=json.loads(ledger.read_text()) if ledger.exists() else dict(spent_wall_s=0.,runs=[])
        remaining=a['hard_wall_s']-bill['spent_wall_s']
        if remaining<=10:raise BudgetStop('Independent grant exhausted')
        for name,h in a['execution_sources'].items():
            if sha_file(ROOT/name)!=h:raise ValueError('Execution source changed before launch')
        env=os.environ.copy();env.update(WIWO_GPU='3',CUDA_VISIBLE_DEVICES='3',PYTHONPATH=str(ROOT/'src'),WIWO_PROJECT_ROOT=str(ROOT),
            WIWO_SDK16_TOKEN=sha_file(grant),WIWO_SDK16_REMAINING_S=str(remaining),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
        run_id=len(bill['runs']);log=(out/f'worker_{run_id:02d}.log').open('ab');start=time.monotonic()
        command=[sys.executable,str(out/'executed_sources/run_platform_sdk16_audit.py'),'worker','--out',str(out),'--grant',str(grant)]
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(out/'worker_pid.json',dict(pid=child.pid,gpu=3,remaining_wall_s=remaining))
        forced=None;peak_nvml=0;next_memory=0.
        try:
            while child.poll() is None:
                elapsed=time.monotonic()-start
                if elapsed>=remaining-2:forced='independent_hard_wall_cap'
                elif folder_bytes(out)>a['max_new_storage_bytes']-1024**2:forced='independent_storage_cap'
                if elapsed>=next_memory:
                    next_memory=elapsed+1
                    try:
                        q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=.5)
                        for row in q.stdout.splitlines():
                            bits=row.split(',')
                            if len(bits)==2 and bits[0].strip()==str(child.pid) and bits[1].strip().isdigit():peak_nvml=max(peak_nvml,int(bits[1])*1024**2)
                    except subprocess.TimeoutExpired:
                        with (out/'monitor_warnings.jsonl').open('a') as f:f.write(json.dumps(dict(elapsed_wall_s=elapsed,warning='nvidia-smi_timeout_allocator_guard_still_active'))+'\n')
                    if peak_nvml>a['max_VRAM_bytes']:forced='independent_process_VRAM_cap'
                save_json(ledger,dict(**bill,active_run=dict(pid=child.pid,elapsed_wall_s=elapsed,remaining_start_s=remaining,peak_process_VRAM_bytes=peak_nvml)))
                if forced:os.killpg(child.pid,signal.SIGKILL);break
                time.sleep(.1)
            code=child.wait()
        finally:
            if child.poll() is None:
                forced=forced or 'supervisor_exception_preserved'
                os.killpg(child.pid,signal.SIGKILL);child.wait()
            elapsed=time.monotonic()-start;log.close();bill['spent_wall_s']+=elapsed
            bill['runs'].append(dict(worker_pid=child.pid,wall_s=elapsed,returncode=child.returncode,forced_stop=forced,peak_process_VRAM_bytes=peak_nvml))
            save_json(ledger,bill)
        if forced:save_json(out/'supervisor_stop.json',dict(reason=forced,preserved=True,no_automatic_retry=True))
    # Always audit committed artifacts, including a stopped or failed worker.
    # No CUDA environment is visible to the separately scoped CPU phase.
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',WIWO_GPU='-1',PYTHONPATH=str(ROOT/'src'),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    command=[sys.executable,str(out/'executed_sources/audit_platform_sdk16_results.py'),'--out',str(out),'--grant',str(grant)]
    with (out/'CPU_audit.log').open('ab') as log:post=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    save_json(out/'supervisor_result.json',dict(worker_returncode=code,CPU_audit_returncode=post.returncode,spent_GPU_wall_s=bill['spent_wall_s'],overall='Partial'))
    return code or post.returncode


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','supervise']);p.add_argument('--out',type=Path,required=True);p.add_argument('--grant',type=Path,required=True);args=p.parse_args()
    if args.mode=='worker':worker(args.out.resolve(),args.grant.resolve())
    else:sys.exit(supervise(args.out.resolve(),args.grant.resolve()))


if __name__=='__main__':main()
