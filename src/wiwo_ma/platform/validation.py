"""Small independent original-SDK gate. Preparing this module never imports CUDA.

Reference scenes are loaded afresh; moving actors are baked from frozen operands
into PLY, rather than updated through the adapter under test. Neither reference
path discovery nor CFR arrangement calls NativeRT or reads its output.
"""
from copy import deepcopy
import hashlib,json,os,signal,subprocess,time,fcntl
from pathlib import Path
import numpy as np
from .config import ROOT,validate,fingerprint,frequency_offsets
from .physics import PhysicalWorld
from .storage import atomic_json,plain,encode,CSIWriter
from .examples import munich_example
from .jobs import inside,Budget,ResourceStop,folder_bytes
from .scene_geometry import scene_xml


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bake_actors(state,folder):
    folder.mkdir(exist_ok=True)
    for a in state['actors']:
        vertices=np.asarray(a['vertices'],np.float32);faces=np.asarray(a['faces'],np.uint32)
        lines=['ply','format ascii 1.0','element vertex '+str(len(vertices)),'property float x','property float y','property float z','element face '+str(len(faces)),'property list uchar int vertex_indices','end_header']
        lines+=[' '.join(format(float(x),'.9g') for x in v) for v in vertices];lines+=['3 '+' '.join(map(str,f)) for f in faces]
        (folder/('reference_actor_'+a['id']+'.ply')).write_text('\n'.join(lines)+'\n')


def prepare(folder):
    folder=inside(folder)
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True)
    full=munich_example();small=deepcopy(full)
    small['base_stations'][0]['panels']=[deepcopy(full['base_stations'][0]['panels'][i]) for i in (0,3)]
    for p in small['base_stations'][0]['panels']:p['array']=dict(kind='single',rows=1,cols=1,spacing_m=.005)
    small['users']=small['users'][:2];small['actions']=[]
    for p in small['base_stations'][0]['panels']:
        small['actions'].append(dict(time_s=0.,bs_id='bs0',panel_id=p['id'],type='target_pose',
            position_m=(np.array(p['position_m'])+[.04,0,0]).tolist(),orientation_deg=(np.array(p['orientation_deg'])+[2.,0,0]).tolist()))
    small=validate(small)
    cases=[]
    for name,c,times in [('full64',full,[1.2]),('small',small,[0.,.1,.2])]:
        atomic_json(folder/(name+'_config.json'),c);w=PhysicalWorld(c)
        for i,t in enumerate(times):
            w.advance_to(t);state=w.state();snap=w.snapshot()
            atomic_json(folder/(name+f'_{i}_state.json'),plain(state));atomic_json(folder/(name+f'_{i}_snapshot.json'),encode(snap))
            cases.append(dict(id=name+f'_{i}',config=name+'_config.json',state=name+f'_{i}_state.json',snapshot=name+f'_{i}_snapshot.json'))
            if name=='small' and i==1:ancestor=snap
        if name=='small':
            w.restore(ancestor);w.advance_to(.2)
            assert np.array_equal(state['tx_port_positions'],w.state()['tx_port_positions'])
            atomic_json(folder/'small_restored_state.json',plain(w.state()));atomic_json(folder/'small_restored_snapshot.json',encode(w.snapshot()))
            cases.append(dict(id='small_restored',config='small_config.json',state='small_restored_state.json',snapshot='small_restored_snapshot.json'))
    for case in cases:bake_actors(json.loads((folder/case['state']).read_text()),folder/case['id'])
    sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/wiwo_ma/platform').glob('*.py') if p.stem not in ('web','controller','scene_geometry','delivery')}
    for name in ['scene.py','mechanics.py','v4/collision.py','v5/actor_mesh.py']:sources['src/wiwo_ma/'+name]=sha(ROOT/'src/wiwo_ma'/name)
    inputs={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}
    proposal=dict(schema='native-platform-independent-gate-v1',status='prepared_not_started',cases=cases,
        configuration_sha256=[fingerprint(full),fingerprint(small)],inputs=inputs,source_sha256=sources,
        limits=dict(max_wall_s=600.,max_allocator_bytes=12*1024**3,max_disk_bytes=512*1024**2),gpu=3,
        expected_wall_s=[180,480],new_Full64_queries=2,small_two_spatial_source_queries=8,
        original_801_Full64_recomputed=0,per_user_NMSE_threshold=.001,
        reference='Fresh original SDK scenes, independent actor baked PLY, same frozen Float32 endpoints/orientations, fixed source chunks/seed, original SDK CFR',
        scope_exclusions=['No long generation','No old grant reuse','No parameter tuning from Full64 answers'])
    atomic_json(folder/'proposal.json',proposal);atomic_json(folder/'state.json',dict(status='prepared_not_started',GPU_started=False))
    return proposal


def frozen_check(folder,p,*,allow_archived=False):
    for name,digest in p['inputs'].items():
        if sha(folder/name)!=digest:raise ValueError('Frozen gate input changed: '+name)
    for name,digest in p['source_sha256'].items():
        saved=folder/'executed_sources'/name
        source=saved if allow_archived and saved.exists() else ROOT/name
        if sha(source)!=digest:raise ValueError('Gate source changed after preparation: '+name)


def reference(config,state,folder,check,complete_path_sink=None):
    # This imports only after explicit authorization in the bounded worker.
    import sionna.rt as rt
    import mitsuba as mi
    import drjit as dr
    scene=rt.load_scene(str(scene_xml(config['scene'])),merge_shapes=False)
    scene.frequency=config['radio']['carrier_hz'];r=config['radio']
    for side in ['tx','rx']:
        setattr(scene,side+'_array',rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern=r[side+'_pattern'],polarization=r[side+'_polarization']))
    if config['scene'].get('material_override'):
        material=rt.ITURadioMaterial(name='reference_static',itu_type=config['scene']['material_override'],thickness=.1);scene.add(material)
        for o in scene.objects.values():o.radio_material=material
    for a in state['actors']:
        path=folder/('reference_actor_'+a['id']+'.ply')
        vertices=np.asarray(a['vertices'],np.float32);faces=np.asarray(a['faces'],np.uint32)
        mat=rt.ITURadioMaterial(name='reference_material_'+a['id'],itu_type=a['material'],thickness=.01)
        obj=rt.SceneObject(name='reference_'+a['id'],fname=str(path),radio_material=mat);scene.add(obj)
        actual=np.asarray(obj.mi_mesh.vertex_positions_buffer()).reshape(-1,3)
        if not np.array_equal(actual.view(np.uint32),vertices.view(np.uint32)):raise ValueError('Baked reference actor differs from frozen operands')
    # Independently enumerate polarization groups; do not call adapter mapping.
    tp=len(scene.tx_array.antenna_pattern.patterns);rp=len(scene.rx_array.antenna_pattern.patterns)
    tx=np.asarray(state['tx_port_positions'],np.float32)[::tp];rx=np.asarray(state['rx_port_positions'],np.float32)[::rp]
    txangles=np.asarray(state['tx_port_orientations_rad'],np.float32)[::tp];rxangles=np.asarray(state['rx_port_orientations_rad'],np.float32)[::rp]
    rm=np.asarray(state['rx_port_map'],int)[::rp]
    for i,xyz in enumerate(rx):scene.add(rt.Receiver(name='reference_rx'+str(i),position=mi.Point3f(xyz.tolist()),orientation=mi.Point3f(rxangles[i].tolist())))
    H=np.zeros((len(config['users']),max(np.bincount(rm[:,0]))*rp,len(tx)*tp,r['num_subcarriers']),np.complex64)
    solver=rt.PathSolver(deterministic=True);solver.loop_mode='evaluated';dr.set_flag(dr.JitFlag.SymbolicCalls,False);dr.set_flag(dr.JitFlag.SymbolicLoops,False)
    names=[];calls=0;started=time.perf_counter();chunks=[]
    settings={k:config['rt'][k] for k in ['max_depth','max_num_paths_per_src','samples_per_src','synthetic_array','los','specular_reflection','diffuse_reflection','refraction','diffraction','edge_diffraction','diffraction_lit_region','seed']}
    for first in range(0,len(tx),config['rt']['tx_device_chunk']):
        check();begin=time.perf_counter()
        for n in names:scene.remove(n)
        names=[];indices=list(range(first,min(len(tx),first+config['rt']['tx_device_chunk'])))
        for i in indices:
            n='reference_tx'+str(i);scene.add(rt.Transmitter(name=n,position=mi.Point3f(tx[i].tolist()),orientation=mi.Point3f(txangles[i].tolist())));names.append(n)
        paths=solver(scene=scene,**settings);calls+=1;dr.eval();dr.sync_thread()
        if not np.array_equal(np.asarray(paths.sources).T.view(np.uint32),tx[indices].view(np.uint32)) or not np.array_equal(np.asarray(paths.targets).T.view(np.uint32),rx.view(np.uint32)):raise ValueError('Reference endpoints changed')
        raw=np.asarray(paths.cfr(frequencies=mi.Float(frequency_offsets(config)),num_time_steps=1,normalize_delays=False,normalize=False,out_type='numpy'))
        if complete_path_sink is not None:
            from .path_capture import complete_V_paths
            complete_path_sink(first,complete_V_paths(paths,scene,config,first,rm))
        for ri,m in enumerate(rm):
            for rpi in range(rp):
                for ti,source in enumerate(indices):
                    for tpi in range(tp):H[m[0],m[1]*rp+rpi,source*tp+tpi,:]=raw[ri,rpi,ti,tpi,0,:]
        chunks.append(time.perf_counter()-begin);check()
    if not np.isfinite(H).all():raise ValueError('Nonfinite independent reference')
    return H,dict(SDK_calls=calls,query_wall_s=time.perf_counter()-started,chunk_wall_s=chunks,independent_baked_actor_geometry='Passed',endpoints='Passed')


def worker(folder):
    proposal=json.loads((inside(folder)/'proposal.json').read_text())
    if proposal.get('workflow')=='physical_dynamic_v1':
        from .physical_gate import worker as physical_worker
        return physical_worker(folder)
    from .native_rt import NativeRT
    from .storage import decode
    folder=inside(folder);p=json.loads((folder/'proposal.json').read_text());frozen_check(folder,p)
    budget=Budget(folder,p['limits']);results=[];start=time.monotonic()
    try:
        for case in p['cases']:
            out=folder/case['id'];out.mkdir(exist_ok=True)
            # A committed case is never solved again on explicit resume.
            if (out/'report.json').exists():results.append(json.loads((out/'report.json').read_text()));continue
            atomic_json(folder/'state.json',dict(status='running_native_gate',case=case['id'],elapsed_wall_s=time.monotonic()-start))
            c=validate(json.loads((folder/case['config']).read_text()));state=json.loads((folder/case['state']).read_text());snapshot=decode(json.loads((folder/case['snapshot']).read_text()))
            if not (out/'native.npz').exists():
                engine=NativeRT(c,check_budget=budget.check);budget.engine=engine;result=engine.solve(state)
                writer=CSIWriter(out/'dataset',c)
                try:writer.stage(result,state,snapshot);writer.commit_pending()
                finally:writer.close()
                np.savez_compressed(out/'native.npz',H=result['H']);atomic_json(out/'native_metadata.json',{k:plain(v) for k,v in result.items() if k!='H'})
                engine.close()
            budget.check()
            if not (out/'reference.npz').exists():
                H,meta=reference(c,state,out,budget.check);np.savez_compressed(out/'reference.npz',H=H);atomic_json(out/'reference_metadata.json',meta)
            a=np.load(out/'native.npz')['H'];b=np.load(out/'reference.npz')['H'];den=np.sum(abs(b.astype(np.complex128))**2,axis=(1,2,3));num=np.sum(abs(a.astype(np.complex128)-b)**2,axis=(1,2,3))
            error=np.divide(num,den,out=np.where(num==0,0.,np.inf),where=den>0)
            report=dict(case=case['id'],status='Passed' if bool(np.all(error<=p['per_user_NMSE_threshold'])) else 'Failed',per_user_NMSE=error.tolist(),max_NMSE=float(max(error)),zero_reference_users=np.flatnonzero(den==0).tolist(),
                native_sha256=sha(out/'native.npz'),reference_sha256=sha(out/'reference.npz'),all_users_included=True)
            atomic_json(out/'report.json',report);results.append(report)
        atomic_json(folder/'report.json',dict(status='Passed' if all(r['status']=='Passed' for r in results) else 'Failed',cases=results,
            elapsed_wall_s=time.monotonic()-start,scope='Five frozen states only; no whole domain certification; long CSI remains Unknown'))
        atomic_json(folder/'state.json',dict(status='finished_independent_gate',results=[r['status'] for r in results]))
    except BaseException as exc:
        atomic_json(folder/'state.json',dict(status='failed_preserved_no_automatic_retry',error=repr(exc),completed_cases=results));raise


def run(folder,authorization):
    """Independent bounded supervisor; no old budget or generic job grant accepted."""
    folder=inside(folder);grant=inside(authorization);a=json.loads(grant.read_text());p=json.loads((folder/'proposal.json').read_text());frozen_check(folder,p)
    if a.get('approved') is not True or a.get('scope')!='native_platform_independent_gate' or a.get('proposal_sha256')!=sha(folder/'proposal.json'):raise ValueError('Matching explicit native gate authorization required')
    if a['gpu']!=p['gpu'] or any(a[k]!=p['limits'][k] for k in p['limits']):raise ValueError('Authorization and proposed resource caps differ')
    lock=grant.with_suffix('.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=grant.with_suffix('.ledger.json');receipt=json.loads(ledger.read_text()) if ledger.exists() else dict(spent_wall_s=0.,runs=[])
    remaining=a['max_wall_s']-receipt['spent_wall_s']
    if remaining<15:raise ResourceStop('Native gate budget exhausted')
    env=os.environ.copy();env.update(WIWO_GPU=str(a['gpu']),CUDA_VISIBLE_DEVICES=str(a['gpu']),WIWO_NATIVE_GATE_TOKEN=sha(folder/'proposal.json'))
    log=(folder/'worker.log').open('ab');start=time.monotonic();proc=subprocess.Popen([str(ROOT/'scripts/platform/run'),'native-gate-worker','--proposal',str(folder)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(folder/'worker_pid.json',dict(pid=proc.pid,gpu=a['gpu'],hard_remaining_wall_s=remaining));forced=False
    try:
        while proc.poll() is None:
            if time.monotonic()-start>=remaining or folder_bytes(folder)>a['max_disk_bytes']:
                os.killpg(proc.pid,signal.SIGKILL);forced=True;break
            time.sleep(.1)
        code=proc.wait()
    finally:
        elapsed=time.monotonic()-start;receipt['spent_wall_s']+=elapsed;receipt['runs'].append(dict(elapsed_wall_s=elapsed,returncode=proc.returncode,forced_kill=forced));atomic_json(ledger,receipt);log.close();lock.close()
    if forced:atomic_json(folder/'state.json',dict(status='interrupted_resource_cap_preserved',no_automatic_retry=True))
    if not forced and code==0 and p.get('workflow')=='physical_dynamic_v1':
        from .physical_gate import close_cpu
        close_cpu(folder)
    return code
