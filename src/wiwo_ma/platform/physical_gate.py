"""Short physical-event SDK gate, prepared on CPU and separately authorized.

Reuses the native-gate budget supervisor, never its old grant. Exact frozen
operands feed an independently loaded, baked-actor SDK reference. Path files
are diagnostic outputs only. Completed native/reference stages are immutable.
"""
from copy import deepcopy
from pathlib import Path
import json,os,time
import numpy as np
from .config import ROOT,validate,fingerprint
from .physics import PhysicalWorld
from .storage import plain,encode,atomic_json,CSIWriter
from .jobs import inside,Budget,folder_bytes,ResourceStop
from .validation import sha,bake_actors,frozen_check
from .path_capture import canonical_paths


def prepare(folder,config):
    out=inside(folder)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);full=validate(config);small=deepcopy(full)
    from .scenario_checks import StaticGeometry
    from .dataset_audit import segment_box
    geometry=StaticGeometry(full)
    def expectation(state,directory):
        tx=np.array(state['tx_port_positions']);rx=np.array(state['rx_port_positions']);blocked=np.zeros((len(rx),len(tx)),bool)
        for actor in state['actors']:blocked|=segment_box(tx[None],rx[:,None],actor['position_m'],actor['rotation_matrix'],actor['size_m'])
        blocked|=np.array([[geometry.blocked(a,b) for a in tx] for b in rx])
        np.savez_compressed(directory/'CPU_LOS_expectation.npz',direct_LOS=~blocked)
    small['base_stations'][0]['panels']=[small['base_stations'][0]['panels'][i] for i in (0,3)]
    for p in small['base_stations'][0]['panels']:p['array']=dict(kind='single',rows=1,cols=1,spacing_m=.005)
    small['users']=small['users'][:2];small=validate(small);cases=[]
    for name,c,times in [('full64',full,[1.56,2.,2.52]),('small',small,[.08,.12,.16,1.56,1.60,2.40,2.44,2.48])]:
        atomic_json(out/(name+'_config.json'),c);w=PhysicalWorld(c)
        for i,t in enumerate(times):
            w.advance_to(t);state=w.state();snapshot=w.snapshot()
            case=dict(id=name+'_'+str(i),config=name+'_config.json',state=name+'_'+str(i)+'_state.json',snapshot=name+'_'+str(i)+'_snapshot.json',group=name,time_s=t)
            atomic_json(out/case['state'],plain(state));atomic_json(out/case['snapshot'],encode(snapshot));bake_actors(plain(state),out/case['id']);expectation(state,out/case['id']);cases.append(case)
            if name=='small' and i==1:ancestor=snapshot
        if name=='small':
            w.restore(ancestor)
            ancestor_case=dict(id='small_restored_ancestor',config='small_config.json',state='small_restored_ancestor_state.json',snapshot='small_restored_ancestor_snapshot.json',group='small',time_s=.12)
            atomic_json(out/ancestor_case['state'],plain(w.state()));atomic_json(out/ancestor_case['snapshot'],encode(w.snapshot()));bake_actors(plain(w.state()),out/ancestor_case['id']);expectation(w.state(),out/ancestor_case['id']);cases.append(ancestor_case)
            w.advance_to(times[-1]);restored=w.state()
            assert np.array_equal(restored['tx_port_positions'],state['tx_port_positions']) and np.array_equal(restored['actors'][0]['vertices'],state['actors'][0]['vertices'])
            case=dict(id='small_restored',config='small_config.json',state='small_restored_state.json',snapshot='small_restored_snapshot.json',group='small',time_s=times[-1],restore_from_case='small_1')
            atomic_json(out/case['state'],plain(restored));atomic_json(out/case['snapshot'],encode(w.snapshot()));bake_actors(plain(restored),out/case['id']);expectation(restored,out/case['id']);cases.append(case)
    sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/wiwo_ma/platform').glob('*.py')}
    for name in ['mechanics.py','scene.py','v4/collision.py','v5/actor_mesh.py']:sources['src/wiwo_ma/'+name]=sha(ROOT/'src/wiwo_ma'/name)
    proposal=dict(schema='native-platform-independent-gate-v1',workflow='physical_dynamic_v1',status='prepared_NOT_authorized_NOT_started',
        cases=cases,inputs={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},source_sha256=sources,
        configuration_sha256=[fingerprint(full),fingerprint(small)],gpu=3,limits=dict(max_wall_s=600.,max_allocator_bytes=12*1024**3,max_disk_bytes=512*1024**2),
        expected_wall_s=[360,540],new_Full64_queries=6,small_two_source_queries=20,original_801_and_3001_recomputed=0,
        per_user_NMSE_threshold=.001,scope='Three full-scale event states + eight selected two-source states and two restored states; not a continuous 4s CSI dataset',
        purposes=['Real native actor LOS/NLOS/LOS event','Same-instance continuous/jump/restored geometry updates','Complete path/CIR/delay and per-UE CSI independent comparison','Real HDF rows and provenance'],
        untouched_old_grants=True,reference='Fresh SDK scene for each state, independently baked actor vertices, no HDF/Full64 answers used as inputs')
    atomic_json(out/'proposal.json',proposal);atomic_json(out/'state.json',dict(status='prepared_NOT_authorized_NOT_started',new_GPU_seconds=0));return proposal


def compare_paths(native_files,reference_files):
    if native_files and reference_files:
        with np.load(native_files[0],allow_pickle=False) as a:polarized='schema' in a and str(a['schema'])=='sdk-public-polarized-path-table-v2'
        if polarized:
            from .polarized_paths import compare_public_tables
            def load(files):
                tables=[]
                for file in files:
                    with np.load(file,allow_pickle=False) as a:tables.append({k:a[k] for k in a.files})
                return tables
            return compare_public_tables(load(native_files),load(reference_files))
    nonfinite=[]
    def collect(files):
        result={}
        for file in files:
            with np.load(file,allow_pickle=False) as a:
                if any(not np.isfinite(a[k]).all() for k in ('vertices','delay_s','coefficient')):nonfinite.append(str(file))
                words=canonical_paths(a)
                for key,indices in words.items():
                    for i in indices:result.setdefault(key,[]).append(tuple(np.asarray(a[k][i]) for k in ('vertices','delay_s','coefficient')))
        return result
    a,b=collect(native_files),collect(reference_files);missing=set(b)-set(a);extra=set(a)-set(b)
    geometry=delay=cir=True;ambiguous=0
    for key in set(a)&set(b):
        if len(a[key])!=len(b[key]):geometry=delay=cir=False;continue
        # Duplicate words require a non-arbitrary matching algorithm. Preserve
        # Unknown rather than silently pair by output order or hide instances.
        if len(a[key])!=1:ambiguous+=1;continue
        x,y=a[key][0],b[key][0];geometry &= np.array_equal(x[0],y[0]);delay &= np.array_equal(x[1],y[1]);cir &= np.array_equal(x[2],y[2])
    passed=not missing and not extra and geometry and delay and cir and not ambiguous
    return dict(status='Failed' if nonfinite else 'Passed' if passed else 'Partial' if not missing and not extra else 'Failed',
        nonfinite_chunks=nonfinite,
        missing_words=len(missing),extra_words=len(extra),native_words=len(a),reference_words=len(b),duplicate_words_Unknown=ambiguous,
        vertices_bitwise=bool(geometry),delays_bitwise=bool(delay),CIR_bitwise=bool(cir))


def audit(folder):
    """Fresh CPU audit from raw outputs; does not trust worker exit status."""
    from .dataset_audit import nmse_all_users
    out=inside(folder);p=json.loads((out/'proposal.json').read_text());frozen_check(out,p,allow_archived=True)
    dest=out/('independent_CPU_audit_'+str(time.time_ns()));dest.mkdir();rows=[]
    for case in p['cases']:
        d=out/case['id'];files=[d/'native.npz',d/'reference.npz']
        if not all(x.exists() for x in files):rows.append(dict(case=case['id'],status='Unknown_missing_raw_stage'));continue
        worker=json.loads((d/'report.json').read_text()) if (d/'report.json').exists() else {}
        for kind in ('native','reference'):
            if worker.get(kind+'_sha256')!=sha(d/(kind+'.npz')):raise ValueError('Uncommitted or changed raw CSI: '+case['id'])
        with np.load(files[0]) as x,np.load(files[1]) as y:error,_,_=nmse_all_users(x['H'],y['H'])
        native=sorted((d/'native_paths').glob('*.npz'));reference=sorted((d/'reference_paths').glob('*.npz'))
        if len(native)!=int(np.ceil(len(json.loads((out/case['state']).read_text())['tx_port_positions'])/json.loads((out/case['config']).read_text())['rt']['tx_device_chunk'])) or len(native)!=len(reference):
            rows.append(dict(case=case['id'],status='Unknown_incomplete_path_chunks'));continue
        for file in native+reference:
            if sha(file)!=json.loads(file.with_suffix('.receipt.json').read_text())['sha256']:raise ValueError('Path receipt mismatch')
        paths=compare_paths(native,reference);expected=np.load(d/'CPU_LOS_expectation.npz')['direct_LOS'];observed=[]
        for side in (native,reference):
            found=np.zeros_like(expected)
            for file in side:
                with np.load(file) as x:
                    los=~np.any(x['interaction_types']!=0,axis=1);found[x['user'][los],x['tx_port'][los]]=True
            observed.append(int(np.count_nonzero(found!=expected)))
        csi_ok=bool(np.isfinite(error).all() and np.all(error<=.001))
        rows.append(dict(case=case['id'],CSI_status='Passed' if csi_ok else 'Failed',per_user_NMSE=error.tolist(),
            paths=paths,geometric_LOS_mismatches_native_reference=observed,
            status='Failed' if not csi_ok or paths['status']=='Failed' or observed!=[0,0] else 'Passed' if paths['status']=='Passed' else 'Partial'))
    counts={k:sum(x['status']==k for x in rows) for k in ('Passed','Partial','Failed')};counts['Unknown']=sum(x['status'].startswith('Unknown') for x in rows)
    result=dict(status='Failed' if counts['Failed'] else 'Passed_selected_states_only' if counts['Passed']==len(p['cases']) else 'Partial',counts=counts,cases=rows,new_GPU_seconds=0,
        old_dataset_labels_unchanged=True,whole_4s_continuous_CSI='Unknown')
    atomic_json(dest/'report.json',result);return result


def close_cpu(folder):
    out=inside(folder);result=audit(out)
    atomic_json(out/'state.json',dict(status='finished_with_independent_CPU_audit',scientific_status=result['status'],whole_platform='Partial'))
    dest=out/('delivery_snapshot_'+str(time.time_ns()));dest.mkdir()
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and dest not in p.parents and p.suffix not in ('.log','.lock','.pending')}
    atomic_json(dest/'manifest.json',dict(status='Partial',files=files,independent_audit=result,old_dataset_labels_unchanged=True))
    return result


def worker(folder):
    from .native_rt import NativeRT
    from .validation import reference
    from .storage import decode
    out=inside(folder);p=json.loads((out/'proposal.json').read_text());frozen_check(out,p)
    budget=Budget(out,p['limits']);engine=None;group=None;results=[]
    # Archive the actual executed platform source before any CUDA work.
    for name in p['source_sha256']:
        dest=out/'executed_sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and sha(dest)!=p['source_sha256'][name]:raise ValueError('Archived source mismatch')
        if not dest.exists():dest.write_bytes((ROOT/name).read_bytes())
    def sink(directory):
        directory.mkdir(exist_ok=True)
        def save(first,arrays):
            budget.check();file=directory/f'chunk_{first:02d}.npz'
            if file.exists():raise FileExistsError('Never overwrite a captured path chunk')
            if folder_bytes(out)+sum(x.nbytes for x in arrays.values())>p['limits']['max_disk_bytes']:raise ResourceStop('Path capture storage reserve')
            temp=file.with_suffix('.pending')
            with temp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
            temp.replace(file)
            atomic_json(file.with_suffix('.receipt.json'),dict(sha256=sha(file),paths=len(arrays['user'])))
        return save
    try:
        for case in p['cases']:
            dest=out/case['id'];budget.check()
            if (dest/'report.json').exists():results.append(json.loads((dest/'report.json').read_text()));continue
            atomic_json(out/'state.json',dict(status='running_physical_native_gate',case=case['id'],elapsed_wall_s=time.monotonic()-budget.started))
            c=validate(json.loads((out/case['config']).read_text()));s=json.loads((out/case['state']).read_text())
            if not (dest/'native.npz').exists():
                if (dest/'native_paths').exists():raise RuntimeError('Uncommitted native path stage preserved; inspect before explicit stage recovery')
                if group!=case['group'] or engine is None:
                    if engine is not None:engine.close()
                    engine=NativeRT(c,check_budget=budget.check);budget.engine=engine;group=case['group']
                engine.complete_path_sink=sink(dest/'native_paths');r=engine.solve(s)
                w=CSIWriter(dest/'dataset',c)
                try:w.stage(r,s,decode(json.loads((out/case['snapshot']).read_text())));w.commit_pending()
                finally:w.close()
                np.savez_compressed(dest/'native.npz',H=r['H']);atomic_json(dest/'native_metadata.json',{k:plain(v) for k,v in r.items() if k!='H'})
            if not (dest/'reference.npz').exists():
                if (dest/'reference_paths').exists():raise RuntimeError('Uncommitted reference path stage preserved; inspect before explicit stage recovery')
                h,meta=reference(c,s,dest,budget.check,complete_path_sink=sink(dest/'reference_paths'))
                np.savez_compressed(dest/'reference.npz',H=h);atomic_json(dest/'reference_metadata.json',meta)
            from .dataset_audit import nmse_all_users
            with np.load(dest/'native.npz') as a,np.load(dest/'reference.npz') as b:error,absolute,relative=nmse_all_users(a['H'],b['H'])
            paths=compare_paths(sorted((dest/'native_paths').glob('*.npz')),sorted((dest/'reference_paths').glob('*.npz')))
            report=dict(case=case['id'],status='Passed' if np.isfinite(error).all() and np.all(error<=.001) else 'Failed',
                per_user_NMSE=error.tolist(),paths=paths,native_sha256=sha(dest/'native.npz'),reference_sha256=sha(dest/'reference.npz'),all_users_included=True)
            atomic_json(dest/'report.json',report);results.append(report)
        atomic_json(out/'report.json',dict(status='Passed' if all(r['status']=='Passed' and r['paths']['status']=='Passed' for r in results) else 'Partial',cases=results,
            limitation='Selected-state gate only. Full 4s continuous native CSI remains Unknown. No label promotion of old datasets.'))
        atomic_json(out/'state.json',dict(status='finished_physical_gate_pending_independent_CPU_audit',completed_cases=len(results)))
    except BaseException as e:
        atomic_json(out/'state.json',dict(status='failed_preserved_no_automatic_retry',error=repr(e),completed_cases=len(results)));raise
    finally:
        if engine is not None:engine.close()
