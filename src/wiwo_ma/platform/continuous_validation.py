"""Independent selected-state SDK checks after closing a continuous writer.

Only geometry/configuration are supplied to the fresh reference solver. HDF
CSI and native path files enter comparison afterwards, never the solver.
The caller's same explicit job budget covers all reference work.
"""
from pathlib import Path
import json,time
import numpy as np
from .storage import CSIReader,atomic_json
from .config import ROOT,fingerprint
from .validation import sha,bake_actors,reference
from .physical_gate import compare_paths
from .dataset_audit import nmse_all_users


def numbers(values):
    # JSON null plus explicit nonfinite-user indices preserves an infinite
    # error failure; never replace a zero-reference denominator with epsilon.
    return [float(x) if np.isfinite(x) else None for x in values]


def select_reference_indices(reader):
    """CPU-only sample selection; observations select *where*, never RT inputs.

    Two additional, deterministic adverse cases: lowest mean user energy and
    largest per-user relative CSI transition, excluding already fixed samples.
    All users/ports remain in each reference. No epsilon hides zero channels.
    """
    fixed=list(reader.config['dataset'].get('independent_reference_frames',[]));selected=list(fixed);choices=[]
    if reader.config['dataset'].get('independent_reference_extremes',0):
        power=[];jump=[];previous=None
        for i in range(reader.count):
            h=reader.frame(i)['H'].astype(np.complex128);axes=tuple(range(1,h.ndim));energy=np.mean(abs(h)**2,axis=axes)
            power.append((float(energy.min()),i,int(energy.argmin())))
            if previous is not None:
                a=np.sum(abs(h-previous)**2,axis=axes);b=np.sum(abs(previous)**2,axis=axes)
                ratio=np.divide(a,b,out=np.where(a==0,0.,np.inf),where=b>0)
                jump.append((float(ratio.max()),i,int(ratio.argmax())))
            previous=h
        for name,ranking in (('minimum_user_mean_channel_energy',sorted(power)),('largest_user_relative_CSI_step',sorted(jump,key=lambda x:(-x[0],x[1],x[2])))):
            choice=next((x for x in ranking if x[1] not in selected),None)
            if choice:
                value,index,user=choice;selected.append(index);choices.append(dict(rule=name,index=index,user=user,score=value if np.isfinite(value) else None,score_infinite=not np.isfinite(value)))
    return sorted(selected),dict(fixed_indices=fixed,additional=choices,indices=sorted(selected),
        scope='Selection only: independent solver receives saved geometry/config, never these CSI arrays or native path answers')


def validate_selected(folder,indices,budget,should_stop=None):
    folder=Path(folder);reader=CSIReader(folder/'csi.h5');rows=[]
    if not reader.complete:raise ValueError('Close complete continuous data before independent checks')
    import drjit as dr
    from types import SimpleNamespace
    budget.engine=SimpleNamespace(dr=dr)
    def check():
        budget.check()
        if should_stop is not None and should_stop():raise InterruptedError('Independent audit paused at a reference boundary; committed native frames preserved')
    for i in indices:
        check();q=reader.frame(i);state=q['state'];dest=folder/'references'/str(i);dest.mkdir(parents=True,exist_ok=True)
        report=dest/'report.json'
        if not report.exists():
            if (dest/'H.npz').exists() or (dest/'paths').exists():raise RuntimeError('Uncommitted reference stage preserved; inspect before explicit recovery')
            bake_actors(state,dest);(dest/'paths').mkdir()
            def sink(first,arrays):
                check();file=dest/'paths'/('chunk_%04d.npz'%first)
                if file.exists():raise FileExistsError(file)
                from .jobs import folder_bytes,ResourceStop
                if folder_bytes(folder)+sum(a.nbytes for a in arrays.values())+2*1024**2>budget.limits['max_disk_bytes']:raise ResourceStop('Reference path reserve exceeds disk cap')
                with file.open('xb') as f:np.savez_compressed(f,**arrays)
            # No H or native paths, no propagation cache passed to reference.
            # Stop only between whole reference states. Interrupting an active
            # reference preserves its uncommitted stage for explicit recovery.
            h,meta=reference(reader.config,state,dest,budget.check,complete_path_sink=sink if q['result'].get('complete_path_files') else None)
            with (dest/'H.npz').open('xb') as f:np.savez_compressed(f,H=h)
            atomic_json(dest/'metadata.json',meta)
            atomic_json(report,dict(status='reference_saved_pending_independent_CPU_check',reference_sha256=sha(dest/'H.npz'),path_sha256={str(p.relative_to(dest)):sha(p) for p in (dest/'paths').glob('*.npz')}))
        saved=json.loads(report.read_text())
        if saved['reference_sha256']!=sha(dest/'H.npz') or any(sha(dest/n)!=v for n,v in saved['path_sha256'].items()):raise ValueError('Saved reference binding changed')
        with np.load(dest/'H.npz',allow_pickle=False) as a:h=a['H']
        errors,energy,squared_error=nmse_all_users(q['H'],h)
        delta=abs(q['H'].astype(np.complex128)-h.astype(np.complex128));axes=tuple(range(1,h.ndim));absolute=np.max(delta,axis=axes)
        with np.errstate(divide='ignore',invalid='ignore'):relative=np.max(np.divide(delta,abs(h),out=np.where(delta==0,0.,np.inf),where=abs(h)>0),axis=axes)
        native=[]
        for receipt in q['result'].get('complete_path_files',[]):
            file=folder/receipt['path']
            if sha(file)!=receipt['sha256']:raise ValueError('Native full path receipt changed')
            native.append(file)
        paths=compare_paths(native,sorted((dest/'paths').glob('*.npz'))) if native else dict(status='Unknown_no_complete_native_path_capture')
        polarized_internal=None
        if native:
            with np.load(native[0],allow_pickle=False) as a:polarized='schema' in a and str(a['schema'])=='sdk-public-polarized-path-table-v2'
            if polarized:
                from .polarized_paths import audit_public_table,reconstruct_cfr
                sides=[]
                for files,expected in ((native,q['H']),(sorted((dest/'paths').glob('*.npz')),h)):
                    tables=[];checks=[]
                    for file in files:
                        with np.load(file,allow_pickle=False) as a:table={key:a[key] for key in a.files}
                        checks.append(audit_public_table(table,reader.config));tables.append(table)
                    e,_,_=nmse_all_users(reconstruct_cfr(tables,reader.config),expected)
                    sides.append(dict(CIR_to_CFR_per_user_NMSE=numbers(e),status='Passed' if np.isfinite(e).all() and np.all(e<=.001) else 'Failed',table_checks=checks))
                polarized_internal=dict(native=sides[0],reference=sides[1])
                if any(side['status']=='Failed' for side in sides):paths['status']='Failed'
        passed=bool(np.isfinite(errors).all() and np.all(errors<=.001))
        import h5py
        with h5py.File(folder/'csi.h5','r') as f:rowsha=f['frame_sha256'].asstr()[i]
        row=dict(index=i,status='Passed' if passed else 'Failed',physical_timestamp=state['timestamp'],frame_sha256=rowsha,configuration_sha256=fingerprint(reader.config),
            per_user_NMSE=numbers(errors),nonfinite_NMSE_users=np.flatnonzero(~np.isfinite(errors)).tolist(),reference_energy=numbers(energy),squared_error=numbers(squared_error),max_absolute_error=numbers(absolute),max_relative_error=numbers(relative),paths=paths,
            reference_array_path=str((dest/'H.npz').relative_to(ROOT)),reference_array_file_sha256=sha(dest/'H.npz'),polarized_internal_consistency=polarized_internal,scope='This independently checked row only; raw HDF Unknown unchanged')
        atomic_json(dest/('CPU_comparison_'+str(time.time_ns())+'.json'),row);rows.append(row)
    counts=dict(Passed=sum(r['status']=='Passed' for r in rows),Failed=sum(r['status']=='Failed' for r in rows),Unknown=reader.count-len(rows))
    path_counts={status:sum(r['paths']['status']==status for r in rows) for status in ('Passed','Partial','Failed')}
    path_counts['Unknown']=sum(r['paths']['status'].startswith('Unknown') for r in rows)+reader.count-len(rows)
    ledger=dict(schema='native-frame-validation-sidecar-v1',rows=rows,counts=counts,path_counts=path_counts,raw_HDF_labels_unchanged=True)
    sidecar=folder/'independent_validation.json'
    if sidecar.exists():sidecar.rename(folder/('independent_validation_previous_'+str(time.time_ns())+'.json'))
    atomic_json(sidecar,ledger);return ledger
