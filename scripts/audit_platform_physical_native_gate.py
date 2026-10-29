#!/usr/bin/env python3
"""Independent CPU audit of native physical-event raw arrays and complete CIR.

Never imports Sionna/NativeRT or consumes reference answers as runtime inputs.
Does not trust worker report/exit code as the scientific verdict.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['WIWO_GPU']='-1'
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import h5py,numpy as np
from wiwo_ma.platform.dataset_audit import nmse_all_users,sha_file
from wiwo_ma.platform.storage import atomic_json,CSIReader


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(x) for x in value]
    return value


def read_paths(files,c,state):
    rows={};arrays=[];bad=[];ntx=len(state['tx_port_positions']);nr=max(u['array']['rows']*u['array']['cols'] for u in c['users']);k=len(c['users']);nf=c['radio']['num_subcarriers']
    h=np.zeros((k*nr*ntx,nf),np.complex128);freq=(np.arange(nf)-(nf-1)/2)*c['radio']['subcarrier_spacing_hz'];los=np.zeros((k,ntx),bool)
    endpoint_errors=0;actor_terms=0;depths=Counter();actor_ids={a['id'] for a in c['actors']}
    for file in files:
        receipt=file.with_suffix('.receipt.json')
        if not receipt.exists() or sha_file(file)!=json.load(open(receipt))['sha256']:raise ValueError('Path receipt mismatch: '+str(file))
        with np.load(file,allow_pickle=False) as f:a={name:f[name] for name in f.files}
        if any(not np.isfinite(a[name]).all() for name in ('vertices','delay_s','coefficient','source_positions','target_positions')):bad.append(str(file))
        if np.any(a['delay_s']<0):bad.append(str(file)+':negative_delay')
        names=json.loads(str(a['object_map_json']));n=len(a['user']);depth=np.count_nonzero(a['interaction_types'],axis=1);depths.update(map(int,depth))
        for i in range(n):
            u,rx,tx=int(a['user'][i]),int(a['rx_element'][i]),int(a['tx_port'][i]);word=[];touches_actor=False
            expected_rx=next(j for j,m in enumerate(state['rx_port_map']) if m[0]==u and m[1]==rx and m[2]==0)
            endpoint_errors+=int(not np.array_equal(a['source_positions'][i],np.asarray(state['tx_port_positions'][tx],np.float32)))
            endpoint_errors+=int(not np.array_equal(a['target_positions'][i],np.asarray(state['rx_port_positions'][expected_rx],np.float32)))
            for typ,obj,prim in zip(a['interaction_types'][i],a['object_ids'][i],a['primitive_ids'][i]):
                if not typ:continue
                name=names[str(int(obj))]
                for actor in actor_ids:
                    if name in ('platform_actor_'+actor,'reference_'+actor):name='actor:'+actor;touches_actor=True;break
                word.append((int(typ),name,int(prim)))
            key=(u,rx,tx,tuple(word));rows.setdefault(key,[]).append((a['vertices'][i].copy(),a['delay_s'][i],a['coefficient'][i]))
            if not word:los[u,tx]=True
            actor_terms+=touches_actor
        link=(a['user']*nr+a['rx_element'])*ntx+a['tx_port']
        vals=a['coefficient'].astype(np.complex128)[:,None]*np.exp(-2j*np.pi*a['delay_s'].astype(float)[:,None]*freq[None])
        np.add.at(h,link,vals)
    return dict(rows=rows,H=h.reshape(k,nr,ntx,nf),LOS=los,nonfinite=bad,endpoint_errors=endpoint_errors,actor_paths=int(actor_terms),depth_counts=dict(depths),path_count=sum(depths.values()))


def path_comparison(a,b):
    x,y=a['rows'],b['rows'];missing=set(y)-set(x);extra=set(x)-set(y);duplicates=0;count_mismatch=0;vbad=dbad=cbad=0;dv=dt=dc=0.;matched=0
    for key in set(x)&set(y):
        if len(x[key])!=len(y[key]):count_mismatch+=1;continue
        if len(x[key])!=1:duplicates+=1;continue
        p,q=x[key][0],y[key][0];matched+=1
        vbad+=not np.array_equal(p[0],q[0]);dbad+=not np.array_equal(p[1],q[1]);cbad+=not np.array_equal(p[2],q[2])
        dv=max(dv,float(np.max(abs(p[0]-q[0]),initial=0)));dt=max(dt,float(abs(float(p[1])-float(q[1]))));dc=max(dc,float(abs(complex(p[2])-complex(q[2]))))
    failed=bool(missing or extra or count_mismatch or a['nonfinite'] or b['nonfinite']);partial=bool(duplicates or vbad or dbad or cbad)
    return dict(status='Failed' if failed else 'Partial' if partial else 'Passed',native_path_count=a['path_count'],reference_path_count=b['path_count'],
        missing_words=len(missing),extra_words=len(extra),multiplicity_mismatch_words=count_mismatch,ambiguous_duplicate_words_Unknown=duplicates,uniquely_matched_paths=matched,
        nonidentical_vertex_paths=int(vbad),nonidentical_delay_paths=int(dbad),nonidentical_CIR_paths=int(cbad),max_vertex_difference_m=dv,max_delay_difference_s=dt,max_CIR_absolute_difference=dc)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();run=Path(a.run);out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False);proposal=json.load(open(run/'proposal.json'));rows=[];data={};binds=[]
    for name,digest in proposal['inputs'].items():
        if sha_file(run/name)!=digest:raise ValueError('Frozen input changed: '+name)
    for name,digest in proposal['source_sha256'].items():
        if sha_file(run/'executed_sources'/name)!=digest:raise ValueError('Executed source archive changed: '+name)
    for case in proposal['cases']:
        dest=run/case['id'];s=json.load(open(run/case['state']));c=json.load(open(run/case['config']));nat=dest/'native.npz';ref=dest/'reference.npz'
        if not nat.exists() or not ref.exists():rows.append(dict(case=case['id'],status='Unknown_missing_stage'));continue
        with np.load(nat,allow_pickle=False) as f:x=f['H']
        with np.load(ref,allow_pickle=False) as f:y=f['H']
        nr=max(u['array']['rows']*u['array']['cols'] for u in c['users']);expected=(len(c['users']),nr,len(s['tx_port_positions']),c['radio']['num_subcarriers'])
        if x.shape!=expected or y.shape!=expected or x.dtype!=np.complex64 or y.dtype!=np.complex64:raise ValueError('CSI shape/dtype contract failure')
        error,den,num=nmse_all_users(x,y);csi=bool(np.isfinite(error).all() and np.all(error<=.001));diff=x.astype(complex)-y
        frame=CSIReader(dest/'dataset/csi.h5').frame(0,verify=True)
        state_identity=frame['state']==s;stored_exact=np.array_equal(frame['H'],x)
        versions=[read_paths(sorted((dest/(kind+'_paths')).glob('*.npz')),c,s) for kind in ('native','reference')]
        expected_chunks=int(np.ceil(expected[2]/c['rt']['tx_device_chunk']));chunks_ok=all(len(list((dest/(kind+'_paths')).glob('*.npz')))==expected_chunks for kind in ('native','reference'))
        pc=path_comparison(*versions);expected_los=np.load(dest/'CPU_LOS_expectation.npz')['direct_LOS'];los_errors=[int(np.count_nonzero(v['LOS']!=expected_los)) for v in versions]
        internal=[nmse_all_users(v['H'],h)[0] for v,h in zip(versions,(x,y))];internal_ok=all(np.isfinite(e).all() and np.all(e<=.001) for e in internal)
        failed=not csi or not state_identity or not stored_exact or not chunks_ok or not internal_ok or any(los_errors) or any(v['endpoint_errors'] for v in versions) or pc['status']=='Failed'
        row=dict(case=case['id'],time_s=s['timestamp'],group=case['group'],status='Failed' if failed else pc['status'],CSI_status='Passed' if csi else 'Failed',
            per_user_NMSE=error,max_user_NMSE=float(error.max()),max_absolute_complex_error=float(abs(diff).max()),weakest_reference_user=int(np.argmin(den)),
            zero_reference_users=np.flatnonzero(den==0),nonfinite_error_users=np.flatnonzero(~np.isfinite(error)),paths=pc,
            CIR_to_CFR_per_user_NMSE_native_reference=internal,state_HDF_exact=state_identity,HDF_CSI_exact=stored_exact,
            LOS_mismatches_native_reference=los_errors,target_UE0_LOS_ports_native_reference=[int(v['LOS'][0].sum()) for v in versions],
            actor_interacting_paths_native_reference=[v['actor_paths'] for v in versions],endpoint_errors_native_reference=[v['endpoint_errors'] for v in versions],
            depth_counts_native_reference=[v['depth_counts'] for v in versions])
        atomic_json(out/(case['id']+'.json'),clean(row));rows.append(row);data[case['id']]=(x,versions[0],s)
        binds += [dict(path=str(f),sha256=sha_file(f)) for f in [nat,ref,dest/'dataset/csi.h5']]
    branches=[]
    for first,second in [('small_1','small_restored_ancestor'),('small_7','small_restored')]:
        if first not in data or second not in data:branches.append(dict(before=first,after=second,status='Unknown'));continue
        x,z,s=data[first];y,w,t=data[second];e=nmse_all_users(x,y)[0];pc=path_comparison(z,w)
        branches.append(dict(before=first,after=second,status='Passed' if s==t and np.isfinite(e).all() and np.all(e<=.001) and pc['status']=='Passed' else 'Failed',
            frozen_state_equal=s==t,CSI_bitwise=np.array_equal(x,y),per_user_NMSE=e,paths=pc))
    life=[]
    for group in ('full64','small'):
        selected=[r for r in rows if r.get('group')==group and 'restored' not in r['case']]
        for first,second in zip(selected,selected[1:]):
            x,y=data[first['case']][1],data[second['case']][1];a0,b0=set(x['rows']),set(y['rows'])
            life.append(dict(before=first['case'],after=second['case'],born_words=len(b0-a0),dead_words=len(a0-b0),persistent_words=len(a0&b0),
                direct_LOS_births=sum(not v[-1] for v in b0-a0),direct_LOS_deaths=sum(not v[-1] for v in a0-b0),scope='SDK-discovered paths, not proof of every physically existing path'))
    counts=dict(Counter(r['status'] for r in rows));csi_counts=dict(Counter(r.get('CSI_status','Unknown') for r in rows))
    summary=dict(status='Failed' if counts.get('Failed') else 'Passed_selected_states_only' if counts.get('Passed')==len(proposal['cases']) and all(b['status']=='Passed' for b in branches) else 'Partial',
        counts=counts,CSI_counts=csi_counts,max_user_NMSE=max((r.get('max_user_NMSE',0) for r in rows),default=None),cases=rows,branches=branches,lifecycle=life,
        raw_array_bindings=binds,old_3001_and_801_labels_unchanged=True,whole_4s_continuous_CSI='Unknown',original29_low_correlation_events='Unknown_no_new_attribution',new_GPU_seconds=0,
        audit_source_sha256=sha_file(Path(__file__)),limitations=['Frozen selected states only','Independent SDK numerical consistency is not physical measurement certification','4s fast-motion setup is not an unaliased Doppler dataset'])
    atomic_json(out/'summary.json',clean(summary))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,1,figsize=(11,6),constrained_layout=True)
    for r in rows:
        if 'per_user_NMSE' in r:axs[0].semilogy(range(len(r['per_user_NMSE'])),np.maximum(r['per_user_NMSE'],1e-30),'.',label=r['case'])
    axs[0].axhline(.001,color='red',ls='--',label='Frozen SDK NMSE gate');axs[0].set(xlabel='UE index',ylabel='Complex CSI NMSE');axs[0].legend(fontsize=6,ncol=4)
    full=[r for r in rows if r.get('group')=='full64']
    if full:axs[1].step([r['time_s'] for r in full],[r['target_UE0_LOS_ports_native_reference'][0] for r in full],where='mid',label='Native SDK discovered direct paths')
    axs[1].set(xlabel='Physical time (s), selected states only',ylabel='UE0 direct LOS TX ports');axs[1].legend();fig.savefig(out/'SDK_gate_NMSE_and_LOS.png',dpi=160);fig.savefig(out/'SDK_gate_NMSE_and_LOS.pdf');plt.close(fig)
    print(json.dumps(clean({k:summary[k] for k in ('status','counts','CSI_counts','max_user_NMSE','branches')})))


if __name__=='__main__':main()
