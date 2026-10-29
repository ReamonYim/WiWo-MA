#!/usr/bin/env python3
"""CPU audit of new full-reference arrays and path evidence, after CUDA exits."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['WIWO_GPU']='-1'
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil
import sys
import time
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path.cwd().resolve()
sys.path.insert(0,str(ROOT/'src'))
from wiwo_ma.platform.dataset_audit import sha_file,save_json,nmse_all_users


def path_CFR(z,freq):
    # Independent CPU reconstruction of ALL reference paths, no filtering.
    c=z['coefficient'].astype(np.complex128);tau=z['delay_s'].astype(float)
    return c[:,None]*np.exp(-2j*np.pi*tau[:,None]*freq)


def read_state(folder,freq):
    data=[];fields=['user','tx_port','rx_port','coefficient','field_coefficient','delay_s','vertices','interaction_types','object_ids','primitive_ids','primitive_triangles','normals','source_positions','target_positions']
    words=[];planes=[]
    for path in sorted(folder.glob('chunk_*.npz')):
        receipt=json.loads(path.with_suffix('.receipt.json').read_text())
        if sha_file(path)!=receipt['sha256']:raise ValueError('Corrupt new reference chunk')
        with np.load(path,allow_pickle=False) as f:z={k:f[k] for k in fields};objects=json.loads(str(f['object_map_json']))
        for k in range(len(z['user'])):
            word=[];plane=[]
            for d,kind in enumerate(z['interaction_types'][k]):
                if not kind:continue
                name=objects[str(int(z['object_ids'][k,d]))]['name']
                word.append((name,int(z['primitive_ids'][k,d]),int(kind)))
                triangle=z['primitive_triangles'][k,d].astype(float)
                normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);normal/=np.linalg.norm(normal)
                offset=float(normal@triangle[0]);first=np.flatnonzero(abs(normal)>1e-9)[0]
                if normal[first]<0:normal=-normal;offset=-offset
                # Co-planar facets are grouped only for OFFLINE attribution.
                # Never used to generate/rewrite RT outputs or numeric gates.
                plane.append((name,tuple(np.round(normal,5)),round(offset,4),int(kind)))
            endpoint=(int(z['user'][k]),int(z['tx_port'][k]),int(z['rx_port'][k]))
            words.append(endpoint+(tuple(word),));planes.append(endpoint+(tuple(plane),))
        data.append(z)
    allz={k:np.concatenate([z[k] for z in data],axis=0) for k in fields}
    allz['words']=words;allz['plane_words']=planes;allz['depth']=np.count_nonzero(allz['interaction_types'],axis=1)
    return allz


def compare_retained(native,z):
    keymap=defaultdict(list)
    for k in np.flatnonzero(z['depth']==0):keymap[(int(z['user'][k]),int(z['tx_port'][k]),int(z['rx_port'][k]))].append(k)
    missing=[];duplicates=[];cerror=[];endpoint_errors=[];delaydiff=[];delaybits=0;cbits=0
    for j,path in enumerate(native):
        key=(path['user'],path['tx_port'],path['rx_spatial_element']);matches=keymap[key]
        if not matches:missing.append(j);continue
        if len(matches)!=1:duplicates.append(dict(original_path=j,reference_matches=len(matches)));continue
        k=matches[0];pts=np.array(path['points'],np.float32)
        if len(pts)!=2:raise ValueError('Expected prior audit strongest subset LOS only')
        endpoint_errors.append(float(max(abs(pts[0]-z['source_positions'][k]).max(),abs(pts[-1]-z['target_positions'][k]).max())))
        coeff=np.complex64(complex(path['coefficient_real'],path['coefficient_imag']));ref=z['coefficient'][k]
        cerror.append(float(abs(complex(coeff)-complex(ref))**2/abs(complex(ref))**2) if ref else (0. if coeff==0 else np.inf))
        dt=np.float32(path['delay_s']);delaydiff.append(float(abs(float(dt)-float(z['delay_s'][k]))))
        delaybits+=int(dt.view(np.uint32)!=z['delay_s'][k].view(np.uint32));cbits+=int(coeff!=ref)
    return dict(saved_subset_count=len(native),matched_count=len(cerror),missing_original_path_indices=missing,
        duplicate_reference_LOS=duplicates,max_endpoint_error_m=max(endpoint_errors,default=None),
        max_delay_difference_s=max(delaydiff,default=None),nonidentical_delay_bits=delaybits,nonidentical_CIR_coefficients=cbits,
        max_per_path_CIR_relative_squared_error=max(cerror,default=None),
        complete_original_path_set='Unknown_original_never_saved',per_path_CIR_gate='Diagnostic_only_no_new_tolerance_invented')


def decomposition(z0,z1,freq,user,tx=None):
    def select(z):return np.flatnonzero((z['user']==user)&((z['tx_port']==tx) if tx is not None else True))
    output={}
    for vocabulary in ['words','plane_words']:
        terms=[]
        for z in [z0,z1]:
            ix=select(z);values=path_CFR({k:z[k][ix] for k in ['coefficient','delay_s']},freq);group=defaultdict(lambda:np.zeros(len(freq),complex))
            for k,val in zip(ix,values):group[z[vocabulary][k]]+=val
            terms.append(group)
        x,y=terms;common=set(x)&set(y);born=set(y)-set(x);dead=set(x)-set(y)
        update=np.zeros((64,len(freq)),complex);birth=update.copy();death=update.copy();before=update.copy();after=update.copy()
        for key in common:update[key[1]]+=y[key]-x[key]
        for key in born:birth[key[1]]+=y[key]
        for key in dead:death[key[1]]-=x[key]
        for key,val in x.items():before[key[1]]+=val
        for key,val in y.items():after[key[1]]+=val
        delta=after-before;energy=float(np.sum(abs(delta)**2));change=update+birth+death
        powers=[float(np.sum(abs(h)**2)) for h in [before,after,delta,update,birth,death]]
        # Coherent terms have cross terms; norm fractions DO NOT sum to1.
        output[vocabulary]=dict(before_terms=len(x),after_terms=len(y),common_terms=len(common),born_terms=len(born),dead_terms=len(dead),
            before_energy=powers[0],after_energy=powers[1],delta_energy=powers[2],common_update_energy=powers[3],birth_energy=powers[4],death_energy=powers[5],
            nonadditive_energy_ratios_to_delta=[v/energy if energy else None for v in powers[3:]],
            complex_decomposition_max_residual=float(abs(change-delta).max()),
            explanation='SDK discovered path support, not all physically existing paths. Co-planar grouping is offline attribution, primitive quantization is not an acceptance gate.')
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--grant',type=Path,required=True);args=p.parse_args();out=args.out;grant=json.loads(args.grant.read_text())
    plan=Path(grant['proposal_folder']);proposal=json.loads((plan/'proposal.json').read_text());job=ROOT/'artifacts/platform/jobs/munich30s_native_audited_attempt01';prior=ROOT/'artifacts/platform/audits/csi30s_comprehensive_attempt02'
    cfg=json.loads((plan/'config.json').read_text());freq=(np.arange(64)-31.5)*30000
    post=out/'independent_CPU_audit';post.mkdir(exist_ok=True);rows=[];states={};start=time.monotonic()
    old=json.loads((prior/'audit_numeric_label_ledger.json').read_text());labels=[dict(v) for v in old['frames']];partial=[]
    with h5py.File(job/'dataset/csi.h5','r') as f:
        for case in proposal['cases']:
            i=case['index'];folder=out/'references'/str(i)
            if not (folder/'complete.json').exists():
                partial.append(dict(index=i,status='Unknown',committed_chunks=len(list(folder.glob('*.receipt.json'))) if folder.exists() else 0));continue
            done=json.loads((folder/'complete.json').read_text())
            for name,h in done['files'].items():
                if sha_file(folder/name)!=h:raise ValueError('Committed state hash mismatch')
            with np.load(folder/'H.npz',allow_pickle=False) as file:ref=file['H']
            tested=f['H'][i];value=json.loads(f['frames_json'].asstr()[i]);frozen=json.loads((plan/str(i)/'state.json').read_text())
            if value['state']!=frozen:raise ValueError('Frozen state binding mismatch')
            err,den,num=nmse_all_users(tested,ref);diff=tested.astype(complex)-ref;rel=np.divide(abs(diff),abs(ref),out=np.where(abs(diff)==0,0,np.inf),where=abs(ref)>0)
            z=read_state(folder,freq);states[i]=z;recon=np.zeros((40,64,64),np.complex128)
            for first in range(0,len(z['user']),8192):
                ix=slice(first,first+8192);vals=path_CFR({key:z[key][ix] for key in ['coefficient','delay_s']},freq)
                np.add.at(recon,(z['user'][ix],z['tx_port'][ix]),vals)
            internal,_,_=nmse_all_users(recon[:,None],ref)
            retained=compare_retained(value['result']['paths'],z)
            status='Passed' if np.isfinite(err).all() and np.all(err<=.001) else 'Failed'
            labels[i].update(status=status,evidence=str(folder/'H.npz'),max_user_NMSE=float(err.max()))
            row=dict(index=i,time_s=i*.01,status=status,per_user_NMSE=err,max_user_NMSE=float(err.max()),
                weakest_reference_user=int(np.argmin(den)),weakest_user_NMSE=float(err[np.argmin(den)]),zero_reference_users=np.flatnonzero(den==0),
                max_absolute_complex_error=float(abs(diff).max()),relative_complex_error_quantiles=np.quantile(rel,[0,.5,.99,1]),
                full_reference_path_count=len(z['user']),reference_path_depth_counts=Counter(map(int,z['depth'])),
                independently_reconstructed_CIR_to_CFR_per_user_NMSE=internal,
                retained_original_paths=retained,full_original_path_set='Unknown_not_saved',
                reference_file_sha256=sha_file(folder/'H.npz'))
            rows.append(row);save_json(post/f'frame_{i}.json',row)
            np.savez_compressed(post/f'frame_{i}_errors.npz',absolute_complex_error=abs(diff),relative_complex_error=rel,per_user_NMSE=err,
                reference_power=np.mean(abs(ref.astype(complex))**2,axis=(1,2,3)))
    pairs=[]
    for left,right,user,tx in [(278,279,28,None),(770,771,23,3),(1433,1434,28,None),(1566,1567,28,None)]:
        if left in states and right in states:pairs.append(dict(before_frame=left,after_frame=right,user=user,tx_port=tx,attribution=decomposition(states[left],states[right],freq,user,tx)))
    save_json(post/'path_change_attribution.json',pairs)
    counts={key:sum(x['status']==key for x in labels) for key in ['Passed','Partial','Failed','Unknown']}
    save_json(post/'audit_numeric_label_ledger.json',dict(counts=counts,frames=labels,original_job_ledger_unchanged=True,prior_audit_counts=old['counts'],numerical_gate=.001))
    original=sha_file(job/'dataset/csi.h5');expected=json.loads((prior/'input_receipt.json').read_text())
    unchanged=original==expected['sha256'] and all(sha_file(job/p)==h for p,h in expected['auxiliary_files'].items())
    ledger=json.loads(args.grant.with_suffix('.ledger.json').read_text())
    summary=dict(overall='Partial',new_selected_CSI_counts=dict(Passed=sum(r['status']=='Passed' for r in rows),Failed=sum(r['status']=='Failed' for r in rows),Unknown=16-len(rows)),
        dataset_audit_overlay_counts=counts,original_job_counts=dict(Passed=7,Failed=0,Unknown=2994),
        max_user_NMSE=max((r['max_user_NMSE'] for r in rows),default=None),completed_indices=[r['index'] for r in rows],incomplete_states=partial,
        source_HDF_unchanged=unchanged,source_HDF_sha256=original,original_full_path_sets='Unknown_missing_original_records',
        full_new_reference_paths=sum(len(z['user']) for z in states.values()),GPU_budget=ledger,CPU_post_wall_s=time.monotonic()-start)
    save_json(post/'summary.json',summary)
    if not unchanged:raise ValueError('Original input changed; do not issue acceptance')
    fig,ax=plt.subplots(figsize=(11,5))
    for r in rows:ax.plot(range(40),r['per_user_NMSE'],'.-',lw=.7,label=f"{r['index']} ({r['time_s']:.2f}s)")
    ax.axhline(.001,color='r',ls='--',label='Frozen1e-3 gate');ax.set(yscale='log',xlabel='UE index, weak users included',ylabel='Complex CSI NMSE',title='Fresh original SDK Full64 vs original30s CSI | K40,64TX,28GHz,Depth3,100k/source')
    ax.legend(ncol=4,fontsize=7);fig.tight_layout();fig.savefig(post/'SDK16_NMSE.png',dpi=180);fig.savefig(post/'SDK16_NMSE.pdf');plt.close(fig)
    table='\n'.join(f"| {r['index']} | {r['time_s']:.2f} | {r['status']} | {r['max_user_NMSE']:.6g} | {r['full_reference_path_count']} |" for r in rows)
    text=f'''# 30秒CSI：新增16状态原SDK Full64独立复算

状态：**Partial（数据集整体）**。本轮已完成{len(rows)}/16状态。预算实际{ledger['spent_wall_s']:.3f}/900秒，独立账本，不借用旧授权。未重跑旧801状态或已受检8帧。原3001帧HDF、原7/2994账本和历史CPU审计均保留。

本轮数值结果：{summary['new_selected_CSI_counts']}；累计新审计覆盖层：{counts}。逐UE原门限1e-3，所有40UE、64物理TX、64频点均参与。完整原始路径集合没有保存，仍Unknown，不能由新参考反推已认证。

| 帧 | 时间(s) | CSI状态 | 最差逐UE NMSE | 新参考完整路径数 |
|---|---|---|---|---|
{table}

参考每状态新建原SDK场景/solver，actor使用独立冻结PLY，源/接收端点及方向逐位核验，synthetic_array=False，固定4源分块/seed42/Depth3/100k每源。复算进程没有打开原HDF，也没有从原CSI、原路径或NativeRT缓存读入答案。每个4源块原子保存完整CIR、Field系数、时延、顶点、primitive ID/三角形和object映射，可恢复且不重复已提交块。process-local object ID跨状态使用稳定名称解析。

CPU收尾从原数组重新计算误差，独立用全部新参考CIR合成频响检查导出完整性，并逐项比较旧最强128条LOS子集。各状态的误差、路径和分层结果保存在本轮independent_CPU_audit/frame_*.json及对应npz。新增SDK仍为数值参考，不是现实物理真值。

四对异常状态的相干增量分解见path_change_attribution.json：区分共同路径相位/系数更新、SDK新发现与消失的路径；primitive序列和离线共面分组分别报告。分量能量有交叉项，比例不能相加作百分比。共面分组只用于离线诊断，从未调整RT输出或门限。

完整存储/时间/机械检查沿用已独立闭合的3001帧CPU证据；全LOS、无直达遮挡、UE15秒瞬时反向、镜像轨迹泄漏及单场景局限不因抽样复算通过而消失。未复算状态继续Unknown。
'''
    doc=ROOT/'docs/platform/CSI_30S_SDK16_REAUDIT.md';doc.write_text(text)
    (out/'REPORT.md').write_text(text)
    delivery=out/'delivery';delivery.mkdir(exist_ok=True)
    rootmanifest=ROOT/'V5_DELIVERY_MANIFEST.json';previous=sha_file(rootmanifest);shutil.copyfile(rootmanifest,delivery/'previous_V5_DELIVERY_MANIFEST.json')
    current=json.loads(rootmanifest.read_text());current.update(snapshot='CSI30s_SDK16_reaudit',status='Partial',previous_active_manifest_sha256=previous)
    current['CSI30s_SDK16_independent_reaudit']=dict(report=str(doc.relative_to(ROOT)),out=str(out.relative_to(ROOT)),counts=counts,
        new_reference_counts=summary['new_selected_CSI_counts'],max_user_NMSE=summary['max_user_NMSE'],independent_GPU_spent_s=ledger['spent_wall_s'],original_job_unchanged=unchanged)
    save_json(rootmanifest,current);shutil.copyfile(rootmanifest,delivery/'V5_DELIVERY_MANIFEST.json');shutil.copyfile(doc,delivery/doc.name)
    with (ROOT/'docs/platform/PLATFORM_PROGRESS.md').open('a') as f:
        f.write(f"\n\n## 30秒CSI新增SDK16独立复算\n\n本轮独立GPU3授权900秒，实际{ledger['spent_wall_s']:.3f}秒；完成{len(rows)}/16状态，本轮{summary['new_selected_CSI_counts']}，累计覆盖层{counts}，原账本不回写。完整数据集仍Partial。详见[CSI_30S_SDK16_REAUDIT.md](CSI_30S_SDK16_REAUDIT.md)。\n")
    save_json(out/'CPU_closure.json',dict(status='CPU_evidence_written',overall='Partial',summary=summary))
    save_json(delivery/'manifest.json',dict(overall='Partial',files={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file() and p.name not in ['manifest.json','CPU_audit.log','supervisor.log']},original_HDF_sha256=original))
    print(json.dumps(dict(status='independent_CPU_audit_complete',counts=counts,max_user_NMSE=summary['max_user_NMSE'],GPU_s=ledger['spent_wall_s'])),flush=True)


if __name__=='__main__':main()
