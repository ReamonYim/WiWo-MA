#!/usr/bin/env python3
"""Offline physics attribution using independently generated full SDK paths.

The ablations below operate on SAVED references, after solving. They are never
used in the reference solver or to adjust any original CSI or acceptance gate.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['WIWO_GPU']='-1'
import argparse,importlib.util,json
from collections import Counter,defaultdict
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path.cwd()/'src'))
from wiwo_ma.platform.dataset_audit import sha_file,save_json


def sdk_phase(tau):
    # Source-defined Float32 multiplication followed by fused canonical-range
    # reduction, independently checked against saved CIR, not fitted to H.
    phase=(np.float32(-2*np.pi)*np.float32(28e9)*np.asarray(tau,np.float32)).astype(np.float32)
    turns=np.floor((phase*np.float32(1/(2*np.pi))).astype(np.float32))
    return (phase.astype(float)-float(np.float32(2*np.pi))*turns.astype(float)).astype(np.float32)


def band_synthesis(coefficient,tau,ports,freq):
    out=np.zeros((64,64),complex)
    np.add.at(out,ports,np.asarray(coefficient,complex)[:,None]*np.exp(-2j*np.pi*np.asarray(tau,float)[:,None]*freq))
    return out


def quantiles(x):
    x=np.asarray(x)
    return np.quantile(x,[0,.5,.9,.99,1]).tolist() if len(x) else []


def analyze_pair(left,right,user,tx,states,freq):
    x,y=states[left],states[right]
    def group(z):
        d=defaultdict(list)
        for k,w in enumerate(z['words']):
            if w[0]==user and (tx is None or w[1]==tx):d[w].append(k)
        return d
    g0,g1=group(x),group(y);common=set(g0)&set(g1);unique=[w for w in common if len(g0[w])==len(g1[w])==1]
    ix=np.array([g0[w][0] for w in unique],int);iy=np.array([g1[w][0] for w in unique],int)
    old=x['field_coefficient'][ix].astype(complex);new=y['field_coefficient'][iy].astype(complex)
    tau0=x['delay_s'][ix];tau1=y['delay_s'][iy];port=x['tx_port'][ix]
    h0=band_synthesis(x['coefficient'][ix],tau0,port,freq);h1=band_synthesis(y['coefficient'][iy],tau1,port,freq)
    phase_only=band_synthesis(old*np.exp(1j*sdk_phase(tau1)),tau1,port,freq)
    field_only=band_synthesis(new*np.exp(1j*sdk_phase(tau0)),tau0,port,freq)
    selected=slice(None) if tx is None else slice(tx,tx+1)
    h0,h1,phase_only,field_only=[h[selected] for h in [h0,h1,phase_only,field_only]]
    den=float(np.sum(abs(h1)**2));nmse_phase=float(np.sum(abs(phase_only-h1)**2)/den) if den else None
    stable_field_rel=abs(new-old)/np.maximum(abs(old),np.finfo(float).tiny)
    p0=float(np.mean(abs(h0)**2));p1=float(np.mean(abs(h1)**2));pp=float(np.mean(abs(phase_only)**2));pf=float(np.mean(abs(field_only)**2))
    recirc0=old*np.exp(1j*sdk_phase(tau0));recirc1=new*np.exp(1j*sdk_phase(tau1))
    circ_err=max(float(abs(recirc0-x['coefficient'][ix]).max()),float(abs(recirc1-y['coefficient'][iy]).max())) if len(ix) else None
    coherent_ratio=[]
    for z,indices,h in [(x,ix,h0),(y,iy,h1)]:
        incoherent=float(np.sum(abs(z['coefficient'][indices].astype(complex))**2))*64
        coherent_ratio.append(float(np.sum(abs(h)**2)/incoherent) if incoherent else None)
    result=dict(before_frame=left,after_frame=right,user=user,tx_port=tx,common_unique_path_terms=len(unique),
        common_all_terms=len(common),before_terms=len(g0),after_terms=len(g1),new_terms=len(set(g1)-set(g0)),lost_terms=len(set(g0)-set(g1)),
        duplicate_words_before=[str(w) for w,items in g0.items() if len(items)>1],duplicate_words_after=[str(w) for w,items in g1.items() if len(items)>1],
        common_bare_Field_relative_change_quantiles=quantiles(stable_field_rel),
        abs_F64_carrier_phase_step_rad_quantiles=quantiles(abs(2*np.pi*28e9*(tau1.astype(float)-tau0.astype(float)))),
        actual_common_CFR_power_step_dB=10*np.log10(p1/p0),phase_only_power_step_dB=10*np.log10(pp/p0),field_only_power_step_dB=10*np.log10(pf/p0),
        phase_only_vs_actual_common_CFR_NMSE=nmse_phase,coherent_to_incoherent_energy_ratio_before_after=coherent_ratio,
        phase_emulation_CIR_max_absolute_difference=circ_err,
        no_runtime_teacher_injection=True,scope='Offline attribution on fresh SDK paths; original full path sets remain unavailable')
    return result,(h0,h1,phase_only,field_only)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    if not (a.run/'supervisor_result.json').exists():raise RuntimeError('Wait for closed GPU and first independent CPU audit')
    a.out.mkdir(parents=True)
    post=a.run/'executed_sources/audit_platform_sdk16_results.py';spec=importlib.util.spec_from_file_location('frozen_post',post);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    freq=(np.arange(64)-31.5)*30000;states={};pairs=[];spectra=[]
    for left,right,u,tx in [(278,279,28,None),(770,771,23,3),(1433,1434,28,None),(1566,1567,28,None)]:
        if not all((a.run/'references'/str(i)/'complete.json').exists() for i in [left,right]):continue
        for i in [left,right]:
            if i not in states:states[i]=m.read_state(a.run/'references'/str(i),freq)
        report,arrays=analyze_pair(left,right,u,tx,states,freq);pairs.append(report);spectra.append(arrays)
    save_json(a.out/'phase_Field_ablations.json',pairs)
    geometry=[]
    for i,z in states.items():
        n=len(z['user']);length=np.zeros(n);previous=z['source_positions'].astype(float);plane_errors=[];reflection_errors=[]
        for d in range(3):
            active=z['interaction_types'][:,d]!=0;vertex=z['vertices'][:,d].astype(float);length[active]+=np.linalg.norm(vertex[active]-previous[active],axis=1)
            if np.any(active):
                normal=z['normals'][active,d].astype(float);normal/=np.linalg.norm(normal,axis=-1)[:,None]
                tri=z['primitive_triangles'][active,d].astype(float)
                plane_errors.extend(abs(np.sum((vertex[active]-tri[:,0])*normal,axis=1)).tolist())
                incoming=vertex[active]-previous[active];incoming/=np.linalg.norm(incoming,axis=-1)[:,None]
                nextpoint=z['target_positions'][active].astype(float)
                if d<2:
                    more=z['interaction_types'][active,d+1]!=0;nextpoint[more]=z['vertices'][active,d+1][more]
                outgoing=nextpoint-vertex[active];outgoing/=np.linalg.norm(outgoing,axis=-1)[:,None]
                ideal=incoming-2*np.sum(incoming*normal,axis=1)[:,None]*normal
                reflection_errors.extend(np.arccos(np.clip(np.sum(ideal*outgoing,axis=1),-1,1)).tolist())
            previous[active]=vertex[active]
        length+=np.linalg.norm(z['target_positions'].astype(float)-previous,axis=1)
        geom=dict(index=i,path_count=n,path_length_vs_saved_delay_error_m_quantiles=quantiles(abs(length-z['delay_s'].astype(float)*299792458)),
            vertex_plane_distance_m_quantiles=quantiles(plane_errors),specular_reflection_direction_error_rad_quantiles=quantiles(reflection_errors),
            thresholds='Diagnostic continuous quantities; no new physical tolerance or reference modification')
        geometry.append(geom)
    save_json(a.out/'reference_path_geometry_physics.json',geometry)
    fig,axs=plt.subplots(len(pairs),2,figsize=(12,3.4*max(1,len(pairs))),squeeze=False)
    for axrow,r,hs in zip(axs,pairs,spectra):
        before,after,phase,field=hs
        for h,label,style in [(before,'Before','-'),(after,'After','-'),(phase,'Phase/delay only','--'),(field,'Bare Field only',':')]:
            axrow[0].plot(freq/1e3,10*np.log10(np.maximum(np.mean(abs(h)**2,axis=0),1e-300)),style,label=label)
        axrow[0].set(title=f"UE{r['user']} {r['before_frame']}→{r['after_frame']}, TX{r['tx_port'] if r['tx_port'] is not None else 'all64'}",xlabel='Frequency offset(kHz)',ylabel='Mean per-port gain(dB)');axrow[0].legend(fontsize=8)
        axrow[1].bar(['Actual','Phase/delay','Bare Field'],[r['actual_common_CFR_power_step_dB'],r['phase_only_power_step_dB'],r['field_only_power_step_dB']]);axrow[1].set(ylabel='Gain change from before(dB)',title='Matched paths only; offline ablation')
    fig.suptitle('Fresh original SDK path attribution | Munich K40,64TX,28GHz,Depth3,100k/source');fig.tight_layout();fig.savefig(a.out/'phase_Field_ablations.png',dpi=180);fig.savefig(a.out/'phase_Field_ablations.pdf');plt.close(fig)
    # Delay/power/path support on the newly captured complete reference paths.
    fig,axs=plt.subplots(2,2,figsize=(12,8));
    for ax,(left,right,u,tx) in zip(axs.ravel(),[(278,279,28,None),(770,771,23,3),(1433,1434,28,None),(1566,1567,28,None)]):
        for i,color in [(left,'tab:blue'),(right,'tab:orange')]:
            if i not in states:continue
            z=states[i];ix=(z['user']==u)&((z['tx_port']==tx) if tx is not None else True)
            # Full64 aggregate, avoid visually misleading every-port duplicate
            # stems by showing the strongest participating port when all64.
            port=tx if tx is not None else int(z['tx_port'][np.flatnonzero(ix)[np.argmax(abs(z['coefficient'][ix]))]])
            ix &= z['tx_port']==port
            ax.scatter(z['delay_s'][ix]*1e9,20*np.log10(abs(z['coefficient'][ix])),s=30,facecolors='none',edgecolors=color,label=f'frame{i} TX{port}')
        ax.set(title=f'UE{u} full reference retained port paths',xlabel='Absolute delay(ns)',ylabel='CIR coefficient20log10|a|(dB)');ax.legend(fontsize=8)
    fig.suptitle('Complete new SDK reference paths for selected links; original dataset CIR incomplete');fig.tight_layout();fig.savefig(a.out/'weak_link_reference_CIR.png',dpi=180);fig.savefig(a.out/'weak_link_reference_CIR.pdf');plt.close(fig)
    shutil_source=a.out/Path(__file__).name;shutil_source.write_bytes(Path(__file__).read_bytes())
    save_json(a.out/'manifest.json',dict(new_GPU_seconds=0,parent_run=str(a.run),source=sha_file(__file__),files={str(p.relative_to(a.out)):sha_file(p) for p in a.out.iterdir() if p.is_file()}))
    print(json.dumps(dict(offline_path_physics_pairs=len(pairs),new_GPU_seconds=0)))


if __name__=='__main__':main()
