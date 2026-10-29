#!/usr/bin/env python3
"""Additional independent offline diagnostics, no CUDA/Sionna import."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['WIWO_GPU']='-1'
import argparse
import base64
from collections import Counter
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wiwo_ma.platform.dataset_audit import (C, sha_file, save_json, temporal_correlation,
    nmse_all_users, decode_plain)


def theta_basis(k):
    z=np.clip(k[...,2],-1,1);phi=np.arctan2(k[...,1],k[...,0])
    return np.stack((z*np.cos(phi),z*np.sin(phi),-np.sqrt(np.maximum(0,1-z*z))),axis=-1)


def los_amplitude(v,rot,distance,wavelength):
    local=np.einsum('...ji,...j->...i',rot,v)
    theta=np.arccos(np.clip(local[...,2],-1,1));phi=np.arctan2(local[...,1],local[...,0])
    av=np.minimum(12*((theta-np.pi/2)/np.deg2rad(65))**2,30)
    ah=np.minimum(12*(phi/np.deg2rad(65))**2,30)
    gain=10**((8-np.minimum(av+ah,30))/20)
    txpol=np.einsum('...ij,...j->...i',rot,theta_basis(local));rxpol=theta_basis(-v)
    projection=np.sum(txpol*rxpol,axis=-1)
    return wavelength/(4*np.pi*distance)*gain*projection


def provenance(job,out):
    packages=[]
    for name in ['sionna-rt','mitsuba','drjit','numpy','scipy','h5py']:
        d=md.distribution(name);bad=[];missing=[];checked=0
        for entry in d.files or []:
            if not entry.hash:continue
            path=d.locate_file(entry)
            if not path.is_file():missing.append(str(entry));continue
            # Include SDK binaries and all PLY/XML resources, not just Python.
            checked+=1;got=base64.urlsafe_b64encode(bytes.fromhex(sha_file(path))).decode().rstrip('=')
            if entry.hash.mode!='sha256':raise ValueError('Unexpected RECORD digest')
            if got!=entry.hash.value:bad.append(str(entry))
        packages.append(dict(name=name,version=d.version,RECORD_checked_files=checked,modified=bad,missing=missing,
            scope='Local installation RECORD comparison, not independent vendor supply-chain attestation'))
    receipt=json.loads((job/'source_receipt.json').read_text());source=[]
    for q in receipt['files']:
        path=Path(q['path']);archive=job/'executed_sources'/q['path']
        source.append(dict(path=str(path),expected=q['sha256'],archived_matches=archive.exists() and sha_file(archive)==q['sha256'],
            current_matches=path.exists() and sha_file(path)==q['sha256']))
    save_json(out/'software_and_source_provenance.json',dict(packages=packages,generation_sources=source,
        loaded_propagation='Unmodified installed Sionna RT according to local RECORD; NativeRT adapts devices and actor operands, not SDK Field formulas',
        source_copies_bound_before_generation=True))


def labels_and_errors(job,out):
    side=json.loads((job/'dataset/independent_validation.json').read_text());ledger=json.loads((job/'label_ledger.json').read_text())
    errors=[];relative=[];rows=[];bindings=[];numerical_status=['Unknown']*3001
    with h5py.File(job/'dataset/csi.h5','r') as f:
        for record in side['rows']:
            i=int(round(record['physical_timestamp']/.01));refpath=Path(record['reference_array_path'])
            if not refpath.is_absolute():refpath=Path.cwd()/refpath
            binding=dict(frame=i,exact_timestamp=float(f['timestamps'][i])==record['physical_timestamp'],
                row_sha256=record['frame_sha256']==f['frame_sha256'].asstr()[i],
                configuration=record['configuration_sha256']==f.attrs['configuration_sha256'],
                reference_file=sha_file(refpath)==record['reference_array_file_sha256'])
            bindings.append(binding)
            with np.load(refpath) as ref:b=ref['H'].astype(complex)
            test=f['H'][i].astype(complex);e=test-b;mag=abs(e);rr=np.divide(mag,abs(b),out=np.where(mag==0,0,np.inf),where=abs(b)>0)
            err,den,num=nmse_all_users(test,b);errors.append(mag[:,0]);relative.append(rr[:,0])
            status='Passed' if np.isfinite(err).all() and np.all(err<=1e-3) and all(v for k,v in binding.items() if k!='frame') else 'Failed';numerical_status[i]=status
            rows.append(dict(index=i,status=status,per_user_NMSE=err,per_element_relative_error_quantiles=np.quantile(rr,[0,.5,.99,1]),
                worst_relative_index=list(np.unravel_index(np.argmax(rr),rr.shape)),zero_reference_elements=int(np.count_nonzero(b==0)),
                weakest_reference_UE=int(np.argmin(den)),weakest_reference_UE_NMSE=float(err[np.argmin(den)]),
                per_user_RMSE=np.sqrt(np.mean(abs(e)**2,axis=(1,2,3)))))
        branch=[]
        for i in [1,2]:
            with np.load(job/f'branch_{i}.npz') as z:
                key='H' if 'H' in z.files else z.files[0];h=z[key]
            with np.load(job/'references'/str(i)/'H.npz') as z:err,_,_=nmse_all_users(h,z['H'])
            to_main,_,_=nmse_all_users(h,f['H'][i]);branch.append(dict(frame=i,per_user_reference_NMSE=err,max_reference_NMSE=err.max(),
                per_user_main_NMSE=to_main,max_main_NMSE=to_main.max(),status='Passed' if np.isfinite(err).all() and np.all(err<=1e-3) else 'Failed'))
    np.savez_compressed(out/'reference_element_errors.npz',frame_indices=[r['index'] for r in rows],absolute_complex_error=errors,relative_complex_error=relative)
    extra=json.loads((out/'prior_gate_reuse_eligibility.json').read_text())
    if extra.get('eligible') and extra.get('additional_offline_comparison_status')=='Passed':numerical_status[120]='Passed'
    # New audit overlay is separate from the original job ledger. Only an
    # actually compared extra existing reference can change this overlay.
    overlay=[dict(index=i,time_s=i*.01,status=status,evidence='saved_long_reference' if i in [0,1,2,102,1500,1502,3000] else 'eligible_prior_gate_full64_reference' if i==120 and status=='Passed' else 'not_independently_compared') for i,status in enumerate(numerical_status)]
    save_json(out/'audit_numeric_label_ledger.json',dict(counts=Counter(numerical_status),frames=overlay,
        original_job_counts=ledger['counts'],original_job_ledger_modified=False,no_new_GPU_seconds=True,
        gate_scope='Per-user full complex CSI only; complete path/CIR/primitive verification remains Unknown'))
    save_json(out/'label_provenance_and_element_errors.json',dict(sidecar_bindings=bindings,rows=rows,branch_reaudit=branch,
        original_ledger_SHA_matches=side['CPU_ledger_sha256']==sha_file(job/'label_ledger.json'),
        original_counts=ledger['counts'],new_audit_counts=Counter(numerical_status),
        correction='Initial scan label_consistency.json omitted the physical_timestamp index conversion; this explicit binding is authoritative. No input artifact changed.'))


def enrich(job,out):
    a=dict(np.load(out/'derived.npz'));paths=np.load(out/'retained_paths.npy');n=len(a['t']);c=json.loads((job/'config.json').read_text());fc=c['radio']['carrier_hz'];wave=C/fc
    q=paths[paths['depth']==0];ii,tt=q['frame'],q['tx'];r=a['rotations'][ii,tt//4];delta=q['points'][:,1].astype(float)-q['points'][:,0];d=np.linalg.norm(delta,axis=-1);v=delta/d[:,None]
    analytical=los_amplitude(v,r,d,wave);amp_error=abs(abs(q['a'])-abs(analytical))/abs(analytical)
    tau=q['delay_s'];f64=np.exp(-2j*np.pi*fc*tau)*analytical
    phase32=(np.float32(-2*np.pi)*np.float32(fc)*tau.astype(np.float32)).astype(np.float32)
    k=np.floor((phase32*np.float32(1/(2*np.pi))).astype(np.float32))
    # Explicit float32 multiply + fused reduction emulations are offline
    # diagnostics only; neither is used to alter any measured CSI.
    reduced32=(phase32-(np.float32(2*np.pi)*k).astype(np.float32)).astype(np.float32)
    reduced_fused=(phase32.astype(float)-float(np.float32(2*np.pi))*k.astype(float)).astype(np.float32)
    phase_summary={}
    for name,phase in [('F64_delay_phase',np.angle(f64)),('F32_separate_reduction',reduced32),('F32_fused_reduction',reduced_fused)]:
        e=abs(np.angle(q['a']*np.exp(-1j*phase)))
        phase_summary[name]=np.quantile(e,[0,.5,.9,.99,1])
    save_json(out/'LOS_physical_math.json',dict(checked_retained_LOS_paths=len(q),
        amplitude_relative_error_quantiles=np.quantile(amp_error,[0,.5,.9,.99,1]),
        amplitude_model='lambda/(4*pi*d) times TR38901 Tx amplitude pattern times world-frame vertical polarization dot product; isotropic V RX',
        wrapped_phase_error_rad_quantiles=phase_summary,
        path_delay_F32_length_error_m_max=float(np.max(abs(tau-d/C))*C),
        immutable_CSI_corrections_applied=False,
        scope='Independent analytic LOS amplitude check, offline floating point phase attribution. Not full multipath SDK certification.',
        source_equations=['sionna/rt/antenna_pattern.py:v_tr38901_pattern','sionna/rt/path_solvers/field_calculator.py:tx/rx spreading and pattern projections','sionna/rt/path_solvers/paths.py:cir/cfr','sionna/rt/utils/misc.py:map_angle_to_canonical_range']))
    # Phase/delay tracking explains expected mm-scale phase sensitivity for
    # unambiguous retained LOS paths. No smoothness acceptance threshold.
    segments=[('before_action',0.,.99),('first_panel_motion',1.,1.64),('cruise_before_reversal',2.,14.99),('second_panel_motion',15.,15.87),('cruise_after_reversal',16.,30.)]
    segment_acf={};coherence=[]
    for name,left,right in segments:
        sel=(a['t']>=left-1e-10)&(a['t']<=right+1e-10);x=a['trace'][sel];rho=np.array([temporal_correlation(x[:,u]) for u in range(40)])
        segment_acf[name]=rho
        for u in range(40):
            crossings=np.flatnonzero(abs(rho[u,1:])<.5);cross=float((crossings[0]+1)*.01) if len(crossings) else None
            coherence.append(dict(segment=name,from_s=left,to_s=right,user=u,first_ACF_0_5_cross_s=cross,
                right_censored=cross is None,observable='One fixed +15kHz tone, all64 TX ports; uncentered overlap-normalized ACF, no stationary-process assertion'))
    np.savez_compressed(out/'segment_ACF.npz',**segment_acf);save_json(out/'segment_coherence.json',coherence)
    # All-port per-frame link gain/phase discontinuities for fixed ports and
    # moving ports separately. A deep-fade phase wrap isn't itself a failure.
    trace=a['trace'].astype(complex);linkcorrphase=np.angle(trace[1:]*trace[:-1].conj());powerdb=10*np.log10(np.maximum(a['power_port'],1e-300));deltadb=np.diff(powerdb,axis=0)
    worst=[]
    for k in np.argpartition(abs(deltadb).ravel(),-30)[-30:]:
        i,u,tx=np.unravel_index(k,deltadb.shape);worst.append(dict(frame=int(i+1),user=int(u),tx_port=int(tx),panel=int(tx//4),time_s=float(a['t'][i+1]),
            link_power_step_dB=float(deltadb[i,u,tx]),before_dB=float(powerdb[i,u,tx]),after_dB=float(powerdb[i+1,u,tx]),
            central_tone_phase_step_rad=float(linkcorrphase[i,u,tx]),user_mean_power_step_dB=float(10*np.log10(a['power_ue'][i+1,u]/a['power_ue'][i,u]))))
    save_json(out/'port_deep_fades.json',dict(largest_30_link_gain_changes=sorted(worst,key=lambda r:-abs(r['link_power_step_dB'])),
        absolute_link_gain_step_dB_quantiles=np.quantile(abs(deltadb),[.5,.9,.99,.999,1]),
        central_tone_phase_step_rad_quantiles=np.quantile(abs(linkcorrphase),[.5,.9,.99,.999,1]),
        diagnosis='Large single-port steps require deep-fade and multipath checks; no arbitrary phase smoothness gate applied'))
    # Long-episode symmetry: same UE geometry t and30-t, fixed panel ports,
    # actor position different. Quantify correlation instead of treating pairs
    # as independent data (potential train/test information leakage).
    fixed=np.array([i for i in range(64) if i//4 not in [0,3,7,12]])
    x=trace[:1501,:,fixed];y=trace[::-1][:1501,:,fixed]
    rho=abs(np.sum(x*y.conj(),axis=-1))/np.sqrt(np.sum(abs(x)**2,axis=-1)*np.sum(abs(y)**2,axis=-1))
    rel=np.sum(abs(x-y)**2,axis=-1)/np.sum(abs(x)**2,axis=-1)
    save_json(out/'time_reversal_dependence.json',dict(central_tone_fixed48_port_mirrored_geometry_correlation_quantiles=np.quantile(rho,[0,.01,.5,.99,1]),
        central_tone_mirrored_geometry_NMSE_quantiles=np.quantile(rel,[0,.01,.5,.99,1]),
        scope='t and30-t have same UE positions and fixed-panel antenna geometry; moving actor may differ. Strong dependence affects split design. Exclude mirrored overlap, not merely adjacent windows.'))
    # Close-up genuine anomaly comparison, not a synthetic curve. Use UE28
    # because the independently scanned largest mean-gain steps select it.
    with h5py.File(job/'dataset/csi.h5','r') as f:
        h=f['H'][:,28,0];selected=np.array([278,279,280,2719,2720,2721,2722])
        z=h[:,0,32];length=np.linalg.norm(a['rx_positions'][:,28]-a['tx_positions'][:,0],axis=-1)
        fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True)
        ix=(a['t']>=2.6)&(a['t']<=3.)
        axs[0].plot(a['t'][ix],10*np.log10(a['power_ue'][ix,28]),'.-',label='All64-port mean gain')
        axs[0].set_ylabel('Channel gain (dB)');axs[0].legend()
        for port in [0,40,44]:axs[1].plot(a['t'][ix],10*np.log10(a['power_port'][ix,28,port]),'.-',label=f'TX{port}')
        axs[1].set_ylabel('Per-link gain (dB)');axs[1].legend()
        axs[2].plot(a['t'][ix],(length[ix]-length[ix][0])*1000,label='TX0-UE28 direct length change')
        axs[2].set(xlabel='Physical time (s)',ylabel='Direct length change (mm)');axs[2].legend()
        fig.suptitle('UE28 frame278/279: finite CSI change with continuous LOS geometry | 28GHz,10ms')
        fig.tight_layout();fig.savefig(out/'figures/11_UE28_unresolved_jump.png',dpi=180);fig.savefig(out/'figures/11_UE28_unresolved_jump.pdf');plt.close(fig)
    fig=plt.figure(figsize=(12,5));ax=fig.add_subplot(121,projection='3d');bx=fig.add_subplot(122,projection='3d')
    for u in range(40):ax.plot(*a['user_positions'][:,u].T,lw=.6)
    ax.plot(*a['actor_positions'][:,0].T,color='black',lw=2,label='Actor')
    for j in range(16):ax.plot(*a['positions'][:,j].T,lw=1)
    ax.set(xlabel='World x (m)',ylabel='World y (m)',zlabel='World z (m)',title='Actual UE, actor and panel trajectories')
    for j in [0,3,7,12]:bx.plot(*((a['positions'][:,j]-a['positions'][0,j])*100).T,label=f'Panel{j}')
    bx.set(xlabel='Delta x (cm)',ylabel='Delta y (cm)',zlabel='Delta z (cm)',title='Moving panel displacements');bx.legend()
    fig.suptitle('Actual geometry, Z-up, SI units | Munich 30s');fig.tight_layout();fig.savefig(out/'figures/12_actual_3D_trajectories.png',dpi=180);fig.savefig(out/'figures/12_actual_3D_trajectories.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));phase=np.load(out/'retained_LOS_phase.npy');keep=(phase['user']==37)&(phase['tx']==41)&(phase['frame']<101)
    axs[0].plot(a['t'][phase['frame'][keep]],phase['observed_step_rad'][keep],label='Observed retained CIR phase step')
    axs[0].plot(a['t'][phase['frame'][keep]],phase['from_saved_delay_step_rad'][keep],ls='--',label='-2pi fc delta tau (F64)')
    axs[0].set(xlabel='Time (s)',ylabel='Wrapped step (rad)',title='UE37/TX41 LOS, initial second');axs[0].legend(fontsize=8)
    axs[1].hist(phase['wrapped_delay_phase_residual_rad'],bins=80,density=True);axs[1].set(xlabel='Observed minus F64 phase-step formula (rad)',ylabel='Density',title='All383992 retained consecutive LOS pairs')
    fig.suptitle('Phase/geometry attribution, Float32 delay/phase limits; no CSI correction | 28GHz');fig.tight_layout();fig.savefig(out/'figures/13_LOS_phase_geometry.png',dpi=180);fig.savefig(out/'figures/13_LOS_phase_geometry.pdf');plt.close(fig)
    # Frequency-selectivity varies by UE even within a narrow 1.92MHz band.
    corr=np.load(out/'correlations.npz');fr=corr['frequency_per_ue'];sp=corr['spatial_per_ue'];within=[];between=[]
    for j in range(64):
        for k in range(j+1,64):
            (within if j//4==k//4 else between).extend(sp[:,j,k].tolist())
    save_json(out/'correlation_summary.json',dict(adjacent_frequency_abs_correlation_quantiles=np.quantile(np.diagonal(fr,1,axis1=1,axis2=2),[0,.01,.5,.99,1]),
        edge_tone_abs_correlation_quantiles=np.quantile(fr[:,0,-1],[0,.01,.5,.99,1]),
        within_panel_abs_spatial_correlation_quantiles=np.quantile(within,[0,.01,.5,.99,1]),
        between_panel_abs_spatial_correlation_quantiles=np.quantile(between,[0,.01,.5,.99,1]),
        normalization='Uncentered complex pair correlation across all observed times and frequencies, per UE; not covariance of an assumed stationary random process'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--job',type=Path,required=True);a=p.parse_args();start=time.monotonic()
    provenance(a.job,a.audit);labels_and_errors(a.job,a.audit);enrich(a.job,a.audit)
    save_json(a.audit/'enrichment_complete.json',dict(new_GPU_seconds=0,wall_s=time.monotonic()-start,figures=13))
    print('Independent CPU enrichment completed; no new GPU solve.',flush=True)


if __name__=='__main__':main()
