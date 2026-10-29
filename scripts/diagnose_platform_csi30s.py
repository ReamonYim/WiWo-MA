#!/usr/bin/env python3
"""CPU-only second pass: real mesh visibility, mechanics, events and plots.

Run only after audit_platform_csi30s has closed its outputs. Input CSI is always
read-only. Scalar Mitsuba is a geometry checker, NOT a new SDK Full64 reference.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['WIWO_GPU']='-1'
import argparse
from collections import Counter, defaultdict
import csv
import importlib.metadata
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wiwo_ma.platform.dataset_audit import (C, save_json, sha_file, nmse_all_users,
    prescribed_pose, temporal_correlation, decode_plain)


def scalar_profile(distance, speed, acceleration, elapsed):
    """Independent trapezoidal/triangular rest-to-rest displacement."""
    t=np.maximum(np.asarray(elapsed),0)
    ramp=min(speed/acceleration,np.sqrt(distance/acceleration))
    peak=acceleration*ramp
    cruise=max(0.,distance/peak-ramp) if peak else 0.
    duration=2*ramp+cruise
    x=np.where(t<ramp,.5*acceleration*t*t,
        np.where(t<ramp+cruise,.5*acceleration*ramp*ramp+peak*(t-ramp),
            distance-.5*acceleration*np.maximum(duration-t,0)**2))
    return np.minimum(np.maximum(x,0),distance),duration


def mechanics(a,config):
    t=a['t'];dt=config['timing']['csi_dt_s'];panels=[p for b in config['base_stations'] for p in b['panels']]
    rows=[]; motion_frames=set(); expected_position=[];expected_rotation=[]
    for j,p in enumerate(panels):
        bs=config['base_stations'][0];bp,br,_=prescribed_pose(bs,0)
        prev_p=np.array(p['position_m']);prev_r=Rotation.from_euler('ZYX',p['orientation_deg'],degrees=True).as_matrix()
        pp=np.broadcast_to(bp+br@prev_p,(len(t),3)).copy();rr=np.broadcast_to(br@prev_r,(len(t),3,3)).copy()
        events=[]
        for action in [x for x in config['actions'] if x['panel_id']==p['id']]:
            target=np.array(action['position_m']);target_r=Rotation.from_euler('ZYX',action['orientation_deg'],degrees=True).as_matrix()
            delta=target-prev_p;distance=np.linalg.norm(delta);rotvec=Rotation.from_matrix(target_r@prev_r.T).as_rotvec();angle=np.linalg.norm(rotvec)
            x,duration=scalar_profile(distance,p['mechanics']['max_speed_m_s'],p['mechanics']['max_acceleration_m_s2'],t-action['time_s'])
            theta,rduration=scalar_profile(angle,p['mechanics']['max_angular_speed_rad_s'],p['mechanics']['max_angular_acceleration_rad_s2'],t-action['time_s'])
            ix=np.flatnonzero(t>=action['time_s']-1e-10)
            pp[ix]=bp+((prev_p+x[ix,None]*(delta/max(distance,1e-300)))@br.T)
            rr[ix]=br@Rotation.from_rotvec(theta[ix,None]*rotvec/max(angle,1e-300)).as_matrix()@prev_r
            events.append(dict(time_s=action['time_s'],translation_duration_s=duration,rotation_duration_s=rduration,
                               last_motion_time_s=action['time_s']+max(duration,rduration)))
            prev_p=target;prev_r=target_r
        speed=np.linalg.norm(a['velocities'][:,j],axis=-1);omega=np.linalg.norm(a['angular_velocities'][:,j],axis=-1)
        acc=np.linalg.norm(np.diff(a['velocities'][:,j],axis=0),axis=-1)/dt
        alpha=np.linalg.norm(np.diff(a['angular_velocities'][:,j],axis=0),axis=-1)/dt
        rotation_difference=Rotation.from_matrix(a['rotations'][:,j]@rr.transpose(0,2,1)).magnitude()
        position_difference=np.linalg.norm(a['positions'][:,j]-pp,axis=-1)
        movement=np.flatnonzero((speed>1e-9)|(omega>1e-9));motion_frames.update(movement.tolist())
        row=dict(panel=j,panel_id=p['id'],max_speed_m_s=speed.max(),max_acceleration_m_s2=acc.max(),
            max_angular_speed_rad_s=omega.max(),max_angular_acceleration_rad_s2=alpha.max(),
            independent_analytic_position_max_error_m=position_difference.max(),independent_analytic_rotation_max_error_rad=rotation_difference.max(),
            displacement_m=float(np.linalg.norm(a['positions'][-1,j]-a['positions'][0,j])),
            first_moving_frame=int(movement[0]) if len(movement) else None,last_moving_frame=int(movement[-1]) if len(movement) else None,
            action_events=events,limit_violations=dict(speed=np.flatnonzero(speed>p['mechanics']['max_speed_m_s']+1e-8),
                acceleration=np.flatnonzero(acc>p['mechanics']['max_acceleration_m_s2']+1e-8)+1,
                angular_speed=np.flatnonzero(omega>p['mechanics']['max_angular_speed_rad_s']+1e-8),
                angular_acceleration=np.flatnonzero(alpha>p['mechanics']['max_angular_acceleration_rad_s2']+1e-8)+1,
                analytic_position=np.flatnonzero(position_difference>1e-8),analytic_rotation=np.flatnonzero(rotation_difference>1e-8)))
        rows.append(row);expected_position.append(pp);expected_rotation.append(rr)
    return dict(panels=rows,motion_frame_indices=sorted(motion_frames),
                status='Passed' if all(not len(v) for r in rows for v in r['limit_violations'].values()) else 'Failed',
                scope='Saved 10ms states and independent analytic bounded profile, plus every state rigid/clock test. No hardware/urban-body collision certification.',
                UE_velocity_reversal=dict(frame=1500,time_s=15.,velocity_jump_m_s=float(np.linalg.norm(a['user_velocities'][1500]-a['user_velocities'][1499],axis=-1).max()),
                explanation='Prescribed piecewise-linear UE route has an instantaneous velocity reversal. Position remains continuous; no UE acceleration bound is implemented.'))


def cpu_visibility(a,config,out):
    import mitsuba as mi
    mi.set_variant('scalar_rgb')
    if mi.variant()!='scalar_rgb':raise RuntimeError('No GPU geometry audit permitted')
    xml=Path(importlib.metadata.distribution('sionna-rt').locate_file('sionna/rt/scenes/'+config['scene']['name']+'/'+config['scene']['name']+'.xml'))
    resource={str(xml):sha_file(xml)};d={'type':'scene'}
    for i,x in enumerate(ET.parse(xml).getroot().findall('shape')):
        if x.find('transform') is not None:raise NotImplementedError('Unvalidated mesh transform')
        path=xml.parent/x.find("string[@name='filename']").get('value');resource[str(path)]=sha_file(path)
        d['shape'+str(i)]={'type':x.get('type'),'filename':str(path),'face_normals':True}
    start=time.monotonic();scene=mi.load_dict(d);build=time.monotonic()-start
    n,u,tx=a['power_port'].shape;blocked=np.zeros((n,u,tx),bool);t0=time.monotonic()
    # Scalar CPU acceleration structure: all 7,682,560 actual endpoint segments.
    # Test strict interior of link; stop 0.1mm short of antenna endpoints.
    for i in range(n):
        for uu in range(u):
            rx=a['rx_positions'][i,uu].astype(float)
            for tt,xyz in enumerate(a['tx_positions'][i]):
                xyz=xyz.astype(float);delta=rx-xyz;length=float(np.linalg.norm(delta));direction=delta/length
                ray=mi.Ray3f(mi.Point3f(xyz+1e-4*direction),mi.Vector3f(direction));ray.maxt=length-2e-4
                blocked[i,uu,tt]=scene.ray_test(ray)
        if i%250==0:print(json.dumps(dict(phase='CPU_static_LOS_geometry',frame=i,elapsed_s=time.monotonic()-t0,new_GPU_seconds=0)),flush=True)
    np.savez_compressed(out/'visibility.npz',static_blocked=blocked,actor_blocked=a['actor_blocked'],
        geometric_LOS=~(blocked|a['actor_blocked']))
    save_json(out/'scene_resource_hashes.json',resource)
    return dict(variant=mi.variant(),new_GPU_seconds=0,scene_build_s=build,CPU_query_wall_s=time.monotonic()-t0,
        actual_link_segment_tests=n*u*tx,objects=len(d)-1,static_blocked_links=int(blocked.sum()),
        actor_intersected_direct_segments=int(a['actor_blocked'].sum()),
        static_visibility_transition_count=int(np.count_nonzero(np.diff(blocked.astype(np.int8),axis=0))),
        actor_visibility_transition_count=int(np.count_nonzero(np.diff(a['actor_blocked'].astype(np.int8),axis=0))),
        endpoint_exclusion_m=1e-4,scope='CPU triangle/OBB geometric LOS, no Field or Full64 SDK propagation solve, no assumption of watertight meshes')


def retained_diagnostics(a,paths,config,visibility,out):
    n,u,tx=a['power_port'].shape;freq=a['freq'];ratios=[];max_abs=[];count_exact=0;los_phase_errors=[];delay_errors=[];trackrows=[]
    # Only links which actually appear in the strongest128 are reconstructible
    # even partially. A subset residual is NOT a full-CIR numerical gate.
    with h5py.File(Path(config['_job'])/'dataset/csi.h5','r') as f:
        for i in range(n):
            subset=paths[paths['frame']==i];h=f['H'][i,:,0];partial=np.zeros_like(h,dtype=np.complex128);covered=np.zeros((u,tx),bool)
            for q in subset:
                uu,tt=int(q['user']),int(q['tx']);partial[uu,tt]+=q['a']*np.exp(-2j*np.pi*freq*q['delay_s']);covered[uu,tt]=True
            den=np.sum(abs(h[covered].astype(complex))**2);err=np.sum(abs(partial[covered]-h[covered])**2)
            ratios.append(err/den if den else None);max_abs.append(float(abs(partial-h).max()));count_exact+=int(np.count_nonzero(covered))
    # Exact endpoint pair and reflection depth; LOS paths are unambiguous in
    # saved subset. Reflected paths tracked only by nearest vertices and retain
    # membership, since primitive/object IDs and full sets were not persisted.
    prev={};life=Counter();matched=Counter();phase_rows=[]
    wavelength=C/config['radio']['carrier_hz']
    for i in range(n):
        subset=paths[paths['frame']==i];current={}
        for q in subset:
            key=(int(q['user']),int(q['tx']),int(q['depth']))
            if key not in current:current[key]=[]
            current[key].append(q)
        for key,qs in current.items():
            life[key]+=1
            if key not in prev:continue
            for q in qs:
                if q['depth']==0:
                    old=prev[key][0];delaydelta=float(q['delay_s']-old['delay_s'])
                    measured=float(np.angle(q['a']*np.conj(old['a'])))
                    ideal=float(np.angle(np.exp(-2j*np.pi*config['radio']['carrier_hz']*delaydelta)))
                    # Float32 fc*tau rounding and angle reduction are deliberately
                    # NOT assumed equal to a float64 delay-difference formula.
                    residual=float(np.angle(np.exp(1j*(measured-ideal))))
                    length0=np.linalg.norm(old['points'][1].astype(float)-old['points'][0]);length1=np.linalg.norm(q['points'][1].astype(float)-q['points'][0])
                    geom=float(np.angle(np.exp(-2j*np.pi*(length1-length0)/wavelength)))
                    phase_rows.append((i,key[0],key[1],measured,ideal,geom,residual,float(q['delay_s']-length1/C)))
                    matched['LOS consecutive retained comparisons']+=1
                else:matched['reflected retained possible matches (not primitive certified)']+=1
        prev=current
    dtype=[('frame','i4'),('user','i2'),('tx','i2'),('observed_step_rad','f8'),('from_saved_delay_step_rad','f8'),('from_F64_length_step_rad','f8'),('wrapped_delay_phase_residual_rad','f8'),('delay_vs_length_s','f8')]
    phase=np.array(phase_rows,dtype=dtype);np.save(out/'retained_LOS_phase.npy',phase,allow_pickle=False)
    np.savez_compressed(out/'partial_CIR_diagnostic.npz',covered_link_partial_CFR_NMSE=ratios)
    save_json(out/'retained_path_diagnostics.json',dict(path_records=len(paths),depth_counts=Counter(map(int,paths['depth'])),
        retained_users=sorted(set(map(int,paths['user']))),retained_UE_time_cells=int(np.count_nonzero(a['retained_counts'])),total_UE_time_cells=n*u,
        partial_CFR_residual_quantiles=np.quantile(ratios,[0,.5,.9,.99,1]),covered_links=count_exact,
        partial_CIR_is_NOT_full_path_validation=True,matched_counts=matched,
        LOS_consecutive_phase_delay_residual_rad_quantiles=np.quantile(abs(phase['wrapped_delay_phase_residual_rad']),[0,.5,.9,.99,1]) if len(phase) else [],
        delay_vs_F64_length_m_quantiles=np.quantile(abs(phase['delay_vs_length_s'])*C,[0,.5,.99,1]) if len(phase) else [],
        lifecycle='Retained subset membership only: top128 ranking truncation prevents global path birth/death certification',
        no_original_primitive_IDs=True))


def anomaly_analysis(a,vis,out):
    p=10*np.log10(a['power_ue']);dp=np.diff(p,axis=0);corr=a['pair_correlation'];step=a['relative_step'];n,u=p.shape
    motion=(np.linalg.norm(a['velocities'],axis=-1).max(axis=-1)>1e-9)|(np.linalg.norm(a['angular_velocities'],axis=-1).max(axis=-1)>1e-9)
    los=vis['geometric_LOS'];los_change=np.zeros((n,u),bool);los_change[1:]=np.any(los[1:]!=los[:-1],axis=-1)
    candidates=set()
    for values in [abs(dp),1-corr[1:],np.log10(np.maximum(step[1:],1e-300))]:
        for k in np.argpartition(values.ravel(),-40)[-40:]:
            ii,uu=np.unravel_index(k,values.shape);candidates.add((ii+1,uu))
    rows=[]
    for i,uu in sorted(candidates):
        membership_changed=int(a['retained_counts'][i,uu])!=int(a['retained_counts'][i-1,uu])
        near_action=any(abs(a['t'][i]-v)<=.03 for v in [1,15])
        if los_change[i,uu]:explanation='Geometric direct visibility transition; full multipath attribution remains Unknown'
        elif motion[i] or motion[i-1]:explanation='During bounded panel motion; CSI/path attribution needs independent SDK sample'
        elif membership_changed:explanation='Retained top128 membership changes; not proof of physical path birth/death'
        elif abs(dp[i-1,uu])>6:explanation='Large gain step without direct LOS change; unexplained multipath/discovery or interference event'
        else:explanation='Low temporal correlation/phase-sensitive superposition; full-path cause unresolved'
        rows.append(dict(frame=i,time_s=float(a['t'][i]),user=int(uu),power_before_dB=float(p[i-1,uu]),power_after_dB=float(p[i,uu]),power_step_dB=float(dp[i-1,uu]),
            correlation=float(corr[i,uu]),relative_CSI_step_energy=float(step[i,uu]),panel_motion=bool(motion[i]),near_action=near_action,
            direct_LOS_change=bool(los_change[i,uu]),retained_membership_count_change=membership_changed,
            explanation=explanation,numerical_validation='Unknown unless in seven checked frames'))
    with (out/'anomalies.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    # All threshold exceedances retained, not just a plotted top-N sample.
    flags=[]
    for i,uu in np.argwhere((abs(dp)>6)|(corr[1:]<.5)|(step[1:]>4)):
        flags.append(dict(frame=int(i+1),user=int(uu),time_s=float(a['t'][i+1]),power_step_dB=float(dp[i,uu]),correlation=float(corr[i+1,uu]),relative_step=float(step[i+1,uu])))
    save_json(out/'all_anomaly_flags.json',dict(thresholds_for_triage_NOT_acceptance=dict(power_step_abs_dB=6,correlation_below=.5,relative_step_above=4),
        flagged_UE_transitions=flags,flagged_frames=sorted(set(r['frame'] for r in flags)),
        note='Flags are diagnostic; no numerical Failed is inferred without reference. Geometry event alone does not explain all CSI components.'))
    save_json(out/'channel_statistics.json',dict(power_gain_dB_quantiles=np.quantile(p,[0,.001,.01,.5,.99,.999,1]),
        consecutive_gain_step_abs_dB_quantiles=np.quantile(abs(dp),[0,.5,.9,.99,.999,1]),
        consecutive_correlation_quantiles=np.quantile(corr[1:],[0,.001,.01,.5,.99,1]),
        relative_step_energy_quantiles=np.quantile(step[1:],[0,.5,.9,.99,.999,1]),
        weakest_mean_gain_UEs=np.argsort(a['power_ue'].mean(axis=0))[:5],flagged_UE_transitions=len(flags),flagged_frames=len(set(r['frame'] for r in flags)),
        stationary_panel_large_steps=int(np.count_nonzero((abs(dp)>6)&(~motion[1:,None]))),
        power_definition='10log10(mean_RX_TX_F |H|^2), no transmit power, no noise; includes directional antenna response',
        path_loss_definition='Negative channel gain in dB, includes antenna patterns; not pure isotropic propagation path loss',
        sampled_Doppler_Nyquist_hz=50.,frequency_delay_bin_s=1/(64*30000),frequency_delay_period_s=1/30000,
        UE_speed_m_s=float(np.linalg.norm(a['user_velocities'],axis=-1).max()),UE_max_oneway_Doppler_bound_hz=float(np.linalg.norm(a['user_velocities'],axis=-1).max()*28e9/C),
        actor_bistatic_speed_bound_hz=.5*2*28e9/C,actor_aliasing_risk='Possible for actor-reflected paths, not a claim that those paths dominate',
        velocity_discontinuities='All UE routes reverse at t15; prescribed trajectories have no bounded UE acceleration'))
    return rows


def plots(a,corr,paths,refs,anomalies,out):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'figure.dpi':140,'savefig.dpi':180,'axes.grid':True,'grid.alpha':.2})
    t=a['t'];p=10*np.log10(a['power_ue']);users=[int(np.argmin(a['power_ue'].mean(axis=0))),int(np.argmax(a['power_ue'].mean(axis=0))),int(np.argsort(a['power_ue'].mean(axis=0))[len(p[0])//2])]
    def done(fig,name):
        fig.suptitle(fig._suptitle.get_text()+' | Munich, 28 GHz, K40, 64 physical TX, 64 tones, 10 ms',fontsize=10)
        if not isinstance(fig.get_layout_engine(),matplotlib.layout_engine.ConstrainedLayoutEngine):fig.tight_layout()
        fig.savefig(folder/(name+'.png'));fig.savefig(folder/(name+'.pdf'));plt.close(fig)
    def events(ax):
        for v in [1,15]:ax.axvline(v,color='black',ls='--',lw=.7,alpha=.6)
    fig,ax=plt.subplots(figsize=(11,5));fig.suptitle('Per-UE mean channel gain (no TX power or noise)')
    ax.plot(t,p,alpha=.45,lw=.6)
    for u in users:ax.plot(t,p[:,u],lw=1.3,label=f'UE{u}')
    events(ax);ax.set(xlabel='Physical time (s)',ylabel='10 log10 mean |H|² (dB)');ax.legend();done(fig,'01_UE_channel_gain')
    fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True);fig.suptitle('Representative complex CSI: fixed TX0, tone32 (+15 kHz)')
    for u in users:
        z=a['trace'][:,u,0];axs[0].plot(t,20*np.log10(np.maximum(abs(z),1e-30)),lw=.7,label=f'UE{u}');axs[1].plot(t,np.angle(z),lw=.45,label=f'UE{u}')
    axs[0].set_ylabel('20 log10 |H| (dB)');axs[1].set(xlabel='Physical time (s)',ylabel='Wrapped phase (rad)');axs[0].legend();[events(x) for x in axs];done(fig,'02_complex_CSI_traces')
    fig,axs=plt.subplots(len(users),1,figsize=(11,8),sharex=True);fig.suptitle('Time-frequency CFR at TX0; no per-frame normalization')
    for ax,u in zip(axs,users):
        im=ax.pcolormesh(t,a['freq']/1e3,20*np.log10(np.maximum(abs(a['first_port_frequency'][:,u].T),1e-30)),shading='auto',cmap='viridis');ax.set(ylabel=f'UE{u}\nOffset (kHz)');fig.colorbar(im,ax=ax,label='20 log10 |H| (dB)');events(ax)
    axs[-1].set_xlabel('Physical time (s)');done(fig,'03_time_frequency_CSI')
    fig,axs=plt.subplots(1,2,figsize=(11,4.7),layout='constrained');fig.suptitle('Uncentered correlations; normalize each pair by its sample energies')
    im=axs[0].imshow(corr['spatial_mean'],vmin=0,vmax=1,cmap='magma');axs[0].set(xlabel='Physical TX port',ylabel='Physical TX port',title='Mean over 40 per-UE |corr|')
    for k in range(4,64,4):axs[0].axhline(k-.5,color='white',lw=.25);axs[0].axvline(k-.5,color='white',lw=.25)
    axs[1].imshow(corr['frequency_mean'],vmin=0,vmax=1,cmap='magma');axs[1].set(xlabel='Tone index',ylabel='Tone index',title='Frequency correlation');fig.colorbar(im,ax=axs.ravel().tolist(),label='|rho|',shrink=.7);done(fig,'04_spatial_frequency_correlation')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5));fig.suptitle('Temporal ACF and sampled Doppler (nonstationary 30 s mixture)')
    for u in users:axs[0].plot(t,abs(corr['temporal_acf'][u]),label=f'UE{u}')
    axs[0].set(xlim=(0,2),xlabel='Lag (s)',ylabel='|uncentered overlap-normalized ACF|');axs[0].axhline(.5,color='k',ls='--');axs[0].legend()
    df=corr['doppler_frequency_hz'];order=np.argsort(df)
    for u in users:
        ps=corr['doppler_density'][u];axs[1].plot(df[order],10*np.log10(np.maximum(ps[order]/ps.max(),1e-30)),label=f'UE{u}')
    axs[1].set(xlabel='Sampled Doppler frequency (Hz)',ylabel='Relative PSD (dB)',xlim=(-50,50),ylim=(-65,2));done(fig,'05_temporal_and_sampled_Doppler')
    fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True);fig.suptitle('Retained strongest128 paths only: not complete CIR or lifetime')
    idx=np.linspace(0,len(paths)-1,min(40000,len(paths))).astype(int);q=paths[idx]
    im=axs[0].scatter(t[q['frame']],q['delay_s']*1e9,c=20*np.log10(abs(q['a'])),s=1,cmap='viridis');axs[0].set_ylabel('Retained delay (ns)');fig.colorbar(im,ax=axs[0],label='20 log10 |a| (dB)')
    for u in sorted(set(map(int,paths['user'])))[:8]:axs[1].plot(t,a['retained_counts'][:,u],lw=.6,label=f'UE{u}')
    axs[1].set(xlabel='Physical time (s)',ylabel='Retained path count');axs[1].legend(ncol=4);[events(x) for x in axs];done(fig,'06_retained_path_delay_and_membership')
    fig,axs=plt.subplots(4,1,figsize=(11,10),sharex=True);fig.suptitle('Action → actual bounded panel motion → channel response')
    for j in [0,3,7,12]:
        axs[0].plot(t,np.linalg.norm(a['positions'][:,j]-a['positions'][0,j],axis=-1)*100,label=f'Panel{j}')
        axs[1].plot(t,np.linalg.norm(a['velocities'][:,j],axis=-1),lw=.8)
        angle=Rotation.from_matrix(a['rotations'][:,j]@a['rotations'][0,j].T).magnitude();axs[2].plot(t,np.degrees(angle),lw=.8)
    axs[0].set_ylabel('Displacement (cm)');axs[0].legend(ncol=4);axs[1].set_ylabel('Speed (m/s)');axs[2].set_ylabel('Rotation from t0 (deg)')
    for u in users:axs[3].plot(t,p[:,u],lw=.6,label=f'UE{u}')
    axs[3].set(xlabel='Physical time (s)',ylabel='Channel gain (dB)');[events(x) for x in axs];done(fig,'07_mechanics_and_CSI')
    fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True);fig.suptitle('Geometric LOS/actor intersections and CSI step diagnostics')
    vis=np.load(out/'visibility.npz');axs[0].plot(t,vis['geometric_LOS'].mean(axis=(1,2)),label='Static+actor geometric LOS fraction');axs[0].plot(t,a['actor_blocked'].mean(axis=(1,2)),label='Direct actor intersection fraction');axs[0].set_ylabel('Link fraction');axs[0].legend()
    axs[1].plot(t[1:],abs(np.diff(p,axis=0)).max(axis=1));axs[1].set_ylabel('Max UE gain step (dB)')
    axs[2].plot(t[1:],np.nanmin(a['pair_correlation'][1:],axis=1));axs[2].set(xlabel='Physical time (s)',ylabel='Min UE adjacent |corr|');[events(x) for x in axs];done(fig,'08_geometry_events_and_jumps')
    top=sorted(anomalies,key=lambda r:abs(r['power_step_dB']),reverse=True)[:4]
    fig,axs=plt.subplots(2,2,figsize=(12,7));fig.suptitle('Largest gain-step diagnostics (flags, not automatically numerical failures)')
    for ax,row in zip(axs.ravel(),top):
        i,u=row['frame'],row['user'];ix=slice(max(0,i-15),min(len(t),i+16));ax.plot(t[ix],p[ix,u],marker='.',label='Mean channel gain');ax.axvline(t[i],color='r',ls='--');ax.set(title=f"UE{u}, frame{i}, step={row['power_step_dB']:.2f} dB",xlabel='Physical time (s)',ylabel='Channel gain (dB)')
    done(fig,'09_extreme_event_diagnosis')
    fig,ax=plt.subplots(figsize=(11,4.5));fig.suptitle('Independent SDK re-audit: all 40 users included, saved seven references')
    for row in refs['rows']:ax.plot(np.arange(40),row['per_user_NMSE'],marker='.',lw=.7,label=f"Frame {row['index']} ({row['time_s']}s)")
    ax.axhline(1e-3,color='red',ls='--',label='Frozen NMSE gate 1e-3');ax.set(yscale='log',xlabel='UE index (zero based)',ylabel='Complex CSI NMSE');ax.legend(ncol=2,fontsize=8);done(fig,'10_independent_SDK_NMSE')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--job',type=Path,required=True);args=p.parse_args()
    out=args.audit;job=args.job;a=dict(np.load(out/'derived.npz',allow_pickle=False));config=json.loads((job/'config.json').read_text());config['_job']=str(job)
    if (out/'diagnosis_complete.json').exists():raise FileExistsError('Do not overwrite finished audit')
    before=sha_file(job/'dataset/csi.h5');start=time.monotonic()
    save_json(out/'mechanics_alignment.json',mechanics(a,config))
    if not (out/'visibility.npz').exists():save_json(out/'visibility_geometry.json',cpu_visibility(a,config,out))
    vis=dict(np.load(out/'visibility.npz'));paths=np.load(out/'retained_paths.npy',allow_pickle=False)
    retained_diagnostics(a,paths,config,vis,out)
    anomalies=anomaly_analysis(a,vis,out)
    plots(a,dict(np.load(out/'correlations.npz')),paths,json.loads((out/'SDK_reaudit.json').read_text()),anomalies,out)
    # References from the earlier small gate are NOT silently treated as
    # episode labels: independently test configuration and every state operand.
    other=Path('artifacts/platform/validation/native_gate_attempt01');frozen=json.loads((other/'full64_0_state.json').read_text());cc=json.loads((other/'full64_config.json').read_text())
    with h5py.File(job/'dataset/csi.h5','r') as f:
        state=json.loads(f['frames_json'].asstr()[120])['state']
        keys=['tx_port_positions','rx_port_positions','tx_port_orientations_rad','rx_port_orientations_rad','actors','user_positions','rotations','positions']
        matches={key:state[key]==frozen[key] for key in keys}
        eligible=cc=={k:v for k,v in config.items() if k!='_job'} and all(matches.values()) and state['timestamp']==frozen['timestamp']
        extra=dict(frame=120,time_s=1.2,configuration_exact=cc=={k:v for k,v in config.items() if k!='_job'},operand_matches=matches,
            eligible=eligible,original_seven_label_ledger_unchanged=True)
        if eligible:
            with np.load(other/'full64_0/reference.npz') as b:err,den,num=nmse_all_users(f['H'][120],b['H'])
            extra.update(per_user_NMSE=err,max_NMSE=err.max(),additional_offline_comparison_status='Passed' if np.isfinite(err).all() and np.all(err<=1e-3) else 'Failed',
                source_reference_sha256=sha_file(other/'full64_0/reference.npz'),scope='Additional audit evidence only; no original HDF/sidecar labels rewritten')
        save_json(out/'prior_gate_reuse_eligibility.json',extra)
    unchanged=before==sha_file(job/'dataset/csi.h5')
    save_json(out/'diagnosis_complete.json',dict(status='CPU_analysis_finished',new_GPU_seconds=0,wall_s=time.monotonic()-start,HDF_unchanged=unchanged,
        scalar_geometry_not_an_SDK_numerical_gate=True,figure_count=len(list((out/'figures').glob('*.png')))))
    print(json.dumps(dict(status='CPU_analysis_finished',new_GPU_seconds=0,HDF_unchanged=unchanged)),flush=True)


if __name__=='__main__':main()
