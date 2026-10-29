"""Independent, read-only CPU audit of a closed native CSI episode.

No production RT, writer, controller, or SDK imports. All derived products are
written outside the input job. SHA256 binds every complex sample AND exact row
JSON; separate checks test physics rather than just trusting the row checksum.
Numerical labels only come from direct comparisons with saved SDK arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
from scipy import fft, signal
from scipy.spatial.transform import Rotation, Slerp

C = 299792458.0


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [clean(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def nmse_all_users(test, reference):
    """No weak-user exclusion or epsilon denominator. Zero cases explicit."""
    a, b = np.asarray(test, np.complex128), np.asarray(reference, np.complex128)
    if a.shape != b.shape or a.ndim < 2: raise ValueError('CSI shape mismatch')
    axes = tuple(range(1, a.ndim))
    den = np.sum(np.abs(b)**2, axis=axes)
    num = np.sum(np.abs(a-b)**2, axis=axes)
    with np.errstate(divide='ignore', invalid='ignore'):
        errors = np.divide(num, den, out=np.where(num == 0, 0., np.inf), where=den > 0)
    finite = np.isfinite(a).all(axis=axes) & np.isfinite(b).all(axis=axes)
    errors[~finite] = np.inf
    return errors, den, num


def temporal_correlation(h):
    """Uncentered complex ACF, exact overlap-energy normalization per lag.

    h: time x arbitrary channels. FFT zero padding excludes circular wrap;
    no mean removal, no assumed stationarity, no padding energy in denominator.
    """
    x = np.asarray(h).reshape(len(h), -1)
    n = len(x); nfft = fft.next_fast_len(2*n-1)
    num = np.zeros(n, np.complex128)
    for first in range(0, x.shape[1], 128):
        z = fft.fft(x[:, first:first+128].astype(np.complex128), n=nfft, axis=0)
        num += fft.ifft((z*z.conj()).sum(axis=1), axis=0)[:n]
    energy = np.sum(np.abs(x.astype(np.complex128))**2, axis=1)
    cum = np.r_[0., np.cumsum(energy)]
    den = np.sqrt(cum[n-np.arange(n)] * (cum[n]-cum[np.arange(n)]))
    return np.divide(num, den, out=np.full(n, np.nan+0j), where=den > 0)


def correlation_matrix(x):
    x = np.asarray(x, np.complex128)
    g = x.conj().T @ x
    d = np.sqrt(np.maximum(g.diagonal().real, 0))
    den = d[:, None]*d[None, :]
    return np.divide(g, den, out=np.full_like(g, np.nan), where=den > 0)


def local_array(a):
    # Independently implement documented column-major Sionna array convention.
    nr, nc, spacing = a['rows'], a['cols'], a['spacing_m']
    col = np.repeat(np.arange(nc), nr)
    row = np.tile(np.arange(nr), nc)
    return np.column_stack((np.zeros(nr*nc), (col-(nc-1)/2)*spacing,
                            ((nr-1)/2-row)*spacing))


def prescribed_pose(entity, t):
    route = entity.get('trajectory', [])
    if not route:
        return (np.array(entity['position_m'], float),
                Rotation.from_euler('ZYX', entity['orientation_deg'], degrees=True).as_matrix(), np.zeros(3))
    ts = np.array([q['time_s'] for q in route])
    pos = np.array([q['position_m'] for q in route])
    rot = Rotation.from_euler('ZYX', [q['orientation_deg'] for q in route], degrees=True)
    if t >= ts[-1]: return pos[-1], rot[-1].as_matrix(), np.zeros(3)
    i = int(np.searchsorted(ts, t, side='right')-1)
    s = (t-ts[i])/(ts[i+1]-ts[i])
    return ((1-s)*pos[i]+s*pos[i+1], Slerp(ts, rot)([t]).as_matrix()[0],
            (pos[i+1]-pos[i])/(ts[i+1]-ts[i]))


def segment_box(a, b, center, rotation, size):
    """Closed segment against OBB, all leading dimensions broadcast."""
    origin = (np.asarray(a)-center) @ rotation
    delta = (np.asarray(b)-np.asarray(a)) @ rotation
    half = np.asarray(size)/2
    parallel = np.abs(delta) < 1e-14
    outside = np.any(parallel & (np.abs(origin) > half), axis=-1)
    d = np.where(parallel, 1., delta)
    lo, hi = (-half-origin)/d, (half-origin)/d
    near = np.where(parallel, -np.inf, np.minimum(lo, hi)).max(axis=-1)
    far = np.where(parallel, np.inf, np.maximum(lo, hi)).min(axis=-1)
    return (~outside) & (np.maximum(near, 1e-6) <= np.minimum(far, 1-1e-6))


def decode_plain(x):
    """Inspect frozen snapshot without instantiating production classes."""
    if isinstance(x, list): return [decode_plain(v) for v in x]
    if not isinstance(x, dict): return x
    tag, val = x.get('@'), x.get('value')
    if tag == 'dict': return {decode_plain(k): decode_plain(v) for k,v in val}
    if tag in ('tuple',): return tuple(decode_plain(v) for v in val)
    if tag == 'array': return np.asarray(val, dtype=x['dtype'])
    if tag in ('MechanicalState', 'Trajectory'): return decode_plain(val)
    if tag == 'infinity': return np.inf * val
    raise ValueError('Unrecognized snapshot type')


def independent_geometry(state, config, snapshot):
    """Rigid transforms, route evaluation, snapshot and clock binding."""
    t = state['timestamp']; p = np.asarray(state['positions']); r = np.asarray(state['rotations'])
    checks = {}; measurements = {}
    measurements['rotation_orthogonality'] = np.max(np.abs(r @ r.transpose(0,2,1)-np.eye(3)))
    measurements['rotation_determinant_error'] = np.max(np.abs(np.linalg.det(r)-1))
    ports, maps, angles = [], [], []
    j = 0
    for bi, bs in enumerate(config['base_stations']):
        bp, br, _ = prescribed_pose(bs, t)
        for panel in bs['panels']:
            q = local_array(panel['array']) @ r[j].T + p[j]
            ports.extend(q); maps.extend((bi,j,e,0) for e in range(len(q)))
            angles.extend([Rotation.from_matrix(r[j]).as_euler('ZYX')]*len(q))
            local = br.T @ (p[j]-bp)
            m = panel['mechanics']
            checks['workspace_'+str(j)] = bool(np.all(local >= np.array(m['workspace_min_m'])-1e-9) and np.all(local <= np.array(m['workspace_max_m'])+1e-9))
            sr = snapshot['controllers'][j]['state']
            checks['snapshot_'+str(j)] = bool(np.allclose(bp+br@sr['position'],p[j],atol=1e-12,rtol=0) and np.allclose(br@sr['rotation'],r[j],atol=1e-12,rtol=0))
            j += 1
    expected = np.asarray(ports, np.float32); actual = np.asarray(state['tx_port_positions'], np.float32)
    measurements['tx_port_max_error_m'] = np.max(np.abs(expected.astype(float)-actual))
    measurements['tx_port_bit_mismatches'] = np.count_nonzero(expected.view(np.uint32) != actual.view(np.uint32))
    # Angle extraction may have a different last bit in SciPy. Compare actual
    # pattern rotation matrices, not just Euler triples near singularities.
    angleR = Rotation.from_euler('ZYX', np.asarray(state['tx_port_orientations_rad'])).as_matrix()
    measurements['tx_orientation_matrix_error'] = np.max(np.abs(angleR-r[np.asarray(maps)[:,1]]))
    checks['tx_map'] = bool(np.array_equal(maps, state['tx_port_map']))
    user_error = 0.; rx_error = 0.; uv_error = 0.; rxmap = []
    for u, user in enumerate(config['users']):
        up, ur, uv = prescribed_pose(user,t)
        user_error = max(user_error, np.max(np.abs(up-state['user_positions'][u])))
        uv_error = max(uv_error,np.max(np.abs(uv-state['user_velocities'][u])))
        loc = (local_array(user['array']) @ ur.T + up).astype(np.float32)
        start = len(rxmap)
        rx_error = max(rx_error,np.max(np.abs(loc.astype(float)-np.array(state['rx_port_positions'])[start:start+len(loc)])))
        rxmap.extend((u,e,0) for e in range(len(loc)))
    checks['rx_map'] = bool(np.array_equal(rxmap,state['rx_port_map']))
    measurements.update(user_position_error_m=user_error, user_velocity_error_m_s=uv_error,rx_port_error_m=rx_error)
    actor_error = 0.; actor_v_error = 0.
    signs = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
    for conf, actor in zip(config['actors'],state['actors']):
        ap, ar, av = prescribed_pose(conf,t)
        vv = ((signs*np.array(conf['size_m'])/2) @ ar.T + ap).astype(np.float32)
        actor_error = max(actor_error,np.max(np.abs(vv.astype(float)-actor['vertices'])))
        actor_v_error = max(actor_v_error,np.max(np.abs(av-actor['velocity_m_s'])))
    measurements.update(actor_vertices_error_m=actor_error,actor_velocity_error_m_s=actor_v_error)
    checks['clock'] = bool(snapshot['tick']==state['mechanics_tick']==round(t/config['timing']['mechanics_dt_s']))
    checks['snapshot_configuration'] = snapshot['configuration']==state['configuration_sha256']
    checks['actions_due'] = state['actions_executed']==sum(a['time_s']<=t+1e-10 for a in config['actions'])
    checks['rigid_geometry'] = bool(measurements['tx_port_max_error_m']<=8e-6 and measurements['tx_orientation_matrix_error']<=3e-7 and user_error<1e-10 and rx_error<=8e-6 and actor_error<=8e-6)
    checks['rotations'] = bool(measurements['rotation_orthogonality']<1e-10 and measurements['rotation_determinant_error']<1e-10)
    return checks, measurements


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    job, out = args.job.resolve(), args.out.resolve()
    if out.is_relative_to(job): raise ValueError('Audit output must be outside original job')
    if out.exists(): raise FileExistsError('Use a new independent audit attempt directory')
    out.mkdir(parents=True); start=time.monotonic()
    def progress(phase, **kw):
        value=dict(phase=phase,elapsed_cpu_wall_s=time.monotonic()-start,new_GPU_seconds=0,**kw)
        save_json(out/'progress.json',value); print(json.dumps(clean(value)),flush=True)
    progress('hash_closed_input')
    source = job/'dataset/csi.h5'; stat_before=source.stat(); inputsha=sha_file(source)
    config=json.loads((job/'config.json').read_text())
    receipts={str(p.relative_to(job)):sha_file(p) for p in job.rglob('*') if p.is_file() and p!=source}
    source_hashes={str(p):sha_file(p) for p in [Path(__file__),Path('scripts/audit_platform_csi30s.py')]}
    input_record=dict(path=str(source),size_bytes=stat_before.st_size,mtime_ns=stat_before.st_mtime_ns,sha256=inputsha,
                      auxiliary_files=receipts,audit_source_sha256=source_hashes)
    save_json(out/'input_receipt.json',input_record)
    with h5py.File(source,'r') as f:
        if not bool(f.attrs.get('complete',False)): raise ValueError('Do not audit an open/incomplete writer')
        n,u,rx,tx,nf=f['H'].shape
        if rx!=1 or config['radio']['tx_polarization']!='V' or config['radio']['rx_polarization']!='V':
            raise NotImplementedError('This audit rigid-map implementation explicitly covers the actual single-pol episode')
        schema={k:dict(shape=list(d.shape),dtype=str(d.dtype),chunks=d.chunks,compression=d.compression,shuffle=d.shuffle,
                       fletcher32=d.fletcher32,storage_bytes=d.id.get_storage_size()) for k,d in f.items()}
        attrs={k:v for k,v in f.attrs.items() if k!='config_json'}
        ts=f['timestamps'][:]; freq=f['frequency_offsets_hz'][:]
        config_hash=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        # Production fingerprint uses the same documented canonical JSON, not
        # its helper. Verify actual representation in config.py if changed.
        expected_n=round(config['timing']['duration_s']/config['timing']['csi_dt_s'])+1
        timing=dict(expected_frames=expected_n,actual_frames=n,min_timestamp=float(ts.min()),max_timestamp=float(ts.max()),
                    min_dt=float(np.diff(ts).min()),max_dt=float(np.diff(ts).max()),
                    bad_grid_indices=np.flatnonzero(np.abs(ts-np.arange(n)*config['timing']['csi_dt_s'])>1e-10).tolist(),
                    duplicate_or_reversed_indices=(np.flatnonzero(np.diff(ts)<=0)+1).tolist(),
                    configuration_matches_HDF=json.loads(f.attrs['config_json'])==config,
                    canonical_configuration_sha256=config_hash,HDF_configuration_sha256=f.attrs['configuration_sha256'])
        save_json(out/'structure.json',dict(schema=schema,attrs=attrs,timing=timing,frequency_offsets_hz=freq,config=config,
                    raw_shape_meaning=['time','UE','RX port per UE','physical TX port','subcarrier'],
                    no_configured_or_added_noise=True,path_scope='Global strongest 128 subset; NOT complete paths',
                    bs_count=len(config['base_stations']),tx_port_map='(BS index, panel global index, spatial element, polarization)'))
        panels=sum(len(b['panels']) for b in config['base_stations'])
        a=dict(t=ts,freq=freq,power_ue=np.zeros((n,u)),power_port=np.zeros((n,u,tx),np.float32),
            trace=np.zeros((n,u,tx),np.complex64),first_port_frequency=np.zeros((n,u,nf),np.complex64),
            pair_correlation=np.ones((n,u)),relative_step=np.zeros((n,u)),absolute_step=np.zeros((n,u)),
            positions=np.zeros((n,panels,3)),rotations=np.zeros((n,panels,3,3)),velocities=np.zeros((n,panels,3)),
            angular_velocities=np.zeros((n,panels,3)),target_positions=np.zeros((n,panels,3)),
            user_positions=np.zeros((n,u,3)),user_velocities=np.zeros((n,u,3)),
            tx_positions=np.zeros((n,tx,3),np.float32),rx_positions=np.zeros((n,u,3),np.float32),
            actor_positions=np.zeros((n,len(config['actors']),3)),actor_blocked=np.zeros((n,u,tx),bool),
            retained_counts=np.zeros((n,u),np.int32),retained_los_counts=np.zeros((n,u),np.int32),
            retained_power=np.zeros((n,u)),retained_delay_mean=np.zeros((n,u)),retained_delay_rms=np.zeros((n,u)),
            solver_wall_s=np.zeros(n),geometry_wall_s=np.zeros(n),SDK_wall_s=np.zeros(n),capture_wall_s=np.zeros(n),
            allocator_peak_bytes=np.zeros(n,np.int64))
        raw_hashes=[]; user_hashes=[hashlib.sha256() for _ in range(u)];port_hashes=[hashlib.sha256() for _ in range(tx)]
        check_rows=[]; geometry_max=defaultdict(float); geometry_failures=defaultdict(list); path_max=defaultdict(float)
        amp_edges=np.linspace(-220,-20,201); amp_hist=np.zeros(200,np.int64);phase_hist=np.zeros(72,np.int64)
        component_min=[np.inf,np.inf];component_max=[-np.inf,-np.inf]; absmin=np.inf;absmax=0.
        path_records=[];path_bad=[];zero_links=[];zero_users=[];zero_samples=0;nonfinite=0; all_equal_adj=[]
        software=Counter(); scene_hashes=Counter();labels=Counter();geometry_identity=Counter(); SDKcalls=Counter(); before=None
        path_dtype=np.dtype([('frame','i4'),('user','i2'),('tx','i2'),('depth','i1'),('delay_s','f8'),('a','c16'),('points','f4',(5,3)),('types','i2',(3,))])
        for i in range(n):
            h=f['H'][i];text=f['frames_json'].asstr()[i];v=json.loads(text);s=v['state'];r=v['result'];snap=decode_plain(v['snapshot'])
            digest=hashlib.sha256(h.tobytes(order='C')+text.encode()).hexdigest()
            rawsha=hashlib.sha256(h.tobytes(order='C')).hexdigest();raw_hashes.append(rawsha)
            finite=bool(np.isfinite(h).all());nonfinite+=int(np.count_nonzero(~np.isfinite(h)))
            check=dict(index=i,time_s=ts[i],row_sha256=digest==f['frame_sha256'].asstr()[i],finite_CSI=finite,
                       clocks=abs(s['timestamp']-ts[i])<1e-12 and abs(r['physical_timestamp']-ts[i])<1e-12,
                       configuration=s['configuration_sha256']==f.attrs['configuration_sha256'],
                       original_native_definition=r['synthetic_array'] is False and r['SDK_calls']==16 and bool(np.all(r['rx_valid'])))
            cc,meas=independent_geometry(s,config,snap)
            for key,good in cc.items():
                if not good:geometry_failures[key].append(i)
            for key,val in meas.items():geometry_max[key]=max(geometry_max[key],float(val))
            check['mechanics_and_rigid_geometry']=all(cc.values())
            check_rows.append(check)
            for name in ['positions','rotations','velocities','angular_velocities','target_positions','user_positions','user_velocities']:
                a[name][i]=s[name]
            a['tx_positions'][i]=s['tx_port_positions'];a['rx_positions'][i]=s['rx_port_positions']
            for ai,actor in enumerate(s['actors']):
                a['actor_positions'][i,ai]=actor['position_m']
                a['actor_blocked'][i] |= segment_box(a['tx_positions'][i][None,:,:],a['rx_positions'][i][:,None,:],
                    np.array(actor['position_m']),np.array(actor['rotation_matrix']),actor['size_m'])
            h64=h[:,0].astype(np.complex128);power=np.abs(h64)**2
            a['power_port'][i]=power.mean(axis=-1);a['power_ue'][i]=power.mean(axis=(1,2))
            a['trace'][i]=h[:,0,:,nf//2];a['first_port_frequency'][i]=h[:,0,0]
            zero_samples+=int(np.count_nonzero(h==0));zeros=np.argwhere(np.all(h[:,0]==0,axis=-1))
            zero_links.extend((i,int(uu),int(tt)) for uu,tt in zeros)
            zero_users.extend((i,int(uu)) for uu in np.flatnonzero(np.all(h==0,axis=(1,2,3))))
            amplitude=np.abs(h64);positive=amplitude[amplitude>0]
            if positive.size:absmin=min(absmin,float(positive.min()));absmax=max(absmax,float(positive.max()))
            amp_hist+=np.histogram(20*np.log10(np.maximum(amplitude,1e-300)),amp_edges)[0]
            phase_hist+=np.histogram(np.angle(h64[h64!=0]),np.linspace(-np.pi,np.pi,73))[0]
            for k,x in enumerate([h.real,h.imag]):component_min[k]=min(component_min[k],float(x.min()));component_max[k]=max(component_max[k],float(x.max()))
            for uu in range(u):user_hashes[uu].update(h[uu].tobytes())
            for tt in range(tx):port_hashes[tt].update(h[:,:,tt,:].tobytes())
            if before is not None:
                den0=np.sum(np.abs(before)**2,axis=(1,2));den1=power.sum(axis=(1,2));num=np.sum(np.abs(h64-before)**2,axis=(1,2))
                a['absolute_step'][i]=np.sqrt(num/(tx*nf))
                a['relative_step'][i]=np.divide(num,den0,out=np.full(u,np.inf),where=den0>0)
                a['pair_correlation'][i]=np.divide(abs(np.sum(h64*before.conj(),axis=(1,2))),np.sqrt(den0*den1),out=np.full(u,np.nan),where=den0*den1>0)
                if rawsha==raw_hashes[-2]:all_equal_adj.append(i)
            before=h64
            software[json.dumps(r['software'],sort_keys=True)]+=1;scene_hashes[r['scene_hash']]+=1
            labels[f['independent_validation'].asstr()[i]]+=1;geometry_identity[r['geometry_identity']]+=1;SDKcalls[r['SDK_calls']]+=1
            for name,key in [('solver_wall_s','solver_wall_seconds'),('geometry_wall_s','geometry_update_seconds'),('allocator_peak_bytes','allocator_peak_bytes')]:a[name][i]=r[key]
            a['SDK_wall_s'][i]=sum(q['SDK_wall_seconds'] for q in r['batches'])
            a['capture_wall_s'][i]=sum(q['CFR_and_capture_wall_seconds'] for q in r['batches'])
            local_paths=[]
            for k,path in enumerate(r['paths']):
                pts=np.asarray(path['points'],float);uu=int(path['user']);tt=int(path['tx_port']);depth=len(pts)-2
                err=max(np.max(abs(pts[0]-s['tx_port_positions'][tt])),np.max(abs(pts[-1]-s['rx_port_positions'][uu])))
                length=np.linalg.norm(np.diff(pts,axis=0),axis=1).sum();dtau=path['delay_s']-length/C
                good=bool(np.isfinite(pts).all() and depth==len(path['interaction_types']) and 0<=depth<=3 and abs(path['timestamp']-ts[i])<1e-12 and err<1e-5 and np.isfinite([path['coefficient_real'],path['coefficient_imag'],path['delay_s']]).all() and path['delay_s']>0)
                if not good:path_bad.append((i,k))
                path_max['endpoint_error_m']=max(path_max['endpoint_error_m'],err)
                path_max['length_delay_discrepancy_m']=max(path_max['length_delay_discrepancy_m'],abs(dtau*C))
                rec=np.zeros((),path_dtype);rec['frame']=i;rec['user']=uu;rec['tx']=tt;rec['depth']=depth;rec['delay_s']=path['delay_s'];rec['a']=complex(path['coefficient_real'],path['coefficient_imag']);rec['points'][:len(pts)]=pts;rec['types'][:depth]=path['interaction_types'];local_paths.append(rec)
                a['retained_counts'][i,uu]+=1;a['retained_los_counts'][i,uu]+=int(depth==0)
            path_records.extend(local_paths)
            for uu in range(u):
                subset=[q for q in local_paths if q['user']==uu]
                if subset:
                    pw=np.array([abs(q['a'])**2 for q in subset]);tau=np.array([q['delay_s'] for q in subset]);m=float(np.sum(pw*tau)/pw.sum())
                    a['retained_power'][i,uu]=pw.sum();a['retained_delay_mean'][i,uu]=m;a['retained_delay_rms'][i,uu]=np.sqrt(np.sum(pw*(tau-m)**2)/pw.sum())
            if i%100==0:progress('stream_all_samples_and_states',frame=i,total=n)
        paths=np.array(path_records,dtype=path_dtype)
        np.save(out/'retained_paths.npy',paths,allow_pickle=False)
        np.savez_compressed(out/'derived.npz',**a)
        with (out/'per_frame_checks.csv').open('w') as stream:
            w=csv.DictWriter(stream,fieldnames=check_rows[0].keys());w.writeheader();w.writerows(check_rows)
        failures={key:[x['index'] for x in check_rows if not x[key]] for key in check_rows[0] if key not in ('index','time_s')}
        bad_frames=sorted(set(i for indices in failures.values() for i in indices))
        duplicates=defaultdict(list)
        for i,h in enumerate(raw_hashes):duplicates[h].append(i)
        integrity=dict(frame_counts=dict(Passed=n-len(bad_frames),Failed=len(bad_frames),Unknown=0),failed_indices=bad_frames,checks_failed_indices=failures,
            samples_examined=int(np.prod(f['H'].shape)),nonfinite_complex_samples=nonfinite,zero_complex_samples=zero_samples,
            zero_all_frequency_links=zero_links,zero_all_port_users=zero_users,duplicate_complete_H_frames=[v for v in duplicates.values() if len(v)>1],
            exact_repeated_adjacent_H=all_equal_adj,amplitude_nonzero_min=absmin,amplitude_max=absmax,component_min=component_min,component_max=component_max,
            user_sequence_sha256=[h.hexdigest() for h in user_hashes],port_sequence_sha256=[h.hexdigest() for h in port_hashes],
            raw_label_counts=labels,software=software,scene_hashes=scene_hashes,SDK_call_counts_per_frame=SDKcalls,
            geometry_identity_self_reports=geometry_identity,geometry_check_maxima=geometry_max,geometry_failure_indices=geometry_failures,
            retained_paths_count=len(paths),retained_path_failures=path_bad,retained_path_check_maxima=path_max,
            checksum_scope='All complex samples plus exact JSON, independently implemented SHA256; every HDF chunk read',
            numeric_accuracy_not_inferred_from_integrity=True)
        save_json(out/'integrity.json',integrity)
        np.savez_compressed(out/'histograms.npz',amp_edges=amp_edges,amplitude_counts=amp_hist,phase_edges=np.linspace(-np.pi,np.pi,73),phase_counts=phase_hist)
        # Existing SDK references: compare original arrays, bind operands and
        # saved receipt; JSON error values are not accepted as evidence.
        progress('recompute_saved_SDK_comparisons')
        refrows=[]
        for folder in sorted((job/'references').iterdir(),key=lambda p:int(p.name)):
            i=int(folder.name);s=json.loads(f['frames_json'].asstr()[i])['state'];frozen=json.loads((folder/'state.json').read_text())
            with np.load(folder/'H.npz',allow_pickle=False) as ref:h_ref=ref['H']
            errors,den,num=nmse_all_users(f['H'][i],h_ref);h_delta=f['H'][i].astype(np.complex128)-h_ref
            original=json.loads((folder/'report.json').read_text())
            operands=s==frozen
            row=dict(index=i,time_s=float(ts[i]),status='Passed' if operands and np.isfinite(errors).all() and np.all(errors<=1e-3) else 'Failed',
                operands_exact_JSON_match=operands,per_user_NMSE=errors,per_user_reference_power=den/(rx*tx*nf),
                max_NMSE=float(errors.max()),max_absolute_complex_error=float(abs(h_delta).max()),
                per_user_max_absolute_error=abs(h_delta).max(axis=(1,2,3)),per_port_NMSE=(np.sum(abs(h_delta)**2,axis=(0,1,3))/np.sum(abs(h_ref.astype(np.complex128))**2,axis=(0,1,3))),
                per_subcarrier_NMSE=np.sum(abs(h_delta)**2,axis=(0,1,2))/np.sum(abs(h_ref.astype(np.complex128))**2,axis=(0,1,2)),
                recorded_report_matches=np.allclose(errors,original['per_user_NMSE'],rtol=1e-12,atol=0),
                reference_receipt_matches=sha_file(folder/'H.npz')==original['reference_sha256'],
                complete_paths_comparison='Unknown_not_saved',geometry_scope='Frozen operands and baked actor plus runtime identity, not independently recomputed path vertices',
                reference_sha256=sha_file(folder/'H.npz'))
            refrows.append(row)
        save_json(out/'SDK_reaudit.json',dict(threshold_per_user_NMSE=1e-3,rows=refrows,
            counts=dict(Passed=sum(x['status']=='Passed' for x in refrows),Failed=sum(x['status']=='Failed' for x in refrows),Unknown=n-len(refrows)),
            new_GPU_seconds=0,complete_path_set='Unknown_not_saved',CIR_delay_primitive_full_comparison='Unknown_not_saved'))
        # Cross-check external ledger and sidecar without rewriting either.
        ledger=json.loads((job/'label_ledger.json').read_text());side=json.loads((job/'dataset/independent_validation.json').read_text())
        save_json(out/'label_consistency.json',dict(original_counts=ledger['counts'],recomputed_indices=[r['index'] for r in refrows],
            sidecar_indices=[r.get('index',r.get('frame')) for r in side['rows']],sidecar_CPU_ledger_hash_matches=side.get('CPU_ledger_sha256')==sha_file(job/'label_ledger.json'),
            immutable_HDF_labels=labels,scope='Raw HDF Unknown; independently re-audited sidecar authorizes only seven selected labels'))
        progress('all_users_full_spatial_frequency_temporal_statistics')
        space=np.zeros((tx,tx));frequency=np.zeros((nf,nf));acf=np.zeros((u,n),complex);space_per_ue=[];freq_per_ue=[];doppler=[];userstats=[];phase_seg=[]
        ff=None
        for uu in range(u):
            h=f['H'][:,uu,0,:,:]  # all time, physical TX and frequency samples
            sp=abs(correlation_matrix(h.transpose(0,2,1).reshape(-1,tx)))
            fr=abs(correlation_matrix(h.reshape(-1,nf)))
            space+=sp/u;frequency+=fr/u;space_per_ue.append(sp);freq_per_ue.append(fr)
            acf[uu]=temporal_correlation(h)
            coherence=[]
            for threshold in [.5,float(np.exp(-1))]:
                where=np.flatnonzero(abs(acf[uu,1:])<threshold)
                coherence.append(float((where[0]+1)*.01) if len(where) else None)
            # Averaged periodograms of every TX/frequency channel. Only the
            # sampled spectrum [-50,50) is observable, not unaliased Doppler.
            ff,ps=signal.welch(h.reshape(n,-1),fs=1/config['timing']['csi_dt_s'],nperseg=256,noverlap=128,
                return_onesided=False,axis=0,detrend=False,scaling='density')
            doppler.append(ps.mean(axis=1))
            amp=abs(h.astype(np.complex128));port=np.argmax(a['power_port'][:,uu,:].mean(axis=0));trace=h[:,port,nf//2]
            phase_seg.append(np.unwrap(np.angle(trace)))
            userstats.append(dict(user=uu,power_gain_dB_quantiles=np.quantile(10*np.log10(a['power_ue'][:,uu]),[0,.01,.5,.99,1]),
                nonzero_amplitude_quantiles=np.quantile(amp[amp>0],[0,.001,.01,.5,.99,.999,1]),
                time_ACF_first_cross_0_5_s=coherence[0],time_ACF_first_cross_exp_minus_1_s=coherence[1],
                strongest_average_TX_port=int(port),retained_path_frame_coverage=int(np.count_nonzero(a['retained_counts'][:,uu])),
                mean_adjacent_frequency_correlation=float(np.diag(fr,1).mean()),
                min_consecutive_time_correlation=float(np.nanmin(a['pair_correlation'][1:,uu]))))
            if uu%5==0:progress('all_users_full_spatial_frequency_temporal_statistics',user=uu,total=u)
        np.savez_compressed(out/'correlations.npz',spatial_mean=space,frequency_mean=frequency,spatial_per_ue=space_per_ue,
            frequency_per_ue=freq_per_ue,temporal_acf=acf,doppler_frequency_hz=ff,doppler_density=np.array(doppler),strongest_link_unwrapped_phase=phase_seg)
        save_json(out/'per_user_statistics.json',userstats)
    # Independent event log scan: do not equate return code with numerical gate.
    events=[json.loads(s) for s in (job/'events.jsonl').read_text().splitlines() if s.strip()]
    starts=[x['frame'] for x in events if x['status']=='solving_native_frame']
    commits=[x['committed_frames']-1 for x in events if x['status']=='frame_committed']
    save_json(out/'execution_audit.json',dict(event_counts=Counter(x['status'] for x in events),frame_starts=len(starts),commits=len(commits),
        start_sequence_exact=starts==list(range(n)),commit_sequence_exact=commits==list(range(n)),
        duplicate_starts=[i for i,k in Counter(starts).items() if k>1],duplicate_commits=[i for i,k in Counter(commits).items() if k>1],
        missing_commits=sorted(set(range(n))-set(commits)),pending_frame_exists=(job/'dataset/pending_frame.npz').exists(),
        checkpoint=json.loads((job/'dataset/checkpoint.json').read_text()),branch_report=json.loads((job/'branch_report.json').read_text()),
        resume_scope='Two saved early snapshot restores, not a forced-kill/power-loss durability experiment',
        state=json.loads((job/'state.json').read_text()),progress=json.loads((job/'progress.json').read_text())))
    # Explicitly re-hash original file after all passes. No write-capable handle
    # ever opens the input. Auxiliary files and generation evidence untouched.
    progress('verify_immutable_input_after_audit')
    sha_after=sha_file(source);aux_changed=[p for p,h in receipts.items() if sha_file(job/p)!=h]
    save_json(out/'immutability.json',dict(HDF_before_sha256=inputsha,HDF_after_sha256=sha_after,
        HDF_unchanged=inputsha==sha_after and source.stat().st_mtime_ns==stat_before.st_mtime_ns,auxiliary_changed=aux_changed,
        input_files_opened_for_write=False,new_GPU_seconds=0,audit_elapsed_wall_s=time.monotonic()-start))
    progress('CPU_scan_complete_plot_and_diagnosis_pending',frames=n)


if __name__=='__main__': main()
