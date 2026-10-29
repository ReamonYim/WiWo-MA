"""Headless real-HDF link analysis. No synthetic CSI, training or GPU use."""
from pathlib import Path
import json
import h5py
import numpy as np
from .dataset_audit import temporal_correlation


def analyze_link(path, user=0, rx=0, tx=0, max_frames=4096):
    if not 2<=max_frames<=4096:raise ValueError('Analysis requires 2..4096 samples')
    with h5py.File(path,'r') as f:
        if f.attrs.get('schema')!='wiwo-native-csi-v1':raise ValueError('Native HDF5 required')
        if not f.attrs.get('complete',False):raise ValueError('Analyze a closed dataset; use committed-frame attachment while writing')
        n=int(f.attrs['committed_frames']);shape=f['H'].shape
        if not (0<=user<shape[1] and 0<=rx<shape[2] and 0<=tx<shape[3]):raise ValueError('Physical link index out of range')
        c=json.loads(f.attrs['config_json']);a=c['users'][user]['array'];pol=2 if c['radio']['rx_polarization'] in ('VH','cross') else 1
        if rx>=a['rows']*a['cols']*pol:raise ValueError('RX index is padding, not a physical antenna')
        # Sampling is explicit. Never interpolate CSI or present it as a higher
        # time resolution. For the 3001-frame baseline every frame is read.
        stride=max(1,int(np.ceil(n/max_frames)));indices=np.arange(0,n,stride)
        t=f['timestamps'][indices];h=f['H'][indices,user,rx,tx,:].astype(np.complex128)
        freq=f['frequency_offsets_hz'][:]
    acf=temporal_correlation(h);valid=np.flatnonzero(abs(acf)<.5);acf_defined=np.isfinite(acf)
    dt=float(t[1]-t[0]) if len(t)>1 else None
    return dict(source=str(path),scope='Actual saved HDF5; no recomputation; no independent numeric-label promotion',
        original_frames=n,sampled_frames=len(t),stride=stride,user=user,rx=rx,tx=tx,timestamps=t,frequency_offsets_hz=freq,
        amplitude_dB=20*np.log10(np.maximum(abs(h),1e-30)),phase_rad=np.angle(h),
        channel_gain_dB=10*np.log10(np.maximum(np.mean(abs(h)**2,axis=-1),1e-30)),
        lag_s=np.arange(len(t))*dt if dt is not None else np.zeros(len(t)),ACF_magnitude=[float(abs(x)) if ok else None for x,ok in zip(acf,acf_defined)],
        ACF_status='Passed_defined_descriptive_statistic' if acf_defined.all() else 'Unknown_zero_overlap_energy_for_null_lags',
        zero_complex_samples=int(np.count_nonzero(h==0)),phase_at_zero='Undefined; numpy angle zero is a display placeholder',amplitude_display_floor_dB=-600.,
        first_0p5_crossing_s=None if not len(valid) or dt is None else float(valid[0]*dt),
        ACF_method='Uncentred complex normalized overlap across frequencies, each lag has its own overlap energy; nonstationary descriptive statistic',
        sampling_note='Single selected physical link; decimation is explicit; no claim of unaliased path Doppler or stationarity')
