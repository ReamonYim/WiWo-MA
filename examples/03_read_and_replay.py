#!/usr/bin/env python3
"""Read/plot/replay a real closed HDF5 supplied by the user; zero new RT."""
import argparse,json
from pathlib import Path
import numpy as np
from wiwo_ma.platform.storage import CSIReader

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--time',type=float,default=0);p.add_argument('--plot',type=Path);p.add_argument('--web',action='store_true');p.add_argument('--port',type=int,default=8080);a=p.parse_args()
    r=CSIReader(a.dataset)
    if not r.complete:raise ValueError('Use a closed HDF5; running workers publish atomic latest frames')
    row=r.seek(a.time);h=row['H'];assert np.isfinite(h).all()
    print(json.dumps(dict(frames=r.count,physical_time_s=row['state']['timestamp'],shape=list(h.shape),dtype=str(h.dtype),axes=['UE','RX_port','TX_port','subcarrier'],raw_validation=row['result']['independent_validation_status'],new_GPU_seconds=0)))
    if a.plot:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from wiwo_ma.platform.replay_analysis import analyze_link
        q=analyze_link(a.dataset,user=0,rx=0,tx=0)
        fig,axes=plt.subplots(2,1,figsize=(10,6));axes[0].plot(q['timestamps'],q['channel_gain_dB']);axes[0].set(ylabel='Mean |H|² (dB)',xlabel='Physical time (s)')
        image=axes[1].imshow(q['amplitude_dB'].T,aspect='auto',origin='lower',extent=[q['timestamps'][0],q['timestamps'][-1],0,len(q['frequency_offsets_hz'])]);axes[1].set(xlabel='Physical time (s)',ylabel='Subcarrier index',title='UE0/RX0/TX0: 20 log10 |H| (dB); stored samples')
        fig.colorbar(image,ax=axes[1],label='20 log10 |H| (dB)')
        fig.tight_layout();a.plot.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.plot,dpi=160);plt.close(fig)
    if a.web:
        from wiwo_ma.platform.web import serve
        serve(r.config,port=a.port,replay=str(a.dataset.resolve()))

if __name__=='__main__':main()
