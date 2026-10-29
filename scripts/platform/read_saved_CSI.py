#!/usr/bin/env python3
"""Standalone saved CSI example: no Web, GPU, task or reference dependency."""
import argparse,json
import numpy as np
from wiwo_ma.platform.storage import CSIReader


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--time',type=float,default=0);a=p.parse_args()
    reader=CSIReader(a.dataset)
    if not reader.complete:raise ValueError('Use a closed HDF5; active jobs publish latest_frame separately')
    f=reader.seek(a.time);H=f['H'];valid=np.asarray(f['result'].get('rx_valid',np.ones(H.shape[:2],bool)))
    print(json.dumps(dict(frames=reader.count,time_s=f['state']['timestamp'],shape=list(H.shape),dtype=str(H.dtype),
        axes=['UE','RX_port','TX_port','subcarrier'],RX_valid=valid.tolist(),
        mean_channel_gain_linear_per_UE=[float(np.mean(abs(H[u,valid[u]].astype(np.complex128))**2)) for u in range(len(H))],
        raw_label=f['result']['independent_validation_status'],label_source=f['result']['label_source'],
        frequency_offsets_hz=f['result']['frequency_offsets_hz'],new_GPU_seconds=0,
        scope='Raw stored label only; no independent reference is inferred by this reader'),default=lambda v:np.asarray(v).tolist(),indent=2))


if __name__=='__main__':main()
