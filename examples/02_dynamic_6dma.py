#!/usr/bin/env python3
"""Town02 16-panel/64TX/K40 mechanics, with no RT query."""
import argparse,json
import numpy as np
from wiwo_ma.platform.config import ROOT,load
from wiwo_ma.platform.physics import PhysicalWorld

def main():
    p=argparse.ArgumentParser();p.add_argument('--headless',action='store_true');p.add_argument('--port',type=int,default=8080);p.add_argument('--config',default=str(ROOT/'configs/platform/town02_strict_itu_64tx_40ue.json'));a=p.parse_args();c=load(a.config)
    if a.headless:
        w=PhysicalWorld(c);initial=w.state();count=0
        times=np.arange(round(c['timing']['duration_s']/c['timing']['csi_dt_s'])+1)*c['timing']['csi_dt_s']
        for t in times:
            w.advance_to(float(t));s=w.state();assert len(s['tx_port_positions'])==64 and len(s['user_positions'])==40 and len(s['actors'])==len(c['actors'])
            assert np.isfinite(s['tx_port_positions']).all();count+=1
        assert not np.array_equal(initial['tx_port_positions'],s['tx_port_positions'])
        print(json.dumps(dict(status='Passed_CPU_dynamic_geometry',scene=c['scene'],samples=count,physical_time_s=s['timestamp'],panels=16,TX_ports=64,UEs=40,actors=len(s['actors']),CSI='Not_computed',GPU_seconds=0)))
    else:
        from wiwo_ma.platform.web import serve
        serve(c,port=a.port)

if __name__=='__main__':main()
