#!/usr/bin/env python3
"""Bounded rail MA, actual mechanical preview; no CSI is fabricated."""
import argparse,json
import numpy as np
from wiwo_ma.platform.config import validate,ROOT
from wiwo_ma.platform.examples import reflector_example
from wiwo_ma.platform.physics import PhysicalWorld,constrained_target

def configuration():
    c=reflector_example();c['actors']=[];c['actions']=[]
    for u in c['users']:u['trajectory']=[]
    b=c['base_stations'][0];p=b['panels'][0]
    b['hardware_mode']='single_ma';p['array'].update(kind='single',rows=1,cols=1)
    p['mechanics']['mechanism']='rail_x'
    c['actions']=[dict(time_s=.1,**constrained_target(c,b['id'],p['id'],[.02,0,0],[0,0,0]))]
    return validate(c)

def main():
    p=argparse.ArgumentParser();p.add_argument('--headless',action='store_true');p.add_argument('--port',type=int,default=8080);a=p.parse_args();c=configuration()
    if a.headless:
        w=PhysicalWorld(c);before=w.state();w.advance_to(1.);after=w.state()
        delta=np.asarray(before['bs_rotations'][0]).T@(np.asarray(after['positions'][0])-before['positions'][0])
        np.testing.assert_allclose(delta,[.02,0,0],atol=1e-10,rtol=0)
        print(json.dumps(dict(status='Passed_CPU_single_MA',physical_time_s=after['timestamp'],actual_BS_local_translation_m=delta.tolist(),TX_ports=len(after['tx_port_positions']),CSI='Not_computed',GPU_seconds=0)))
    else:
        from wiwo_ma.platform.web import serve
        serve(c,port=a.port)

if __name__=='__main__':main()
