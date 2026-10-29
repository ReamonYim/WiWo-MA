"""Identity of SDK operands, independent of CSI/path answers or CUDA."""
import hashlib
import numpy as np


def geometry_digest(state):
    digest=hashlib.sha256()
    digest.update(str(state['configuration_sha256']).encode())
    digest.update(np.asarray([state['timestamp']],dtype='<f8').tobytes())
    def array(name,value,dtype):
        a=np.ascontiguousarray(value,dtype=dtype)
        digest.update(name.encode()+str(a.shape).encode()+a.tobytes())
    for side in ('tx','rx'):
        for suffix,dtype in [('positions','<f4'),('orientations_rad','<f4'),('map','<i4')]:
            key=side+'_port_'+suffix
            array(key,state.get(key,[]),dtype)
    for actor in state.get('actors',[]):
        digest.update(str(actor['id']).encode()+str(actor.get('material','')).encode())
        for key,dtype in [('vertices','<f4'),('faces','<i4'),('velocity_m_s','<f4')]:
            array(key,actor.get(key,[]),dtype)
    return digest.hexdigest()
