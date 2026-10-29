"""Lossless diagnostic export of compact original SDK V/V path buffers.

Serialization only. No path discovery, geometric correction or Field changes.
Native and fresh-scene reference solvers remain independent.
"""
import json
import numpy as np


def complete_V_paths(paths, scene, config, first_source, rx_map):
    import mitsuba as mi
    import drjit as dr
    if config['radio']['tx_polarization'] not in ('V','H') or config['radio']['rx_polarization'] not in ('V','H'):
        from .polarized_paths import capture_public_paths
        return capture_public_paths(paths,scene,config,first_source,rx_map)
    b=paths._paths_buffer;n=b.buffer_size;depth=config['rt']['max_depth'];active=dr.full(mi.Bool,True,n)
    src=np.asarray(b.source_indices,np.int32);target=np.asarray(b.target_indices,np.int32);pid=np.asarray(paths._path_ind,np.int32)
    if not np.asarray(b.valid).all():raise ValueError('Expected compact valid SDK path buffer')
    cir,tau=paths.cir(normalize_delays=False,num_time_steps=1,out_type='numpy')
    v=np.zeros((n,depth,3),np.float32);typ=np.zeros((n,depth),np.uint32);obj=typ.copy();prim=typ.copy()
    for j in range(depth):
        kind=b.get_interaction_type(mi.UInt(j+1),active);mask=kind!=0
        typ[:,j]=np.asarray(kind);v[:,j]=np.asarray(b.get_vertex(mi.UInt(j+1),mask)).T
        obj[:,j]=np.asarray(b.get_shape(mi.UInt(j+1),mask));prim[:,j]=np.asarray(b.get_primitive(mi.UInt(j+1),mask))
    objects={str(int(o.object_id)):name for name,o in scene.objects.items()}
    return dict(user=np.asarray(rx_map)[target,0],rx_element=np.asarray(rx_map)[target,1],tx_port=src+first_source,
        coefficient=np.asarray(cir[target,0,src,0,pid,0],np.complex64),delay_s=np.asarray(tau[target,0,src,0,pid],np.float32),
        vertices=v,interaction_types=typ,object_ids=obj,primitive_ids=prim,
        source_positions=np.asarray(paths.sources).T[src],target_positions=np.asarray(paths.targets).T[target],
        object_map_json=np.array(json.dumps(objects,sort_keys=True)))


def canonical_paths(arrays):
    """Stable path words for independent post-run matching, never solver input."""
    names=json.loads(str(arrays['object_map_json']));rows={}
    for i in range(len(arrays['user'])):
        word=[]
        for k,oid,pid in zip(arrays['interaction_types'][i],arrays['object_ids'][i],arrays['primitive_ids'][i]):
            if not k:continue
            name=names[str(int(oid))]
            if name.startswith('platform_actor_'):name='actor_'+name[len('platform_actor_'):]
            if name.startswith('reference_'):name='actor_'+name[len('reference_'):]
            word.append((int(k),name,int(pid)))
        key=(int(arrays['user'][i]),int(arrays['rx_port'][i] if 'rx_port' in arrays else arrays['rx_element'][i]),int(arrays['tx_port'][i]),tuple(word))
        rows.setdefault(key,[]).append(i)
    return rows
