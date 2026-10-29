"""Lossless table of SDK-valid polarized path terms, not invented rays.

Pinned SDK2.2: public valid/tau axes [rxDevice,rxPattern,txDevice,
txPattern,pathSlot], public CIR adds time=1. Each device in this adapter has
one spatial location. Pattern axes are read from tensor shapes and checked
against the actual scene; VH/cross are never treated as spatial elements.
"""
import json
from collections import Counter
import numpy as np
from .config import estimated_storage,frequency_offsets


def tx_spatial_map(config):
    return np.asarray([(bi,pi,e) for pi,(bi,p) in enumerate(
        (bi,p) for bi,b in enumerate(config['base_stations']) for p in b['panels'])
        for e in range(p['array']['rows']*p['array']['cols'])],np.int32)


def pack_public_paths(config, first_source, rx_map, *, valid, cir, tau,
                      vertices, interactions, objects, primitives, sources, targets,
                      object_names, tx_patterns, rx_patterns):
    """CPU-only serialization of exact public SDK arrays; no solver/math."""
    valid=np.asarray(valid);cir=np.asarray(cir);tau=np.asarray(tau)
    if valid.ndim!=5 or valid.dtype!=np.bool_ or cir.shape!=valid.shape+(1,) or tau.shape!=valid.shape:
        raise ValueError('Unsupported SDK public path/CIR axes')
    nr,rp,nt,tp,npath=valid.shape
    if rp!=rx_patterns or tp!=tx_patterns or tp not in (1,2) or rp not in (1,2):
        raise ValueError('SDK antenna-pattern dimensions differ from declared scene')
    expected_tp=2 if config['radio']['tx_polarization'] in ('VH','cross') else 1
    expected_rp=2 if config['radio']['rx_polarization'] in ('VH','cross') else 1
    if (tp,rp)!=(expected_tp,expected_rp):raise ValueError('Configured polarization differs from SDK pattern axes')
    rx_map=np.asarray(rx_map,np.int32);sources=np.asarray(sources,np.float32);targets=np.asarray(targets,np.float32)
    if sources.shape!=(nt,3) or targets.shape!=(nr,3) or rx_map.shape!=(nr,3):raise ValueError('Spatial endpoint/map mismatch')
    if np.any(rx_map[:,2]!=0) or len(np.unique(rx_map[:,:2],axis=0))!=nr:raise ValueError('RX spatial map must contain one row per physical location')
    txmap=tx_spatial_map(config)
    if first_source<0 or first_source+nt>len(txmap):raise ValueError('TX chunk outside physical source map')
    vertices=np.asarray(vertices);interactions=np.asarray(interactions);objects=np.asarray(objects);primitives=np.asarray(primitives)
    if interactions.shape[1:]!=valid.shape or vertices.shape!=interactions.shape+(3,) or objects.shape!=interactions.shape or primitives.shape!=interactions.shape:
        raise ValueError('SDK public geometry axes changed')
    for u,element,_ in rx_map:
        if not 0<=u<len(config['users']) or not 0<=element<config['users'][u]['array']['rows']*config['users'][u]['array']['cols']:
            raise ValueError('RX element points into padding or missing user')
    # The set is SDK.valid, including valid zero coefficients. No power filter.
    ri,rpi,ti,tpi,pid=np.nonzero(valid)
    index=(ri,rpi,ti,tpi,pid)
    coeff=np.asarray(cir[index+(np.zeros(len(ri),int),)],np.complex64)
    delay=np.asarray(tau[index],np.float32)
    if not np.isfinite(coeff).all() or not np.isfinite(delay).all() or np.any(delay<0):raise ValueError('Invalid SDK-valid CIR/delay')
    global_source=ti+first_source;meta=txmap[global_source]
    result=dict(schema=np.array('sdk-public-polarized-path-table-v2'),
        user=rx_map[ri,0],rx_element=rx_map[ri,1],rx_polarization=rpi.astype(np.int32),
        rx_port=(rx_map[ri,1]*rp+rpi).astype(np.int32),tx_port=(global_source*tp+tpi).astype(np.int32),
        tx_polarization=tpi.astype(np.int32),tx_spatial_source=global_source.astype(np.int32),
        tx_bs_index=meta[:,0],tx_panel_index=meta[:,1],tx_element=meta[:,2],
        sdk_indices=np.stack(index,axis=1).astype(np.int32),coefficient=coeff,delay_s=delay,
        vertices=vertices[(slice(None),)+index].transpose(1,0,2).copy(),
        interaction_types=interactions[(slice(None),)+index].T.copy(),
        object_ids=objects[(slice(None),)+index].T.copy(),primitive_ids=primitives[(slice(None),)+index].T.copy(),
        source_positions=sources[ti],target_positions=targets[ri],object_map_json=np.array(json.dumps(object_names,sort_keys=True)),
        BS_ids_json=np.array(json.dumps([b['id'] for b in config['base_stations']])),
        first_spatial_source=np.array(first_source),
        SDK_valid=valid,SDK_CIR=cir,SDK_tau=tau,SDK_vertices=vertices,SDK_interactions=interactions,
        SDK_objects=objects,SDK_primitives=primitives,SDK_sources=sources,SDK_targets=targets,SDK_rx_spatial_map=rx_map,
        semantics=np.array('One row per actual public SDK.valid term, including zero CIR; shared geometric ray per pattern pair is not an extra discovered ray'))
    return result


def capture_public_paths(paths,scene,config,first_source,rx_map):
    if paths.synthetic_array:raise ValueError('Full polarized path export requires real physical ports')
    if paths._eff_tx_array_size!=1 or paths._eff_rx_array_size!=1:
        raise NotImplementedError('Only one spatial location per SDK device is validated')
    cir,tau=paths.cir(normalize_delays=False,num_time_steps=1,out_type='numpy')
    return pack_public_paths(config,first_source,rx_map,valid=np.asarray(paths.valid),cir=cir,tau=tau,
        vertices=np.asarray(paths.vertices),interactions=np.asarray(paths.interactions),objects=np.asarray(paths.objects),
        primitives=np.asarray(paths.primitives),sources=np.asarray(paths.sources).T,targets=np.asarray(paths.targets).T,
        object_names={str(int(o.object_id)):name for name,o in scene.objects.items()},
        tx_patterns=len(scene.tx_array.antenna_pattern.patterns),rx_patterns=len(scene.rx_array.antenna_pattern.patterns))


def audit_public_table(a,config):
    """Independent table coverage/gather check against retained raw SDK tensors."""
    ix=np.asarray(a['sdk_indices']);valid=np.asarray(a['SDK_valid']);expected=np.argwhere(valid)
    if ix.shape!=expected.shape or not np.array_equal(ix,expected):raise ValueError('Missing, duplicated or reordered SDK-valid terms')
    t=tuple(ix.T);n=len(ix);rp,tp=valid.shape[1],valid.shape[3]
    np.testing.assert_array_equal(a['coefficient'],a['SDK_CIR'][t+(np.zeros(n,int),)])
    np.testing.assert_array_equal(a['delay_s'],a['SDK_tau'][t])
    np.testing.assert_array_equal(a['vertices'],a['SDK_vertices'][(slice(None),)+t].transpose(1,0,2))
    for row,raw in [('interaction_types','SDK_interactions'),('object_ids','SDK_objects'),('primitive_ids','SDK_primitives')]:
        np.testing.assert_array_equal(a[row],a[raw][(slice(None),)+t].T)
    rm=a['SDK_rx_spatial_map'];tm=tx_spatial_map(config);first=int(a['first_spatial_source'])
    np.testing.assert_array_equal(a['user'],rm[ix[:,0],0])
    np.testing.assert_array_equal(a['rx_element'],rm[ix[:,0],1])
    np.testing.assert_array_equal(a['rx_polarization'],ix[:,1])
    np.testing.assert_array_equal(a['tx_polarization'],ix[:,3])
    np.testing.assert_array_equal(a['tx_spatial_source'],first+ix[:,2])
    np.testing.assert_array_equal(a['rx_port'],rm[ix[:,0],1]*rp+ix[:,1])
    np.testing.assert_array_equal(a['tx_port'],(first+ix[:,2])*tp+ix[:,3])
    for key,column in [('tx_bs_index',0),('tx_panel_index',1),('tx_element',2)]:
        np.testing.assert_array_equal(a[key],tm[first+ix[:,2],column])
    np.testing.assert_array_equal(a['source_positions'],a['SDK_sources'][ix[:,2]])
    np.testing.assert_array_equal(a['target_positions'],a['SDK_targets'][ix[:,0]])
    return dict(status='Passed_CPU_serialization_only',polarized_terms=n,
        spatial_path_slots=len(np.unique(ix[:,[0,2,4]],axis=0)),valid_zero_CIR_terms=int(np.count_nonzero(a['coefficient']==0)))


def reconstruct_cfr(tables,config):
    shape=estimated_storage(config);H=np.zeros((shape['users'],shape['rx_ports_max'],shape['tx_ports'],shape['subcarriers']),np.complex128)
    f=frequency_offsets(config).astype(np.float64)
    for a in tables:
        rx=a['rx_port'] if 'rx_port' in a else a['rx_element']
        values=a['coefficient'].astype(np.complex128)[:,None]*np.exp(-2j*np.pi*a['delay_s'].astype(np.float64)[:,None]*f)
        np.add.at(H,(a['user'],rx,a['tx_port']),values)
    return H


def compare_public_tables(native,reference):
    """Exact multisets preserve duplicate topology, without arbitrary pairing."""
    def collect(tables):
        answer={};nonfinite=0
        for a in tables:
            names=json.loads(str(a['object_map_json']))
            for i in range(len(a['user'])):
                word=[]
                for kind,obj,primitive in zip(a['interaction_types'][i],a['object_ids'][i],a['primitive_ids'][i]):
                    if not kind:continue
                    name=names[str(int(obj))]
                    if name.startswith('platform_actor_'):name='actor_'+name[len('platform_actor_'):]
                    if name.startswith('reference_'):name='actor_'+name[len('reference_'):]
                    word.append((int(kind),name,int(primitive)))
                key=(int(a['user'][i]),int(a['rx_port'][i]),int(a['tx_port'][i]),tuple(word))
                fields=[a[k][i] for k in ('vertices','delay_s','coefficient','source_positions','target_positions')]
                nonfinite+=int(any(not np.isfinite(v).all() for v in fields))
                answer.setdefault(key,[]).append(tuple(np.asarray(v).tobytes() for v in fields))
        return answer,nonfinite
    x,xbad=collect(native);y,ybad=collect(reference);missing=set(y)-set(x);extra=set(x)-set(y)
    count_bad=sum(len(x[k])!=len(y[k]) for k in x.keys()&y.keys())
    exact=not missing and not extra and all(Counter(x[k])==Counter(y[k]) for k in x.keys()&y.keys())
    def field_same(index):return not missing and not extra and all(Counter(v[index] for v in x[k])==Counter(v[index] for v in y[k]) for k in x.keys()&y.keys())
    return dict(status='Failed' if missing or extra or count_bad or xbad or ybad else 'Passed' if exact else 'Partial',
        missing_words=len(missing),extra_words=len(extra),multiplicity_mismatches=count_bad,
        native_words=len(x),reference_words=len(y),native_terms=sum(map(len,x.values())),reference_terms=sum(map(len,y.values())),
        exact_duplicate_multisets=exact,duplicate_words_Unknown=0,nonfinite_chunks=xbad+ybad,
        vertices_bitwise=field_same(0),delays_bitwise=field_same(1),CIR_bitwise=field_same(2),endpoints_bitwise=field_same(3) and field_same(4))
