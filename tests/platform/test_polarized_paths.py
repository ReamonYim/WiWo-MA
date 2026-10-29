"""Explicit CPU tensors exercise SDK axes; not native propagation evidence."""
from copy import deepcopy
import numpy as np
import pytest
from wiwo_ma.platform.examples import reflector_example
from wiwo_ma.platform.config import validate,frequency_offsets
from wiwo_ma.platform.polarized_paths import pack_public_paths,audit_public_table,reconstruct_cfr,compare_public_tables


def fixture():
    c=reflector_example();c.update(actions=[],actors=[])
    other=deepcopy(c['base_stations'][0]);other['id']='bs1';other['position_m'][0]+=20;c['base_stations'].append(other)
    user=deepcopy(c['users'][0]);c['users']=[]
    for i in range(3):
        u=deepcopy(user);u.update(id='ue'+str(i),trajectory=[]);u['position_m'][1]+=i
        u['array']=dict(kind='ULA',rows=1,cols=2 if i==1 else 1,spacing_m=.03);c['users'].append(u)
    c['radio'].update(tx_polarization='VH',rx_polarization='cross',num_subcarriers=7);c=validate(c)
    shape=(4,2,2,2,3);grid=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)
    valid=np.ones(shape,bool);valid[0,1,1,0,1]=False
    cir=(grid+1+1j*(grid+2)).astype(np.complex64)[...,None];cir[0,0,0,0,0,0]=0
    tau=(grid+1)*np.float32(1e-9)
    geometry=(2,*shape)
    inputs=dict(valid=valid,cir=cir,tau=tau,
        vertices=np.arange(np.prod(geometry)*3,dtype=np.float32).reshape(geometry+(3,)),
        interactions=np.ones(geometry,np.uint32),objects=np.ones(geometry,np.uint32),primitives=np.zeros(geometry,np.uint32),
        sources=np.arange(6,dtype=np.float32).reshape(2,3),targets=np.arange(12,dtype=np.float32).reshape(4,3),
        object_names={'1':'wall'},tx_patterns=2,rx_patterns=2)
    rm=np.array([[0,0,0],[1,0,0],[1,1,0],[2,0,0]],np.int32)
    return c,rm,inputs


def test_public_tensor_polarizations_multibs_rx_padding_zero_CIR_and_all_valid_terms(tmp_path):
    c,rm,kw=fixture();a=pack_public_paths(c,3,rm,**kw)
    assert len(a['user'])==int(kw['valid'].sum())==95
    assert set(a['tx_bs_index'])=={0,1} and set(a['tx_port'])=={6,7,8,9}
    assert audit_public_table(a,c)['valid_zero_CIR_terms']==1
    out=reconstruct_cfr([a],c);expected=np.zeros_like(out)
    # Independently enumerate every raw public tensor coordinate, not table rows.
    f=frequency_offsets(c).astype(float)
    for rx in range(4):
        for rp in range(2):
            for tx in range(2):
                for tp in range(2):
                    for path in range(3):
                        if kw['valid'][rx,rp,tx,tp,path]:
                            expected[rm[rx,0],2*rm[rx,1]+rp,2*(3+tx)+tp]+=complex(kw['cir'][rx,rp,tx,tp,path,0])*np.exp(-2j*np.pi*float(kw['tau'][rx,rp,tx,tp,path])*f)
    np.testing.assert_allclose(out,expected,rtol=1e-14,atol=1e-12)
    assert np.count_nonzero(out[0,2:])==np.count_nonzero(out[2,2:])==0
    file=tmp_path/'paths.npz';np.savez_compressed(file,**a)
    with np.load(file,allow_pickle=False) as f:
        assert audit_public_table({k:f[k] for k in f.files},c)['polarized_terms']==95


@pytest.mark.parametrize('fault',['missing','duplicate','wrong_pol','wrong_pol_identity','wrong_BS','swapped_phase','geometry'])
def test_independent_saved_tensor_audit_rejects_corrupt_port_or_path_rows(fault):
    c,rm,kw=fixture();a=pack_public_paths(c,3,rm,**kw)
    if fault=='missing':a['sdk_indices']=a['sdk_indices'][:-1]
    elif fault=='duplicate':a['sdk_indices'][1]=a['sdk_indices'][0]
    elif fault=='wrong_pol':a['rx_port'][0]+=1
    elif fault=='wrong_pol_identity':a['tx_polarization'][0]+=1
    elif fault=='wrong_BS':a['tx_bs_index'][0]+=1
    elif fault=='swapped_phase':a['coefficient'][2]*=1j
    elif fault=='geometry':a['vertices'][0,0,0]+=1
    with pytest.raises((ValueError,AssertionError)):audit_public_table(a,c)


def test_topology_multiplicity_preserved_not_silently_deduplicated():
    c,rm,kw=fixture();a=pack_public_paths(c,3,rm,**kw);b={k:v.copy() for k,v in a.items()}
    assert compare_public_tables([a],[b])['status']=='Passed'
    # Identical path words for different SDK slots are retained as a multiset.
    b['coefficient'][2]+=np.complex64(1)
    assert compare_public_tables([a],[b])['status']=='Partial'
    b={k:v.copy() for k,v in a.items()};b['rx_port'][0]+=1
    assert compare_public_tables([a],[b])['status']=='Failed'


def test_changed_SDK_tensor_pattern_or_spatial_semantics_fail_closed():
    c,rm,kw=fixture();kw['tx_patterns']=1
    with pytest.raises(ValueError,match='pattern'):pack_public_paths(c,3,rm,**kw)
