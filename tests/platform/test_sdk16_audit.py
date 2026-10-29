"""Independent reference export/closure math; no SDK or CUDA needed."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest


def script(name):
    p=Path(__file__).resolve().parents[2]/'scripts'/name
    s=importlib.util.spec_from_file_location(name.replace('.','_'),p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_full_physical_port_axes_independent_enumeration():
    m=script('run_platform_sdk16_audit.py');raw=np.arange(40*4*64).reshape(40,1,4,1,1,64).astype(np.complex64)
    mapping=np.column_stack((np.arange(40)[::-1],np.zeros(40,int),np.zeros(40,int)))
    h=m.assemble_chunk(raw,mapping)
    assert h.shape==(40,1,4,64)
    np.testing.assert_array_equal(h[0,0,3],raw[39,0,3,0,0]);np.testing.assert_array_equal(h[39,0,0],raw[0,0,0,0,0])
    with pytest.raises(ValueError):m.assemble_chunk(raw[:2],mapping[:2])


def test_chunk_receipt_corruption_cannot_be_silently_reused(tmp_path):
    m=script('run_platform_sdk16_audit.py');p=tmp_path/'chunk.npz'
    assert not m.committed_chunk(p)
    m.atomic_npz(p,H=np.ones((2,3),complex));assert not m.committed_chunk(p)
    p.with_suffix('.receipt.json').write_text(json.dumps(dict(sha256=m.sha_file(p))))
    assert m.committed_chunk(p)
    p.write_bytes(b'corrupt')
    with pytest.raises(ValueError):m.committed_chunk(p)


def test_frozen_input_and_proposal_binding(tmp_path):
    m=script('run_platform_sdk16_audit.py');p=tmp_path/'state.json';p.write_text('{}')
    proposal=tmp_path/'proposal.json';proposal.write_text(json.dumps(dict(frozen_inputs={'state.json':m.sha_file(p)})))
    h=m.sha_file(proposal);assert m.verify_proposal(tmp_path,h)['frozen_inputs']
    p.write_text('{"bad":true}')
    with pytest.raises(ValueError):m.verify_proposal(tmp_path,h)


def test_cir_full_path_synthesis_and_coherent_change_decomposition():
    m=script('audit_platform_sdk16_results.py');freq=np.array([-30000.,0.,30000.]);word=(28,3,0,(('building',1,1),));los=(28,3,0,())
    a=dict(coefficient=np.array([1+0j,1j]),delay_s=np.array([0.,1e-7]),user=np.array([28,28]),tx_port=np.array([3,3]),words=[word,los],plane_words=[word,los])
    b=dict(coefficient=np.array([2+0j,1j]),delay_s=np.array([0.,1e-7]),user=np.array([28,28]),tx_port=np.array([3,3]),words=[word,los],plane_words=[word,los])
    np.testing.assert_allclose(m.path_CFR(a,freq)[:,1],[1,1j])
    d=m.decomposition(a,b,freq,28,3)
    assert d['words']['common_terms']==2 and d['words']['born_terms']==0
    assert d['words']['complex_decomposition_max_residual']<1e-14
    np.testing.assert_allclose(d['words']['nonadditive_energy_ratios_to_delta'],[1,0,0])


def test_offline_ablation_uses_coherent_sum_and_carrier_sign():
    m=script('diagnose_sdk16_path_physics.py');freq=(np.arange(64)-31.5)*30000
    h=m.band_synthesis(np.array([1,-1]),np.array([1e-7,1e-7]),np.array([3,3]),freq)
    np.testing.assert_allclose(h,0,atol=1e-15)
    phase=m.sdk_phase(np.array([0.,1/28e9],np.float32))
    np.testing.assert_allclose(np.exp(1j*phase),1,atol=1e-6)


def test_exact_plane_reference_one_reflection_and_finite_triangle():
    m=script('diagnose_sdk16_short_segments.py')
    tri=np.zeros((1,3,3,3));tri[0,0]=[[-10,-10,0],[10,-10,0],[0,10,0]]
    v,length,inside=m.exact_plane_images(np.array([[0.,0,1]]),np.array([[2.,0,1]]),tri,np.array([1]))
    np.testing.assert_allclose(v[0,0],[1,0,0]);np.testing.assert_allclose(length,[np.sqrt(8)]);assert inside[0]
    tri[0,0]+=np.array([100.,0,0])
    _,_,inside=m.exact_plane_images(np.array([[0.,0,1]]),np.array([[2.,0,1]]),tri,np.array([1]))
    assert not inside[0]
