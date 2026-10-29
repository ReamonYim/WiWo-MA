"""Analytic frequency response and exact endpoint checks for the CPU auditor."""
import importlib.util
import json
from pathlib import Path
import numpy as np


def module():
    p=Path(__file__).resolve().parents[2]/'scripts/audit_platform_physical_native_gate.py'
    spec=importlib.util.spec_from_file_location('independent_physical_audit',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_independent_CIR_synthesis_sign_frequency_order_and_port_axis(tmp_path):
    m=module();c=dict(radio=dict(num_subcarriers=3,subcarrier_spacing_hz=1.),users=[dict(array=dict(rows=1,cols=1))],actors=[])
    s=dict(tx_port_positions=[[0,0,1],[0,1,1]],rx_port_positions=[[3,0,1]],rx_port_map=[[0,0,0]])
    a=dict(user=np.array([0,0]),rx_element=np.array([0,0]),tx_port=np.array([0,1]),
        coefficient=np.array([1,2],np.complex64),delay_s=np.array([.25,0],np.float32),
        vertices=np.zeros((2,1,3),np.float32),interaction_types=np.zeros((2,1),int),object_ids=np.zeros((2,1),int),primitive_ids=np.zeros((2,1),int),
        source_positions=np.asarray(s['tx_port_positions'],np.float32),target_positions=np.tile(np.asarray(s['rx_port_positions'],np.float32),(2,1)),object_map_json=np.array('{}'))
    p=tmp_path/'chunk_00.npz';np.savez(p,**a);p.with_suffix('.receipt.json').write_text(json.dumps(dict(sha256=m.sha_file(p))))
    x=m.read_paths([p],c,s)
    np.testing.assert_allclose(x['H'][0,0],[[1j,1,-1j],[2,2,2]],atol=1e-15)
    assert x['endpoint_errors']==0 and x['path_count']==2 and x['LOS'].all()
    s['tx_port_positions'][1][1]=2
    assert m.read_paths([p],c,s)['endpoint_errors']==1


def test_independent_geometry_and_CIR_layers_do_not_collapse_to_CSI_label():
    m=module();key=(0,0,0,())
    def path(value):return dict(rows={key:[(np.zeros((1,3)),np.float32(.1),np.complex64(value))]},nonfinite=[],path_count=1)
    a,b=path(1),path(2);r=m.path_comparison(a,b)
    assert r['status']=='Partial' and r['nonidentical_CIR_paths']==1 and r['nonidentical_vertex_paths']==0
    a['nonfinite']=['explicit_nonfinite_chunk'];assert m.path_comparison(a,b)['status']=='Failed'
