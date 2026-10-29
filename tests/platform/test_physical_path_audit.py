import json
import numpy as np
from wiwo_ma.platform.path_capture import canonical_paths
from wiwo_ma.platform.physical_gate import compare_paths


def arrays(name='platform_actor_van',n=1):
    return dict(user=np.zeros(n,int),rx_element=np.zeros(n,int),tx_port=np.zeros(n,int),coefficient=np.ones(n,np.complex64),delay_s=np.ones(n,np.float32),
        vertices=np.ones((n,1,3),np.float32),interaction_types=np.ones((n,1),int),object_ids=np.ones((n,1),int)*9,primitive_ids=np.ones((n,1),int)*3,
        object_map_json=np.array(json.dumps({'9':name})))


def test_stable_words_resolve_process_local_IDs_and_keep_duplicate_instances(tmp_path):
    a=arrays(n=2);b=arrays('reference_van',n=2);b['object_ids'][:]=12;b['object_map_json']=np.array(json.dumps({'12':'reference_van'}))
    assert list(canonical_paths(a))==list(canonical_paths(b))
    assert list(canonical_paths(a).values())==[[0,1]]
    np.savez(tmp_path/'a.npz',**a);np.savez(tmp_path/'b.npz',**b)
    q=compare_paths([tmp_path/'a.npz'],[tmp_path/'b.npz']);assert q['status']=='Partial' and q['duplicate_words_Unknown']==1


def test_path_mismatch_is_independent_of_CSI_equality(tmp_path):
    a,b=arrays(),arrays('reference_van');np.savez(tmp_path/'a.npz',**a);np.savez(tmp_path/'b.npz',**b)
    assert compare_paths([tmp_path/'a.npz'],[tmp_path/'b.npz'])['status']=='Passed'
    b['vertices'][0,0,0]=np.nextafter(np.float32(1),np.float32(2));np.savez(tmp_path/'b.npz',**b)
    q=compare_paths([tmp_path/'a.npz'],[tmp_path/'b.npz']);assert q['status']=='Partial' and not q['vertices_bitwise']
    b['primitive_ids'][:]=4;np.savez(tmp_path/'b.npz',**b)
    assert compare_paths([tmp_path/'a.npz'],[tmp_path/'b.npz'])['status']=='Failed'


def test_nonfinite_path_never_passes_or_becomes_only_Partial(tmp_path):
    a=arrays();a['coefficient'][0]=np.nan
    np.savez(tmp_path/'a.npz',**a);np.savez(tmp_path/'b.npz',**a)
    assert compare_paths([tmp_path/'a.npz'],[tmp_path/'b.npz'])['status']=='Failed'
