"""A claimed Passed JSON must never override a failing complex-array check."""
import json
import hashlib
import h5py,numpy as np,pytest
from wiwo_ma.platform import controller
from wiwo_ma.platform.config import SCHEMA,fingerprint
from wiwo_ma.platform.physics import PhysicalWorld
from wiwo_ma.platform.storage import CSIWriter


@pytest.mark.parametrize('reference_scale',[1.,2.])
def test_replay_independently_checks_raw_weak_user(tmp_path,monkeypatch,reference_scale):
    c=dict(schema=SCHEMA,base_stations=[dict(id='bs',position_m=[0,0,3],panels=[dict(id='p',position_m=[0,0,0])])],users=[dict(id='u',position_m=[3,0,1.5])])
    w=PhysicalWorld(c);h=np.full((1,1,4,64),1e-12+1e-12j,np.complex64);writer=CSIWriter(tmp_path,c)
    writer.stage(dict(H=h,physical_timestamp=0.,label_source='test_fixture',independent_validation_status='Unknown',paths=[]),w.state(),w.snapshot());writer.commit_pending();writer.complete();writer.close()
    reference=tmp_path/'reference.npz';np.savez(reference,H=h*reference_scale)
    with h5py.File(tmp_path/'csi.h5','r') as f:rowhash=f['frame_sha256'].asstr()[0]
    row=dict(status='Passed',per_user_NMSE=[0.],reference_array_path=str(reference),reference_array_file_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        physical_timestamp=0.,frame_sha256=rowhash,configuration_sha256=fingerprint(c))
    (tmp_path/'independent_validation.json').write_text(json.dumps(dict(schema='native-frame-validation-sidecar-v1',rows=[row])))
    monkeypatch.setattr(controller,'inside',lambda p:__import__('pathlib').Path(p));monkeypatch.setattr(controller,'ROOT',tmp_path)
    replay=controller.Replay(tmp_path/'csi.h5');result=replay.frame(0)[1]
    if reference_scale==1:assert result['independent_validation_status'].startswith('Passed')
    else:assert result['independent_validation_status']=='Unknown' and result['independent_validation_issue'].startswith('Failed')
