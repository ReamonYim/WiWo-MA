import json
import h5py,numpy as np,pytest
from wiwo_ma.platform.physics import PhysicalWorld
from wiwo_ma.platform.storage import CSIWriter,CSIReader,encode,decode
from test_physics import config,target


def frame(w):
    s=w.state();h=np.full((1,1,4,64),s['timestamp']+1j,np.complex64)
    return dict(H=h,physical_timestamp=s['timestamp'],label_source='test_fixture',independent_validation_status='Unknown',paths=[]),s,w.snapshot()


def test_streaming_complex_axes_resume_and_state_roundtrip(tmp_path):
    c=config();world=PhysicalWorld(c);world.command(target());writer=CSIWriter(tmp_path,c)
    for t in [0,.01]:
        world.advance_to(t);writer.stage(*frame(world));writer.commit_pending()
    snapshot=writer.latest_snapshot();writer.close();w=PhysicalWorld(c);w.restore(snapshot);assert w.time==.01
    # Continuation comes from the actual pending acceleration trajectory.
    writer=CSIWriter(tmp_path,c,resume=True);w.advance_to(.02);writer.stage(*frame(w));writer.commit_pending();writer.complete();writer.close()
    r=CSIReader(tmp_path/'csi.h5');assert r.count==3 and r.complete
    assert r.frame(2)['H'].dtype==np.complex64 and r.frame(2)['H'].shape==(1,1,4,64)
    assert r.seek(.015)['state']['timestamp']==.01
    world.advance_to(.02);np.testing.assert_array_equal(world.state()['tx_port_positions'],w.state()['tx_port_positions'])


def test_write_ahead_recovery_no_second_RT_query(tmp_path):
    c=config();w=PhysicalWorld(c);writer=CSIWriter(tmp_path,c);writer.stage(*frame(w));writer.close()
    assert (tmp_path/'pending_frame.npz').exists()
    writer=CSIWriter(tmp_path,c,resume=True);assert writer.commit_pending() and writer.count==1
    assert not (tmp_path/'pending_frame.npz').exists();writer.close()


def test_uncommitted_HDF_tail_truncated_and_corruption_detected(tmp_path):
    c=config();w=PhysicalWorld(c);writer=CSIWriter(tmp_path,c);writer.stage(*frame(w));writer.commit_pending();writer.close()
    with h5py.File(tmp_path/'csi.h5','r+') as f:
        for key in ['H','timestamps','frames_json','frame_sha256','label_source','independent_validation']:f[key].resize(2,axis=0)
    writer=CSIWriter(tmp_path,c,resume=True);assert writer.count==len(writer.file['H'])==1;writer.close()
    with h5py.File(tmp_path/'csi.h5','r+') as f:f['H'][0,0,0,0,0]=999
    with pytest.raises(ValueError,match='hash'):CSIReader(tmp_path/'csi.h5').frame(0)


def test_timestamp_gaps_and_wrong_configuration_are_rejected(tmp_path):
    c=config();w=PhysicalWorld(c);writer=CSIWriter(tmp_path,c);writer.stage(*frame(w));writer.commit_pending();w.advance_to(.02);writer.stage(*frame(w))
    with pytest.raises(ValueError,match='gap'):writer.commit_pending()
    writer.close();c['radio']['carrier_hz']=28e9
    with pytest.raises(ValueError,match='configuration'):CSIWriter(tmp_path,c,resume=True)


def test_velocity_snapshot_is_json_not_pickle_and_unknown_tag_rejected():
    w=PhysicalWorld(config());w.command(dict(bs_id='bs0',panel_id='0',type='velocity',velocity_m_s=[.05,0,0],duration_s=.1));w.advance_to(.02)
    encoded=json.dumps(encode(w.snapshot()),allow_nan=False);copy=PhysicalWorld(config());copy.restore(decode(json.loads(encoded)))
    w.advance_to(.3);copy.advance_to(.3);np.testing.assert_array_equal(w.state()['tx_port_positions'],copy.state()['tx_port_positions'])
    with pytest.raises(ValueError):decode({'@':'python','value':'unsafe'})


def test_second_writer_cannot_duplicate_job(tmp_path):
    c=config();writer=CSIWriter(tmp_path,c)
    with pytest.raises(BlockingIOError):CSIWriter(tmp_path,c,resume=True)
    writer.close()
