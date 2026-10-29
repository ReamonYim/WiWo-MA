"""Restricted historical SDK artifacts, deliberately outside public CPU CI."""
import json,shutil,time,hashlib
import numpy as np
import pytest
from wiwo_ma.platform.config import ROOT
from wiwo_ma.platform.examples import reflector_example,munich_example
from wiwo_ma.platform.controller import PlatformController,Replay
from wiwo_ma.platform.longstudy import nmse,verify_inputs,supervise
pytestmark=pytest.mark.restricted

def test_stored_native_data_are_not_editable_and_seek_uses_saved_timestamps(tmp_path):
    # Existing writer tests cover synthetic storage fixtures. This integration
    # uses a real retained historical HDF, with its Partial status unchanged.
    from wiwo_ma.platform.config import ROOT
    path=ROOT/'artifacts/v4_release/outdoor40_35ghz_30s/data.h5'
    if not path.exists():pytest.skip('Optional retained historical replay dataset unavailable')
    x=PlatformController(reflector_example())
    try:
        x.command('load_dataset',dict(path=str(path)));s,h,stats=x.snapshot()
        assert h['H'].shape[:3]==(40,1,64) and 'Partial' in h['label_source']
        x.command('seek',dict(time_s=30.));s,h,_=x.snapshot();assert s['timestamp']==h['physical_timestamp']==30.
        with pytest.raises(ValueError,match='read-only'):x.action(dict(bs_id='legacyBS',panel_id='0',type='stop'))
    finally:x.close()


def test_attach_actual_native_publication_is_readonly_and_never_advances_from_UI_clock(tmp_path,monkeypatch):
    import wiwo_ma.platform.controller as module
    source=ROOT/'artifacts/platform/validation/native_gate_attempt01/full64_0/dataset/latest_frame.npz'
    if not source.exists():pytest.skip('Optional actual native output unavailable')
    config=ROOT/'artifacts/platform/validation/native_gate_attempt01/full64_config.json'
    folder=tmp_path/'job';(folder/'dataset').mkdir(parents=True);shutil.copyfile(config,folder/'config.json');shutil.copyfile(source,folder/'dataset/latest_frame.npz')
    monkeypatch.setattr(module,'inside',lambda _:folder);monkeypatch.setattr(module,'ROOT',tmp_path)
    x=PlatformController(munich_example())
    try:
        x.command('attach_job',dict(path='job'));s,h,stats=x.snapshot()
        assert s['timestamp']==h['physical_timestamp']==1.2 and h['H'].shape==(40,1,64,64)
        assert stats['mode']=='native_background_frames' and not stats['GPU_enabled']
        x.command('start');time.sleep(.07);assert x.state()['timestamp']==1.2 and not x.playing
        with pytest.raises(ValueError,match='frozen'):x.action(dict(bs_id='bs0',panel_id='panel00',type='stop'))
        x.command('stop');assert x.state()['timestamp']==1.2 and not x.playing
        x.command('reset');assert x.job_kind is None and x.state()['timestamp']==0.
    finally:x.close()


def test_long_job_rejects_unapproved_grant_before_any_CUDA(tmp_path,monkeypatch):
    import wiwo_ma.platform.longstudy as module
    folder=ROOT/'artifacts/platform/jobs/munich30s_native_audited_attempt01'
    if not (folder/'proposal.json').exists():pytest.skip('Prepared project job is optional')
    # Bypass only source-version checks here to isolate grant admission.
    # A forbidden grant must reject without constructing NativeRT.
    monkeypatch.setattr(module,'verify_inputs',lambda *_:None)
    grant=tmp_path/'unapproved.json';grant.write_text(json.dumps(dict(approved=False)))
    monkeypatch.setattr(module,'inside',lambda x:folder if str(x)=='job' else grant)
    with pytest.raises(ValueError,match='authorization'):supervise('job','grant')


def test_native_replay_passed_sidecar_is_bound_to_real_HDF_row_and_reference():
    path=ROOT/'artifacts/platform/validation/native_gate_attempt01/full64_0/dataset/csi.h5'
    if not path.exists():pytest.skip('Optional actual native diagnostic artifact unavailable')
    reader=Replay(path);s,h,_=reader.frame(0)
    assert h['H'].shape==(40,1,64,64) and h['recorded_independent_validation_status']=='Unknown'
    assert h['independent_validation_status']=='Passed_original_SDK_checked_frame_sidecar'
    # A proof for another row cannot promote this frame.
    reader.verified_rows={};_,h,_=reader.frame(0);assert h['independent_validation_status']=='Unknown'

