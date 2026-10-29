"""Attach actual committed native data without launching another GPU solver."""
import shutil,time
import json,hashlib
import numpy as np
import pytest
from wiwo_ma.platform.config import ROOT
from wiwo_ma.platform.examples import munich_example
from wiwo_ma.platform.controller import PlatformController




def test_current_pose_attach_restores_frozen_geometry_before_first_frame_and_remains_readonly(tmp_path,monkeypatch):
    from wiwo_ma.platform.examples import reflector_example
    from wiwo_ma.platform.physics import PhysicalWorld
    from wiwo_ma.platform.storage import encode
    from wiwo_ma.platform.state_identity import geometry_digest
    import wiwo_ma.platform.controller as module
    c=reflector_example();c.update(actions=[],actors=[])
    for u in c['users']:u['trajectory']=[]
    world=PhysicalWorld(c)
    world.submit(dict(bs_id='bs0',panel_id='panel0',type='target_pose',position_m=[.01,0,0]))
    world.advance_to(.4)
    folder=tmp_path/'job';folder.mkdir();(folder/'config.json').write_text(json.dumps(c))
    file=folder/'initial_snapshot.json';file.write_text(json.dumps(encode(world.snapshot())))
    (folder/'plan.json').write_text(json.dumps(dict(query_mode='current_state',initial_snapshot_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),initial_geometry_sha256=geometry_digest(world.state()))))
    monkeypatch.setattr(module,'inside',lambda _:folder);monkeypatch.setattr(module,'ROOT',tmp_path)
    x=PlatformController(c)
    try:
        x.authorization=tmp_path/'old_other_job_grant.json'
        x.command('attach_job',dict(path='job'));s,h,stats=x.snapshot()
        assert s['timestamp']==.4 and h is None
        assert geometry_digest(s)==geometry_digest(world.state())
        assert stats['current_CSI']=='Unknown_current_pose' and stats['mode']=='native_current_state'
        assert x.authorization is None
        x.command('start');time.sleep(.05);assert x.state()['timestamp']==.4 and not x.playing
        with pytest.raises(ValueError,match='frozen'):x.command('step')
    finally:x.close()


@pytest.mark.parametrize('corruption',['snapshot_hash','geometry_hash','missing_snapshot'])
def test_invalid_current_pose_attachment_is_atomic_and_does_not_adopt_the_job(tmp_path,monkeypatch,corruption):
    from wiwo_ma.platform.examples import reflector_example
    from wiwo_ma.platform.physics import PhysicalWorld
    from wiwo_ma.platform.storage import encode
    from wiwo_ma.platform.state_identity import geometry_digest
    import wiwo_ma.platform.controller as module
    c=reflector_example();world=PhysicalWorld(c);world.advance_to(.02)
    folder=tmp_path/'job';folder.mkdir();(folder/'config.json').write_text(json.dumps(c))
    file=folder/'initial_snapshot.json';file.write_text(json.dumps(encode(world.snapshot())))
    plan=dict(query_mode='current_state',initial_snapshot_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),initial_geometry_sha256=geometry_digest(world.state()))
    if corruption=='snapshot_hash':plan['initial_snapshot_sha256']='invalid'
    if corruption=='geometry_hash':plan['initial_geometry_sha256']='invalid'
    if corruption=='missing_snapshot':file.unlink()
    (folder/'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(module,'inside',lambda _:folder);monkeypatch.setattr(module,'ROOT',tmp_path)
    x=PlatformController(c)
    try:
        initial=geometry_digest(x.state())
        with pytest.raises(ValueError,match='snapshot|geometry'):x.command('attach_job',dict(path='job'))
        assert geometry_digest(x.state())==initial and x.job is None
    finally:x.close()
