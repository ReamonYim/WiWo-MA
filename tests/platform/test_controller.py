from copy import deepcopy
import time
import numpy as np
import pytest
from wiwo_ma.platform.controller import PlatformController
from wiwo_ma.platform.examples import reflector_example,munich_example


def test_UI_preview_clock_pending_restore_and_no_CSI_without_GPU():
    c=reflector_example();c['actions']=[];x=PlatformController(c)
    try:
        x.command('step',dict(seconds=.02));x.action(dict(bs_id='bs0',panel_id='panel0',type='target_pose',position_m=[.05,0,0]));x.command('save')
        x.command('step',dict(seconds=.2));a=x.state()['tx_port_positions'];x.command('restore');x.command('step',dict(seconds=.2))
        assert np.array_equal(a,x.state()['tx_port_positions'])
        assert x.result is None and x.snapshot()[2]['current_CSI']=='Unknown_current_pose'
        for kind in ('generate','solve_current'):
            with pytest.raises(ValueError,match='disabled'):x.command(kind)
    finally:x.close()


def test_fullscale_Stop_batches_without_preflighting_16_futures_and_brakes():
    c=munich_example();c['actions']=[];x=PlatformController(c)
    try:
        x.action(dict(bs_id='bs0',panel_id='panel00',type='target_pose',position_m=(np.array(c['base_stations'][0]['panels'][0]['position_m'])+[.04,0,0]).tolist()))
        x.command('step',dict(seconds=.1));assert np.linalg.norm(x.state()['velocities'])>0
        before=x.world.time;started=time.monotonic();x.command('stop');latency=time.monotonic()-started
        assert latency<2.,'Stop must not re-simulate sixteen complete futures'
        deadline=time.monotonic()+8
        while x.playing and time.monotonic()<deadline:time.sleep(.01)
        assert not x.playing and not x.error and x.world.time>before
        assert np.max(abs(x.state()['velocities']))<1e-8
        from wiwo_ma.platform.config import validate
        assert validate(x.export_actions())['actions'], 'Emergency braking must remain reproducible/exportable'
    finally:x.close()


