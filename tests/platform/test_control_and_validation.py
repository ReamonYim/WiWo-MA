from copy import deepcopy
import json
import numpy as np
import pytest
from wiwo_ma.platform.physics import PhysicalWorld
from wiwo_ma.platform.config import validate,hardware_mode
from wiwo_ma.platform.examples import reflector_example
from wiwo_ma.platform.storage import encode,decode
from wiwo_ma.platform.validation import prepare,frozen_check


def test_control_grid_queue_and_branch_pending_inheritance():
    c=reflector_example();c['actions']=[];w=PhysicalWorld(c);w.advance_to(.02)
    q=dict(bs_id='bs0',panel_id='panel0',type='target_pose',position_m=[.04,0,0])
    receipt=w.submit(q);assert receipt['effective_time_s']==.1
    assert w.controllers[0].command_counter==0
    snap=decode(json.loads(json.dumps(encode(w.snapshot()))));w.advance_to(.3);expected=w.state()['tx_port_positions'].copy()
    w.restore(snap);w.advance_to(.3);assert np.array_equal(expected,w.state()['tx_port_positions'])
    assert w.executed[0]['time_s']==.1 and not w.scheduled


def test_precision_and_script_control_grid_are_never_silently_changed():
    c=reflector_example();c['rt']['precision']='float64'
    with pytest.raises(ValueError,match='Float32'):validate(c)
    c=reflector_example();c['actions'][0]['time_s']=.102
    with pytest.raises(ValueError,match='control ticks'):validate(c)


def test_declared_hardware_must_match_real_mechanisms():
    c=reflector_example();assert hardware_mode(c['base_stations'][0])=='panel_6dma'
    c['base_stations'][0]['hardware_mode']='single_ma'
    with pytest.raises(ValueError):validate(c)
    c['base_stations'][0]['hardware_mode']='fixed_array'
    with pytest.raises(ValueError):validate(c)
    c['base_stations'][0]['panels'][0]['mechanics']['mechanism']='fixed'
    assert hardware_mode(validate(c)['base_stations'][0])=='fixed_array'


def test_small_native_gate_is_frozen_CPU_only_and_reference_has_no_adapter_call(tmp_path,monkeypatch):
    import wiwo_ma.platform.validation as module
    monkeypatch.setattr(module,'inside',lambda x:tmp_path/'gate')
    proposal=prepare('gate');assert len(proposal['cases'])==5
    assert proposal['new_Full64_queries']==2 and proposal['small_two_spatial_source_queries']==8
    frozen_check(tmp_path/'gate',proposal)
    path=tmp_path/'gate/small_0_state.json';state=json.loads(path.read_text());state['timestamp']=.7;path.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='Frozen gate input'):frozen_check(tmp_path/'gate',proposal)
    import inspect
    ref=inspect.getsource(module.reference)
    assert 'NativeRT' not in ref and 'arrange_cfr' not in ref
