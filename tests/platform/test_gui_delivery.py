"""Permanent regression for lost GUI fields and observable Action states."""
import asyncio,hashlib,json,re,shutil,subprocess,threading
from types import SimpleNamespace
import pytest
from wiwo_ma.platform.config import ROOT
from wiwo_ma.platform.gui_delivery import patch_html,base88_decode,base88_encode,GUI_ORDERED_BRANCH
from wiwo_ma.platform.web import PlatformUI
from wiwo_ma.platform.controller import PlatformController
from wiwo_ma.platform.examples import reflector_example
from wiwo_ma.platform.physics import constrained_target

def test_base88_roundtrip():
    for n in range(257):
        data=bytes(i%256 for i in range(n));assert base88_decode(base88_encode(data))==data

def test_frontend_patch_fails_closed_for_an_unreviewed_build():
    with pytest.raises(RuntimeError,match='Unreviewed'):patch_html(b'unknown frontend')

def test_real_compiled_sender_preserves_cross_field_values_before_click():
    if not shutil.which('node'):pytest.skip('Node needed for actual compiled JS regression')
    import importlib.metadata,zstandard
    source=importlib.metadata.distribution('viser').locate_file('viser/client/build/index.html').read_bytes()
    def sender(html):
        s=html.decode();ss=int(re.search(r'data-ss="(\d+)"',s)[1]);raw=zstandard.ZstdDecompressor().decompress(base88_decode(re.search(r'data-p="([^"]+)"',s)[1]));js=raw[ss:].decode()
        return js[js.index('function RL(e,t){'):js.index('function zL(e){')]
    before=sender(source);after=sender(patch_html(source));assert GUI_ORDERED_BRANCH in after
    def execute(function):
        code='let sent=[],timers=[];function setTimeout(fn){timers.push(fn);return 1}function clearTimeout(){};'+function+''';let x=RL({mutable:{current:{sendMessage:m=>sent.push(m)}}},50);for(const [uuid,value] of [['speed',.1],['target',[.003,0,0]],['execute',true]])x.send({type:'GuiUpdateMessage',uuid,updates:{value}});timers.shift()?.();console.log(JSON.stringify(sent));'''
        return json.loads(subprocess.check_output(['node','-e',code],text=True))
    assert [x['uuid'] for x in execute(before)]==['speed','execute']
    assert [x['uuid'] for x in execute(after)]==['speed','target','execute']

def test_critical_button_callback_is_awaited_before_next_edit():
    import inspect
    ui=PlatformUI.__new__(PlatformUI);ui.render_lock=threading.RLock();ui.sessions=SimpleNamespace(require=lambda _:None);ui.message=SimpleNamespace(content='')
    observed=[];fn=ui._ordered(lambda:observed.append('Action'))
    assert inspect.iscoroutinefunction(fn)
    asyncio.run(fn(SimpleNamespace(client=object(),client_id=1)));assert observed==['Action']

def test_no_op_and_real_action_have_distinct_receipts():
    config=reflector_example();config['actions']=[]
    ui=PlatformUI.__new__(PlatformUI);ui.controller=PlatformController(config);ui.draft_revision=3;ui.draft_unsubmitted=True;ui.action_status=SimpleNamespace(content='')
    c=ui.controller.config;b=c['base_stations'][0];p=b['panels'][0]
    try:
        ui._submit_action(constrained_target(c,b['id'],p['id'],[0,0,0],[0,0,0]));assert ui.action_submission['stage']=='NoOp'
        ui._update_action_status(ui.controller.state());assert '没有新的位移' in ui.action_status.content
        ui._submit_action(constrained_target(c,b['id'],p['id'],[.003,0,0],[0,0,0]));assert ui.action_submission['stage']=='Submitted'
        ui.controller.world.advance_to(.1);ui._update_action_status(ui.controller.state());assert ui.action_submission['stage'] in ('Executing','Completed')
        ui.controller.world.advance_to(1.);ui._update_action_status(ui.controller.state());assert ui.action_submission['stage']=='Completed'
        ui._mark_draft();assert ui.draft_unsubmitted
    finally:ui.controller.close()

def test_pending_control_tick_and_superseding_script_are_not_false_completion():
    config=reflector_example();config['actions']=[]
    ui=PlatformUI.__new__(PlatformUI);ui.controller=PlatformController(config);ui.draft_revision=0;ui.draft_unsubmitted=True;ui.action_status=SimpleNamespace(content='')
    c=ui.controller.config;b=c['base_stations'][0];p=b['panels'][0]
    try:
        ui.controller.world.advance_to(.002)
        ui._submit_action(constrained_target(c,b['id'],p['id'],[.003,0,0],[0,0,0]))
        ui._update_action_status(ui.controller.state());assert ui.action_submission['stage']=='Submitted'
        ui.controller.world.advance_to(.1);ui._update_action_status(ui.controller.state());assert ui.action_submission['stage']=='Executing'
        ui.controller.world.command(constrained_target(c,b['id'],p['id'],[-.003,0,0],[0,0,0]))
        ui._update_action_status(ui.controller.state());assert ui.action_submission['stage']=='Superseded'
    finally:ui.controller.close()
