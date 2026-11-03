"""Capture setup, cleanup and screenshot policy without real browser sockets."""
import errno
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest


SCRIPT=Path(__file__).resolve().parents[2]/'scripts/media/capture_town02.py'
spec=importlib.util.spec_from_file_location('town02_capture',SCRIPT)
capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)


class FakePage:
    def __init__(self,errors):
        self.errors=list(errors);self.calls=[];self.viewport_size=dict(width=1440,height=960);self.resizes=[]

    def screenshot(self,**options):
        self.calls.append(options)
        error=self.errors.pop(0) if self.errors else None
        if error:raise error
        Image.new('RGB',(self.viewport_size['width'],self.viewport_size['height']),'#304050').save(options['path'])

    def set_viewport_size(self,size):self.viewport_size=size;self.resizes.append(size)
    def wait_for_timeout(self,ms):pass


def test_capture_has_explicit_long_timeout_and_validates_native_png(tmp_path):
    page=FakePage([]);target=tmp_path/'shot.png'
    report=capture.save_browser_screenshot(page,target,180000)
    assert target.is_file() and report['width']==1440 and report['height']==960
    assert page.calls[0]['timeout']==180000 and page.calls[0]['scale']=='css'
    assert page.calls[0]['animations']=='disabled' and not page.calls[0]['full_page']
    assert not page.resizes and not list(tmp_path.glob('*.pending.png'))


def test_only_a_timeout_retries_at_a_smaller_real_viewport(tmp_path):
    page=FakePage([TimeoutError('GPU readback stalled')])
    report=capture.save_browser_screenshot(page,tmp_path/'shot.png',180000)
    assert len(page.calls)==2 and page.resizes==[dict(width=1024,height=768)]
    assert report['attempts'][0]['status']=='timeout' and report['attempts'][1]['status']=='captured'
    assert report['width']==1024 and report['height']==768


def test_repeated_timeout_is_bounded_and_preserves_an_existing_output(tmp_path):
    target=tmp_path/'shot.png';target.write_bytes(b'previous capture')
    page=FakePage([TimeoutError('first'),TimeoutError('second')])
    with pytest.raises(TimeoutError,match='second'):
        capture.save_browser_screenshot(page,target,180000)
    assert len(page.calls)==2 and target.read_bytes()==b'previous capture'
    assert not list(tmp_path.glob('*.pending.png'))


def test_non_timeout_failure_is_not_retried_or_masked(tmp_path):
    page=FakePage([RuntimeError('WebGL context was lost')])
    with pytest.raises(RuntimeError,match='context was lost'):
        capture.save_browser_screenshot(page,tmp_path/'shot.png',180000)
    assert len(page.calls)==1 and not page.resizes


@pytest.mark.parametrize('failure',[None,KeyboardInterrupt(),RuntimeError('render failed')])
def test_preview_sets_rate_before_running_and_closes_on_every_exit(monkeypatch,failure):
    from wiwo_ma.platform import controller,web
    events=[];config={'scene':'test'}
    control=SimpleNamespace(close=lambda:events.append('controller closed'))

    class UI:
        def __init__(self,actual,host,port):
            assert actual is control and host=='127.0.0.1' and port==45000
            self.render_hz=20.;self.render_rate=SimpleNamespace(value=20)
        def run(self):
            assert self.render_hz==2. and self.render_rate.value==2
            events.append('run')
            if failure:raise failure
        def close(self):events.append('UI closed')

    def create(actual):
        assert actual is config
        return control

    monkeypatch.setattr(controller,'PlatformController',create)
    monkeypatch.setattr(web,'PlatformUI',UI)
    if isinstance(failure,RuntimeError):
        with pytest.raises(RuntimeError,match='render failed'):capture.run_capture_preview(config,45000)
    else:capture.run_capture_preview(config,45000)
    assert events==['run','UI closed']


def test_preview_constructor_failure_closes_controller(monkeypatch):
    from wiwo_ma.platform import controller,web
    closed=[]
    monkeypatch.setattr(controller,'PlatformController',lambda _:SimpleNamespace(close=lambda:closed.append(True)))

    def broken(*args):raise RuntimeError('cannot open preview port')

    monkeypatch.setattr(web,'PlatformUI',broken)
    with pytest.raises(RuntimeError,match='cannot open preview port'):capture.run_capture_preview({},45000)
    assert closed==[True]


def test_private_port_selection_retries_a_busy_adjacent_api_port_and_closes_sockets(monkeypatch):
    opened=[];bound=[];closed=[]

    class Socket:
        def __init__(self):self.index=len(opened);opened.append(self)
        def __enter__(self):return self
        def __exit__(self,*args):closed.append(self.index)
        def getsockname(self):return ('127.0.0.1',45000+self.index)
        def bind(self,address):
            bound.append(address)
            if self.index==1:raise OSError(errno.EADDRINUSE,'busy')

    monkeypatch.setattr(capture.socket,'socket',Socket)
    assert capture.choose_capture_port()==45002
    assert bound==[('127.0.0.1',0),('127.0.0.1',45001),('127.0.0.1',0),('127.0.0.1',45003)]
    assert sorted(closed)==[0,1,2,3]


def test_private_port_selection_does_not_retry_denied_socket_permissions(monkeypatch):
    attempts=[]

    def denied():
        attempts.append(True)
        raise PermissionError(errno.EPERM,'socket permission denied')

    monkeypatch.setattr(capture.socket,'socket',denied)
    with pytest.raises(PermissionError,match='socket permission denied'):capture.choose_capture_port()
    assert attempts==[True]
