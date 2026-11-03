"""Screenshot timeout policy; no browser dependency or network sockets."""
import importlib.util
from pathlib import Path

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
