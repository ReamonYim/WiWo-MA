"""Detached service ownership/readiness checks without opening sockets."""
import errno
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2]/'scripts/platform/town02_service.py'
spec = importlib.util.spec_from_file_location('town02_service',SCRIPT)
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


@pytest.fixture
def service(tmp_path,monkeypatch):
    config = tmp_path/'configs/platform/town02_strict_itu_64tx_40ue.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(dict(scene=dict(name='town02',material_policy=manager.POLICY,
        xml_path='assets/town02/town2_strict_itu.xml',xml_sha256='latest'),radio=dict(carrier_hz=2.8e9))))
    monkeypatch.setattr(manager,'ROOT',tmp_path)
    monkeypatch.setattr(manager,'CONFIG',config)
    monkeypatch.setattr(manager,'PYTHON',tmp_path/'.venv-full/bin/python')
    monkeypatch.setattr(manager,'ports_owned',lambda row:True)
    return manager.Service(8080)


def record():
    return dict(status='managed_running',pid=777,start_ticks='123',host='0.0.0.0',port=8080)


def process_identity():
    return dict(start_ticks='123',cwd=str(manager.ROOT),argv=[str(manager.PYTHON),'-m',
        'wiwo_ma.platform.cli','serve','--config',str(manager.CONFIG),'--host','0.0.0.0','--port','8080'])


@pytest.mark.parametrize('change',[dict(start_ticks='999'),dict(cwd='/other'),dict(argv=['python','unrelated.py'])])
def test_reused_or_unrelated_pid_is_never_signaled(service,monkeypatch,change):
    current = dict(process_identity(),**change)
    monkeypatch.setattr(manager,'identity',lambda pid:current)
    service.save(record())
    monkeypatch.setattr(manager.os,'kill',lambda *_:pytest.fail('Must not signal another process'))
    assert service.status()['status']=='failed'
    assert service.stop()['status']=='stopped'


def test_verified_stop_sends_only_sigint(service,monkeypatch):
    alive = [True];signals=[]
    monkeypatch.setattr(manager,'identity',lambda pid:process_identity() if alive[0] and pid==777 else None)
    def send(pid,kind):
        signals.append((pid,kind));alive[0]=False
    monkeypatch.setattr(manager.os,'kill',send)
    service.save(record())
    assert service.stop()['status']=='stopped'
    assert signals==[(777,signal.SIGINT)]


def test_duplicate_start_keeps_existing_owner(service,monkeypatch):
    service.save(record())
    monkeypatch.setattr(manager,'identity',lambda pid:process_identity())
    monkeypatch.setattr(manager,'ready',lambda *_:(True,None))
    monkeypatch.setattr(manager.subprocess,'Popen',lambda *_a,**_k:pytest.fail('Duplicate server'))
    assert service.start('0.0.0.0',8080)['status']=='ready'
    with service.lock():
        with pytest.raises(BlockingIOError):
            with service.lock():pass


def test_bind_permission_failure_is_visible_and_never_launches(service,monkeypatch):
    def denied(*_):raise PermissionError(errno.EPERM,'Socket creation denied')
    monkeypatch.setattr(manager,'check_ports',denied)
    monkeypatch.setattr(manager.subprocess,'Popen',lambda *_a,**_k:pytest.fail('Bind failure must prevent launch'))
    result=service.start('0.0.0.0',8080)
    assert result['status']=='failed' and 'Socket creation denied' in result['reason']
    assert 'START FAILED' in service.log.read_text()


def test_both_ports_are_checked_before_either_is_released(monkeypatch):
    events=[]
    class Socket:
        def setsockopt(self,*_):pass
        def bind(self,address):events.append(('bind',address[1]))
        def close(self):events.append(('close',None))
    monkeypatch.setattr(manager.socket,'getaddrinfo',lambda host,port,**_: [(2,1,6,'',(host,port))])
    monkeypatch.setattr(manager.socket,'socket',lambda *_:Socket())
    manager.check_ports('127.0.0.1',8080)
    assert events==[('bind',8080),('bind',8081),('close',None),('close',None)]


def test_launch_is_detached_pinned_and_readiness_checked(service,monkeypatch):
    calls=[]
    monkeypatch.setattr(manager,'check_ports',lambda *_:None)
    monkeypatch.setattr(manager,'identity',lambda pid:process_identity() if pid==777 else None)
    monkeypatch.setattr(manager,'ready',lambda row,expected:(expected['scene']['xml_sha256']=='latest',None))
    def launch(argv,**kwargs):
        calls.append((argv,kwargs));return SimpleNamespace(pid=777,poll=lambda:None)
    monkeypatch.setattr(manager.subprocess,'Popen',launch)
    assert service.start('0.0.0.0',8080)['status']=='ready'
    argv,kwargs=calls[0]
    assert argv==[str(manager.ROOT/'scripts/platform/town02'),'--host','0.0.0.0','--port','8080']
    assert kwargs['start_new_session'] and kwargs['stdin']==subprocess.DEVNULL
    assert kwargs['env']['WIWO_TOWN02_CONFIG']==str(manager.CONFIG)
    assert kwargs['env']['WIWO_GPU']=='-1' and kwargs['env']['WIWO_PYTHON']==str(manager.PYTHON)


def test_exited_start_records_failure_and_log_path(service,monkeypatch):
    monkeypatch.setattr(manager,'check_ports',lambda *_:None)
    monkeypatch.setattr(manager,'identity',lambda pid:None)
    monkeypatch.setattr(manager.subprocess,'Popen',lambda *_a,**_k:SimpleNamespace(pid=777))
    result=service.start('0.0.0.0',8080)
    assert result['status']=='failed' and 'exited' in result['reason']
    assert result['log']==str(service.log)


def test_readiness_requires_current_strict_config_and_completed_map(service,monkeypatch):
    expected=service.expected();actual=json.loads(json.dumps(expected));actual['scene']['xml_sha256']='old'
    ui=dict(map_faces=100,server_render_updates=1,last_action_error=None)
    urls=[]
    def fetch(url,as_json=True):
        urls.append(url)
        if not as_json:return b'web page'
        return {'health':dict(status='ready',port=8080),'config':actual,'ui':ui}[url.rsplit('/',1)[1]]
    monkeypatch.setattr(manager,'fetch',fetch)
    assert not manager.ready(record(),expected)[0]
    actual['scene']['xml_sha256']='latest';ui['map_faces']=0
    assert not manager.ready(record(),expected)[0]
    ui['map_faces']=100
    assert manager.ready(record(),expected)==(True,None)
    assert all('127.0.0.1' in url for url in urls)


def test_stop_timeout_preserves_identity_without_stronger_signal(service,monkeypatch):
    service.save(record());signals=[]
    monkeypatch.setattr(manager,'identity',lambda pid:process_identity())
    monkeypatch.setattr(manager.os,'kill',lambda *args:signals.append(args))
    # A healthy HTTP server may ignore SIGINT; stop must still report failure.
    monkeypatch.setattr(manager,'ready',lambda *_:(True,None))
    assert service.stop(timeout=0)['status']=='failed'
    assert service.read()['pid']==777 and signals==[(777,signal.SIGINT)]


def test_readiness_rejects_a_foreign_listener(service,monkeypatch):
    monkeypatch.setattr(manager,'ports_owned',lambda row:False)
    monkeypatch.setattr(manager,'fetch',lambda *_a,**_k:pytest.fail('Do not probe another process'))
    assert not manager.ready(record(),service.expected())[0]


def test_both_listener_inodes_must_appear_in_managed_pid_fds(tmp_path,monkeypatch):
    monkeypatch.setattr(manager,'PROC',tmp_path)
    folder=tmp_path/'777';(folder/'fd').mkdir(parents=True);(folder/'net').mkdir()
    (folder/'net/tcp').write_text('header\n0: 00000000:1F90 00000000:0000 0A 0:0 0:0 0 0 0 501\n'
                                  '1: 00000000:1F91 00000000:0000 0A 0:0 0:0 0 0 0 502\n')
    (folder/'fd/3').symlink_to('socket:[501]')
    assert not manager.ports_owned(record())
    (folder/'fd/4').symlink_to('socket:[502]')
    assert manager.ports_owned(record())
