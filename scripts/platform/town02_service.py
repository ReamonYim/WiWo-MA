"""Project-local detached Town02 preview; no native jobs or automatic retries."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'configs/platform/town02_strict_itu_64tx_40ue.json'
PYTHON = ROOT/'.venv-full/bin/python'
POLICY = 'town02-source-label-itu-v1'
PROC = Path('/proc')


def identity(pid):
    try:
        folder = PROC/str(pid)
        fields = (folder/'stat').read_text().rsplit(') ',1)[1].split()
        if fields[0] == 'Z':return None
        return dict(start_ticks=fields[19],cwd=str((folder/'cwd').resolve()),
                    argv=(folder/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))
    except (OSError,ValueError,IndexError):return None


def owned(row):
    current = identity(row.get('pid',0))
    expected = [str(PYTHON),'-m','wiwo_ma.platform.cli','serve','--config',str(CONFIG),
                '--host',row.get('host',''),'--port',str(row.get('port',''))]
    return bool(current and current['start_ticks'] == row.get('start_ticks')
                and current['cwd'] == str(ROOT) and current['argv'] == expected)


def contains(actual,wanted):
    if isinstance(wanted,dict):
        return isinstance(actual,dict) and all(k in actual and contains(actual[k],v) for k,v in wanted.items())
    if isinstance(wanted,list):
        return isinstance(actual,list) and len(actual)==len(wanted) and all(contains(a,b) for a,b in zip(actual,wanted))
    return actual == wanted


def ports_owned(row):
    """Both requested listeners must belong to this exact managed PID."""
    try:
        folder = PROC/str(row['pid'])
        links = [os.readlink(fd) for fd in (folder/'fd').iterdir()]
        sockets = {link[8:-1] for link in links if link.startswith('socket:[') and link.endswith(']')}
        ports = set()
        for name in ('tcp','tcp6'):
            table = folder/'net'/name
            if not table.exists():continue
            for line in table.read_text().splitlines()[1:]:
                fields = line.split()
                if fields[3]=='0A' and fields[9] in sockets:
                    ports.add(int(fields[1].rsplit(':',1)[1],16))
        return {row['port'],row['port']+1} <= ports
    except (OSError,ValueError,IndexError,KeyError):return False


def fetch(url,as_json=True):
    with urllib.request.urlopen(url,timeout=2) as response:
        if response.status != 200:raise ValueError('HTTP '+str(response.status))
        data = response.read()
    return json.loads(data) if as_json else data


def ready(row,expected):
    if not ports_owned(row):return False,'Both requested listening ports must belong to the managed PID'
    host = {'0.0.0.0':'127.0.0.1','::':'::1'}.get(row['host'],row['host'])
    host = '['+host+']' if ':' in host else host
    web = 'http://'+host+':'+str(row['port'])
    api = 'http://'+host+':'+str(row['port']+1)+'/api/'
    try:
        fetch(web+'/',False)
        health = fetch(api+'health')
        if health.get('status')!='ready' or health.get('port')!=row['port']:return False,'API is not ready'
        if not contains(fetch(api+'config'),expected):return False,'Running configuration differs from the latest strict preset'
        ui = fetch(api+'ui')
        if ui.get('map_faces',0)<=0 or ui.get('server_render_updates',0)<=0 or ui.get('last_action_error'):
            return False,'Map has not completed its first successful draw'
        return True,None
    except Exception as error:return False,str(error)


def check_ports(host,port):
    held = []
    try:
        for number in (port,port+1):
            addresses = {(family,address) for family,_,_,_,address in socket.getaddrinfo(host,number,type=socket.SOCK_STREAM)}
            for family,address in addresses:
                sock = socket.socket(family,socket.SOCK_STREAM);held.append(sock)
                sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                sock.bind(address)
    finally:
        for sock in held:sock.close()


class Service:
    def __init__(self,port):
        self.folder = ROOT/'.state/town02-service'/str(port)
        self.folder.mkdir(parents=True,exist_ok=True)
        self.state = self.folder/'state.json';self.log = self.folder/'server.log'

    def read(self):
        return json.loads(self.state.read_text()) if self.state.exists() else dict(status='stopped')

    def save(self,row):
        temporary = self.state.with_suffix('.tmp')
        temporary.write_text(json.dumps(row,indent=2)+'\n');temporary.replace(self.state)

    @contextmanager
    def lock(self):
        with (self.folder/'manager.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            yield

    def expected(self):
        config = json.loads(CONFIG.read_text())
        if config['scene'].get('material_policy')!=POLICY or config['scene'].get('xml_path')!='assets/town02/town2_strict_itu.xml':
            raise ValueError('The pinned Town02 strict preset is missing its material policy')
        return config

    def status(self):
        row = self.read()
        if owned(row):
            ok,reason = ready(row,self.expected())
            row = dict(row,status='ready' if ok else row['status'] if row['status']=='failed' else 'managed_running',reason=reason)
        elif row.get('status') in ('ready','managed_running'):
            row = dict(row,status='failed',reason='Recorded PID is absent or belongs to another process; no signal sent')
        return dict(row,log=str(self.log),state_file=str(self.state))

    def stop(self,timeout=15):
        row = self.read()
        if owned(row):
            os.kill(row['pid'],signal.SIGINT)
            deadline = time.monotonic()+timeout
            while owned(row) and time.monotonic()<deadline:time.sleep(.2)
            if owned(row):
                row.update(status='failed',reason='SIGINT stop timed out; no stronger signal sent')
                self.save(row);return dict(row,log=str(self.log),state_file=str(self.state))
        self.save(dict(status='stopped',reason='Only the verified managed process was eligible for signaling'))
        return self.status()

    def start(self,host,port,timeout=120):
        previous = self.read()
        if owned(previous):return self.status()
        row = dict(status='managed_running',host=host,port=port,config=str(CONFIG))
        process = None
        try:
            expected = self.expected();check_ports(host,port)
            environment = dict(os.environ,WIWO_GPU='-1',WIWO_PYTHON=str(PYTHON),WIWO_TOWN02_CONFIG=str(CONFIG),PYTHONUNBUFFERED='1')
            with self.log.open('ab') as log:
                log.write(('\nTown02 start '+time.strftime('%Y-%m-%d %H:%M:%S')+'\n').encode());log.flush()
                process = subprocess.Popen([str(ROOT/'scripts/platform/town02'),'--host',host,'--port',str(port)],
                    cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row['pid'] = process.pid
            current = identity(process.pid)
            if current is None:raise RuntimeError('Server exited before its PID identity could be recorded')
            row['start_ticks'] = current['start_ticks'];self.save(row)
            deadline = time.monotonic()+timeout
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('Server exited with code '+str(process.returncode))
                if owned(row):
                    ok,reason = ready(row,expected)
                    if ok:
                        row['status']='ready';self.save(row);return self.status()
                time.sleep(.25)
            raise TimeoutError('Startup timed out; '+str(ready(row,expected)[1]))
        except Exception as error:
            if process is not None and owned(row):
                self.save(row);self.stop(5)
            row.update(status='failed',reason=str(error));self.save(row)
            with self.log.open('a') as log:log.write('START FAILED: '+str(error)+'\n')
            return dict(row,log=str(self.log),state_file=str(self.state))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('start','restart','stop','status'))
    parser.add_argument('--host',default=os.environ.get('WIWO_SERVICE_HOST','0.0.0.0'))
    parser.add_argument('--port',type=int,default=int(os.environ.get('WIWO_SERVICE_PORT','8080')))
    parser.add_argument('--timeout',type=float)
    args = parser.parse_args()
    if not 1<=args.port<=65534:parser.error('port must leave room for its adjacent API port')
    if args.timeout is not None and args.timeout<=0:parser.error('timeout must be positive')
    service = Service(args.port)
    try:
        if args.command=='status':result=service.status()
        else:
            with service.lock():
                result = service.stop(args.timeout or 15) if args.command in ('stop','restart') else None
                if args.command in ('start','restart') and (result is None or result['status']=='stopped'):
                    result=service.start(args.host,args.port,args.timeout or 120)
    except Exception as error:result=dict(status='failed',reason=str(error),log=str(service.log))
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 1 if result['status']=='failed' else 0


if __name__=='__main__':raise SystemExit(main())
