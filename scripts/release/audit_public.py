#!/usr/bin/env python3
"""Inspect worktree, Git index and all reachable history; never print secrets.

This scanner is a deterministic preflight, not proof that arbitrary secrets
cannot exist. Reports contain rule/path/line only, never matching values.
"""
import argparse,hashlib,ipaddress,json,re,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PATTERNS={
    'private_key':rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----',
    'GitHub_token':rb'(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{30,})',
    'cloud_access_key':rb'(?:AKIA|ASIA)[A-Z0-9]{16}',
    'API_secret':rb'(?:sk-[A-Za-z0-9_-]{32,}|AIza[A-Za-z0-9_-]{35})',
    'URL_password':rb'https?://[^\s/@:]+:[^\s/@]+@',
    'machine_absolute_path':rb'(?:/'+b'root'+rb'/|/'+b'home'+rb'/[A-Za-z0-9_.-]+/|[A-Za-z]:\\Users\\)',
}
FORBIDDEN_PREFIXES=('.releasework/','.venv/','.envs/','.cache/','.state/','artifacts/','experiments/','dist/','build/','releases/')
FORBIDDEN_SUFFIXES=('.h5','.hdf5','.npy','.npz','.pt','.pth','.ckpt','.pickle','.pkl','.pem','.key','.so','.dll','.log','.jsonl','.blend')

def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])

def scan(name,data,layer,findings):
    if name.startswith(FORBIDDEN_PREFIXES) or name.endswith(FORBIDDEN_SUFFIXES) or '/__pycache__/' in name:
        findings.append(dict(layer=layer,path=name,rule='forbidden_release_path'));return
    if len(data)>5*1024**2:findings.append(dict(layer=layer,path=name,rule='file_over_5MiB'))
    if name.endswith(('.png','.gif','.jpg','.jpeg')):return
    for rule,pattern in PATTERNS.items():
        for match in re.finditer(pattern,data):findings.append(dict(layer=layer,path=name,rule=rule,line=data[:match.start()].count(b'\n')+1))
    for match in re.finditer(rb'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])',data):
        try:address=ipaddress.ip_address(match[0].decode())
        except ValueError:continue
        if not address.is_loopback and not address.is_unspecified and address.is_private:findings.append(dict(layer=layer,path=name,rule='private_network_address',line=data[:match.start()].count(b'\n')+1))
    if name.endswith('.json'):
        try:q=json.loads(data)
        except (ValueError,UnicodeDecodeError):return
        if isinstance(q,dict) and q.get('approved') is True and any(k in q for k in ('total_wall_s','gpu','max_allocator_bytes')):findings.append(dict(layer=layer,path=name,rule='approved_resource_grant'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();findings=[];entries=[]
    for item in git('ls-files','--stage','-z').split(b'\0'):
        if not item:continue
        meta,name=item.split(b'\t',1);mode,oid,stage=meta.decode().split();name=name.decode()
        if stage!='0' or mode not in ('100644','100755'):findings.append(dict(layer='index',path=name,rule='unsupported_index_mode'));continue
        data=git('cat-file','blob',oid);scan(name,data,'index',findings)
        file=ROOT/name
        if not file.is_file() or hashlib.sha256(file.read_bytes()).digest()!=hashlib.sha256(data).digest():findings.append(dict(layer='worktree',path=name,rule='index_worktree_mismatch'))
        elif file.is_file():scan(name,file.read_bytes(),'worktree',findings)
        entries.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    history=git('rev-list','--objects','--all').decode().splitlines();blobs=0
    for row in history:
        oid,_,name=row.partition(' ')
        if git('cat-file','-t',oid).strip()!=b'blob':continue
        blobs+=1;scan(name,git('cat-file','blob',oid),'history',findings)
    if not entries:findings.append(dict(layer='index',path='',rule='empty_staged_release'))
    report=dict(status='Passed_scoped_security_scan' if not findings else 'Failed',index_files=len(entries),index_bytes=sum(x['bytes'] for x in entries),history_blobs=blobs,history_scope='All reachable refs; empty history is explicit, not a scanned published commit',findings=findings,files=entries,
        scope='Deterministic path/secret-pattern checks of staged blobs and matching worktree/history; media also require visual review; licensing/publication approval is separate')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],index_files=len(entries),index_bytes=report['index_bytes'],history_blobs=blobs,findings=findings)))
    if findings:raise SystemExit(1)

if __name__=='__main__':main()
