"""Release guards use synthetic secrets and isolated temporary Git histories."""
import importlib.util,json,subprocess,sys,tarfile
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]

def tool(name):
    p=ROOT/'scripts/release'/name;spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def repository(path):
    subprocess.run(['git','init',str(path)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(path),'symbolic-ref','HEAD','refs/heads/main'],check=True,capture_output=True)
    # Explicitly synthetic test identity; never used in the release repository.
    for key,value in [('user.name','Synthetic test fixture'),('user.email','fixture@example.invalid')]:subprocess.run(['git','-C',str(path),'config',key,value],check=True)

def run(path,*args):return subprocess.run(['git','-C',str(path),*args],check=True,capture_output=True)

def test_scan_reports_rule_and_line_without_repeating_secret():
    m=tool('audit_public.py');secret=('ghp_'+'A'*36).encode();findings=[];m.scan('sample.py',b'x\n'+secret,'index',findings)
    assert findings==[dict(layer='index',path='sample.py',rule='GitHub_token',line=2)]
    assert secret.decode() not in json.dumps(findings)

def test_scan_catches_reachable_deleted_history(tmp_path,monkeypatch,capsys):
    m=tool('audit_public.py');repository(tmp_path);p=tmp_path/'sample.txt';secret='ghp_'+'A'*36;p.write_text(secret);run(tmp_path,'add','.');run(tmp_path,'commit','-m','synthetic fixture')
    p.write_text('clean\n');run(tmp_path,'add','.');run(tmp_path,'commit','-m','remove fixture secret');out=tmp_path/'scan.json'
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(sys,'argv',['scan','--output',str(out)])
    with pytest.raises(SystemExit):m.main()
    q=json.loads(out.read_text());assert q['history_blobs']==2 and any(x['layer']=='history' and x['rule']=='GitHub_token' for x in q['findings'])
    assert secret not in capsys.readouterr().out

def test_source_bundle_uses_staged_bytes_and_rejects_later_edits(tmp_path,monkeypatch):
    m=tool('source_bundle.py');repository(tmp_path);p=tmp_path/'README.md';p.write_text('Synthetic bundle fixture\n');run(tmp_path,'add','README.md');out=tmp_path/'candidate.tar.gz'
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(sys,'argv',['bundle','--output',str(out)]);m.main()
    with tarfile.open(out) as tar:assert tar.extractfile('WiWo-MA-v1.0.0/README.md').read()==p.read_bytes()
    p.write_text('unstaged\n');other=tmp_path/'other.tar.gz';monkeypatch.setattr(sys,'argv',['bundle','--output',str(other)])
    with pytest.raises(AssertionError,match='Unstaged'):m.main()
    assert not other.exists()

def test_resource_approval_is_not_a_public_json_asset():
    m=tool('audit_public.py');findings=[];m.scan('example.json',json.dumps(dict(approved=True,gpu=0,total_wall_s=10)).encode(),'index',findings)
    assert any(x['rule']=='approved_resource_grant' for x in findings)

def test_banned_data_and_private_paths_are_caught_without_printing_them():
    m=tool('audit_public.py');findings=[];m.scan('data/raw.h5',b'fixture','index',findings)
    m.scan('settings.txt',('/'+'root'+'/someone/private').encode(),'index',findings)
    assert {x['rule'] for x in findings}=={'forbidden_release_path','machine_absolute_path'}
