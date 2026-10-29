#!/usr/bin/env python3
"""Reproducible, staged-file source release. Never uploads or includes data."""
import argparse,gzip,hashlib,io,json,subprocess,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    payload={};modes={}
    for record in subprocess.check_output(['git','-C',str(ROOT),'ls-files','--stage','-z']).split(b'\0'):
        if not record:continue
        meta,name=record.split(b'\t',1);mode,oid,stage=meta.decode().split();name=name.decode()
        assert stage=='0' and mode in ('100644','100755')
        if name.startswith(('.releasework/','artifacts/','experiments/','dist/','data/demo','.venv/','.cache/')) or name.endswith(('.h5','.npz','.npy','.so','.log','.jsonl')):raise ValueError('Forbidden bundle input: '+name)
        data=subprocess.check_output(['git','-C',str(ROOT),'cat-file','blob',oid]);assert data==(ROOT/name).read_bytes(),'Unstaged modification: '+name
        payload[name]=data;modes[name]=int(mode[-3:],8)
    assert payload
    manifest=dict(version='1.0.0',status='Source_release_limited_validated_scope',license='MIT',repository='https://github.com/ReamonYim/WiWo-MA',files={n:hashlib.sha256(v).hexdigest() for n,v in sorted(payload.items())},new_GPU_seconds=0)
    payload['BUNDLE_MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode();modes['BUNDLE_MANIFEST.json']=0o644
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for name,data in sorted(payload.items()):
            info=tarfile.TarInfo('WiWo-MA-v1.0.0/'+name);info.size=len(data);info.mode=modes[name];info.mtime=0;info.uid=info.gid=0;tar.addfile(info,io.BytesIO(data))
    digest=hashlib.sha256(a.output.read_bytes()).hexdigest();Path(str(a.output)+'.sha256').write_text(digest+'  '+a.output.name+'\n')
    print(json.dumps(dict(status='Built_source_release_not_uploaded',files=len(payload),bytes=a.output.stat().st_size,sha256=digest)))

if __name__=='__main__':main()
