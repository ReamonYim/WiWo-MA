"""Pinned project-owned Viser frontend adapter. Installed SDKs stay untouched.

Viser1.1.1 shares one latest-message throttle between ALL GUI UUIDs. A later
field or click can erase another field, while React still displays its value.
GUI updates are discrete state/command messages: send them in order without
that lossy throttle. Continuous camera/drag senders retain upstream behavior.
"""
import hashlib
import importlib.metadata
import json
import re
import threading
from pathlib import Path

import zstandard

from .config import ROOT

UPSTREAM_HTML_SHA256 = '8c05b8bfea1ed54773a7616d1d868ebf0b45055ff8da8fc1603ccb2a7858f079'
GUI_SEND_PREFIX = 'function RL(e,t){let n=!0,r=!1,i=null,a=null;function o(s){let c=e.mutable.current;'
GUI_ORDERED_BRANCH = 'if(s.type==="GuiUpdateMessage"){if(c.sendMessage!==null)c.sendMessage(s);return}'
_LOCK = threading.Lock()


def base88_decode(text):
    alphabet = ''.join(chr(i) for i in range(33,127) if chr(i) not in '\"&<>\'`')
    table = {ch:i for i,ch in enumerate(alphabet)}
    result = bytearray()
    for i in range(0,len(text),5):
        group=text[i:i+5];value=0
        for ch in group:value=value*88+table[ch]
        result.extend(value.to_bytes(4 if len(group)==5 else len(group)-1,'big'))
    return bytes(result)


def base88_encode(data):
    alphabet = ''.join(chr(i) for i in range(33,127) if chr(i) not in '\"&<>\'`')
    result=[]
    for i in range(0,len(data),4):
        chunk=data[i:i+4];value=int.from_bytes(chunk,'big');group=[]
        for _ in range(len(chunk)+1):group.append(alphabet[value%88]);value//=88
        result.extend(reversed(group))
    return ''.join(result)


def patch_html(source):
    if hashlib.sha256(source).hexdigest()!=UPSTREAM_HTML_SHA256:
        raise RuntimeError('Unreviewed Viser frontend build; refuse an unverified patch')
    html=source.decode()
    match=re.search(r'data-p="([^"]+)"',html)
    size=int(re.search(r'data-ss="(\d+)"',html)[1])
    raw=zstandard.ZstdDecompressor().decompress(base88_decode(match[1]))
    css=raw[:size];js=raw[size:].decode()
    if js.count(GUI_SEND_PREFIX)!=1:raise RuntimeError('Pinned GUI sender signature changed')
    fixed=js.replace(GUI_SEND_PREFIX,GUI_SEND_PREFIX+GUI_ORDERED_BRANCH)
    payload=base88_encode(zstandard.ZstdCompressor(level=12).compress(css+fixed.encode()))
    html=html[:match.start(1)]+payload+html[match.end(1):]
    html=re.sub(r'data-cs="\d+"',f'data-cs="{len(fixed.encode())}"',html,count=1)
    html=html.replace('</body>','<script src="./input_guard.js"></script></body>')
    return html.encode()


def frontend_root():
    distribution=importlib.metadata.distribution('viser')
    if distribution.version!='1.1.1':raise RuntimeError('Only Viser1.1.1 frontend is certified')
    folder=ROOT/'assets/platform/viser-ordered-gui-v1'
    source=Path(distribution.locate_file('viser/client/build/index.html')).read_bytes()
    expected=patch_html(source)
    with _LOCK:
        folder.mkdir(parents=True,exist_ok=True)
        target=folder/'index.html'
        guard=folder/'input_guard.js'
        if not guard.is_file():raise RuntimeError('Missing numeric draft guard asset')
        if target.exists():
            if target.read_bytes()!=expected:raise RuntimeError('Project frontend differs from deterministic adapter')
        else:target.write_bytes(expected)
        receipt=folder/'adapter.json'
        metadata=dict(upstream_version='1.1.1',upstream_html_sha256=UPSTREAM_HTML_SHA256,
            patched_html_sha256=hashlib.sha256(expected).hexdigest(),numeric_guard_sha256=hashlib.sha256(guard.read_bytes()).hexdigest(),
            purpose='Ordered lossless GUI delivery and explicit unfinished-number rejection; SDK unchanged')
        if receipt.exists():
            if json.loads(receipt.read_text())!=metadata:raise RuntimeError('Project frontend receipt differs from assets')
        else:receipt.write_text(json.dumps(metadata,sort_keys=True)+'\n')
    return folder


def create_server(**kwargs):
    """Narrow constructor injection because pinned Viser has no root option.

    Only this process's server construction is wrapped, then restored. No
    installed files, existing service or propagation function is patched.
    """
    import viser
    root=frontend_root()
    with _LOCK:
        original=viser.infra.WebsockServer
        class BoundServer(original):
            def __init__(self,*args,**kw):
                kw['http_server_root']=root
                super().__init__(*args,**kw)
        viser.infra.WebsockServer=BoundServer
        try:return viser.ViserServer(**kwargs)
        finally:viser.infra.WebsockServer=original
