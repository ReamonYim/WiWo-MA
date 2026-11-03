#!/usr/bin/env python3
"""Restore missing upstream gallery files from the downloaded source revision."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REVISION = '9d198db1d75349310ff115a695cd849fabdf3b56'
FILES = {
    '01_munich_overview.png':'e74daa1195f48316a25d2367e992d3100f9e1036',
    '02_full64_array.png':'a2a73515cbc460abdcfd1edf6b03a04197ce16c5',
    '03_mechanics_controls.png':'0ae53de2ea9ea6b780b02d07f39b40aafb50776d',
    '04_panel_motion.png':'34e882ae086dbf3cf440071b8e321b0ac201b4b9',
    '05_actor_occlusion.png':'5213537bf34570af9ced561715094306d511104e',
    '06_csi_response.png':'916414a92af53556de04a3ce7796d9b9ef9cd50c',
    '07_full_sequence_replay.png':'708cacaf1db292eaa29fb22802aaf19c0aaa97e9',
    '08_csi_time_frequency.png':'324d88195b4aa75d0c92f3af13744f93703e1d1d',
}


def main():
    for name,expected in FILES.items():
        destination = ROOT/'docs/media'/name
        existed = destination.exists()
        if existed:
            data = destination.read_bytes()
        else:
            url = f'https://raw.githubusercontent.com/ReamonYim/WiWo-MA/{REVISION}/docs/media/{name}'
            with urllib.request.urlopen(url,timeout=20) as response:
                data = response.read()
        actual = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual != expected:
            raise ValueError('Upstream gallery checksum mismatch: '+name)
        if not existed:
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(data)
        print('Verified',name)


if __name__ == '__main__':
    main()
