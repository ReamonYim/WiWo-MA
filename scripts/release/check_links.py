#!/usr/bin/env python3
"""Check repository-local Markdown destinations; no network or GPU."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]

def main():
    errors=[];checked=0
    for p in [ROOT/'README.md',*sorted((ROOT/'docs').rglob('*.md'))]:
        text=re.sub(r'```.*?```','',p.read_text(),flags=re.S)
        for match in re.finditer(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            dest=match[1].strip().split(' ',1)[0].strip('<>')
            if not dest or dest.startswith(('#','http://','https://','mailto:')):continue
            checked+=1;target=(p.parent/dest.split('#')[0]).resolve()
            if not target.is_relative_to(ROOT) or not target.exists():errors.append(str(p.relative_to(ROOT))+': '+dest)
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Passed {checked} local documentation links')

if __name__=='__main__':main()
