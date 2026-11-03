#!/usr/bin/env python3
"""Launch the installed automation browser and check software WebGL2."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True,args=[
            '--no-sandbox','--disable-dev-shm-usage','--use-angle=swiftshader',
            '--enable-unsafe-swiftshader'])
        try:
            page = browser.new_page()
            page.set_content('<title>WiWo-MA browser check</title><canvas id="view"></canvas>')
            assert page.title() == 'WiWo-MA browser check'
            webgl = page.evaluate("!!document.getElementById('view').getContext('webgl2')")
            if not webgl:
                raise RuntimeError('Browser started, but WebGL2 is unavailable')
            report = dict(status='Passed_browser_launch_and_WebGL2',browser=browser.version,
                          rendering='software',native_CSI_computed=False)
        finally:
            browser.close()
    output = ROOT/'artifacts/installation/browser-check.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
