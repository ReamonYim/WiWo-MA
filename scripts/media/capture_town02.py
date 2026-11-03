#!/usr/bin/env python3
"""Capture the real Town02 Web UI on an isolated CPU-preview service.

No native RT, stored Munich dataset, generated image or DOM replacement is
used. The normal background service is not changed. Run from a terminal that
can open local TCP sockets and launch the installed Playwright Chromium.
"""
import argparse
from contextlib import closing
from datetime import datetime,timezone
import hashlib
import errno
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.request import urlopen

import numpy as np
from wiwo_ma.platform.config import ROOT,load
from wiwo_ma.platform.town02_resources import require_resources


def choose_capture_port(requested=0):
    """Reserve-check a fresh adjacent pair; never retry permission failures."""
    for _ in range(8):
        with socket.socket() as web,socket.socket() as api_socket:
            web.bind(('127.0.0.1',requested))
            port=web.getsockname()[1]
            if port>=65535:continue
            try:api_socket.bind(('127.0.0.1',port+1))
            except OSError as error:
                if not requested and error.errno==errno.EADDRINUSE:continue
                raise
            return port
    raise RuntimeError('Could not find an unused adjacent capture port pair')


def run_capture_preview(config,port,prepare=None):
    """Use the production UI, configuring its rate before the browser connects."""
    from wiwo_ma.platform.controller import PlatformController
    from wiwo_ma.platform.web import PlatformUI
    controller=PlatformController(config);ui=None
    try:
        ui=PlatformUI(controller,'127.0.0.1',port)
        ui.render_hz=2.
        ui.render_rate.value=2
        if prepare is not None:prepare(ui)
        ui.run()
    except KeyboardInterrupt:pass
    finally:
        if ui is not None:ui.close()
        else:controller.close()


def prepare_capture_views(ui):
    """Native scene-camera controls on the private service, without moving objects."""
    from wiwo_ma.platform.camera_views import fit_camera,padded_bounds
    from wiwo_ma.platform.scene_geometry import load_display_meshes,scene_xml
    meshes,_=load_display_meshes(str(scene_xml(ui.controller.config['scene'])))
    vertices=np.concatenate([mesh['vertices'] for mesh in meshes])
    bs_label=ui.server.scene.add_label('/gallery/BS',text='BS · 16 panels / 64 TX',visible=False)
    ue_label=ui.server.scene.add_label('/gallery/UE',text='40 UE · road positions',visible=False)

    def apply(name):
        state=ui.controller.state();panels=np.asarray(state['positions']);users=np.asarray(state['user_positions'])
        bs=np.asarray(state['bs_positions'][0]);bs_label.position=bs+[0,0,3];ue_label.position=users.mean(0)+[0,0,2]
        bs_label.visible=ue_label.visible=name in ('site','platform')
        ui.gizmo.visible=False
        minimum=.1;fill=.86
        if name in ('panorama','topdown'):
            points=vertices;direction=[.12,-.48,1.] if name=='panorama' else [0.,-.08,1.]
        elif name in ('site','platform'):
            anchors=np.concatenate([users,panels]) if name=='platform' else bs[None,:]
            margin=np.array([28.,32.]) if name=='platform' else np.array([52.,48.])
            lo=anchors[:,:2].min(0)-margin;hi=anchors[:,:2].max(0)+margin
            points=vertices[np.all(vertices[:,:2]>=lo,axis=1)&np.all(vertices[:,:2]<=hi,axis=1)]
            direction=[.18,-.65,1.] if name=='platform' else [-.65,-.5,1.]
        elif name=='antennas':
            points=padded_bounds(panels,.15);direction=[0.,1.,1.1];minimum=4.5;fill=.78
        elif name=='detail':
            selected=ui.selected;rotation=np.asarray(state['rotations'][selected]);normal=rotation[:,0]
            points=padded_bounds(panels[selected:selected+1],.12)
            direction=normal+[0.,0.,.3];minimum=.65;fill=.75
        else:raise ValueError('Unknown capture view: '+name)
        if not len(points):raise ValueError('No actual geometry in the requested view')
        for client in ui.server.get_clients().values():
            view=fit_camera(points,direction,aspect=client.camera.aspect,fill=fill,minimum_distance=minimum)
            ui._aim(client,view['look_at'],view['position']-view['look_at'])

    for name in ('panorama','topdown','site','antennas','detail','platform'):
        ui._button('Capture view: '+name,lambda name=name:apply(name))


def save_browser_screenshot(page,target,timeout_ms,timeout_error=TimeoutError,on_resize=None):
    """Bound capture time, retry a stalled compositor at a smaller viewport.

    Both attempts use the real browser and unchanged scene. The PNG is only
    published after it decodes and matches the browser's current viewport.
    """
    from PIL import Image
    target=Path(target);pending=target.with_name('.'+target.stem+'.pending.png')
    attempts=[]
    for attempt in range(2):
        pending.unlink(missing_ok=True)
        try:
            page.screenshot(path=str(pending),timeout=timeout_ms,animations='disabled',
                            caret='hide',scale='css',full_page=False)
            with Image.open(pending) as picture:
                picture.verify()
            with Image.open(pending) as picture:
                size=picture.size
                if size!=(page.viewport_size['width'],page.viewport_size['height']):
                    raise ValueError('Screenshot dimensions differ from the current viewport')
            attempts.append(dict(status='captured',width=size[0],height=size[1]))
            pending.replace(target)
            return dict(attempts=attempts,width=size[0],height=size[1])
        except timeout_error:
            attempts.append(dict(status='timeout',timeout_ms=timeout_ms))
            pending.unlink(missing_ok=True)
            if attempt:raise
            viewport=page.viewport_size
            smaller=dict(width=min(viewport['width'],1024),height=min(viewport['height'],768))
            print('Screenshot compositor timed out; retrying the unchanged scene at '+str(smaller),flush=True)
            page.set_viewport_size(smaller);page.wait_for_timeout(3000)
            if on_resize is not None:on_resize()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=0,help='Private capture port; default selects a fresh unused pair')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/media')
    parser.add_argument('--width',type=int,default=1600)
    parser.add_argument('--height',type=int,default=1000)
    parser.add_argument('--screenshot-timeout',type=float,default=180.,help='Maximum seconds per screenshot attempt')
    parser.add_argument('--serve-preview',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.width<800 or args.height<600 or not 0<args.screenshot_timeout<=600:
        parser.error('Use a viewport of at least 800x600 and a screenshot timeout of 1-600 seconds')
    require_resources()
    config=load(ROOT/'configs/platform/town02_strict_itu_64tx_40ue.json')
    if args.serve_preview:
        if not 1024<=args.port<=65534:parser.error('The private preview needs an explicit unprivileged port')
        return run_capture_preview(config,args.port,prepare=prepare_capture_views)
    if args.port and not 1024<=args.port<=65534:parser.error('port must be 0 or between 1024 and 65534')
    args.port=choose_capture_port(args.port)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    work=ROOT/'artifacts/town02-gallery';work.mkdir(parents=True,exist_ok=True)
    environment=dict(os.environ,WIWO_GPU='-1',PYTHONUNBUFFERED='1',
                     CUDA_VISIBLE_DEVICES='-1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                     WIWO_PYTHON=str(ROOT/'.venv-full/bin/python'),
                     WIWO_TOWN02_CONFIG=str(ROOT/'configs/platform/town02_strict_itu_64tx_40ue.json'),
                     PLAYWRIGHT_BROWSERS_PATH=str(ROOT/'.cache/ms-playwright'))
    os.environ['PLAYWRIGHT_BROWSERS_PATH']=environment['PLAYWRIGHT_BROWSERS_PATH']

    def api(path='state'):
        with urlopen(f'http://127.0.0.1:{args.port+1}/api/'+path,timeout=30) as response:
            return json.load(response)

    def wait_for(predicate,seconds=180):
        deadline=time.monotonic()+seconds;last=None
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('Preview exited; inspect artifacts/town02-gallery/server.log')
            try:
                last=predicate()
                if last:return last
            except (OSError,ValueError):pass
            time.sleep(.2)
        raise TimeoutError('Timed out waiting for the actual Town02 UI: '+str(last))

    report=dict(status='capturing',scene='Town02',mode='physical_preview_no_synthetic_CSI',
                captured_utc=datetime.now(timezone.utc).isoformat(),native_CSI_computed=False,
                config_scene=config['scene'],images=[],page_errors=[],
                browser_channel='chromium',rendering='software',preview_update_hz=2,
                screenshot_timeout_s=args.screenshot_timeout)
    print(f'Starting isolated Town02 preview on private ports {args.port}/{args.port+1}; loading the complete map...',flush=True)
    with (work/'server.log').open('ab') as log:
        process=subprocess.Popen([str(ROOT/'.venv-full/bin/python'),str(Path(__file__).resolve()),
                                  '--serve-preview','--port',str(args.port)],
                                 cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    try:
        wait_for(lambda:api('ui').get('map_faces',0)>0)
        actual=api('config')
        if actual['scene']!=config['scene']:raise ValueError('Capture service is not using the current strict Town02 scene')
        from playwright.sync_api import sync_playwright,TimeoutError as PlaywrightTimeoutError
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch(channel='chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage',
                '--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
            with closing(browser):
                page=browser.new_page(viewport=dict(width=args.width,height=args.height),device_scale_factor=1)
                page.set_default_timeout(90000)
                report['browser_version']=browser.version
                page.on('pageerror',lambda error:report['page_errors'].append(str(error)))
                page.goto(f'http://127.0.0.1:{args.port}',wait_until='domcontentloaded')
                page.get_by_role('button',name='释放平台操作权',exact=True).wait_for(timeout=180000)
                wait_for(lambda:len(api('ui').get('cameras',[]))>0)
                notice=page.get_by_text('Software WebGL rendering detected',exact=True)
                if notice.count():notice.locator('xpath=../..').get_by_role('button').click()
    
                def button(label):
                    control=page.get_by_role('button',name=label,exact=True)
                    for _ in range(40):
                        if control.is_visible():break
                        closed=page.locator('button[aria-expanded="false"]')
                        if not closed.count():break
                        closed.first.click();page.wait_for_timeout(100)
                    control.scroll_into_view_if_needed();control.click()
                    page.wait_for_timeout(1500)
    
                def top():
                    page.get_by_role('heading',name='WiWo MA / 6DMA',exact=True).scroll_into_view_if_needed()
    
                if len(api('ui').get('cameras',[]))!=1:
                    raise RuntimeError('The private capture service has another browser connected; choose a fresh port')
                print('Town02 is loaded. Capture preview rate: 2 Hz; native CSI: not computed.',flush=True)
    
                def shot(name,title,view):
                    state=api()['state'];telemetry=api('ui')
                    if api('csi')['available']:raise ValueError('Gallery must remain an actual mechanics preview without CSI')
                    if telemetry['last_action_error']:raise ValueError(telemetry['last_action_error'])
                    np.testing.assert_allclose(telemetry['displayed_tx_positions'],np.asarray(state['tx_port_positions'],np.float32),atol=0,rtol=0)
                    top();page.wait_for_timeout(2000)
                    print('Capturing '+name+'; allowing '+str(args.screenshot_timeout)+' seconds per attempt...',flush=True)
                    def reframe():
                        button('Capture view: '+view);top();page.wait_for_timeout(2000)
                    capture=save_browser_screenshot(page,output/name,args.screenshot_timeout*1000,PlaywrightTimeoutError,on_resize=reframe)
                    if api()['state']['timestamp']!=state['timestamp']:
                        raise AssertionError('Physical time changed during a supposedly paused capture')
                    report['images'].append(dict(file=name,title=title,physical_time_s=state['timestamp'],
                        mode=report['mode'],view=view,capture=capture,cameras=api('ui')['cameras'],
                        scene_faces=telemetry['map_faces'],sha256=hashlib.sha256((output/name).read_bytes()).hexdigest()))
                    (work/'capture.json').write_text(json.dumps(report,indent=2)+'\n')
                    print('Captured '+name,flush=True)
    
                button('Capture view: panorama')
                shot('town02-panorama.png','Town02 panorama in the running Web platform','panorama')
                button('Capture view: topdown')
                shot('town02-topdown.png','Town02 street layout from above','topdown')
                button('Capture view: site')
                shot('town02-site.png','Town02 blocks around the actual base station','site')
                button('Capture view: antennas')
                initial=api()['state']
                shot('town02-antennas.png','Town02 movable antenna panels and physical TX ports before motion','antennas')
                button('Capture view: detail')
                shot('town02-panel-detail.png','Selected movable antenna panel and its four physical TX ports','detail')
                button('Start / 开始预览')
                wait_for(lambda:api()['state']['timestamp']>=1.)
                button('Pause / 暂停')
                final=api()['state']
                if np.array_equal(initial['positions'],final['positions']):raise AssertionError('Scheduled antenna motion did not execute')
                report['motion']=dict(initial_panel_positions=initial['positions'],final_panel_positions=final['positions'],
                                      initial_time_s=initial['timestamp'],final_time_s=final['timestamp'])
                button('Capture view: platform')
                shot('town02-platform.png','Running Town02 platform after the scheduled bounded antenna motion','platform')
                if report['page_errors']:raise AssertionError(report['page_errors'])
        report['status']='Passed_actual_Town02_browser_gallery'
    except BaseException as error:
        report.update(status='Failed_preserved',error=str(error));raise
    finally:
        (work/'capture.json').write_text(json.dumps(report,indent=2)+'\n')
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10)
    (output/'town02-gallery.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Town02 screenshots and provenance saved in '+str(output),flush=True)


if __name__=='__main__':main()
