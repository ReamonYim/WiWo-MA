#!/usr/bin/env python3
"""Actual browser gallery, from a supplied real full401 HDF and CPU mechanics.

Use a dedicated service/port: this script changes that service's preview and
replay state. No native RT call, fake CSI, interpolation or screenshot synthesis.
The footer is an explicit presentation annotation with source attribution.
"""
import argparse,hashlib,json,time
from pathlib import Path
from urllib.request import urlopen
import numpy as np
from scipy.spatial.transform import Rotation
from wiwo_ma.platform.config import ROOT
from wiwo_ma.platform.storage import CSIReader

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--port',type=int,default=8190);p.add_argument('--output',type=Path,default=Path('docs/media'));p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    dataset=a.dataset.resolve();assert dataset.is_relative_to(ROOT),'Use an authorized local copy inside the source tree'
    r=CSIReader(dataset);assert r.complete and r.count==401 and r.frame(0)['H'].shape[:3]==(40,1,64)
    def api(path='/api/state'):return json.load(urlopen(f'http://127.0.0.1:{a.port+1}'+path,timeout=40))
    def wait(fn,seconds=50):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            q=api()
            if fn(q):return q
            time.sleep(.03)
        raise AssertionError('Actual state receipt timeout')
    # Read the actual recorded LOS metadata, not a fabricated event time.
    counts=[]
    import h5py
    with h5py.File(dataset,'r') as f:
        for raw in f['frames_json'].asstr():counts.append(json.loads(raw)['result']['direct_LOS_port_counts'][0])
    blocked=next(i for i,n in enumerate(counts) if n==0);assert counts[0]==counts[-1]==64
    report=dict(status='Unknown',new_GPU_seconds=0,images=[],page_errors=[],blocked_frame=blocked,LOS_UE0=[counts[0],counts[blocked],counts[-1]],notes='Footer annotations identify mode/time/attribution. Geometry/CSI are actual UI outputs.')
    a.output.mkdir(parents=True,exist_ok=True);a.report.parent.mkdir(parents=True,exist_ok=True)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']);page=browser.new_page(viewport={'width':1680,'height':1100});page.on('pageerror',lambda e:report['page_errors'].append(str(e)))
        def button(label):
            n=page.get_by_role('button',name=label,exact=True);n.scroll_into_view_if_needed();n.click()
        def select(label,value):
            n=page.get_by_role('combobox',name=label,exact=True);n.scroll_into_view_if_needed();n.click();page.get_by_role('option',name=value,exact=True).click()
        def field(label,value):
            n=page.get_by_text(label,exact=True).locator('xpath=../../..').locator('input').first;n.scroll_into_view_if_needed();n.fill(str(value));n.press('Tab')
        def vector(label,values):
            nodes=page.get_by_text(label,exact=True).locator('xpath=../../..').locator('input');nodes.first.scroll_into_view_if_needed()
            for i,v in enumerate(values):nodes.nth(i).fill(str(v))
        def recorded(index):
            if index in (0,400):
                n=page.get_by_text('回放物理时间s',exact=True).locator('xpath=../../..').locator('[role="slider"]');n.scroll_into_view_if_needed();n.focus();n.press('Home' if index==0 else 'End')
            else:
                n=page.get_by_text('回放物理时间s',exact=True).locator('xpath=../../..').locator('[role="slider"]');n.scroll_into_view_if_needed();n.focus();n.press('Home')
                for _ in range(index):n.press('ArrowRight')
            button('跳到时间轴所选帧');wait(lambda q:q['statistics'].get('replay_index')==index)
            expected=r.frame(index);h=api('/api/csi');np.testing.assert_array_equal(np.asarray(h['real'])+1j*np.asarray(h['imag']),expected['H'])
            wait(lambda q:api('/api/ui')['displayed_geometry_time_s']==expected['state']['timestamp'])
        def shot(name,title):
            q=api();mode=q['statistics']['mode'];t=q['state']['timestamp']
            footer=f'{title} | {mode} | physical t={t:.3f} s | No new RT during capture | Map data © OpenStreetMap contributors, ODbL · openstreetmap.org/copyright · Sionna RT Munich'
            page.evaluate('''text=>{let x=document.getElementById('wiwo-gallery-credit');if(!x){x=document.createElement('div');x.id='wiwo-gallery-credit';document.body.appendChild(x)}x.textContent=text;Object.assign(x.style,{position:'fixed',bottom:'0',left:'0',right:'0',zIndex:2147483646,background:'#0b1220',color:'#f8fafc',font:'13px sans-serif',padding:'9px',borderTop:'1px solid #64748b',pointerEvents:'none'})}''',footer)
            page.evaluate('()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))')
            path=a.output/name
            if path.exists():raise FileExistsError(path)
            page.screenshot(path=str(path));report['images'].append(dict(file=name,title=title,physical_time_s=t,mode=mode,CSI_source='Recorded native HDF5' if mode=='stored_replay' else 'No CSI computed',sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            a.report.write_text(json.dumps(report,indent=2)+'\n');print(name,mode,t,flush=True)
        try:
            page.goto(f'http://127.0.0.1:{a.port}');page.get_by_role('button',name='释放平台操作权',exact=True).wait_for(timeout=90000)
            notice=page.get_by_text('Software WebGL rendering detected',exact=True)
            if notice.count():notice.locator('xpath=../..').get_by_role('button').click()
            button('Pause / 暂停');recorded(0)
            wait(lambda q:api('/api/ui')['map_faces']>30000)
            page.mouse.move(500,450);page.mouse.wheel(0,800);page.evaluate('()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');shot('01_munich_overview.png','Actual Munich scene · 16 panels / 64 TX / 40 UE')
            button('聚焦选中面板');shot('02_full64_array.png','Actual stored antenna geometry · 64 physical TX ports')
            # A real physical preset, not replay, for action demonstration.
            select('预设','Munich continuous / 64TX / K40 / 3.5GHz / 100Hz');button('加载所选预设');wait(lambda q:q['statistics']['mode']=='physical_preview_no_synthetic_CSI')
            c=api('/api/config');c['actions']=[]
            box=page.get_by_text('完整配置 / Action / 轨迹 JSON',exact=True).locator('xpath=../../..').locator('textarea');box.scroll_into_view_if_needed();box.fill(json.dumps(c));button('验证并加载配置JSON');wait(lambda q:q['state']['timestamp']==0)
            initial=api()['state'];key=initial['panel_ids'][0];select('选择面板',' / '.join(key));select('目标输入坐标','constrained_BS_local')
            vector('目标相对初始BS局部位移m',[.006,0,.003]);vector('目标相对初始BS局部旋转向量rad',[0,0,.003]);shot('03_mechanics_controls.png','Mechanical input draft · bounded translation and rotation')
            button('执行目标姿态');button('Start / 开始预览');wait(lambda q:q['state']['timestamp']>=.8 and q['state']['mechanics'][0]['completed']);button('Pause / 暂停')
            final=api()['state'];br=np.asarray(initial['bs_rotations'][0]);delta=br.T@(np.asarray(final['positions'][0])-initial['positions'][0]);np.testing.assert_allclose(delta,[.006,0,.003],atol=1e-10,rtol=0)
            expected=br@Rotation.from_rotvec([0,0,.003]).as_matrix()@br.T@np.asarray(initial['rotations'][0]);np.testing.assert_allclose(final['rotations'][0],expected,atol=1e-10,rtol=0);assert not api('/api/csi')['available'];report['actual_manual_action']=dict(status='Passed_geometry_only',translation_m=delta.tolist(),rotation_vector_rad=[0,0,.003],CSI='Not_computed')
            button('聚焦选中面板');shot('04_panel_motion.png','Executed translation and rotation · geometry preview only')
            field('HDF5项目相对路径',str(dataset.relative_to(ROOT)));button('加载真实保存数据');wait(lambda q:q['statistics'].get('replay_frames')==401);recorded(blocked);button('聚焦选中面板');shot('05_actor_occlusion.png','Actual recorded actor occlusion · UE0 LOS ports 64 → 0 → 64')
            page.get_by_text('各TX端口相位rad',exact=True).scroll_into_view_if_needed();shot('06_csi_response.png','Actual complex CSI amplitude / phase and channel gain')
            button('Reset / 重置');button('Resume / 继续');wait(lambda q:q['statistics'].get('replay_index')==400 and not q['statistics']['playing'],90);recorded(400);shot('07_full_sequence_replay.png','Complete 401-sample HDF replay · physical interval 0–4 s')
            assert not report['page_errors'];report['status']='Passed_actual_browser_gallery_no_new_GPU'
        except BaseException as e:report.update(status='Failed_preserved',error=repr(e));page.screenshot(path=str(a.report.parent/'capture_failure.png'));raise
        finally:a.report.write_text(json.dumps(report,indent=2)+'\n');browser.close()
    public=dict(status=report['status'],images=report['images'],source='Frozen native full401 dataset; no new native query',LOS_UE0=report['LOS_UE0'],mode_annotation='Preview and recorded CSI explicitly distinguished',attribution='Map data © OpenStreetMap contributors, ODbL; Sionna RT Munich scene')
    (a.output/'gallery.json').write_text(json.dumps(public,indent=2)+'\n')

if __name__=='__main__':main()
