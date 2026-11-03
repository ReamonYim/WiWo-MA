"""Camera framing must expose the scene without changing physical geometry."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from wiwo_ma.platform.camera_views import fit_camera,padded_bounds


@pytest.mark.parametrize('aspect',[.7,1.2,2.2])
def test_world_points_fit_both_image_axes_with_an_explicit_margin(aspect):
    points=padded_bounds([[100,-220,0],[450,-30,45]],0);before=points.copy()
    view=fit_camera(points,[.12,-.48,1.],aspect=aspect)
    forward=view['look_at']-view['position'];forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward)
    relative=points-view['position'];depth=relative@forward
    assert np.min(depth)>0
    tangent=np.tan(np.deg2rad(view['fov_deg'])/2)
    screen=np.column_stack(((relative@right)/(depth*tangent*aspect),(relative@up)/(depth*tangent)))
    assert np.max(abs(screen))<=.86000001
    assert np.max(abs(screen))>=.85999999
    np.testing.assert_array_equal(points,before)


def test_camera_follows_translated_world_coordinates_without_scaling_the_scene():
    points=padded_bounds([[0,0,0],[2,3,4]],.15);translation=np.array([1200,-890,23])
    original=fit_camera(points,[0,1,1.1]);translated=fit_camera(points+translation,[0,1,1.1])
    for key in ('position','look_at'):np.testing.assert_allclose(translated[key]-original[key],translation)
    assert translated['distance_m']==pytest.approx(original['distance_m'])


@pytest.mark.parametrize('options',[{'aspect':0},{'fill':1},{'direction':[0,0,1]}])
def test_invalid_projection_is_rejected(options):
    parameters=dict(direction=[0,1,1]);parameters.update(options)
    with pytest.raises(ValueError):fit_camera([[0,0,0],[1,1,1]],**parameters)


def test_gallery_views_move_only_cameras_and_use_distinct_actual_scene_regions(monkeypatch):
    import importlib.util
    from pathlib import Path
    from wiwo_ma.platform import scene_geometry
    script=Path(__file__).resolve().parents[2]/'scripts/media/capture_town02.py'
    spec=importlib.util.spec_from_file_location('capture_views_test',script)
    capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)
    points=np.concatenate([padded_bounds([[-30,-50,0],[60,50,20]],0),
                           padded_bounds([[-20,-20,0],[20,20,10]],0)])
    state=dict(positions=np.column_stack((np.linspace(-1,1,16),np.zeros(16),np.linspace(9,11,16))),
               user_positions=np.column_stack((np.linspace(-15,25,40),np.full(40,-10),np.ones(40))),
               bs_positions=[[0,0,10]],rotations=np.tile(np.eye(3),(16,1,1)))
    original=deepcopy(state);buttons={};labels=[];aims=[]
    def label(*args,**kwargs):
        handle=SimpleNamespace(**kwargs);labels.append(handle);return handle
    monkeypatch.setattr(scene_geometry,'scene_xml',lambda _: 'actual-scene.xml')
    monkeypatch.setattr(scene_geometry,'load_display_meshes',lambda _:([dict(vertices=points)],[]))
    ui=SimpleNamespace(controller=SimpleNamespace(config={'scene':{}},state=lambda:state),
                       server=SimpleNamespace(scene=SimpleNamespace(add_label=label),get_clients=lambda:{0:SimpleNamespace(camera=SimpleNamespace(aspect=1.2))}),
                       selected=0,gizmo=SimpleNamespace(visible=True),
                       _button=lambda name,callback:buttons.update({name:callback}),
                       _aim=lambda client,center,offset:aims.append((center.copy(),offset.copy())))
    capture.prepare_capture_views(ui)
    for name in ('panorama','topdown','site','antennas','detail','platform'):buttons['Capture view: '+name]()
    assert len(aims)==6
    assert not np.allclose(aims[0][1],aims[1][1])
    assert np.linalg.norm(aims[3][1])>=4.5
    assert np.linalg.norm(aims[4][1])<1
    assert labels[0].visible and labels[1].visible
    for key in original:np.testing.assert_array_equal(state[key],original[key])


def test_screenshot_retry_reframes_for_the_actual_smaller_viewport(tmp_path):
    import importlib.util
    from pathlib import Path
    from PIL import Image
    script=Path(__file__).resolve().parents[2]/'scripts/media/capture_town02.py'
    spec=importlib.util.spec_from_file_location('capture_retry_test',script)
    capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)
    events=[]
    class Page:
        viewport_size={'width':1600,'height':1000}
        def screenshot(self,**options):
            events.append('screenshot')
            if len(events)==1:raise TimeoutError('compositor stalled')
            Image.new('RGB',(self.viewport_size['width'],self.viewport_size['height'])).save(options['path'])
        def set_viewport_size(self,viewport):self.viewport_size=viewport;events.append('resize')
        def wait_for_timeout(self,ms):pass
    page=Page()
    def reframe():
        assert page.viewport_size==dict(width=1024,height=768)
        events.append('reframe')
    capture.save_browser_screenshot(page,tmp_path/'capture.png',180000,on_resize=reframe)
    assert events==['screenshot','resize','reframe','screenshot']
