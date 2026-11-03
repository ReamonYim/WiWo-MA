"""Local Town02 resource integrity and nonzero-world-coordinate regressions."""
from copy import deepcopy
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from wiwo_ma.platform.config import validate
from wiwo_ma.platform.physics import PhysicalWorld
from wiwo_ma.platform.scene_geometry import scene_xml
from wiwo_ma.platform.town02 import ASSET_ROOT, example, resource_report


LOCAL_ASSETS = pytest.mark.skipif(not (ASSET_ROOT/'import-manifest.json').is_file(),
                                 reason='User-supplied Town02 resources are local assets')


@LOCAL_ASSETS
def test_copied_map_meshes_and_three_original_ITU_materials_are_intact():
    report = resource_report()
    assert report['shapes'] == report['unique_meshes'] == 6085
    assert report['materials'] == dict(mat_wood='wood',mat_floorboard='concrete',mat_brick='brick')
    assert report['material_bindings'] == dict(mat_wood=551,mat_floorboard=614,mat_brick=4920)
    assert report['verified_imported_files'] == 8635


@LOCAL_ASSETS
def test_single_MA_reaches_BS_local_target_in_the_original_Town02_world_frame():
    config = example('single')
    world = PhysicalWorld(config)
    np.testing.assert_allclose(world.state()['positions'][0],[103.211403,-102.730927,10.],rtol=0,atol=1e-12)
    final = world.advance_to(1.)
    np.testing.assert_allclose(final['positions'][0],[103.231403,-102.730927,10.],rtol=0,atol=1e-10)
    assert final['configuration_sha256'] == world.configuration


@LOCAL_ASSETS
def test_full_Town02_preserves_physical_port_counts_and_bounded_road_motion():
    config = example()
    state = PhysicalWorld(config).state()
    assert state['tx_port_positions'].shape == (64,3)
    assert state['user_positions'].shape == (40,3)
    assert config['scene']['name'] == 'town02'
    assert 'material_override' not in config['scene']
    assert all(user['motion']['model'] == 'bounded_quintic' for user in config['users'])


@LOCAL_ASSETS
def test_preview_and_native_loader_resolve_the_same_XML_from_another_CWD(tmp_path,monkeypatch):
    from wiwo_ma.scene import load_asset

    config = example('single')
    expected = scene_xml(config['scene'])
    seen = []

    class PathCaptured(Exception):
        pass

    def capture(path,**kwargs):
        seen.append(path)
        raise PathCaptured()

    rt = SimpleNamespace(load_scene=capture)
    monkeypatch.setitem(sys.modules,'sionna',SimpleNamespace(rt=rt))
    monkeypatch.setitem(sys.modules,'sionna.rt',rt)
    monkeypatch.setitem(sys.modules,'drjit',SimpleNamespace())
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PathCaptured):
        load_asset(config)
    assert seen == [str(expected)]


@LOCAL_ASSETS
@pytest.mark.parametrize('xml_path',[None,'assets/town02/town2_3mat.xml'])
def test_Town02_default_XML_is_explicit_in_validated_configuration(xml_path):
    config = example('single')
    config['scene']['xml_path'] = xml_path
    assert validate(config)['scene']['xml_path'] == 'assets/town02/town2_3mat.xml'


@LOCAL_ASSETS
def test_scene_validation_rejects_an_external_XML_and_keeps_historical_config_unchanged(tmp_path):
    config = example('single')
    foreign = tmp_path/'map.xml'
    foreign.write_text('<scene/>')
    config['scene']['xml_path'] = str(foreign)
    with pytest.raises(ValueError,match='inside project'):
        validate(config)
    from wiwo_ma.platform.examples import reflector_example
    original = reflector_example()
    assert validate(deepcopy(original)) == original


def test_resource_verification_detects_mesh_changes_with_valid_XML_bindings(tmp_path):
    xml = '<scene><bsdf type="itu-radio-material" id="mat_wood"><string name="type" value="wood"/></bsdf><shape type="ply"><string name="filename" value="mesh.ply"/><ref id="mat_wood"/></shape></scene>'
    (tmp_path/'town2_3mat.xml').write_text(xml)
    data = b'synthetic checksum fixture'
    (tmp_path/'mesh.ply').write_bytes(data)
    manifest = dict(files=[dict(path='mesh.ply',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())])
    (tmp_path/'import-manifest.json').write_text(json.dumps(manifest))
    resource_report(tmp_path)
    (tmp_path/'mesh.ply').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Imported resource changed'):
        resource_report(tmp_path)
