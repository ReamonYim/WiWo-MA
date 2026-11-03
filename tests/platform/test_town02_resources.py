"""External-resource discovery and a tiny, real PLY import for public CI.

The fixture geometry is authored here: 25 independent quadrilaterals, not a
reduced or redistributed Town02 map. It exercises the real importer, strict
material conversion, display mesh reader and physical configurations.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pytest

from wiwo_ma.platform import config,town02,town02_materials,scene_geometry
from wiwo_ma.platform.physics import PhysicalWorld
from wiwo_ma.platform.town02_resources import resource_status,require_resources

REAL_ROOT=Path(__file__).resolve().parents[2]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def tiny_package(tmp_path,monkeypatch):
    project=tmp_path/'project';source=tmp_path/'source';originals=source/'Town2_xml'
    (originals/'meshes').mkdir(parents=True)
    (project/'configs/materials').mkdir(parents=True)
    (project/'configs/platform').mkdir(parents=True)
    rules=REAL_ROOT/'configs/materials/town02_strict_itu.json'
    shutil.copy2(rules,project/'configs/materials'/rules.name)
    ids=[row['source_material_id'] for row in json.loads(rules.read_text())['rules']]
    ids+=['mat-M_GlassContainer','mat-M_Window','mat-M_Aluminium','mat-M_Plastic']
    baseline=ET.Element('scene',version='2.1.0');original=ET.Element('scene',version='2.1.0')
    for ident,kind in [('mat_brick','brick'),('mat_wood','wood'),('mat_floorboard','concrete')]:
        material=ET.SubElement(baseline,'bsdf',type='itu-radio-material',id=ident)
        ET.SubElement(material,'string',name='type',value=kind)
    for i,ident in enumerate(ids):
        material=ET.SubElement(original,'bsdf',type='principled',id=ident)
        ET.SubElement(material,'float',name='metallic',value='1')
        name=f'meshes/quad_{i:02d}.ply';x=100+i%5;y=-110+i//5
        data=('ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n'
              'element face 2\nproperty list uchar int vertex_indices\nend_header\n'
              f'{x} {y} 0\n{x+1} {y} 0\n{x+1} {y+1} 0\n{x} {y+1} 0\n3 0 1 2\n3 0 2 3\n')
        (originals/name).write_text(data)
        for root,material_id in [(baseline,'mat_brick'),(original,ident)]:
            shape=ET.SubElement(root,'shape',type='ply',id=f'quad_{i:02d}')
            ET.SubElement(shape,'string',name='filename',value=name)
            ET.SubElement(shape,'ref',name='bsdf',id=material_id)
    for root,name in [(baseline,'town2_3mat.xml'),(original,'town2_xml.xml')]:
        (originals/name).write_bytes(ET.tostring(root,encoding='utf-8',xml_declaration=True))
    trajectory=originals/town02.TRAJECTORY;trajectory.parent.mkdir(parents=True)
    with trajectory.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['x_b','y_b','z_b']);writer.writeheader()
        for i in range(81):writer.writerow(dict(x_b=100+i*.1,y_b=-108,z_b=1.5))
    (originals/'material_manager.py').write_text('# Synthetic fixture; never executed.\n')
    files=[dict(path=str(p.relative_to(source)),bytes=p.stat().st_size,sha256=digest(p))
           for p in sorted(originals.rglob('*')) if p.is_file()]
    (source/'资源SHA256.json').write_text(json.dumps(dict(files=files)))
    for module in [config,town02,town02_materials,scene_geometry]:monkeypatch.setattr(module,'ROOT',project)
    monkeypatch.setattr(town02,'ASSET_ROOT',project/'assets/town02')
    return project,source,files


def test_committed_configuration_does_not_imply_actual_map_availability(tmp_path):
    path=tmp_path/'configs/platform/town02_strict_itu_64tx_40ue.json';path.parent.mkdir(parents=True)
    path.write_text('{}')
    report=resource_status(tmp_path)
    assert report['status']=='not_imported'
    assert 'assets/town02/town2_3mat.xml' in report['missing']
    assert 'assets/town02/town2_xml.xml' in report['missing']
    with pytest.raises(FileNotFoundError,match='import --source'):require_resources(tmp_path)


def test_existing_manifest_without_xml_is_an_incomplete_import(tmp_path):
    folder=tmp_path/'assets/town02';folder.mkdir(parents=True)
    (folder/'import-manifest.json').write_text(json.dumps(dict(files=[dict(path='meshes/missing.ply')])))
    report=resource_status(tmp_path)
    assert report['status']=='incomplete'
    assert 'assets/town02/town2_3mat.xml' in report['missing']
    assert 'assets/town02/meshes/missing.ply' in report['missing']


def test_invalid_manifest_is_an_error_instead_of_optional_absence(tmp_path):
    folder=tmp_path/'assets/town02';folder.mkdir(parents=True)
    (folder/'import-manifest.json').write_text('{')
    report=resource_status(tmp_path)
    assert report['status']=='incomplete' and report['errors']


def test_tiny_real_ply_import_exercises_strict_materials_display_and_mechanics(tiny_package):
    project,source,files=tiny_package
    report=town02.import_resources(source)
    assert report['unique_meshes']==25 and report['verified_imported_files']==len(files)
    assert resource_status(project)['status']=='available'
    strict=town02_materials.check_variant()
    assert set(strict['materials'].values())=={'brick','concrete','wood','glass','metal'}
    assert strict['counts']['total_meshes']==25
    assert strict['counts']['matched_meshes']==21 and strict['counts']['changed_meshes']==17
    assert strict['counts']['unchanged_unverified_meshes']==4
    full=town02_materials.example('full')
    meshes,_=scene_geometry.load_display_meshes(str(scene_geometry.scene_xml(full['scene'])))
    assert len(meshes)==25 and sum(len(m['vertices']) for m in meshes)==100
    assert sum(len(m['faces']) for m in meshes)==50
    world=PhysicalWorld(full)
    assert world.state()['tx_port_positions'].shape==(64,3)
    assert world.state()['user_positions'].shape==(40,3)
    single=PhysicalWorld(town02_materials.example('single'));start=single.state()['positions'][0].copy()
    end=single.advance_to(1.)['positions'][0]
    assert end[0]-start[0]==pytest.approx(.02)


@pytest.mark.parametrize('name',['town2_3mat.xml','town2_xml.xml','town2_strict_itu.xml',
                                  'town2_strict_itu.materials.json','meshes/quad_00.ply'])
def test_each_actual_runtime_resource_is_required_even_when_configs_exist(tiny_package,name):
    project,source,_=tiny_package;town02.import_resources(source)
    (project/'assets/town02'/name).unlink()
    report=resource_status(project)
    assert report['status']=='incomplete'
    assert 'assets/town02/'+name in report['missing']
    with pytest.raises(FileNotFoundError):require_resources(project)


def test_invalid_source_does_not_overwrite_a_working_import(tiny_package):
    project,source,_=tiny_package;town02.import_resources(source)
    original=project/'assets/town02/meshes/quad_00.ply';before=digest(original)
    (source/'Town2_xml/meshes/quad_00.ply').write_text('corrupted source')
    with pytest.raises(ValueError,match='checksum mismatch'):town02.import_resources(source)
    assert digest(original)==before and resource_status(project)['status']=='available'


def test_missing_source_package_has_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError,match='WIWO_TOWN02_SOURCE'):
        town02.import_resources(tmp_path/'absent')


def test_missing_town02_scene_reports_import_command_before_loading_a_backend(tiny_package):
    project,_,_=tiny_package
    preset=json.loads((REAL_ROOT/'configs/platform/town02_strict_itu_single_ma.json').read_text())
    assert not (project/preset['scene']['xml_path']).exists()
    with pytest.raises(ValueError,match='Town02 scene XML is missing.*import --source'):
        config.validate(preset)


def test_missing_referenced_source_mesh_cannot_damage_a_working_import(tiny_package):
    project,source,_=tiny_package;town02.import_resources(source)
    imported=project/'assets/town02/meshes/quad_00.ply';before=digest(imported)
    (source/'Town2_xml/meshes/quad_00.ply').unlink()
    with pytest.raises(FileNotFoundError,match='source mesh'):town02.import_resources(source)
    assert digest(imported)==before


def test_panorama_uses_actual_world_bounds_without_moving_the_system(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from wiwo_ma.platform import web
    ui=object.__new__(web.PlatformUI);scene=dict(name='town02');seen=[]
    ui.controller=SimpleNamespace(config={'scene':scene})
    ui.server=SimpleNamespace(get_clients=lambda:{1:'viewer'})
    monkeypatch.setattr(web,'scene_xml',lambda current:'actual.xml' if current is scene else None)
    monkeypatch.setattr(web,'load_display_meshes',lambda path:([
        dict(vertices=np.array([[100,-200,0],[130,-180,10]],float))],[]))
    monkeypatch.setattr(ui,'_aim',lambda client,center,offset:seen.append((client,center,offset)))
    ui._overview()
    assert seen[0][0]=='viewer'
    np.testing.assert_allclose(seen[0][1],[115,-190,5])
    assert np.linalg.norm(seen[0][2])>30
    assert ui.controller.config['scene'] is scene
