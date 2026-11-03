"""Strict material provenance, conservative bindings, and tamper regressions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from wiwo_ma.platform import town02_materials as materials
from wiwo_ma.platform.itu_catalog import load_catalog
from wiwo_ma.platform.town02 import resource_report


REAL_ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_xml(path, root):
    path.write_bytes(ET.tostring(root,encoding='utf-8',xml_declaration=True))


@pytest.fixture
def local_variant(tmp_path,monkeypatch):
    monkeypatch.setattr(materials,'ROOT',tmp_path)
    folder = tmp_path/'assets/town02'
    (folder/'meshes').mkdir(parents=True)
    baseline = ET.Element('scene',version='2.1.0')
    original = ET.Element('scene',version='2.1.0')
    for ident, kind in [('mat_brick','brick'),('mat_wood','wood'),('mat_floorboard','concrete')]:
        node = ET.SubElement(baseline,'bsdf',type='itu-radio-material',id=ident)
        ET.SubElement(node,'string',name='type',value=kind)
    entries = [
        ('glass','mat-Glass','mat_brick'),
        ('container','mat-M_GlassContainer','mat_brick'),
        ('window','mat-M_Window','mat_wood'),
        ('alloy','mat-M_Aluminium','mat_floorboard'),
        ('plastic','mat-M_Plastic','mat_brick'),
    ]
    mesh_paths = []
    for name,source_id,before in entries:
        optical = ET.SubElement(original,'bsdf',type='principled',id=source_id)
        # Intentionally uninformative optical numbers must not drive RF rules.
        ET.SubElement(optical,'float',name='metallic',value='1')
        filename = 'meshes/'+name+'.ply'
        mesh = folder/filename
        mesh.write_bytes(('synthetic checksum fixture '+name).encode())
        mesh_paths.append(mesh)
        for root,ident in [(baseline,before),(original,source_id)]:
            shape = ET.SubElement(root,'shape',type='ply',id='mesh-'+name)
            ET.SubElement(shape,'string',name='filename',value=filename)
            ET.SubElement(shape,'ref',name='bsdf',id=ident)
    baseline_path = folder/materials.BASE_XML
    source_path = folder/materials.SOURCE_XML
    write_xml(baseline_path,baseline)
    write_xml(source_path,original)
    policy = dict(schema=materials.POLICY,frequency_hz=2_800_000_000,
                  scope='Original asset material labels; no measured physical identification',
                  rules=[dict(source_material_id='mat-Glass',itu_type='glass',
                              evidence='Original material label explicitly names Glass')],
                  excluded_material_ids={source_id:'Identity not approved; preserve baseline'
                                         for _,source_id,_ in entries if source_id!='mat-Glass'})
    rules_path = tmp_path/'configs/materials/policy.json'
    rules_path.parent.mkdir(parents=True)
    rules_path.write_text(json.dumps(policy))

    def seal_import_manifest():
        paths = [baseline_path,source_path,*mesh_paths]
        records = [dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]
        (folder/'import-manifest.json').write_text(json.dumps(dict(files=records)))

    seal_import_manifest()
    return SimpleNamespace(folder=folder,baseline=baseline_path,source=source_path,
                           rules=rules_path,meshes=mesh_paths,seal=seal_import_manifest)


def build(fixture):
    return materials.build_variant(folder=fixture.folder,rules_path=fixture.rules,write_configs=False)


def rebind_xml_and_audit(fixture,scene):
    """Update editable hashes, leaving independent provenance rows unchanged."""
    scene = deepcopy(scene)
    report_path = fixture.folder/materials.AUDIT_NAME
    report = json.loads(report_path.read_text())
    scene['xml_sha256'] = sha(fixture.folder/materials.STRICT_XML)
    report['output_xml_sha256'] = scene['xml_sha256']
    report_path.write_text(json.dumps(report))
    scene['material_audit_sha256'] = sha(report_path)
    return scene


def test_exact_source_allowlist_changes_only_glass_and_preserves_geometry(local_variant):
    fixture = local_variant
    before = {p:sha(p) for p in [fixture.baseline,fixture.source,*fixture.meshes]}
    scene,report = build(fixture)
    rows = {r['source_material_id']:r for r in report['meshes']}
    glass = rows['mat-Glass']
    assert glass['itu_type']=='glass' and glass['changed']
    assert glass['status']=='source_label_confirmed'
    for ident in ['mat-M_GlassContainer','mat-M_Window','mat-M_Aluminium','mat-M_Plastic']:
        row = rows[ident]
        assert row['status']=='unchanged_unverified' and not row['changed']
        assert row['material_id']==row['baseline_material_id']
    assert report['counts']['matched_meshes']==report['counts']['changed_meshes']==1
    assert report['counts']['unchanged_unverified_meshes']==4
    assert before=={p:sha(p) for p in before}
    original_geometry = materials._geometry_digest(ET.parse(fixture.baseline).getroot())
    new_geometry = materials._geometry_digest(ET.parse(fixture.folder/materials.STRICT_XML).getroot())
    assert original_geometry==new_geometry==report['geometry_sha256']
    assert resource_report(fixture.folder)['verified_imported_files']==7
    materials.validate_strict_scene(scene,2.8e9)


def test_only_builtin_models_and_sdk_defaults_are_written(local_variant):
    fixture = local_variant
    _,report = build(fixture)
    root = ET.parse(fixture.folder/materials.STRICT_XML).getroot()
    for node in root.findall('bsdf'):
        assert node.get('type')=='itu-radio-material'
        assert [(child.tag,child.attrib) for child in node]==[
            ('string',dict(name='type',value=report['materials'][node.get('id')]))]
    assert report['parameters']['glass']['relative_permittivity']==6.31
    assert report['parameters']['glass']['conductivity_S_per_m']==pytest.approx(.0036*2.8**1.3394)
    assert report['itu_catalog_sha256']==load_catalog()['source_sha256']


@pytest.mark.parametrize('target',['xml','audit','rules','source','baseline'])
def test_changed_provenance_or_variant_files_are_rejected(local_variant,target):
    fixture = local_variant
    scene,_ = build(fixture)
    path = dict(xml=fixture.folder/materials.STRICT_XML,audit=fixture.folder/materials.AUDIT_NAME,
                rules=fixture.rules,source=fixture.source,baseline=fixture.baseline)[target]
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='checksum|source changed'):
        materials.validate_strict_scene(scene,2.8e9)


def test_changed_imported_mesh_is_rejected_before_build(local_variant):
    fixture = local_variant
    fixture.meshes[0].write_bytes(b'modified geometry bytes')
    with pytest.raises(ValueError,match='Imported resource changed'):
        build(fixture)


def test_changed_imported_mesh_is_rejected_after_build_with_full_verification(local_variant):
    fixture = local_variant
    scene,_ = build(fixture)
    fixture.meshes[0].write_bytes(b'modified geometry bytes after variant creation')
    with pytest.raises(ValueError,match='Imported resource changed'):
        materials.validate_strict_scene(scene,2.8e9,verify_meshes=True)


def test_changed_import_manifest_is_rejected_after_build(local_variant):
    fixture = local_variant
    scene,_ = build(fixture)
    path = fixture.folder/'import-manifest.json'
    manifest = json.loads(path.read_text())
    manifest['files'][2]['sha256']='0'*64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='resource manifest changed'):
        materials.validate_strict_scene(scene,2.8e9,verify_meshes=True)


def test_cached_semantic_validation_rechecks_source_hash_before_return(local_variant,monkeypatch):
    fixture = local_variant
    monkeypatch.setattr(materials,'_VALIDATED_BUNDLES',{})
    scene,_ = build(fixture)
    materials.validate_strict_scene(scene,2.8e9,verify_meshes=False)

    def unexpected_parse(*args,**kwargs):
        raise AssertionError('Cached semantic validation must not parse XML again')

    monkeypatch.setattr(materials.ET,'parse',unexpected_parse)
    materials.validate_strict_scene(scene,2.8e9,verify_meshes=False)
    materials.validate_strict_scene(scene,2.8e9,verify_meshes=False)
    fixture.source.write_bytes(fixture.source.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='source changed: source_xml'):
        materials.validate_strict_scene(scene,2.8e9,verify_meshes=False)


def test_global_material_override_is_rejected(local_variant):
    scene,_ = build(local_variant)
    scene['material_override']='metal'
    with pytest.raises(ValueError,match='global material overrides'):
        materials.validate_strict_scene(scene,2.8e9)


def test_carrier_change_outside_any_retained_builtin_band_is_rejected(local_variant):
    scene,_ = build(local_variant)
    # Retained brick is defined at 1-40 GHz and 110-330 GHz, not at 70 GHz.
    with pytest.raises(ValueError,match='brick.*not defined'):
        materials.validate_strict_scene(scene,70e9)


@pytest.mark.parametrize('field,value,message',[
    ('itu_type','rubber','Unknown Sionna ITU material'),
    ('source_material_id','mat-*','patterns are forbidden'),
    ('evidence','','undocumented material rule'),
])
def test_invalid_or_undocumented_assignment_rules_are_rejected(local_variant,field,value,message):
    fixture = local_variant
    policy = json.loads(fixture.rules.read_text())
    policy['rules'][0][field]=value
    fixture.rules.write_text(json.dumps(policy))
    with pytest.raises(ValueError,match=message):
        build(fixture)
    assert not (fixture.folder/materials.STRICT_XML).exists()


def test_duplicate_original_filename_cannot_choose_an_arbitrary_material(local_variant):
    fixture = local_variant
    root = ET.parse(fixture.source).getroot()
    duplicate = deepcopy(root.findall('shape')[0])
    duplicate.set('id','another-original-object')
    duplicate.find('ref').set('id','mat-M_Plastic')
    root.append(duplicate)
    write_xml(fixture.source,root)
    fixture.seal()
    with pytest.raises(ValueError,match='Ambiguous duplicate source mesh'):
        build(fixture)


def test_manual_EM_coefficients_are_rejected_even_with_rebound_editable_hashes(local_variant):
    fixture = local_variant
    scene,_ = build(fixture)
    path = fixture.folder/materials.STRICT_XML
    root = ET.parse(path).getroot()
    ET.SubElement(root.find("bsdf[@id='mat_glass']"),'float',name='relative_permittivity',value='2.1')
    write_xml(path,root)
    scene = rebind_xml_and_audit(fixture,scene)
    with pytest.raises(ValueError,match='SDK parameter defaults'):
        materials.validate_strict_scene(scene,2.8e9)


def test_unapproved_binding_is_rejected_even_with_rebound_editable_hashes(local_variant):
    fixture = local_variant
    scene,_ = build(fixture)
    path = fixture.folder/materials.STRICT_XML
    root = ET.parse(path).getroot()
    root.find("shape[@id='mesh-plastic']/ref").set('id','mat_glass')
    write_xml(path,root)
    scene = rebind_xml_and_audit(fixture,scene)
    with pytest.raises(ValueError):
        materials.validate_strict_scene(scene,2.8e9)


def test_map_ticks_reuse_verified_scene_but_configuration_change_resolves_again(monkeypatch,tmp_path):
    from wiwo_ma.platform import web

    resolved = []
    loaded = []

    def resolve(scene):
        resolved.append(deepcopy(scene))
        return tmp_path/'strict.xml'

    def display_meshes(xml):
        loaded.append(xml)
        return [],[]

    monkeypatch.setattr(web,'scene_xml',resolve)
    monkeypatch.setattr(web,'load_display_meshes',display_meshes)
    ui = object.__new__(web.PlatformUI)
    ui.controller = SimpleNamespace(config={'scene':dict(name='town02',material_audit_sha256='first')})
    ui.map_signature = None

    ui._map({})
    ui._map({})
    # Equivalent imported configuration should preserve the cache, regardless of key order.
    ui.controller.config['scene'] = dict(material_audit_sha256='first',name='town02')
    ui._map({})
    assert len(resolved)==len(loaded)==1
    assert ui.map_faces==ui.map_full_faces==0

    # A changed scene must verify again even when its XML path remains identical.
    ui.controller.config['scene']['material_audit_sha256']='second'
    ui._map({})
    ui._map({})
    assert len(resolved)==len(loaded)==2
    assert [scene['material_audit_sha256'] for scene in resolved]==['first','second']


@pytest.mark.town02
@pytest.mark.usefixtures('require_town02')
def test_actual_variant_has_only_five_builtin_types_and_preserves_original_imports():
    folder = REAL_ROOT/'assets/town02'
    paths = [folder/materials.BASE_XML,folder/materials.SOURCE_XML]
    before = {path:sha(path) for path in paths}
    report = materials.check_variant()
    assert set(report['materials'].values())=={'brick','concrete','wood','glass','metal'}
    assert report['counts']['total_meshes']==6085
    assert report['counts']['matched_meshes']==408
    assert report['counts']['changed_meshes']==395
    assert report['counts']['unchanged_unverified_meshes']==5677
    assert resource_report(folder)['verified_imported_files']==8635
    assert before=={path:sha(path) for path in paths}
