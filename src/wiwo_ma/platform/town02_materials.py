"""Conservative, traceable Town02 material bindings from original asset labels."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .config import ROOT
from .itu_catalog import load_catalog, material_parameters


POLICY = 'town02-source-label-itu-v1'
BASE_XML = 'town2_3mat.xml'
SOURCE_XML = 'town2_xml.xml'
STRICT_XML = 'town2_strict_itu.xml'
AUDIT_NAME = 'town2_strict_itu.materials.json'
STRICT_CONFIGS = {'single':'town02_strict_itu_single_ma.json',
                  'full':'town02_strict_itu_64tx_40ue.json'}
# Only semantic replay is reused. Every call still hashes the XML, audit,
# original XMLs, rules, manifest and SDK; actual asset loads also verify meshes.
_VALIDATED_BUNDLES = {}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _local(path):
    path = (ROOT/path).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError('Material resource must exist inside project: '+str(path))
    return path


def _definitions(root):
    result = {}
    for node in root.findall('bsdf'):
        kind = node.find("string[@name='type']")
        if node.get('type') != 'itu-radio-material' or kind is None:
            raise ValueError('Strict variant permits only built-in ITU radio materials')
        if node.get('id') in result:
            raise ValueError('Duplicate radio material identifier')
        # No manually supplied EM coefficients, roughness, or thickness.
        if len(node) != 1:
            raise ValueError('Strict ITU definitions must retain SDK parameter defaults')
        result[node.get('id')] = kind.get('value')
    return result


def _geometry_digest(root):
    copy = deepcopy(root)
    for node in list(copy.findall('bsdf')):
        copy.remove(node)
    for shape in copy.findall('shape'):
        for ref in list(shape.findall('ref')):
            if ref.get('name') in (None,'bsdf'):
                shape.remove(ref)
    for node in copy.iter():
        node.text = node.text.strip() if node.text and node.text.strip() else None
        node.tail = None
    return hashlib.sha256(ET.tostring(copy,encoding='utf-8')).hexdigest()


def _read_rules(path,catalog,frequency_hz):
    data = json.loads(path.read_text())
    if data.get('schema') != POLICY:
        raise ValueError('Unknown strict material assignment policy')
    rules = {}
    for row in data['rules']:
        source = row['source_material_id']
        if not isinstance(source,str) or not source or any(c in source for c in '*?[]'):
            raise ValueError('Exact material IDs are required; patterns are forbidden')
        if source in rules or not row.get('evidence'):
            raise ValueError('Duplicate or undocumented material rule')
        material_parameters(row['itu_type'],frequency_hz,catalog)
        rules[source] = row
    return data,rules


def build_variant(folder=None,rules_path=None,write_configs=True):
    """Change only allowlisted bindings; preserve unconfirmed baseline values."""
    from .town02 import CONFIGS,resource_report

    folder = _local((folder or ROOT/'assets/town02')/BASE_XML).parent
    rules_path = _local(rules_path or ROOT/'configs/materials/town02_strict_itu.json')
    resource_report(folder)  # Validate original copies against import SHA256s.
    original_path = _local(folder/SOURCE_XML)
    baseline_path = _local(folder/BASE_XML)
    original = ET.parse(original_path).getroot()
    baseline = ET.parse(baseline_path).getroot()
    output = deepcopy(baseline)
    catalog = load_catalog()
    policy_data = json.loads(rules_path.read_text())
    frequency_hz = policy_data['frequency_hz']
    policy_data,rules = _read_rules(rules_path,catalog,frequency_hz)
    original_materials = {n.get('id') for n in original.findall('bsdf')}
    sources = {}
    for node in original.findall('shape'):
        filename = node.find("string[@name='filename']")
        ref = node.find('ref')
        if filename is None or ref is None or ref.get('id') not in original_materials:
            raise ValueError('Original shape has no declared material binding')
        key = filename.get('value')
        if key in sources:
            raise ValueError('Ambiguous duplicate source mesh: '+key)
        sources[key] = dict(material=ref.get('id'),shape=node.get('id'))

    materials = _definitions(output)
    targets = {kind:ident for ident,kind in materials.items()}
    for kind in sorted({row['itu_type'] for row in rules.values()}):
        if kind not in targets:
            ident = 'mat_'+kind
            if ident in materials:
                raise ValueError('Conflicting material identifier')
            node = ET.Element('bsdf',{'type':'itu-radio-material','id':ident})
            ET.SubElement(node,'string',{'name':'type','value':kind})
            nodes = output.findall('bsdf')
            index = list(output).index(nodes[-1])+1 if nodes else 0
            output.insert(index,node)
            materials[ident] = kind
            targets[kind] = ident
    rows = []
    used_rules = set()
    for node in output.findall('shape'):
        filename = node.find("string[@name='filename']").get('value')
        if filename not in sources:
            raise ValueError('Retained mesh has no original material provenance: '+filename)
        source = sources[filename]
        ref = node.find('ref')
        before = ref.get('id')
        source_id = source['material']
        rule = rules.get(source_id)
        after = targets[rule['itu_type']] if rule else before
        if rule:
            used_rules.add(source_id)
            ref.set('id',after)
        rows.append(dict(mesh=filename,shape_id=node.get('id'),source_shape_id=source['shape'],
                         source_material_id=source_id,baseline_material_id=before,
                         baseline_itu_type=materials[before],material_id=after,itu_type=materials[after],
                         changed=before!=after,status='source_label_confirmed' if rule else 'unchanged_unverified',
                         evidence=rule['evidence'] if rule else policy_data.get('excluded_material_ids',{}).get(
                             source_id,'No approved exact original material ID; baseline retained without physical confirmation')))
    if used_rules != set(rules):
        raise ValueError('Allowlist contains material IDs absent from retained meshes: '+str(sorted(set(rules)-used_rules)))
    geometry_hash = _geometry_digest(baseline)
    if _geometry_digest(output) != geometry_hash:
        raise ValueError('Strict material conversion changed scene geometry')
    ET.indent(output,space='  ')
    output_bytes = ET.tostring(output,encoding='utf-8',xml_declaration=True)+b'\n'
    output_path = folder/STRICT_XML
    output_path.write_bytes(output_bytes)
    output_hash = _sha(output_path)
    report = dict(schema=POLICY,scope='Original asset material labels; no measured physical identification',
                  source_xml=str(original_path.relative_to(ROOT)),source_xml_sha256=_sha(original_path),
                  baseline_xml=str(baseline_path.relative_to(ROOT)),baseline_xml_sha256=_sha(baseline_path),
                  import_manifest_sha256=_sha(folder/'import-manifest.json'),
                  rules_path=str(rules_path.relative_to(ROOT)),rules_sha256=_sha(rules_path),
                  output_xml=str(output_path.relative_to(ROOT)),output_xml_sha256=output_hash,
                  geometry_sha256=geometry_hash,frequency_hz=frequency_hz,
                  itu_catalog_version=catalog['version'],itu_catalog_sha256=catalog['source_sha256'],
                  materials=materials,
                  parameters={kind:material_parameters(kind,frequency_hz,catalog) for kind in sorted(set(materials.values()))},
                  defaults='Sionna SDK defaults retained; no authored EM coefficients, thickness, scattering or weather values',
                  counts=dict(total_meshes=len(rows),matched_meshes=sum(r['status']=='source_label_confirmed' for r in rows),
                              changed_meshes=sum(r['changed'] for r in rows),
                              unchanged_unverified_meshes=sum(r['status']=='unchanged_unverified' for r in rows),
                              material_bindings=dict(Counter(r['material_id'] for r in rows))),
                  meshes=rows)
    audit_path = folder/AUDIT_NAME
    audit_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    scene = dict(name='town02',xml_path=str(output_path.relative_to(ROOT)),xml_sha256=output_hash,
                 material_policy=POLICY,material_audit_path=str(audit_path.relative_to(ROOT)),
                 material_audit_sha256=_sha(audit_path),material_rules_sha256=_sha(rules_path),
                 material_itu_catalog_sha256=catalog['source_sha256'])
    if write_configs:
        from .config import validate
        for mode,filename in STRICT_CONFIGS.items():
            config = json.loads((ROOT/'configs/platform'/CONFIGS[mode]).read_text())
            if config['scene'].get('material_override'):
                raise ValueError('Strict material policy forbids global material overrides')
            config['scene'] = deepcopy(scene)
            config['town02_provenance']['static_materials'] = 'Exact original material ID allowlist -> SDK ITU models; unconfirmed baseline bindings retained and marked'
            config['town02_provenance']['material_assignment_counts'] = report['counts']
            config = validate(config)
            (ROOT/'configs/platform'/filename).write_text(json.dumps(config,indent=2)+'\n')
    return scene,report


def validate_strict_scene(scene,frequency_hz=None,verify_meshes=False):
    """Reject changed assets, custom coefficients, invalid bands or overrides."""
    if scene.get('material_policy') != POLICY:
        if scene.get('material_policy') is None:
            return
        raise ValueError('Unknown strict material assignment policy')
    if scene.get('material_override'):
        raise ValueError('Strict material policy forbids global material overrides')
    xml = _local(scene['xml_path'])
    audit_path = _local(scene['material_audit_path'])
    if _sha(xml) != scene.get('xml_sha256'):
        raise ValueError('Strict material XML checksum mismatch')
    audit_hash = _sha(audit_path)
    if audit_hash != scene.get('material_audit_sha256'):
        raise ValueError('Strict material audit checksum mismatch')
    report = json.loads(audit_path.read_text())
    if report.get('schema') != POLICY or report['output_xml_sha256'] != scene['xml_sha256']:
        raise ValueError('Strict material audit is not bound to the scene')
    for path_key,sha_key in [('source_xml','source_xml_sha256'),('baseline_xml','baseline_xml_sha256'),
                             ('rules_path','rules_sha256')]:
        if _sha(_local(report[path_key])) != report[sha_key]:
            raise ValueError('Strict material source changed: '+path_key)
    if report['rules_sha256'] != scene.get('material_rules_sha256'):
        raise ValueError('Strict material rules are not bound to configuration')
    if _sha(xml.parent/'import-manifest.json') != report['import_manifest_sha256']:
        raise ValueError('Original resource manifest changed')
    catalog = load_catalog()
    if catalog['source_sha256'] != scene.get('material_itu_catalog_sha256') or report['itu_catalog_sha256'] != catalog['source_sha256']:
        raise ValueError('SDK ITU catalog changed; rebuild the material variant')
    frequency_hz = frequency_hz if frequency_hz is not None else report['frequency_hz']
    bundle = (audit_hash,catalog['source_sha256'],frequency_hz)
    if not verify_meshes and bundle in _VALIDATED_BUNDLES:
        return
    tree = ET.parse(xml).getroot()
    materials = _definitions(tree)
    if materials != report['materials']:
        raise ValueError('Strict material definitions differ from audit')
    for kind in materials.values():
        material_parameters(kind,frequency_hz if frequency_hz is not None else report['frequency_hz'],catalog)
    if _geometry_digest(tree) != report['geometry_sha256']:
        raise ValueError('Strict material variant geometry differs from audit')
    baseline = ET.parse(_local(report['baseline_xml'])).getroot()
    original = ET.parse(_local(report['source_xml'])).getroot()
    if _geometry_digest(baseline) != report['geometry_sha256']:
        raise ValueError('Strict geometry differs from original baseline')
    baselines = {n.find("string[@name='filename']").get('value'):n.find('ref').get('id') for n in baseline.findall('shape')}
    sources = {}
    for node in original.findall('shape'):
        mesh = node.find("string[@name='filename']").get('value')
        if mesh in sources:
            raise ValueError('Ambiguous duplicate source mesh: '+mesh)
        sources[mesh] = node.find('ref').get('id')
    _,rules = _read_rules(_local(report['rules_path']),catalog,
                         frequency_hz if frequency_hz is not None else report['frequency_hz'])
    rows = {row['mesh']:row for row in report['meshes']}
    shapes = tree.findall('shape')
    if len(rows) != len(shapes) or len(rows) != len(report['meshes']):
        raise ValueError('Material audit mesh coverage mismatch')
    matched=changed=unverified=0
    bindings=Counter()
    for node in shapes:
        mesh = node.find("string[@name='filename']").get('value')
        row = rows.get(mesh)
        source = sources.get(mesh)
        before = baselines.get(mesh)
        after = node.find('ref').get('id')
        if row is None or source is None or before is None:
            raise ValueError('Missing exact per-mesh material provenance')
        rule = rules.get(source)
        if (row['source_material_id'] != source or row['baseline_material_id'] != before
                or row['shape_id'] != node.get('id') or row['material_id'] != after
                or row['itu_type'] != materials.get(after) or row['changed'] != (before!=after)):
            raise ValueError('Material audit binding mismatch: '+mesh)
        if rule:
            if materials[after] != rule['itu_type'] or row['status'] != 'source_label_confirmed':
                raise ValueError('Material binding violates exact original-ID rule: '+mesh)
            matched+=1
        else:
            if before != after or row['status'] != 'unchanged_unverified':
                raise ValueError('Unconfirmed material was replaced: '+mesh)
            unverified+=1
        changed+=before!=after
        bindings[after]+=1
    expected=dict(total_meshes=len(shapes),matched_meshes=matched,changed_meshes=changed,
                  unchanged_unverified_meshes=unverified,material_bindings=dict(bindings))
    if report['counts'] != expected:
        raise ValueError('Material audit counts do not describe the actual bindings')
    if verify_meshes:
        from .town02 import resource_report
        resource_report(xml.parent)
    _VALIDATED_BUNDLES[bundle] = True
    if len(_VALIDATED_BUNDLES)>32:
        _VALIDATED_BUNDLES.pop(next(iter(_VALIDATED_BUNDLES)))


def check_variant(config_path=None):
    from .config import load
    config = load(config_path or ROOT/'configs/platform'/STRICT_CONFIGS['full'])
    validate_strict_scene(config['scene'],config['radio']['carrier_hz'],verify_meshes=True)
    report = json.loads(_local(config['scene']['material_audit_path']).read_text())
    catalog = load_catalog()
    frequency_hz = config['radio']['carrier_hz']
    return dict(status='Passed_strict_ITU_material_assignment',materials=report['materials'],frequency_hz=frequency_hz,
                parameters={kind:material_parameters(kind,frequency_hz,catalog) for kind in sorted(set(report['materials'].values()))},counts=report['counts'],
                basis=report['scope'],native_CSI='Not_computed')


def example(mode='full'):
    from .config import load
    return load(ROOT/'configs/platform'/STRICT_CONFIGS[mode])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['build','check'])
    args = parser.parse_args()
    if args.command == 'build':
        build_variant()
    print(json.dumps(check_variant(),indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
