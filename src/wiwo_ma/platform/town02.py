"""Import the user's Town02 resources and prepare explicit local experiments."""
import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .config import ROOT, load, validate


ASSET_ROOT = ROOT / 'assets/town02'
XML_NAME = 'town2_3mat.xml'
TRAJECTORY = 'car_coordinate/OneCar/Straight 1 (103.211403, -102.730927, 20.000000)/car1.csv'
CONFIGS = {'single': 'town02_single_ma.json', 'full': 'town02_64tx_40ue.json'}


def example(mode='full'):
    return load(ROOT / 'configs/platform' / CONFIGS[mode])


def resource_report(folder=None, verify_hashes=True):
    """Check real XML bindings and mesh files without initializing an SDK."""
    folder = Path(folder or ASSET_ROOT)
    if not (folder/XML_NAME).is_file():
        from .town02_resources import IMPORT_HINT
        raise FileNotFoundError('Missing Town02 baseline XML: '+str(folder/XML_NAME)+'. '+IMPORT_HINT)
    tree = ET.parse(folder / XML_NAME).getroot()
    materials = {
        node.get('id'): node.find("string[@name='type']").get('value')
        for node in tree.findall("bsdf[@type='itu-radio-material']")
    }
    references = []
    bindings = Counter()
    for shape in tree.findall('shape'):
        filename = shape.find("string[@name='filename']")
        material = shape.find('ref')
        if filename is None or material is None or material.get('id') not in materials:
            raise ValueError('Town02 mesh has a missing resource or material binding')
        resource = (folder / filename.get('value')).resolve()
        if not resource.is_relative_to(folder.resolve()) or not resource.is_file():
            raise ValueError('Missing or external Town02 mesh: ' + filename.get('value'))
        references.append(resource)
        bindings[material.get('id')] += 1
    checked = 0
    manifest_path = folder / 'import-manifest.json'
    if verify_hashes:
        manifest = json.loads(manifest_path.read_text())
        for item in manifest['files']:
            resource = (folder / item['path']).resolve()
            if not resource.is_relative_to(folder.resolve()):
                raise ValueError('External imported resource')
            data = resource.read_bytes()
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Imported resource changed: ' + item['path'])
            checked += 1
    return dict(status='Passed_resource_files',scene=XML_NAME,shapes=len(references),
                unique_meshes=len(set(references)),materials=materials,material_bindings=dict(bindings),
                verified_imported_files=checked,SDK_scene_load='Not_tested',CSI='Not_computed')


def configuration(mode='full'):
    """Slow bounded UE motion anchored to archived Town02 road coordinates.

    These are new experiment trajectories, not a replay of the recorded car.
    The original XML's ITU materials are retained without a global override.
    """
    from .examples import munich_example
    from .physics import constrained_target

    with (ASSET_ROOT / TRAJECTORY).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    positions = np.array([[float(r[k]) for k in ('x_b','y_b','z_b')] for r in rows])
    users = 40 if mode == 'full' else 1
    config = munich_example(users=users, carrier_hz=2.8e9)
    config['scene'] = dict(name='town02',xml_path='assets/town02/' + XML_NAME)
    config['timing'].update(duration_s=4. if mode == 'full' else 1.,csi_dt_s=.01)
    config['base_stations'][0]['position_m'] = [103.211403,-102.730927,10.]
    duration = config['timing']['duration_s']
    indices = np.linspace(0,len(positions)-2,users,dtype=int) if users > 1 else [len(positions)-2]
    for user, index in zip(config['users'],indices):
        start = positions[index]
        direction = positions[index+1]-start
        direction /= max(float(np.linalg.norm(direction)),1e-12)
        end = start + direction * .02 * duration
        user.update(position_m=start.tolist(),orientation_deg=[0.,0.,0.],
                    motion=dict(model='bounded_quintic',max_speed_m_s=.05,max_acceleration_m_s2=.2,
                                max_angular_speed_rad_s=.1,max_angular_acceleration_rad_s2=.2),
                    trajectory=[dict(time_s=0.,position_m=start.tolist()),
                                dict(time_s=duration,position_m=end.tolist())])
    config['actors'] = []
    config['actions'] = []
    bs = config['base_stations'][0]
    if mode == 'single':
        panel = bs['panels'][0]
        panel.update(position_m=[0.,0.,0.],orientation_deg=[0.,0.,0.])
        panel['array'] = dict(kind='single',rows=1,cols=1,spacing_m=.04)
        panel['mechanics'].update(mechanism='rail_x',workspace_min_m=[-.25,-.25,-.25],
                                  workspace_max_m=[.25,.25,.25])
        bs['panels'] = [panel]
        bs['hardware_mode'] = 'single_ma'
    for i, panel in enumerate(bs['panels'][:4]):
        config['actions'].append(dict(time_s=.1,**constrained_target(
            config,bs['id'],panel['id'],[.02 if i%2 == 0 else -.02,0,0],[0,0,0])))
    config['town02_provenance'] = dict(
        trajectory_resource=TRAJECTORY,coordinate_system='Original XML/PLY and CSV x_b,y_b,z_b; metres; no map rescaling',
        BS_position_source='Archived Straight 1 TX coordinates, with the original generator height adjustment 20m to 10m',
        UE_motion='New slow bounded quintic motions from sampled CSV road positions; not recorded vehicle playback',
        static_materials='Original XML ITU wood/concrete/brick; no weather or global material override',
        static_geometry_clearance='Not_yet_checked_by_native_scene_preflight')
    return validate(config)


def import_resources(source):
    """Copy complete map meshes and archived trajectories, checking source SHA256."""
    source = Path(source).expanduser().resolve()
    originals = source / 'Town2_xml'
    required=[source/'资源SHA256.json',originals/XML_NAME,originals/'town2_xml.xml',
              originals/'material_manager.py',originals/TRAJECTORY]
    missing=[str(path.relative_to(source)) for path in required if not path.is_file()]
    if missing or not (originals/'meshes').is_dir():
        raise FileNotFoundError('Incomplete external Town02 package: '+', '.join(missing or ['Town2_xml/meshes'])+
                                '. Supply the original resource package and set WIWO_TOWN02_SOURCE or --source.')
    records=json.loads((source/'资源SHA256.json').read_text(encoding='utf-8-sig'))['files']
    expected = {x['path']:x for x in records}
    if len(expected)!=len(records):raise ValueError('Duplicate paths in source checksum manifest')
    selected = [*sorted((originals/'meshes').glob('*.ply')),
                *sorted(originals.glob('town2*.xml')),
                *sorted((originals/'car_coordinate').rglob('*.csv')),
                originals/'material_manager.py']
    if not list((originals/'meshes').glob('*.ply')):
        raise FileNotFoundError('Town02 XML and meshes are required in ' + str(originals))
    # Validate every source before changing an existing import.
    for path in selected:
        key=str(path.relative_to(source))
        if key not in expected:raise ValueError('Source resource is absent from checksum manifest: '+key)
        item=expected[key];data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=item['sha256'] or len(data)!=item['bytes']:
            raise ValueError('Source resource checksum mismatch: '+key)
    for name in (XML_NAME,'town2_xml.xml'):
        for shape in ET.parse(originals/name).getroot().findall('shape'):
            filename=shape.find("string[@name='filename']")
            if filename is None:raise ValueError('Source shape has no mesh filename: '+name)
            path=(originals/filename.get('value')).resolve()
            if not path.is_relative_to(originals) or not path.is_file():
                raise FileNotFoundError('Missing or external source mesh: '+filename.get('value'))
    imported = []
    for path in selected:
        relative = path.relative_to(originals)
        item = expected[str(path.relative_to(source))]
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != item['sha256'] or len(data) != item['bytes']:
            raise ValueError('Source resource checksum mismatch: ' + str(relative))
        destination = ASSET_ROOT/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        imported.append(dict(path=str(relative),bytes=len(data),sha256=digest))
    manifest = dict(source=str(source),source_scene='Town2_xml/'+XML_NAME,
                    total_bytes=sum(x['bytes'] for x in imported),files=imported)
    (ASSET_ROOT/'import-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report = resource_report(ASSET_ROOT)
    for mode, filename in CONFIGS.items():
        target = ROOT/'configs/platform'/filename
        if not target.exists():
            target.write_text(json.dumps(configuration(mode),indent=2)+'\n')
    from .town02_materials import build_variant
    _,strict_report = build_variant()
    report['strict_material_variant'] = strict_report['counts']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    commands.add_parser('import').add_argument('--source',type=Path,
        default=Path(os.environ.get('WIWO_TOWN02_SOURCE',str(Path.home()/'project/Sionna_Town02'))))
    commands.add_parser('check')
    status=commands.add_parser('status',help='Inspect actual external XML/PLY prerequisites without loading a backend')
    status.add_argument('--require',action='store_true',help='Exit nonzero unless the full strict import is available')
    args = parser.parse_args()
    if args.command=='status':
        from .town02_resources import resource_status
        result=resource_status()
        print(json.dumps(result,indent=2))
        if result['status']=='incomplete' or args.require and result['status']!='available':raise SystemExit(2)
        return
    result = import_resources(args.source) if args.command == 'import' else resource_report()
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
