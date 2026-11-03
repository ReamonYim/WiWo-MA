"""Discover an external Town02 import without initializing Sionna or CUDA."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .config import ROOT

IMPORT_HINT = 'Import the complete resource package with: python -m wiwo_ma.platform.town02 import --source /path/to/Sionna_Town02'
BASE_FILES = ('town2_3mat.xml','town2_xml.xml','import-manifest.json')
STRICT_FILES = ('town2_strict_itu.xml','town2_strict_itu.materials.json')


def resource_status(root=None,strict=True):
    """Distinguish a clean checkout from an incomplete or malformed import.

    Committed experiment JSON files do not establish map availability. Only
    an entirely absent import is optional; a partial import is an error.
    This checks presence and structure, not hashes; resource_report and the
    strict material loader still perform the full integrity checks.
    """
    root=Path(root or ROOT).resolve();folder=root/'assets/town02'
    required=[folder/name for name in (*BASE_FILES,*(STRICT_FILES if strict else ()))]
    present=any(path.exists() for path in required) or (folder/'meshes').exists()
    missing=set();errors=[]

    def require(path):
        path=path.resolve()
        if not path.is_relative_to(root):
            errors.append('Resource path escapes the project');return
        if not path.is_file():missing.add(str(path.relative_to(root)))

    for path in required:require(path)
    if present:
        manifest=folder/'import-manifest.json'
        if manifest.is_file():
            try:
                data=json.loads(manifest.read_text())
                if not isinstance(data.get('files'),list) or not data['files']:
                    raise ValueError('Expected a nonempty imported file list')
                for item in data['files']:require(folder/item['path'])
            except (ValueError,TypeError,KeyError) as error:errors.append('Invalid import manifest: '+str(error))
        xml_files=('town2_3mat.xml','town2_strict_itu.xml') if strict else ('town2_3mat.xml',)
        for name in xml_files:
            path=folder/name
            if not path.is_file():continue
            try:
                for node in ET.parse(path).getroot().findall('shape'):
                    filename=node.find("string[@name='filename']")
                    if filename is None or not filename.get('value'):
                        raise ValueError('Mesh filename is missing')
                    require(folder/filename.get('value'))
            except (ET.ParseError,ValueError) as error:errors.append('Invalid '+name+': '+str(error))
        configs=('town02_single_ma.json','town02_64tx_40ue.json')
        if strict:configs+=('town02_strict_itu_single_ma.json','town02_strict_itu_64tx_40ue.json')
        for name in configs:require(root/'configs/platform'/name)
        if strict:require(root/'configs/materials/town02_strict_itu.json')
    state='not_imported' if not present else 'incomplete' if missing or errors else 'available'
    return dict(status=state,external_resources=True,strict=strict,missing=sorted(missing),errors=errors,hint=IMPORT_HINT)


def require_resources(root=None,strict=True):
    report=resource_status(root,strict)
    if report['status']!='available':
        details='; '.join(report['missing'][:8]+report['errors'])
        raise FileNotFoundError('Town02 resources are '+report['status']+': '+details+'. '+IMPORT_HINT)
    return report
