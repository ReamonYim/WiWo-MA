"""CPU display geometry from the actual SDK scene resources; no mock CSI."""
import hashlib,importlib.metadata,xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
import numpy as np
from .config import ROOT


def scene_xml(scene):
    if scene.get('material_policy'):
        from .town02_materials import validate_strict_scene
        validate_strict_scene(scene,verify_meshes=True)
    if scene.get('xml_path'):return (ROOT/scene['xml_path']).resolve()
    if scene['name']=='town02':return ROOT/'assets/town02/town2_3mat.xml'
    if scene['name']=='reflector':return ROOT/'assets/reflector.xml'
    package=importlib.metadata.distribution('sionna-rt')
    return Path(package.locate_file('sionna/rt/scenes/'+scene['name']+'/'+scene['name']+'.xml'))


@lru_cache(maxsize=4)
def load_display_meshes(xml):
    import trimesh
    path=Path(xml);meshes=[];hashes=[]
    root=ET.parse(path).getroot()
    colors={'wood':(151,112,72),'concrete':(145,149,151),'brick':(176,104,83),
            'glass':(88,130,190),'metal':(112,123,138)}
    materials={}
    for node in root.findall('bsdf'):
        kind=node.find("string[@name='type']")
        if kind is not None:materials[node.get('id')]=kind.get('value')
    for i,node in enumerate(root.findall('shape')):
        if node.find('transform') is not None:raise NotImplementedError('Display XML transform adapter not validated; do not silently show wrong geometry')
        name=node.find("string[@name='filename']")
        if node.get('type') not in ('ply','obj') or name is None:raise NotImplementedError('Display requires actual PLY/OBJ mesh, no geometric placeholder')
        source=(path.parent/name.get('value')).resolve()
        mesh=trimesh.load_mesh(source,process=False,maintain_order=True)
        if isinstance(mesh,trimesh.Scene):mesh=mesh.to_mesh()
        vertices=np.asarray(mesh.vertices,np.float32);faces=np.asarray(mesh.faces,np.uint32)
        material=node.find('ref');material_id=material.get('id') if material is not None else 'default'
        meshes.append(dict(name=node.get('id','obj_'+str(i)),vertices=vertices,faces=faces,
            material_id=material_id,color=colors.get(materials.get(material_id),(84,99,115))))
        # A valid independently installed SDK need not live below this source
        # checkout (e.g. an external clean venv). Preserve resource provenance
        # without assuming that layout or weakening custom-scene validation.
        resource=str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)
        hashes.append(dict(path=resource,sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    return meshes,hashes
