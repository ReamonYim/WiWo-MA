from __future__ import annotations
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .config import PROJECT_ROOT

@dataclass
class SceneAsset:
    name: str
    xml_path: str
    scene_hash: str
    meshes: list
    load_ms: float
    bounds: list
    source: str

def load_asset(config):
    """Load a private Sionna scene. Read meshes from the actual Mitsuba BVH geometry."""
    import sionna.rt as rt
    import drjit as dr
    start = time.perf_counter()
    name = config["scene"]["name"]
    custom = config["scene"].get("xml_path")
    if custom:
        path = Path(custom).resolve()
        source = "user-supplied XML"
    elif name == "reflector":
        path = PROJECT_ROOT / "assets/reflector.xml"
        source = "WiWo-MA self-built finite wall and ground; metre Z-up"
    elif name in ("munich", "etoile"):
        path = Path(getattr(rt.scene, name))
        source = "NVIDIA Sionna RT; OpenStreetMap contributors (ODbL source data); Blender-OSM conversion"
    else:
        raise ValueError(f"Unknown scene {name}; set scene.xml_path for your map")
    if not path.is_file():
        raise FileNotFoundError(path)
    scene = rt.load_scene(str(path), merge_shapes=False)
    meshes = []
    digest = hashlib.sha256(path.read_bytes())
    for obj in scene.objects.values():
        mesh = obj.mi_mesh
        vertices = np.asarray(mesh.vertex_positions_buffer()).reshape(-1, 3).copy()
        faces = np.asarray(mesh.faces_buffer()).reshape(-1, 3).copy()
        digest.update(vertices.tobytes()); digest.update(faces.tobytes())
        # Radio material definitions in XML are included in the digest as well.
        meshes.append({"name": obj.name, "vertices": vertices, "faces": faces})
    dr.sync_thread()
    vertices = np.concatenate([m["vertices"] for m in meshes]) if meshes else np.zeros((1,3))
    asset = SceneAsset(name, str(path), digest.hexdigest(), meshes,
                       (time.perf_counter()-start)*1000,
                       [vertices.min(0).tolist(), vertices.max(0).tolist()], source)
    return scene, asset
