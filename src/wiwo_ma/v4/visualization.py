"""V4 Viser client of the full sixteen-panel physical controller.

Rendering and HTTP handlers never perform propagation queries. Static scene
geometry is uploaded once; all motion comes from completed controller snapshots.
An optional injected controller supports explicitly labelled CPU UI tests.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import PROJECT_ROOT
from ..mechanics import euler_to_matrix, matrix_to_euler


def jsonable(value):
    """Finite JSON telemetry, with complex data explicitly encoded if requested."""
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": jsonable(value.real), "imag": jsonable(value.imag)}
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, complex):
        return [jsonable(value.real), jsonable(value.imag)]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def quaternion(matrix):
    q = Rotation.from_matrix(np.asarray(matrix)).as_quat()
    return q[..., [3, 0, 1, 2]]


def rotation(wxyz):
    q = np.asarray(wxyz)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]]).as_matrix()


def _box(size):
    vertex = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5)
                       for z in (-.5, .5)], np.float32)*np.asarray(size, np.float32)
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                      [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.uint32)
    return vertex, faces


def _cylinder(radius, sides=12):
    angles = np.arange(sides)*2*np.pi/sides
    disk = np.column_stack((radius*np.cos(angles), radius*np.sin(angles)))
    vertices = np.vstack((np.column_stack((disk, np.zeros(sides))),
                          np.column_stack((disk, np.ones(sides)))))
    faces = []
    for i in range(sides):
        j = (i+1) % sides
        faces.extend([[i, j, sides+j], [i, sides+j, sides+i]])
    return vertices.astype(np.float32), np.asarray(faces, np.uint32)


def _rod_rotation(direction):
    z = np.asarray(direction, float)
    z = z/max(np.linalg.norm(z), 1e-12)
    reference = [0., 0., 1.] if abs(z[2]) < .95 else [0., 1., 0.]
    x = np.cross(reference, z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def _scene_meshes(asset, center, radius):
    """Merge actual asset geometry, optionally cropping display faces only."""
    vertices, faces, offset = [], [], 0
    full_faces = 0
    for mesh in getattr(asset, "meshes", ()):
        vv = np.asarray(mesh["vertices"], np.float32)
        ff = np.asarray(mesh["faces"], np.int64)
        full_faces += len(ff)
        if radius is not None and len(ff):
            # Keep any triangle whose XY bounding rectangle overlaps the ROI.
            xyz = vv[ff, :2]
            lower, upper = xyz.min(1), xyz.max(1)
            nearest = np.maximum(lower, np.minimum(np.asarray(center)[:2], upper))
            ff = ff[np.sum((nearest-np.asarray(center)[:2])**2, 1) <= radius**2]
        if not len(ff):
            continue
        used, inverse = np.unique(ff, return_inverse=True)
        vertices.append(vv[used]); faces.append(inverse.reshape(-1, 3)+offset)
        offset += len(used)
    return (np.concatenate(vertices) if vertices else np.zeros((0, 3), np.float32),
            np.concatenate(faces).astype(np.uint32) if faces else np.zeros((0, 3), np.uint32),
            full_faces)


def _depth(path):
    kinds = path.get("interaction_types", ())
    return len(kinds) if len(kinds) else max(0, len(path.get("points", ()))-2)


def _path_power(path):
    gain = np.asarray(path.get("passband_gain", [0., 0.]), float)
    return float(np.sum(gain*gain))


def _svg(values, title, x=None, color="#38bdf8"):
    values = np.asarray(values, float).reshape(-1)
    if not len(values):
        return f"<div>{title}: awaiting data</div>"
    indexes = np.linspace(0, len(values)-1, min(128, len(values)), dtype=int)
    values = values[indexes]
    if not np.isfinite(values).all():
        return f"<div>{title}: nonfinite solver output</div>"
    lo, hi = values.min(), values.max()
    xx = np.linspace(8, 350, len(values))
    if x is not None and len(x) == len(indexes):
        positions = np.asarray(x, float)
        xx = 8+342*(positions-positions.min())/max(float(np.ptp(positions)), 1e-12)
    yy = 95-68*(values-lo)/max(float(hi-lo), 1e-15)
    line = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(xx, yy))
    return (f'<svg viewBox="0 0 360 120" style="width:100%;background:#111827;border-radius:8px">'
            f'<text x="8" y="17" fill="#e5e7eb" font-size="12">{title}</text>'
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.6"/>'
            f'<text x="8" y="114" fill="#9ca3af" font-size="10">{lo:.4g} … {hi:.4g}</text></svg>')


def _time(state):
    return float(state.get("timestamp", state.get("time", state.get("t", 0.))))


def _users(state):
    world = state.get("world", {})
    return np.asarray(world.get("user_positions", world.get("ues", state.get("user_positions", []))), float).reshape(-1, 3)


def _center(state):
    return np.asarray(state.get("centre_position", state.get("bscenter", np.mean(state["positions"], 0))), float)


def _result_field(result, name, default=None):
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


class V4ControlAPI:
    """Loopback action API and observation of explicitly historical live CSI."""
    def __init__(self, controller, ui, port, host="127.0.0.1"):
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Control API must bind to loopback")
        self.controller, self.ui = controller, ui
        outer = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def reply(self, status, value):
                data = json.dumps(jsonable(value), allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                self.end_headers(); self.wfile.write(data)
            def do_OPTIONS(self):
                self.reply(200, {})
            def do_GET(self):
                if self.path == "/api/state":
                    self.reply(200, outer.state_payload())
                elif self.path == "/api/ui":
                    self.reply(200, ui.telemetry())
                elif self.path == "/api/csi":
                    state, result, _ = controller.snapshot()
                    self.reply(200, {"t": _result_field(result, "t"),
                                     "cfr": _result_field(result, "cfr"),
                                     "frequency_offsets_hz": _result_field(result, "frequency_offsets_hz"),
                                     "scope": "historical_live_CSI_not_policy_future_labels"})
                else:
                    self.reply(404, {"error": "Unknown route"})
            def do_POST(self):
                try:
                    size = int(self.headers.get("Content-Length", 0))
                    if not 0 < size <= 131072:
                        raise ValueError("Request size must lie in 1..131072 bytes")
                    payload = json.loads(self.rfile.read(size))
                    if self.path == "/api/action":
                        result = controller.action(payload)
                    elif self.path == "/api/command":
                        if payload.get("kind") not in {"play", "pause", "step", "reset", "stop", "home", "mode", "speed", "preset", "mechanics", "seek", "episode"}:
                            raise ValueError("Unsupported controller command")
                        result = controller.command(payload["kind"], payload.get("payload"))
                    elif self.path == "/api/fps":
                        fps = float(payload["fps"])
                        if not np.isfinite(fps) or not 0 <= fps <= 1000:
                            raise ValueError("Invalid FPS")
                        with ui.lock:
                            ui.browser_fps[str(payload.get("client_id", "browser"))] = {"fps": fps, "wall": time.monotonic(), "kind": payload.get("kind", "RAF_cadence")}
                        result = {"received": True}
                    else:
                        self.reply(404, {"error": "Unknown route"}); return
                    self.reply(202, {"submitted": True, "result": result})
                except Exception as exc:
                    self.reply(400, {"error": f"{type(exc).__name__}: {exc}"})
        self.server = ThreadingHTTPServer((host, int(port)), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, name="v4-control-api", daemon=True)
        self.thread.start()

    def state_payload(self):
        state, result, stats = self.controller.snapshot()
        out = {"state": state, "statistics": stats, "ui": self.ui.telemetry(),
               "scope": "read_only_saved_episode_replay" if getattr(self.controller, "is_replay", False) else "interactive_live_asynchronous_historical_CSI"}
        if result is not None:
            h = np.asarray(_result_field(result, "cfr"))
            paths = _result_field(result, "paths", _result_field(result, "path_metadata", []))
            out["csi"] = {"t": _result_field(result, "t", _result_field(result, "timestamp")),
                          "request_id": _result_field(result, "request_id"),
                          "session": _result_field(result, "session", state.get("session")),
                          "positions": _result_field(result, "positions"),
                          "rotations": _result_field(result, "rotations"),
                          "user_positions": _result_field(result, "user_positions"),
                          "shape": h.shape, "finite": bool(np.isfinite(h).all()),
                          "level": _result_field(result, "level"),
                          "reason": _result_field(result, "reason"),
                          "label_model": _result_field(result, "label_model"),
                          "latency_ms": _result_field(result, "latency_ms"),
                          "rt_calls": _result_field(result, "rt_calls", 0),
                          "path_metadata_count": len(paths),
                          "path_events": _result_field(result, "path_events", {}),
                          "cache_metadata": _result_field(result, "cache_metadata", {}),
                          "metrics": _result_field(result, "metrics", {})}
        else:
            out["csi"] = None
        return out

    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)


class V4Visualization:
    """Small scene-node updates and throttled diagnostics for sixteen panels."""
    def __init__(self, controller, config, port=8092, host="127.0.0.1", api_port=None, render_hz=30.):
        import viser
        import trimesh
        if render_hz <= 0:
            raise ValueError("render_hz must be positive")
        self.controller, self.config = controller, deepcopy(config)
        self.server = viser.ViserServer(host=host, port=int(port))
        self.server.scene.set_up_direction("+z")
        self.server.gui.configure_theme(control_layout="collapsible", control_width="large", dark_mode=True,
                                        show_share_button=False, brand_color=(40, 150, 220))
        self.render_hz = float(render_hz)
        self.lock = threading.RLock()
        self.selected = 0; self.selected_panels = {0}
        self.gizmo_events = 0; self.gizmo_drag_ends = 0; self.panel_click_events = 0
        self.render_frames = 0; self.selection_events = 0; self.error = None
        self.browser_fps = {}; self.last_action_result = None
        self.last_mouse_wall = -1e9; self.dragging = False; self.syncing = False
        self.started_wall = time.monotonic(); self.last_stats_wall = -1e9
        self.last_version = None; self.last_user_positions = None; self.asset_signature = None
        self.last_channel_key = None; self.last_path_key = None; self.last_path_detail_key = None
        self.path_handles = []; self.map_handles = []; self.displayed_paths = []
        self.actor_handles = {}
        self.last_state, _, _ = controller.snapshot()
        state = self.last_state
        if np.asarray(state["positions"]).shape != (16, 3):
            raise ValueError("UI requires all sixteen actual panel states")
        self.physical_config = config.get("mechanics_v4", {})
        if getattr(controller, "bs", None) is not None:
            self.physical_config = controller.bs.mechanical_config
        self.panel_size = np.asarray(self.physical_config.get("panel_size", [.008, .09, .09]))
        self.center = _center(state)
        self.map_radius = float(config.get("visualization_v4", {}).get("map_radius_m", 120.))
        self.trail = deque(maxlen=300)
        self.user_trail = deque(maxlen=300); self.last_selected_user = None
        self.rate_history = deque(maxlen=256)
        self.controls = {}
        self._build_gui()
        p, r = np.asarray(state["positions"]), np.asarray(state["rotations"])
        vv, ff = _box(self.panel_size)
        self.panel_mesh = self.server.scene.add_batched_meshes_simple("/bs/panels", vv, ff, quaternion(r), p,
            batched_colors=self._panel_colors(), side="double", cast_shadow=False, receive_shadow=False, lod="off")
        self.target_mesh = self.server.scene.add_batched_meshes_simple("/bs/targets", vv, ff,
            quaternion(state["target_rotations"]), state["target_positions"], batched_colors=(255, 181, 65),
            wireframe=True, opacity=.6, cast_shadow=False, receive_shadow=False, lod="off")
        element_radius = float(config.get("bs", {}).get("element_spacing_m", .0625))*.085
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=element_radius)
        self.elements = self.server.scene.add_batched_meshes_simple("/bs/64_elements", np.asarray(sphere.vertices, np.float32),
            np.asarray(sphere.faces, np.uint32), np.tile([1., 0., 0., 0.], (64, 1)),
            np.asarray(state["element_positions"]).reshape(64, 3), batched_colors=(210, 235, 255),
            cast_shadow=False, receive_shadow=False, lod="off")
        self.panel_labels = [self.server.scene.add_label(f"/bs/panel_{b:02d}/label", f"Panel {b+1:02d}",
            position=p[b]+[0, 0, .12], font_screen_scale=.65, depth_test=False) for b in range(16)]
        self.panel_frame = self.server.scene.add_frame("/bs/selected_actual", position=p[0], wxyz=quaternion(r[0]), axes_length=.19, axes_radius=.008)
        self.target_frame = self.server.scene.add_frame("/bs/selected_target", position=state["target_positions"][0],
            wxyz=quaternion(state["target_rotations"][0]), axes_length=.14, axes_radius=.004)
        self.gizmo = self.server.scene.add_transform_controls("/bs/target_gizmo", scale=.45, position=state["target_positions"][0],
            wxyz=quaternion(state["target_rotations"][0]), depth_test=False, disable_rotations=True, line_width=4.)
        self.panel_mesh.on_click(self._panel_click)
        self.panel_frame.on_click(lambda e: self._select(self.selected, source="scene_frame"))
        self.gizmo.on_update(self._gizmo_update)
        hub_radius = float(self.physical_config.get("main_structure_radius", .15))
        hub = trimesh.creation.icosphere(subdivisions=2, radius=hub_radius)
        self.server.scene.add_mesh_simple("/bs/cpu_hub", np.asarray(hub.vertices, np.float32), np.asarray(hub.faces, np.uint32),
            position=self.center, color=(65, 75, 85), cast_shadow=False, receive_shadow=False)
        self.server.scene.add_label("/bs/cpu_label", "BS central processor · common RF clock", position=self.center+[0, 0, .2], font_screen_scale=.7)
        self.rods = []
        anchors = state.get("rod_anchors")
        if anchors is not None:
            rv, rf = _cylinder(float(self.physical_config.get("rod_radius", .012)))
            for b, anchor in enumerate(anchors):
                d = p[b]-np.asarray(anchor)
                self.rods.append(self.server.scene.add_mesh_simple(f"/bs/rod_{b:02d}", rv, rf,
                    position=anchor, wxyz=quaternion(_rod_rotation(d)), scale=(1., 1., float(np.linalg.norm(d))),
                    color=(115, 125, 140), cast_shadow=False, receive_shadow=False))
        users = _users(state)
        self.user_cloud = self.server.scene.add_point_cloud("/users/all", users.astype(np.float32), (255, 99, 75),
            point_size=.45, precision="float32", point_shape="circle")
        self.user_selected = self.server.scene.add_point_cloud("/users/selected", users[:1].astype(np.float32), (255, 220, 80),
            point_size=.75, precision="float32", point_shape="circle")
        self.user_selected_label = self.server.scene.add_label("/users/selected_label", "UE 01", position=users[0]+[0, 0, 1.] if len(users) else self.center, visible=bool(len(users)), font_screen_scale=.7)
        self.user_trail_handle = self.server.scene.add_line_segments("/users/selected_trail", np.zeros((1, 2, 3), np.float32), (255, 198, 65), thickness=1.2, thickness_units="screen")
        self.trail_handle = self.server.scene.add_line_segments("/bs/selected_trail", np.zeros((1, 2, 3), np.float32), (35, 185, 230), thickness=.006)
        self.workspace_handles = []
        if anchors is not None:
            limits = self.physical_config.get("rod_length_range", [.8, 1.5])
            for i, length in enumerate(limits):
                shell = trimesh.creation.icosphere(subdivisions=2, radius=float(length))
                self.workspace_handles.append(self.server.scene.add_mesh_simple(f"/bs/selected_rod_length_shell_{i}",
                    np.asarray(shell.vertices, np.float32), np.asarray(shell.faces, np.uint32), wireframe=True, opacity=.16,
                    color=(55, 175, 205), position=anchors[0], visible=False, cast_shadow=False, receive_shadow=False))
        else:
            cubev, cubef = _box([float(config.get("bs", {}).get("region_side_length", 1.))]*3)
            self.workspace_handles.append(self.server.scene.add_mesh_simple("/bs/canonical_cube", cubev, cubef, wireframe=True,
                opacity=.2, color=(55, 175, 205), position=self.center, visible=False, cast_shadow=False, receive_shadow=False))
        self.csi_pose = self.server.scene.add_frame("/csi/historical_selected", show_axes=True, axes_length=.11, axes_radius=.004, visible=False)
        self.map_info = {"scope": "actual_scene_mesh_display_only", "rt_geometry_unchanged": True}
        self._change_mesh()
        self.server.on_client_connect(self._aim_bs)
        self.api = V4ControlAPI(controller, self, int(port)+1 if api_port is None else api_port, host)
        self._select(0, source="initial")
        self.source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        print(f"WiWo-MA V4: http://{host}:{port}; loopback API http://{host}:{int(port)+1 if api_port is None else api_port}; 16 panels / 64 elements / {len(users)} UEs", flush=True)

    def _build_gui(self):
        gui = self.server.gui
        gui.add_markdown("# WiWo-MA V4\n\n**16 independent panels · 64 coherent BS antennas**\n\n"+
            ("Read-only saved episode replay; displayed frames are actual stored labels." if getattr(self.controller, "is_replay", False) else "Actual motion and historical completed CSI have separate timestamps."))
        if getattr(self.controller, "is_replay", False):
            with gui.add_folder("Saved episode replay"):
                self.replay_episode = gui.add_dropdown("Replay episode", options=tuple(self.controller.episode_ids), initial_value=self.controller.episode_id)
                self.replay_episode.on_update(lambda _: self._command("episode", self.replay_episode.value))
                self.replay_frame = gui.add_slider("Replay saved frame", min=0, max=max(1, self.controller.frame_count-1), step=1, initial_value=0)
                self.replay_frame.on_update(lambda _: None if self.syncing else self._command("seek", int(self.replay_frame.value)))
                gui.add_markdown("Playback can skip display frames to follow elapsed wall time; the HDF5 labels are read unchanged and no ray tracing is recomputed. Action controls are rejected in read-only replay.")
        with gui.add_folder("Simulation"):
            self.controls["play"] = gui.add_checkbox("Play", initial_value=False)
            self.controls["play"].on_update(lambda _: None if self.syncing else self._command("play", self.controls["play"].value))
            gui.add_button("Pause").on_click(lambda _: self._command("pause"))
            gui.add_button("Step").on_click(lambda _: self._command("step"))
            gui.add_button("Stop and brake").on_click(lambda _: self._command("stop"))
            gui.add_button("Reset episode").on_click(lambda _: self._command("reset"))
            gui.add_button("Return home").on_click(lambda _: self._command("home"))
            speed = gui.add_slider("Simulation speed", min=.1, max=4., step=.1, initial_value=1.)
            speed.on_update(lambda _: self._command("speed", speed.value))
            self.mode = gui.add_dropdown("Motion mode", options=("Manual", "Joint sweep", "Rotation sweep", "External policy"), initial_value="Manual")
            self.mode.on_update(lambda _: self._command("mode", self.mode.value))
            gui.add_button("Focus BS").on_click(lambda e: self._aim_bs(e.client))
            gui.add_button("Focus selected panel").on_click(lambda e: self._aim_panel(e.client))
            gui.add_button("Focus BS and users").on_click(lambda e: self._aim_world(e.client))
            self.status = gui.add_markdown("Starting complete system…")
        with gui.add_folder("Independent / joint panel control"):
            self.panel_select = gui.add_dropdown("Select Panel", options=tuple(f"Panel {b+1:02d}" for b in range(16)), initial_value="Panel 01")
            self.panel_select.on_update(lambda _: None if self.syncing else self._select(int(self.panel_select.value.split()[-1])-1, source="dropdown"))
            gui.add_button("Select All").on_click(lambda _: self._select_all())
            gui.add_button("Select only current").on_click(lambda _: self._select_only())
            self.checkboxes = [gui.add_checkbox(f"Include Panel {b+1:02d}", initial_value=b==0) for b in range(16)]
            for b, checkbox in enumerate(self.checkboxes):
                checkbox.on_update(lambda _, b=b: self._multi_update(b))
            self.selection_info = gui.add_markdown("1 selected panel")
            self.target_numbers = [gui.add_number("Target "+axis, initial_value=0., step=.002) for axis in "XYZ"]
            self.target_angles = [gui.add_number(name, initial_value=0., step=.2) for name in ("Yaw", "Pitch", "Roll")]
            self.translation_speed = gui.add_number("Translation ceiling (m/s)", initial_value=float(self.physical_config.get("max_center_velocity", .6)), min=.001, step=.01)
            self.rotation_speed = gui.add_number("Rotation ceiling (rad/s)", initial_value=float(self.physical_config.get("max_panel_angular_velocity", .2)), min=.001, step=.01)
            gui.add_button("Apply speed ceilings").on_click(lambda _: self._command("mechanics", {"max_center_velocity": self.translation_speed.value, "max_panel_angular_velocity": self.rotation_speed.value}))
            self.rod_length = gui.add_number("Rod target length (m)", initial_value=1.1, min=.01, step=.002)
            gui.add_button("Send rod length").on_click(lambda _: self._submit_rod())
            self.policy = gui.add_dropdown("Command policy", options=("replace", "queue"), initial_value="replace")
            gui.add_button("Send target").on_click(lambda _: self._numeric_target())
            self.preset = gui.add_dropdown("Action preset", options=("Joint", "Translation", "Rotation", "Stay"), initial_value="Joint")
            gui.add_button("Execute selected preset").on_click(lambda _: self._command("preset", {"kind": self.preset.value.lower(), "panels": sorted(self.selected_panels)}))
            self.gizmo_mode = gui.add_dropdown("Mouse gizmo", options=("Translation", "Rotation", "Translation + Rotation", "Hidden"), initial_value="Translation")
            self.gizmo_mode.on_update(self._gizmo_mode_update)
            gui.add_markdown("Click a physical panel in 3D to select it. Drag its target gizmo; the actual panel follows bounded mechanics. Multi-panel numeric/mouse commands apply the selected panel's rigid displacement to each selected target. Safety rejects infeasible requests with a reason.")
            self.action_info = gui.add_markdown("No submitted action.")
        with gui.add_folder("Mechanical safety"):
            self.workspace_visible = gui.add_checkbox("Show selected workspace bounds", initial_value=False)
            self.workspace_visible.on_update(lambda _: [setattr(h, "visible", self.workspace_visible.value) for h in getattr(self, "workspace_handles", [])])
            gui.add_markdown("Rod length shells show bounds for the selected anchor; angles, collisions and the full trajectory further restrict feasible reach. Ideal mode displays the canonical cube.")
            self.mechanics_info = gui.add_markdown("Waiting for mechanics.")
        with gui.add_folder("Map and visible propagation paths"):
            self.map_mode = gui.add_dropdown("Display map", options=("Nearby actual geometry", "Entire city", "Hidden"), initial_value="Nearby actual geometry")
            self.map_mode.on_update(lambda _: self._map_visibility())
            self.map_description = gui.add_markdown("RT uses the complete scene; visual ROI changes display only.")
            self.ray_visible = gui.add_checkbox("Show rays", initial_value=True)
            self.ray_scope = gui.add_dropdown("Path scope", options=("Selected panel and UE", "Selected UE", "All metadata"), initial_value="Selected panel and UE")
            self.depth_filter = gui.add_dropdown("Reflection depth", options=("All", "0", "1", "2", "3+"), initial_value="All")
            self.top_k = gui.add_slider("Strongest visible paths", min=1, max=64, step=1, initial_value=12)
            self.path_select = gui.add_dropdown("Path details", options=("None",), initial_value="None")
            self.paths_info = gui.add_markdown("Waiting for path metadata.")
            self.path_info = gui.add_markdown("Select a path to inspect its actual parameters.")
        with gui.add_folder("Multiuser wireless · completed CSI"):
            count = len(_users(self.last_state))
            self.user_select = gui.add_dropdown("Select UE", options=tuple(f"UE {i+1:02d}" for i in range(count)) or ("No UE",), initial_value="UE 01" if count else "No UE")
            self.subcarrier = gui.add_dropdown("Stored subcarrier", options=("0",), initial_value="0")
            self.antenna = gui.add_dropdown("Frequency trace antenna", options=tuple(str(i) for i in range(64)), initial_value="0")
            self.wireless_info = gui.add_markdown("No completed CSI; awaiting the actual propagation worker.")
            self.channel_plot = gui.add_html("Waiting for the selected user's 64-antenna channel.")
            self.frequency_plot = gui.add_html("Waiting for stored frequency samples.")
            self.delay_plot = gui.add_html("Waiting for path delays.")
            self.rate_plot = gui.add_html("Waiting for rates.")
        with gui.add_folder("Performance and timestamps"):
            self.diagnostics = gui.add_markdown("Waiting for measured clocks.")
            gui.add_markdown("Live CSI may arrive after physical motion. Strict offline simulation must retain every requested label. Browser draw-frame proxy, server render rate, mechanics tick rate and CSI completion rate are distinct measurements.")

    def _panel_colors(self):
        return np.asarray([(50, 210, 165) if b==self.selected else ((70, 155, 250) if b in self.selected_panels else (120, 145, 175)) for b in range(16)], np.uint8)

    def _command(self, kind, payload=None):
        try:
            result = self.controller.command(kind, payload)
            self.last_action_result = {"kind": kind, "result": jsonable(result)}
            return result
        except Exception as exc:
            self.last_action_result = {"kind": kind, "error": f"{type(exc).__name__}: {exc}"}
            return None

    def _action(self, action, source):
        try:
            result = self.controller.action(action)
            self.last_action_result = {"source": source, "request": jsonable(action), "result": jsonable(result)}
            return result
        except Exception as exc:
            self.last_action_result = {"source": source, "request": jsonable(action), "error": f"{type(exc).__name__}: {exc}"}
            return None

    def _select(self, b, source="ui"):
        with self.lock:
            self.selected = int(b)
            if not 0 <= self.selected < 16:
                raise ValueError("Invalid panel index")
            self.selected_panels.add(self.selected)
            self.selection_events += 1
            self.syncing = True
            self.panel_select.value = f"Panel {b+1:02d}"
            for j, c in enumerate(self.checkboxes):
                c.value = j in self.selected_panels
            state, _, _ = self.controller.snapshot()
            p, r = state["target_positions"][b], state["target_rotations"][b]
            for control, v in zip(self.target_numbers, p):
                control.value = float(v)
            for control, v in zip(self.target_angles, matrix_to_euler(r)):
                control.value = float(v)
            rod = state.get("rod_states", [None]*16)[b]
            self.rod_length.disabled = rod is None
            if rod is not None:
                self.rod_length.value = float(rod["length"])
            self.gizmo.position = p; self.gizmo.wxyz = quaternion(r)
            self.syncing = False
            self.panel_mesh.batched_colors = self._panel_colors()
            self.selection_info.content = f"**Selected panel: Panel {b+1:02d}** · {len(self.selected_panels)} included: "+", ".join(f"{j+1:02d}" for j in sorted(self.selected_panels))
            self.trail.clear()

    def _multi_update(self, b):
        if self.syncing:
            return
        if self.checkboxes[b].value:
            self.selected_panels.add(b)
        else:
            self.selected_panels.discard(b)
        self.panel_mesh.batched_colors = self._panel_colors()
        self.selection_info.content = f"**Selected panel: Panel {self.selected+1:02d}** · {len(self.selected_panels)} included: "+", ".join(f"{j+1:02d}" for j in sorted(self.selected_panels))

    def _select_all(self):
        self.selected_panels = set(range(16)); self._select(self.selected)

    def _select_only(self):
        self.selected_panels = {self.selected}; self._select(self.selected)

    def _panel_click(self, event):
        state, _, _ = self.controller.snapshot()
        b = getattr(event, "instance_index", None)
        if b is None:
            ray = np.asarray(event.ray_direction, float); ray /= np.linalg.norm(ray)
            rel = np.asarray(state["positions"])-np.asarray(event.ray_origin)
            residual = rel-np.sum(rel*ray, axis=1)[:, None]*ray
            b = int(np.argmin(np.linalg.norm(residual, axis=1)))
        with self.lock:
            self.panel_click_events += 1
        self._select(b, source="actual_scene_mouse")

    def _joint_target(self, p, r, source):
        state, _, _ = self.controller.snapshot()
        b = self.selected
        original_p = np.asarray(state["target_positions"])[b]
        original_r = np.asarray(state["target_rotations"])[b]
        delta = np.asarray(r)@original_r.T
        displacement = np.asarray(p)-original_p
        panels = {j: {"position": (np.asarray(state["target_positions"])[j]+displacement).tolist(),
                       "rotation_matrix": (delta@np.asarray(state["target_rotations"])[j]).tolist()}
                  for j in sorted(self.selected_panels)}
        if not panels:
            self.last_action_result = {"error": "No panels selected"}; return
        return self._action({"panels": panels, "policy": self.policy.value}, source)

    def _numeric_target(self):
        p = np.asarray([x.value for x in self.target_numbers])
        r = euler_to_matrix(np.asarray([x.value for x in self.target_angles]))
        self._joint_target(p, r, "numeric_target")

    def _submit_rod(self):
        state, _, _ = self.controller.snapshot()
        rods = state.get("rod_states", [None]*16)
        anchors = state.get("rod_anchors")
        if anchors is None or any(rods[j] is None for j in self.selected_panels):
            self.last_action_result = {"error": "Rod control is unavailable in ideal pose mode"}; return
        panels = {j: {"position": (np.asarray(anchors[j])+self.rod_length.value*np.asarray(rods[j]["direction"])).tolist(),
                       "rotation_matrix": np.asarray(state["target_rotations"])[j].tolist()}
                  for j in sorted(self.selected_panels)}
        self._action({"panels": panels, "policy": self.policy.value}, "rod_length_FK_target")

    def _gizmo_update(self, event):
        phase = getattr(event, "phase", None)
        self.dragging = phase != "end"; self.last_mouse_wall = time.monotonic()
        with self.lock:
            self.gizmo_events += 1; self.gizmo_drag_ends += int(phase == "end")
        p, r = np.asarray(self.gizmo.position), rotation(self.gizmo.wxyz)
        self.syncing = True
        for control, v in zip(self.target_numbers, p):
            control.value = float(v)
        for control, v in zip(self.target_angles, matrix_to_euler(r)):
            control.value = float(v)
        self.syncing = False
        # Send only completed drags: avoid invoking full safety planning at every
        # pointer packet, and preserve the last certified target while dragging.
        if phase in {None, "end"}:
            self._joint_target(p, r, "actual_mouse_gizmo")

    def _gizmo_mode_update(self, _):
        self.gizmo.visible = self.gizmo_mode.value != "Hidden"
        self.gizmo.disable_axes = self.gizmo_mode.value == "Rotation"
        self.gizmo.disable_sliders = self.gizmo_mode.value == "Rotation"
        self.gizmo.disable_rotations = self.gizmo_mode.value == "Translation"

    def _aim_bs(self, client):
        if client is None:
            return
        state, _, _ = self.controller.snapshot()
        center = _center(state)
        radius = max(.7, float(np.max(np.linalg.norm(np.asarray(state["positions"])-center, axis=1))))
        client.camera.position = center+np.array([2.1, -2.3, 1.5])*radius
        client.camera.look_at = center
        client.camera.up_direction = np.array([0., 0., 1.])
        client.camera.fov = np.deg2rad(52.)

    def _aim_panel(self, client):
        if client is None:
            return
        state, _, _ = self.controller.snapshot()
        p = np.asarray(state["positions"])[self.selected]
        client.camera.position = p+np.array([1.3, -1.5, .8])
        client.camera.look_at = p
        client.camera.up_direction = np.array([0., 0., 1.])
        client.camera.fov = np.deg2rad(45.)

    def _aim_world(self, client):
        if client is None:
            return
        state, _, _ = self.controller.snapshot()
        users, center = _users(state), _center(state)
        focus = center if not len(users) else (center+users.mean(0))/2
        distance = 15. if not len(users) else max(15., np.max(np.linalg.norm(users-center, axis=1))*.8)
        client.camera.position = focus+np.array([.8, -.7, .65])*distance
        client.camera.look_at = focus

    def _change_mesh(self):
        asset = getattr(self.controller, "asset", None)
        if asset is None:
            engine = getattr(self.controller, "engine", None)
            asset = getattr(engine, "asset", None)
        if asset is None:
            return
        signature = getattr(asset, "scene_hash", id(asset))
        if signature == self.asset_signature:
            return
        for handle in self.map_handles:
            handle.remove()
        self.map_handles.clear()
        fullv, fullf, count = _scene_meshes(asset, self.center, None)
        nearv, nearf, _ = _scene_meshes(asset, self.center, self.map_radius)
        for name, vv, ff in (("full", fullv, fullf), ("nearby", nearv, nearf)):
            self.map_handles.append(self.server.scene.add_mesh_simple("/map/"+name, vv, ff, color=(89, 101, 116),
                opacity=1., side="double", cast_shadow=False, receive_shadow=False, visible=name=="nearby"))
        self.asset_signature = signature
        self.map_info = {"scene_name": getattr(asset, "name", "unknown"), "scene_hash": str(signature),
                         "source": getattr(asset, "source", "unreported"), "full_faces": count,
                         "nearby_faces": len(nearf), "visual_roi_radius_m": self.map_radius,
                         "rt_geometry_unchanged": True, "mesh_uploads_per_asset": 2}
        self.map_description.content = (f"**{self.map_info['scene_name']}** · {count:,} actual mesh triangles; nearby display {len(nearf):,} triangles within {self.map_radius:g} m.\n\nFull RT geometry is unchanged. The map buffers are uploaded once per asset; the selector changes visibility only.")
        self._map_visibility()

    def _map_visibility(self):
        if len(self.map_handles) == 2:
            self.map_handles[0].visible = self.map_mode.value == "Entire city"
            self.map_handles[1].visible = self.map_mode.value == "Nearby actual geometry"

    def telemetry(self):
        clients = []
        for client in self.server.get_clients().values():
            try:
                clients.append({"client_id": client.client_id, "camera": {"position": client.camera.position,
                    "wxyz": client.camera.wxyz, "look_at": client.camera.look_at, "fov": client.camera.fov,
                    "image_width": client.camera.image_width, "image_height": client.camera.image_height}})
            except AssertionError:
                pass
        with self.lock:
            return jsonable({"selected_panel": self.selected, "included_panels": sorted(self.selected_panels),
                "gizmo_events": self.gizmo_events, "gizmo_drag_ends": self.gizmo_drag_ends,
                "panel_click_events": self.panel_click_events, "selection_events": self.selection_events,
                "render_frames": self.render_frames, "server_render_average_hz": self.render_frames/max(time.monotonic()-self.started_wall, 1e-12),
                "render_target_hz": self.render_hz, "browser_fps": self.browser_fps,
                "map": self.map_info, "clients": clients, "error": self.error,
                "latest_action_submission": self.last_action_result, "ui_source_sha256": getattr(self, "source_hash", None),
                "visible_paths_only": len(self.displayed_paths), "gpu_display_present_fps_measured": False})

    def _update_geometry(self, state):
        p, r = np.asarray(state["positions"]), np.asarray(state["rotations"])
        target_p, target_r = np.asarray(state["target_positions"]), np.asarray(state["target_rotations"])
        version = (state.get("session"), state.get("mechanical_version"), _time(state), self.selected)
        if version != self.last_version:
            with self.server.atomic():
                self.panel_mesh.batched_positions = p; self.panel_mesh.batched_wxyzs = quaternion(r)
                self.target_mesh.batched_positions = target_p; self.target_mesh.batched_wxyzs = quaternion(target_r)
                self.elements.batched_positions = np.asarray(state["element_positions"]).reshape(64, 3)
                self.panel_frame.position = p[self.selected]; self.panel_frame.wxyz = quaternion(r[self.selected])
                self.target_frame.position = target_p[self.selected]; self.target_frame.wxyz = quaternion(target_r[self.selected])
                if not self.dragging and time.monotonic()-self.last_mouse_wall > .4:
                    self.gizmo.position = target_p[self.selected]; self.gizmo.wxyz = quaternion(target_r[self.selected])
                for j, label in enumerate(self.panel_labels):
                    label.position = p[j]+[0, 0, .12]
                anchors = state.get("rod_anchors")
                if anchors is not None:
                    for j, rod in enumerate(self.rods):
                        direction = p[j]-anchors[j]
                        rod.wxyz = quaternion(_rod_rotation(direction))
                        rod.scale = (1., 1., float(np.linalg.norm(direction)))
                if not self.trail or np.linalg.norm(p[self.selected]-self.trail[-1]) > .0005:
                    self.trail.append(p[self.selected].copy())
                if len(self.trail)>1:
                    tr = np.asarray(self.trail, np.float32)
                    self.trail_handle.points = np.stack((tr[:-1], tr[1:]), axis=1)
            self.last_version = version
        users = _users(state)
        user_options = tuple(f"UE {i+1:02d}" for i in range(len(users))) or ("No UE",)
        if self.user_select.options != user_options:
            self.user_select.options = user_options; self.user_select.value = user_options[0]
        if self.last_user_positions is None or not np.array_equal(users, self.last_user_positions):
            self.user_cloud.points = users.astype(np.float32)
            self.last_user_positions = users.copy()
        k = self._selected_user(users)
        self.user_selected.points = users[k:k+1].astype(np.float32)
        self.user_selected_label.visible = bool(len(users))
        if len(users):
            self.user_selected_label.position = users[k]+[0, 0, 1.]
            self.user_selected_label.text = f"UE {k+1:02d} · stored user index {k}"
            if self.last_selected_user != k:
                self.user_trail.clear(); self.last_selected_user = k
            if not self.user_trail or np.linalg.norm(users[k]-self.user_trail[-1])>.005:
                self.user_trail.append(users[k].copy())
            if len(self.user_trail)>1:
                ut = np.asarray(self.user_trail, np.float32)
                self.user_trail_handle.points = np.stack((ut[:-1], ut[1:]), axis=1)
        if state.get("rod_anchors") is not None:
            for h in self.workspace_handles:
                h.position = state["rod_anchors"][self.selected]
        active = set()
        for actor in state.get("world", {}).get("actors", []):
            identifier = str(actor["id"])
            active.add(identifier)
            if identifier not in self.actor_handles:
                vv, ff = _box(actor["size"])
                node = self.server.scene.add_mesh_simple("/actors/"+identifier.replace("/", "_")+"/mesh", vv, ff,
                    color=(220, 105, 65), cast_shadow=False, receive_shadow=False)
                self.actor_handles[identifier] = node
            node = self.actor_handles[identifier]
            node.position = actor["position"]
            node.wxyz = quaternion(actor.get("rotation_matrix", np.eye(3)))
        for identifier in set(self.actor_handles)-active:
            self.actor_handles.pop(identifier).remove()

    def _selected_user(self, users):
        value = 0 if self.user_select.value == "No UE" else int(self.user_select.value.split()[-1])-1
        return min(max(value, 0), max(len(users)-1, 0))

    def _update_channel(self, state, result, stats):
        if result is None:
            return
        h = np.asarray(_result_field(result, "cfr"))
        if h.ndim != 3 or h.shape[1] != 64:
            raise ValueError(f"Completed V4 CSI must have shape [K,64,F], got {h.shape}")
        users = _users(state); k = min(self._selected_user(users), max(len(h)-1, 0))
        if not len(h):
            self.wireless_info.content = "Poisson realization has zero users; complete channel shape is [0,64,F]."; return
        offsets = _result_field(result, "frequency_offsets_hz", _result_field(result, "frequency_offsetsHz"))
        radio = self.config.get("radio", {})
        logical_count = int(radio.get("num_subcarriers", h.shape[2]))
        logical_indices = np.asarray(radio.get("subcarrier_indices", np.arange(h.shape[2])), int)
        if offsets is None:
            if len(logical_indices) != h.shape[2]:
                raise ValueError("CSI frequency mapping does not match the configured stored subcarriers")
            offsets = (logical_indices-logical_count//2)*float(radio.get("subcarrier_spacing", 15000.))
        else:
            offsets = np.asarray(offsets, float)
        if offsets.shape != (h.shape[2],):
            raise ValueError("Completed CSI frequency offsets do not match its frequency dimension")
        options = tuple(f"{i}: SC {logical_indices[i] if len(logical_indices)==len(offsets) else '?'} · {v/1e6:+.4g} MHz" for i, v in enumerate(offsets))
        if self.subcarrier.options != options:
            self.subcarrier.options = options; self.subcarrier.value = options[0]
        sc = int(self.subcarrier.value.split(":")[0]); antenna = int(self.antenna.value)
        csi_t = float(_result_field(result, "t", _result_field(result, "timestamp", 0.)))
        key = (_result_field(result, "session", state.get("session")), _result_field(result, "request_id", csi_t), k, sc, antenna)
        if key != self.last_channel_key:
            self.channel_plot.content = (_svg(20*np.log10(np.maximum(np.abs(h[k, :, sc]), 1e-30)), "All 64 BS antennas · magnitude dB")+
                _svg(np.angle(h[k, :, sc]), "All 64 BS antennas · phase rad", color="#fbbf24"))
            self.frequency_plot.content = (_svg(20*np.log10(np.maximum(np.abs(h[k, antenna]), 1e-30)), f"Antenna {antenna} · stored frequency response", offsets/1e6)+
                _svg(np.angle(h[k, antenna]), "Stored frequency response · phase rad", offsets/1e6, color="#fbbf24"))
            if len(offsets)>1 and np.allclose(np.diff(offsets), np.diff(offsets)[0], rtol=1e-9, atol=1e-9):
                self.delay_plot.content = _svg(np.abs(np.fft.ifft(np.fft.ifftshift(h[k, antenna])))**2, "IFFT over stored uniformly spaced CFR · finite-band delay power", color="#e879f9")
            else:
                paths = [x for x in _result_field(result, "paths", _result_field(result, "path_metadata", [])) if int(x.get("user", -1))==k]
                self.delay_plot.content = _svg([_path_power(x) for x in paths], "Actual path power · delays available in Path details", color="#e879f9")+"<div>Stored frequencies are sparse/nonuniform; a full-band IFFT CIR is not reconstructed from missing labels.</div>"
            self.last_channel_key = key
            metrics = _result_field(result, "metrics", {})
            rate = metrics.get("uplink_capacity_bits_per_s_hz") if metrics.get("mode", "uplink")=="uplink" else metrics.get("downlink_sum_rate_bps")
            if rate is not None and np.isfinite(rate):
                if not self.rate_history or csi_t>self.rate_history[-1][0]:
                    self.rate_history.append((csi_t, float(rate)))
                self.rate_plot.content = _svg([x[1] for x in self.rate_history],
                    "UL joint-decoding capacity · bit/s/Hz" if metrics.get("mode", "uplink")=="uplink" else "Scheduled DL sum rate · bit/s")
        if time.monotonic()-self.last_stats_wall < .5:
            return
        metrics = _result_field(result, "metrics", {})
        metadata = _result_field(result, "cache_metadata", {})
        selected_metrics = {}
        for name, value in metrics.items():
            arr = np.asarray(value) if isinstance(value, (list, tuple, np.ndarray)) else None
            selected_metrics[name] = jsonable(arr[k] if arr is not None and arr.ndim and len(arr)==len(h) else value)
        self.wireless_info.content = (f"**Full CSI tensor: {h.shape[0]} users × 64 antennas × {h.shape[2]} stored frequencies**\n\n"
            f"Last completed CSI timestamp: **{csi_t:.6f} s** · current mechanics: {_time(state):.6f} s · CSI age: {max(0., _time(state)-csi_t):.6f} s\n\n"
            f"Selected UE {k+1:02d}; solver **{_result_field(result, 'level', 'unreported')}**, latency {_result_field(result, 'latency_ms', 'unreported')} ms.\n\n"
            f"Label model: `{_result_field(result, 'label_model', 'unreported')}`.\n\n"
            "```json\n"+json.dumps(jsonable(selected_metrics), ensure_ascii=False, indent=2)[:2000]+"\n```\n"
            f"Cache / propagation provenance: `{json.dumps(jsonable(metadata))[:500]}`")
        cp = _result_field(result, "positions")
        cr = _result_field(result, "rotations")
        if cp is not None and cr is not None:
            self.csi_pose.visible = True; self.csi_pose.position = np.asarray(cp)[self.selected]; self.csi_pose.wxyz = quaternion(np.asarray(cr)[self.selected])

    def _update_paths(self, state, result):
        if result is None:
            return
        k = self._selected_user(_users(state))
        key = (_result_field(result, "request_id", _result_field(result, "t")), self.selected, k,
               self.ray_scope.value, self.depth_filter.value, self.top_k.value, self.ray_visible.value)
        if key != self.last_path_key:
            all_paths = _result_field(result, "paths", _result_field(result, "path_metadata", []))
            paths = all_paths
            if self.ray_scope.value != "All metadata":
                paths = [p for p in paths if int(p.get("user", -1))==k]
            if self.ray_scope.value == "Selected panel and UE":
                paths = [p for p in paths if int(p.get("panel", -1))==self.selected]
            if self.depth_filter.value != "All":
                d = self.depth_filter.value
                paths = [p for p in paths if (_depth(p)>=3 if d=="3+" else _depth(p)==int(d))]
            self.displayed_paths = sorted(paths, key=_path_power, reverse=True)[:int(self.top_k.value)]
            for handle in self.path_handles:
                handle.remove()
            self.path_handles.clear()
            palette = [(60, 205, 100), (60, 150, 245), (255, 171, 50), (226, 99, 217)]
            for d in range(4):
                lines, points = [], []
                for path in self.displayed_paths:
                    if min(_depth(path), 3) != d:
                        continue
                    pp = np.asarray(path.get("points", []), np.float32)
                    if len(pp)>1:
                        lines.extend(np.stack((pp[:-1], pp[1:]), axis=1)); points.extend(pp[1:-1])
                if lines:
                    self.path_handles.append(self.server.scene.add_line_segments(f"/paths/depth_{d}", np.asarray(lines), palette[d],
                        thickness=1.5, thickness_units="screen", visible=self.ray_visible.value))
                if points:
                    self.path_handles.append(self.server.scene.add_point_cloud(f"/paths/interactions_{d}", np.asarray(points), palette[d],
                        point_size=.2, visible=self.ray_visible.value))
            options = ("None",)+tuple(f"{i}: P{p.get('panel', '?')} UE{p.get('user', '?')} · {p.get('path_id', 'unknown')}" for i, p in enumerate(self.displayed_paths))
            self.path_select.options = options; self.path_select.value = "None"
            self.paths_info.content = f"Showing {len(self.displayed_paths)} strongest paths from {len(all_paths)} display metadata records.\n\nLoS **green**, single **blue**, double **orange**, triple+ **magenta**. Backend packed propagation paths and full CSI are unaffected by display filters.\n\nPath events: `{json.dumps(jsonable(_result_field(result, 'path_events', {})))[:500]}`"
            self.last_path_key = key
        detail_key = (self.last_path_key, self.path_select.value)
        if detail_key != self.last_path_detail_key:
            if self.path_select.value != "None":
                i = int(self.path_select.value.split(":")[0])
                self.path_info.content = "```json\n"+json.dumps(jsonable(self.displayed_paths[i]), indent=2)[:5000]+"\n```"
            else:
                self.path_info.content = "Select a path to inspect its actual delay, gain, reflection sequence and Doppler."
            self.last_path_detail_key = detail_key

    def tick(self):
        state, result, stats = self.controller.snapshot()
        self.last_state = state
        self._change_mesh()
        self._update_geometry(state)
        self._update_channel(state, result, stats)
        self._update_paths(state, result)
        if time.monotonic()-self.last_stats_wall >= .5:
            self.syncing = True; self.controls["play"].value = bool(state.get("playing", False)); self.syncing = False
            if getattr(self.controller, "is_replay", False):
                self.syncing = True
                self.replay_frame.max = max(1, self.controller.frame_count-1)
                self.replay_frame.value = self.controller.frame_index
                self.syncing = False
            self.status.content = f"**{'Playing' if state.get('playing') else 'Paused'}** · physical time **{_time(state):.6f} s**\n\n16 panels · 4 elements each · 64 BS antennas · **{len(_users(state))} users**\n\nSession {state.get('session', 0)} · mechanical version {state.get('mechanical_version', 0)}"
            b = self.selected
            p, target = np.asarray(state["positions"])[b], np.asarray(state["target_positions"])[b]
            velocity = np.asarray(state.get("velocities", np.zeros((16, 3))))[b]
            omega = np.asarray(state.get("angular_velocities", np.zeros((16, 3))))[b]
            remaining = np.asarray(state.get("remaining_time", np.zeros(16)))[b]
            rod = state.get("rod_states", [None]*16)[b]
            remain_text = str(remaining) if "remaining_time" in state else "not stored in replay"
            self.mechanics_info.content = (f"**Panel {b+1:02d}** · actual {np.round(p, 5).tolist()}\n\nTarget {np.round(target, 5).tolist()} · remaining distance {np.linalg.norm(target-p):.6f} m\n\n"
                f"Linear speed {np.linalg.norm(velocity):.6f} m/s; angular speed {np.linalg.norm(omega):.6f} rad/s ({np.rad2deg(np.linalg.norm(omega)):.4f} °/s).\n\n"
                f"Remaining travel estimate {remain_text}; rod length {None if rod is None else rod.get('length')} m.\n\nSafety: `{json.dumps(jsonable(state.get('safety', {})))[:900]}`")
            actual_result = stats.get("action_result", self.last_action_result)
            rejection = ""
            if isinstance(actual_result, dict):
                if actual_result.get("accepted") is False:
                    rejection = "**Action rejected:** "+str(actual_result.get("reasons", actual_result.get("reason", "unreported")))+"\n\n"
                elif actual_result.get("accepted") is True:
                    rejection = "**Action accepted by the mechanical safety checker.**\n\n"
            self.action_info.content = rejection+"Latest request / execution / rejection:\n\n```json\n"+json.dumps(jsonable(actual_result), indent=2)[:2200]+"\n```"
            self.diagnostics.content = f"Server render average **{self.render_frames/max(time.monotonic()-self.started_wall, 1e-12):.2f} Hz** (target {self.render_hz:g}).\n\n```json\n"+json.dumps(jsonable(stats), indent=2)[:3000]+"\n```"
            error = state.get("error") or stats.get("error")
            if error:
                self.status.content += f"\n\n**Backend error:** `{error}`"
            self.last_stats_wall = time.monotonic()
        self.render_frames += 1

    def run(self, stop_event=None):
        deadline = time.monotonic()
        try:
            while stop_event is None or not stop_event.is_set():
                self.tick()
                deadline += 1/self.render_hz
                delay = deadline-time.monotonic()
                if delay>0:
                    time.sleep(min(delay, .05))
                elif delay < -.5:
                    deadline = time.monotonic()
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.close()

    def close(self):
        self.api.close(); self.server.stop()


def start_v4_visualization(config, port=8092, *, host="127.0.0.1", api_port=None,
                           controller=None, controller_factory=None, stop_event=None, render_hz=30.,
                           dataset_path=None, episode_id=None):
    """Run V4 live visualizer; an injected controller owns its own lifecycle."""
    own_controller = controller is None
    if own_controller:
        if dataset_path is not None:
            controller = V4ReplayController(dataset_path, episode_id).start()
            config = controller.config
        elif controller_factory is None:
            from .live import V4LiveController
            controller_factory = V4LiveController
            controller = controller_factory(config).start()
        else:
            controller = controller_factory(config).start()
    try:
        viewer = V4Visualization(controller, config, port=port, host=host, api_port=api_port, render_hz=render_hz)
        viewer.run(stop_event)
    finally:
        if own_controller:
            controller.close()


class V4ReplayController:
    """Read-only, one-CSI-frame-at-a-time HDF5 replay; never imports RT."""
    is_replay = True

    def __init__(self, dataset_path, episode_id=None):
        import h5py
        from types import SimpleNamespace
        from .mechanics import kinematic_preset
        self.path = Path(dataset_path).resolve()
        self.file = h5py.File(self.path, "r")
        if self.file.attrs.get("schema_version") != "wiwo-ma-v4-fullscale-1.0":
            self.file.close(); raise ValueError("Replay requires an actual V4 full-scale HDF5 dataset")
        self.config = json.loads(self.file.attrs["config_json"])
        self.episode_ids = tuple(self.file["episodes"].keys())
        if not self.episode_ids:
            self.file.close(); raise ValueError("Dataset contains no saved episodes")
        self.episode_id = self.episode_ids[0] if episode_id is None else str(episode_id)
        self.lock = threading.RLock(); self.closed = threading.Event()
        self.playing = False; self.speed = 1.; self.session = 1; self.error = None
        self.bs = SimpleNamespace(mechanical_config=kinematic_preset(self.config.get("mechanics_v4", {}).get("preset", "kinematic_twc2026_conservative"), self.config.get("mechanics_v4", {})))
        meshes = []
        if "scene_geometry" in self.file:
            for key, node in self.file["scene_geometry"].items():
                meshes.append({"name": str(node.attrs.get("name", key)), "vertices": node["vertices"][:], "faces": node["faces"][:]})
        self.asset = SimpleNamespace(name=self.config["scene"]["name"],
            source="Actual geometry stored in the dataset; replay does not recompute RT" if meshes else "No scene geometry stored in this analytic episode; replay does not launch RT",
            scene_hash=str(self.file.get("scene_geometry", {}).attrs.get("scene_hash", "unreported")) if "scene_geometry" in self.file else "missing_stored_geometry",
            meshes=meshes)
        self.command_result = None; self._select_episode(self.episode_id)
        self.thread = threading.Thread(target=self._run, name="v4-readonly-replay", daemon=True)

    def _select_episode(self, identifier):
        if identifier not in self.episode_ids:
            raise ValueError("Unknown saved episode")
        self.episode_id = identifier; self.group = self.file["episodes"][identifier]
        self.timestamps = self.group["timestamps"][:]
        if not len(self.timestamps) or self.group["cfr"].ndim != 4 or self.group["cfr"].shape[2] != 64:
            raise ValueError("Saved episode must contain real nonempty [T,K,64,F] labels")
        self.frame_count = len(self.timestamps); self.frame_index = 0
        from .world import ExternalWorld
        self.replay_world = ExternalWorld(self.config)
        self._load(0)

    @staticmethod
    def _text(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    def _load(self, index):
        self.frame_index = int(np.clip(index, 0, self.frame_count-1))
        g, i = self.group, self.frame_index
        center = np.asarray(self.config["bs"]["centre_position"])
        rod_states = json.loads(self._text(g["rod_states"][i])) if "rod_states" in g else [None]*16
        positions = g["positions"][i]
        anchors = None
        if all(r is not None for r in rod_states):
            anchors = positions-np.asarray([r["length"]*np.asarray(r["direction"]) for r in rod_states])
        world = self.replay_world.state_at(float(self.timestamps[i]))
        if "world_metadata" in g:
            world.update(json.loads(self._text(g["world_metadata"][i])))
        world["user_positions"] = g["users"][i]
        world["user_velocities"] = g["user_velocities"][i]
        self.state = {"positions": positions, "rotations": g["rotations"][i],
            "element_positions": g["element_positions"][i], "velocities": g["velocities"][i],
            "angular_velocities": g["angular_velocities"][i], "target_positions": g["target_positions"][i],
            "target_rotations": g["target_rotations"][i], "timestamp": float(self.timestamps[i]), "time": float(self.timestamps[i]),
            "centre_position": center, "rod_states": rod_states, "rod_anchors": anchors,
            "completed": g["completed"][i] if "completed" in g else np.zeros(16, bool),
            "world": world, "playing": self.playing, "session": self.session,
            "mechanical_version": i, "error": None, "safety": {"scope": "saved_episode_records_not_recertified_in_replay"}}
        for name in ("progress", "remaining_time", "pending_latency", "moving", "command_ids"):
            if name in g:
                self.state[name] = g[name][i]
        if "mechanical_status" in g:
            self.state.update(json.loads(self._text(g["mechanical_status"][i])))
        meta = json.loads(self._text(g["metadata"][i])) if "metadata" in g else {}
        metrics = dict(meta.get("metrics", {}))
        if "wireless" in g:
            for name, dataset in g["wireless"].items():
                metrics[name] = dataset[i]
        self.result = {"cfr": g["cfr"][i], "t": float(self.timestamps[i]), "request_id": i,
            "session": self.session, "positions": self.state["positions"], "rotations": self.state["rotations"],
            "user_positions": world["user_positions"], "frequency_offsets_hz": g["frequency_offsets_hz"][:],
            "paths": json.loads(self._text(g["paths"][i])) if "paths" in g else [],
            "metrics": metrics, "path_events": meta.get("path_events", {}),
            "cache_metadata": meta.get("cache_metadata", {}), "level": self._text(g["level"][i]) if "level" in g else "saved",
            "label_model": meta.get("label_model", "unreported"), "latency_ms": 0.,
            "reason": "read_only_saved_HDF5_label_no_RT_recomputation"}

    def start(self):
        self.thread.start(); return self

    def _run(self):
        last = time.monotonic(); logical = float(self.timestamps[self.frame_index])
        while not self.closed.wait(.02):
            with self.lock:
                now = time.monotonic()
                if self.playing:
                    logical += (now-last)*self.speed
                    index = min(self.frame_count-1, int(np.searchsorted(self.timestamps, logical, side="right")-1))
                    if index != self.frame_index:
                        self._load(index)
                    if self.frame_index==self.frame_count-1:
                        self.playing = False
                else:
                    logical = float(self.timestamps[self.frame_index])
                last = now

    def action(self, request):
        self.command_result = {"accepted": False, "reasons": ["Replay is read-only; stored actions and CSI are unchanged"], "requested": jsonable(request)}
        return self.command_result

    def command(self, kind, payload=None):
        with self.lock:
            if kind=="play":
                self.playing = payload is not False
            elif kind in {"pause", "stop"}:
                self.playing = False
            elif kind in {"reset", "home"}:
                self.playing = False; self.session += 1; self._load(0)
            elif kind=="step":
                self._load(min(self.frame_count-1, self.frame_index+1))
            elif kind=="seek":
                self.playing = False; self._load(int(payload))
            elif kind=="episode":
                self.playing = False; self.session += 1; self._select_episode(str(payload))
            elif kind=="speed":
                value = float(payload)
                if not .1<=value<=4:
                    raise ValueError("Replay speed must lie in .1..4")
                self.speed = value
            else:
                return self.action({"kind": kind, "payload": payload})
            return {"accepted": True, "kind": kind, "saved_frame": self.frame_index}

    def snapshot(self):
        with self.lock:
            state = deepcopy(self.state); state["playing"] = self.playing; state["session"] = self.session
            return state, self.result, {"scope": "read_only_saved_V4_HDF5_replay", "queued": 0, "running": 0,
                "saved_frame": self.frame_index, "saved_frames": self.frame_count, "action_result": self.command_result,
                "dataset": str(self.path), "error": self.error, "rt_queries_in_replay": 0}

    def close(self):
        self.closed.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        self.file.close()
