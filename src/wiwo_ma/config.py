from __future__ import annotations
import copy
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = {
    "scene": {"name": "reflector", "xml_path": None},
    "radio": {"carrier_frequency": 3.5e9, "num_tx_rows": 2, "num_tx_cols": 2,
              "subcarrier_spacing": 30000., "num_subcarriers": 128,
              "pattern": "tr38901", "polarization": "V", "tx_power_dbm": 30.,
              "noise_figure_db": 7., "pilot_stride": 8},
    "simulation": {"dt": .005, "control_dt": .1, "csi_dt": .01,
                   "feedback_delay": .02, "duration": 5., "seed": 42,
                   "user_positions": [[8., 1., 1.5]], "user_velocities": [[.1, 0., 0.]],
                   "initial_position": [0., 0., 3.], "initial_orientation_deg": [0., 0., 0.],
                   "movement_penalty": 0., "rotation_penalty": 0.},
    "mechanics": {"model": "velocity", "max_speed": .2, "max_angular_speed_deg": 30.,
                  "max_acceleration": .5, "max_angular_acceleration_deg": 90.,
                  "settling_time": .02, "command_latency": .01,
                  "workspace_min": [-1., -1., 2.], "workspace_max": [1., 1., 4.],
                  "orientation_min_deg": [-180., -80., -180.],
                  "orientation_max_deg": [180., 80., 180.],
                  "communicate_while_moving": True, "execution_policy": "replace"},
    "propagation": {"method": "incremental", "anchor_radius": .15,
                    "refresh_interval": .25, "samples_per_src": 10000,
                    "max_depth": 2, "synthetic_array": True,
                    "local_radius": .01, "audit_interval": 0., "error_threshold": .01,
                    "seed": 42, "diffuse_reflection": False, "diffraction": False,
                    "loop_mode": "evaluated"},
    "dataset": {"generation_mode": "serial", "chunk_size": 256,
                "compression": "gzip", "branching_states_per_episode": 1,
                "branch_horizon_steps": 1, "action_distribution": "uniform_pose",
                "split_unit": "trajectory"},
}

def merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else copy.deepcopy(value)
    return result

def load_config(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path) as stream:
            cfg = merge(cfg, yaml.safe_load(stream) or {})
    cfg = merge(cfg, overrides)
    for key in ("dt", "control_dt", "csi_dt", "duration"):
        if cfg["simulation"][key] <= 0:
            raise ValueError(f"simulation.{key} must be positive")
    if cfg["radio"]["num_subcarriers"] < 1 or cfg["radio"]["carrier_frequency"] <= 0:
        raise ValueError("Invalid radio dimensions/frequency")
    return cfg
