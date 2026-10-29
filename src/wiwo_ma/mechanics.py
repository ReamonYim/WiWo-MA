"""Rigid-array pose execution; metres, seconds, world-frame rad/s.

Euler input is [yaw, pitch, roll], R = Rz(yaw) Ry(pitch) Rx(roll).
This is an abstract end-effector model, not a robot inverse-kinematics solver.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Protocol

import numpy as np


def euler_to_matrix(angles_deg):
    yaw, pitch, roll = np.deg2rad(np.asarray(angles_deg, dtype=float))
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]], dtype=float)


def matrix_to_euler(rotation):
    r = np.asarray(rotation, dtype=float)
    pitch = math.asin(float(np.clip(-r[2, 0], -1., 1.)))
    if abs(math.cos(pitch)) > 1e-9:
        yaw, roll = math.atan2(r[1, 0], r[0, 0]), math.atan2(r[2, 1], r[2, 2])
    else:
        yaw, roll = math.atan2(-r[0, 1], r[1, 1]), 0.
    return np.rad2deg([yaw, pitch, roll])


def rotation_exp(vector):
    v = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(v))
    k = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    if angle < 1e-8:
        return np.eye(3) + k + .5*k@k
    return np.eye(3) + math.sin(angle)/angle*k + (1-math.cos(angle))/(angle*angle)*k@k


def rotation_log(rotation):
    """Principal SO(3) logarithm, including the numerically delicate pi case."""
    r = np.asarray(rotation, dtype=float)
    angle = math.acos(float(np.clip((np.trace(r)-1)/2, -1., 1.)))
    vee = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]])
    if angle < 1e-8:
        return .5*vee
    if math.pi-angle < 1e-5:
        # A pi rotation has a symmetric rank-one (R+I)/2 matrix.
        eigval, eigvec = np.linalg.eigh((r+r.T+2*np.eye(3))/4)
        axis = eigvec[:, np.argmax(eigval)]
        if np.dot(axis, vee) < 0:
            axis = -axis
        return axis*angle
    return vee*(angle/(2*math.sin(angle)))


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return result.copy()


def _bounded_norm(v, limit):
    n = float(np.linalg.norm(v))
    return v if n <= limit or n == 0 else v*(limit/n)


@dataclass
class MechanicalState:
    position: np.ndarray
    rotation: np.ndarray
    target_position: np.ndarray
    target_rotation: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    movement_state: str = "idle"
    time: float = 0.
    completed: bool = True
    remaining_time: float = 0.
    command_id: int = 0

    def pose(self, target=False):
        p = self.target_position if target else self.position
        r = self.target_rotation if target else self.rotation
        return {"position": p.copy(), "orientation_deg": matrix_to_euler(r), "rotation_matrix": r.copy()}


class KinematicAdapter(Protocol):
    """Future hardware-specific adapter contract; no industrial arm is implied."""
    def command(self, action: dict, timestamp: float) -> None: ...
    def advance(self, dt: float, timestamp_end: float) -> MechanicalState: ...
    def snapshot(self) -> dict: ...
    def restore(self, snapshot: dict) -> None: ...


class _Trajectory:
    """Exact bounded-acceleration scalar profile following an initial brake.

    Replacing a moving command first brakes continuously, then follows a
    straight translation / shortest SO(3) arc from rest. Consequently velocity
    and acceleration bounds hold even when the new target reverses direction.
    """
    def __init__(self, start, target, velocity, speed, acceleration, angular=False):
        self.start, self.target = start.copy(), target.copy()
        self.angular = angular
        self.speed, self.acceleration = float(speed), acceleration
        v = np.asarray(velocity, dtype=float)
        self.initial_velocity = v.copy()
        self.brake_time = float(np.linalg.norm(v)/acceleration) if acceleration else 0.
        brake_displacement = v*self.brake_time/2
        self.stop = rotation_exp(brake_displacement)@start if angular else start+brake_displacement
        delta = rotation_log(target@self.stop.T) if angular else target-self.stop
        self.distance = float(np.linalg.norm(delta))
        self.direction = delta/self.distance if self.distance > 1e-14 else np.zeros(3)
        if acceleration:
            self.accel_time = min(speed/acceleration, math.sqrt(self.distance/acceleration))
            self.peak_speed = acceleration*self.accel_time
            self.cruise_time = max(0., (self.distance-acceleration*self.accel_time**2)/self.peak_speed) if self.peak_speed else 0.
        else:
            self.accel_time, self.peak_speed = 0., speed
            self.cruise_time = self.distance/speed
        self.duration = self.brake_time+2*self.accel_time+self.cruise_time

    def evaluate(self, elapsed):
        t = max(0., float(elapsed))
        if t >= self.duration-1e-12:
            return self.target.copy(), np.zeros(3)
        if t < self.brake_time:
            fraction = t/self.brake_time
            displacement = self.initial_velocity*(t-.5*t*fraction)
            pose = rotation_exp(displacement)@self.start if self.angular else self.start+displacement
            return pose, self.initial_velocity*(1-fraction)
        t -= self.brake_time
        if self.acceleration is None:
            distance, speed = self.speed*t, self.speed
        elif t < self.accel_time:
            distance, speed = .5*self.acceleration*t*t, self.acceleration*t
        elif t < self.accel_time+self.cruise_time:
            distance = .5*self.acceleration*self.accel_time**2 + self.peak_speed*(t-self.accel_time)
            speed = self.peak_speed
        else:
            remaining = 2*self.accel_time+self.cruise_time-t
            distance, speed = self.distance-.5*self.acceleration*remaining**2, self.acceleration*remaining
        displacement = self.direction*distance
        pose = rotation_exp(displacement)@self.stop if self.angular else self.stop+displacement
        return pose, self.direction*speed


class MechanicalController:
    def __init__(self, config, initial_position=None, initial_orientation_deg=None):
        self.config = deepcopy(config.get("mechanics", config))
        sim = config.get("simulation", {})
        p = _vector(initial_position if initial_position is not None else sim.get("initial_position", [0, 0, 2]), "initial_position")
        r = euler_to_matrix(initial_orientation_deg if initial_orientation_deg is not None else sim.get("initial_orientation_deg", [0, 0, 0]))
        self.lower = _vector(self.config.get("workspace_min", [-1, -1, 1]), "workspace_min")
        self.upper = _vector(self.config.get("workspace_max", [1, 1, 3]), "workspace_max")
        if np.any(self.lower > self.upper) or np.any(p < self.lower) or np.any(p > self.upper):
            raise ValueError("Initial position must lie in the valid mechanical workspace")
        self.orientation_min = _vector(self.config.get("orientation_min_deg", [-180, -89, -180]), "orientation_min_deg")
        self.orientation_max = _vector(self.config.get("orientation_max_deg", [180, 89, 180]), "orientation_max_deg")
        if np.any(self.orientation_min > self.orientation_max):
            raise ValueError("orientation_min_deg must not exceed orientation_max_deg")
        self.max_speed = float(self.config.get("max_speed", .2))
        self.max_angular_speed = math.radians(float(self.config.get("max_angular_speed_deg", 60)))
        self.max_acceleration = float(self.config.get("max_acceleration", .5))
        self.max_angular_acceleration = math.radians(float(self.config.get("max_angular_acceleration_deg", 120)))
        for name in ("max_speed", "max_angular_speed", "max_acceleration", "max_angular_acceleration"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        self.model = str(self.config.get("model", "velocity")).lower()
        self.model = {"a": "instant", "b": "velocity", "c": "acceleration", "velocity_limited": "velocity", "acceleration_limited": "acceleration"}.get(self.model, self.model)
        if self.model not in {"instant", "velocity", "acceleration"}:
            raise NotImplementedError("Hardware-specific Model D requires a real KinematicAdapter")
        self.execution_policy = self.config.get("execution_policy", "replace")
        if self.execution_policy not in {"replace", "queue"}:
            raise ValueError("execution_policy must be replace or queue")
        self.latency = float(self.config.get("command_latency", 0))
        self.settling_time = float(self.config.get("settling_time", 0))
        if self.latency < 0 or self.settling_time < 0:
            raise ValueError("Mechanical command latency and settling time cannot be negative")
        self.state = MechanicalState(p, r, p.copy(), r.copy())
        self.pending = []
        self.active_action = {"type": "stay"}
        self.translation = self.rotation_trajectory = None
        self.started = self.motion_end = self.settle_end = 0.
        self.velocity_command = None
        self.command_counter = 0

    def command(self, action, timestamp=None):
        if action is None:
            return
        if not isinstance(action, dict):
            raise TypeError("Action must be a dictionary")
        arrival = float(self.state.time if timestamp is None else timestamp)+self.latency
        if arrival < self.state.time-1e-9:
            raise ValueError("Cannot submit a command in the past")
        kind = action.get("type", "target_pose")
        if kind not in {"target_pose", "incremental", "velocity", "discrete", "stop", "stay"}:
            raise ValueError(f"Unknown mechanical action type: {kind}")
        self.command_counter += 1
        item = (arrival, self.command_counter, deepcopy(action))
        # A dwell command must not cancel a target still waiting for actuator
        # latency. It also does not interrupt an already executing target.
        if self.execution_policy == "replace" and kind != "stay":
            self.pending = [item]
        else:
            self.pending.append(item)
        self._activate_due()

    def _activate_due(self):
        if not self.pending or self.pending[0][0] > self.state.time+1e-12:
            return
        if self.execution_policy == "queue" and not self.state.completed:
            return
        _, command_id, action = self.pending.pop(0)
        kind = action.get("type", "target_pose")
        if kind == "stay":
            # Dwell never interrupts an already executing target.
            if self.state.completed:
                self.active_action = action
            self._activate_due()
            return
        if kind == "discrete":
            poses = action.get("poses", self.config.get("discrete_poses", []))
            index = int(action.get("index", 0))
            if not poses or index < 0 or index >= len(poses):
                raise ValueError("A discrete action requires a valid index and configured poses")
            action = {"type": "target_pose", **deepcopy(poses[index])}
            kind = "target_pose"
        p = self.state.position.copy()
        r = self.state.rotation.copy()
        if kind == "incremental":
            p += _vector(action.get("translation", action.get("delta_position", [0, 0, 0])), "translation")
            r = euler_to_matrix(_vector(action.get("rotation_deg", action.get("delta_orientation_deg", [0, 0, 0])), "rotation_deg"))@r
        elif kind == "target_pose":
            p = _vector(action.get("position", action.get("target_position", p)), "position")
            r = euler_to_matrix(_vector(action.get("orientation_deg", action.get("target_orientation_deg", matrix_to_euler(r))), "orientation_deg"))
        elif kind == "stop" and self.model == "acceleration":
            p += self.state.velocity*np.linalg.norm(self.state.velocity)/(2*self.max_acceleration)
            r = rotation_exp(self.state.angular_velocity*np.linalg.norm(self.state.angular_velocity)/(2*self.max_angular_acceleration))@r
        elif kind == "velocity":
            velocity = _bounded_norm(_vector(action.get("velocity", [0, 0, 0]), "velocity"), self.max_speed)
            if "angular_velocity" in action:
                omega = _vector(action["angular_velocity"], "angular_velocity")
            else:
                omega = np.deg2rad(_vector(action.get("angular_velocity_deg", [0, 0, 0]), "angular_velocity_deg"))
            self.velocity_command = (velocity, _bounded_norm(omega, self.max_angular_speed))
            self.active_action, self.state.command_id = deepcopy(action), command_id
            self.state.completed = False
            self.state.movement_state = "moving"
            self.state.remaining_time = float("inf")
            return
        # A stop brakes the current motion continuously for acceleration model.
        p = np.clip(p, self.lower, self.upper)
        orientation = np.clip(matrix_to_euler(r), self.orientation_min, self.orientation_max)
        r = euler_to_matrix(orientation)
        self.velocity_command = None
        self.active_action, self.state.command_id = deepcopy(action), command_id
        self.state.target_position, self.state.target_rotation = p.copy(), r.copy()
        self.started = self.state.time
        if self.model == "instant":
            self.state.position, self.state.rotation = p.copy(), r.copy()
            self.state.velocity, self.state.angular_velocity = np.zeros(3), np.zeros(3)
            self.motion_end = self.started
        else:
            accel = self.max_acceleration if self.model == "acceleration" else None
            angular_accel = self.max_angular_acceleration if self.model == "acceleration" else None
            self.translation = _Trajectory(self.state.position, p, self.state.velocity, self.max_speed, accel)
            self.rotation_trajectory = _Trajectory(self.state.rotation, r, self.state.angular_velocity, self.max_angular_speed, angular_accel, True)
            self.motion_end = self.started+max(self.translation.duration, self.rotation_trajectory.duration)
        self.settle_end = self.motion_end+self.settling_time
        self._update_status()

    def _update_status(self):
        remaining = max(0., self.settle_end-self.state.time)
        self.state.remaining_time = remaining
        self.state.completed = remaining < 1e-11
        self.state.movement_state = "idle" if self.state.completed else ("settling" if self.state.time >= self.motion_end-1e-11 else "moving")

    @property
    def next_event_time(self):
        events = []
        if self.pending:
            if self.execution_policy == "replace" or self.state.completed:
                events.append(max(self.state.time, self.pending[0][0]))
            elif self.velocity_command is None:
                events.append(max(self.settle_end, self.pending[0][0]))
        if not self.state.completed and self.velocity_command is None:
            events.append(self.motion_end if self.state.time < self.motion_end-1e-11 else self.settle_end)
        return min(events) if events else float("inf")

    def advance(self, dt, timestamp_end=None):
        if dt < 0:
            raise ValueError("Mechanical dt cannot be negative")
        end = self.state.time+float(dt) if timestamp_end is None else float(timestamp_end)
        if abs(end-self.state.time-float(dt)) > 1e-8:
            raise ValueError("Mechanical timestamp_end does not match dt")
        while self.state.time < end-1e-12:
            self._activate_due()
            stop = min(end, self.next_event_time)
            if stop <= self.state.time+1e-12:
                stop = end
            interval = stop-self.state.time
            if self.velocity_command is not None:
                v, omega = self.velocity_command
                old_v, old_omega = self.state.velocity.copy(), self.state.angular_velocity.copy()
                if self.model == "acceleration":
                    v = old_v+_bounded_norm(v-old_v, self.max_acceleration*interval)
                    omega = old_omega+_bounded_norm(omega-old_omega, self.max_angular_acceleration*interval)
                    displacement, angle = .5*(old_v+v)*interval, .5*(old_omega+omega)*interval
                else:
                    displacement, angle = v*interval, omega*interval
                unclipped = self.state.position+displacement
                clipped = np.clip(unclipped, self.lower, self.upper)
                proposed_rotation = rotation_exp(angle)@self.state.rotation
                proposed_angles = matrix_to_euler(proposed_rotation)
                bounded_angles = np.clip(proposed_angles, self.orientation_min, self.orientation_max)
                hits_position_boundary = np.any(np.abs(unclipped-clipped) > 1e-12)
                hits_orientation_boundary = np.any(np.abs(proposed_angles-bounded_angles) > 1e-9)
                if self.model == "acceleration" and (hits_position_boundary or hits_orientation_boundary):
                    raise RuntimeError("Velocity command reaches a pose boundary without braking clearance; submit a feasible target pose or stop command")
                self.state.position = clipped
                v = np.where(np.abs(unclipped-clipped) > 1e-12, 0., v)
                bounded_rotation = euler_to_matrix(bounded_angles)
                if hits_orientation_boundary:
                    omega = rotation_log(bounded_rotation@self.state.rotation.T)/interval
                self.state.rotation = bounded_rotation
                self.state.velocity, self.state.angular_velocity = v, omega
                self.state.target_position, self.state.target_rotation = self.state.position.copy(), self.state.rotation.copy()
            elif self.translation is not None and self.model != "instant":
                self.state.position, self.state.velocity = self.translation.evaluate(stop-self.started)
                self.state.rotation, self.state.angular_velocity = self.rotation_trajectory.evaluate(stop-self.started)
                if np.any(self.state.position < self.lower-1e-9) or np.any(self.state.position > self.upper+1e-9):
                    raise RuntimeError("A braking trajectory leaves the workspace; increase workspace clearance or use a hardware adapter")
                angles = matrix_to_euler(self.state.rotation)
                if np.any(angles < self.orientation_min-1e-7) or np.any(angles > self.orientation_max+1e-7):
                    raise RuntimeError("The SO(3) trajectory leaves the configured Euler pose envelope; use wider end-effector limits or a hardware-specific planner")
            self.state.time = stop
            if self.velocity_command is None:
                self._update_status()
            self._activate_due()
        self._activate_due()
        return self.state

    def snapshot(self):
        return deepcopy({key: value for key, value in self.__dict__.items()})

    def restore(self, snapshot):
        self.__dict__.clear()
        self.__dict__.update(deepcopy(snapshot))


def mechanical_step(controller, action, dt):
    """Convenience transition through the stateful execution controller."""
    controller.command(action)
    return controller.advance(dt)
