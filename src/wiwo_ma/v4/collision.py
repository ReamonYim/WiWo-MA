"""Finite panel boxes, rod capsules and continuous-time clearance guards.

The published angle inequalities are geometric proxies. They are checked in
addition to actual box/capsule intersections, rather than substituted for them.
BS-relative coordinates are used for the published inequalities.
"""
from __future__ import annotations

import math
import numpy as np


def segment_distance(a, b, c, d):
    """Exact shortest distance between two closed 3-D line segments."""
    u, v, w = b-a, d-c, a-c
    aa, bb, cc = float(u@u), float(u@v), float(v@v)
    dd, ee = float(u@w), float(v@w)
    if aa < 1e-24 and cc < 1e-24:
        return float(np.linalg.norm(w))
    if aa < 1e-24:
        t = np.clip(ee/cc, 0., 1.)
        return float(np.linalg.norm(w-t*v))
    if cc < 1e-24:
        s = np.clip(-dd/aa, 0., 1.)
        return float(np.linalg.norm(w+s*u))
    denominator = aa*cc-bb*bb
    s = np.clip((bb*ee-cc*dd)/denominator, 0., 1.) if denominator > 1e-24 else 0.
    t = (bb*s+ee)/cc
    if t < 0.:
        t, s = 0., np.clip(-dd/aa, 0., 1.)
    elif t > 1.:
        t, s = 1., np.clip((bb-dd)/aa, 0., 1.)
    return float(np.linalg.norm(w+s*u-t*v))


def point_segment_distance(point, a, b):
    d = b-a
    t = np.clip(float((point-a)@d)/float(d@d), 0., 1.) if d@d > 1e-24 else 0.
    return float(np.linalg.norm(point-(a+t*d)))


def pairwise_segment_distances(starts, ends):
    """Vectorised version of segment_distance for all nondegenerate rods."""
    u = ends-starts
    w = starts[:, None, :]-starts[None, :, :]
    aa = np.einsum("bi,bi->b", u, u)
    bb = u@u.T
    dd = np.einsum("bi,bji->bj", u, w)
    ee = np.einsum("ji,bji->bj", u, w)
    denominator = aa[:, None]*aa[None, :]-bb*bb
    s = np.zeros_like(bb)
    np.divide(bb*ee-aa[None, :]*dd, denominator, out=s, where=denominator > 1e-20)
    s = np.clip(s, 0, 1)
    t = (bb*s+ee)/aa[None, :]
    low, high = t < 0, t > 1
    s = np.where(low, np.clip(-dd/aa[:, None], 0, 1), s)
    s = np.where(high, np.clip((bb-dd)/aa[:, None], 0, 1), s)
    t = np.clip(t, 0, 1)
    return np.linalg.norm(w+s[:, :, None]*u[:, None, :]-t[:, :, None]*u[None, :, :], axis=-1)


def segment_box_distance(a, b, centre, rotation, half_extents):
    """Exact segment-to-OBB distance by piecewise quadratic minimisation."""
    start, direction = rotation.T@(a-centre), rotation.T@(b-a)
    knots = [0., 1.]
    for k in range(3):
        if abs(direction[k]) > 1e-15:
            knots.extend(float(x) for x in ((-half_extents[k]-start[k])/direction[k],
                                          (half_extents[k]-start[k])/direction[k]) if 0. < x < 1.)
    knots = sorted(set(knots))
    best = math.inf
    def distance(t):
        outside = np.maximum(np.abs(start+t*direction)-half_extents, 0.)
        return float(outside@outside)
    for lo, hi in zip(knots[:-1], knots[1:]):
        middle = start+(lo+hi)*.5*direction
        active = np.abs(middle) > half_extents
        offset = start-np.sign(middle)*half_extents
        denominator = float(direction[active]@direction[active])
        t = np.clip(-float(offset[active]@direction[active])/denominator, lo, hi) if denominator else lo
        best = min(best, distance(lo), distance(hi), distance(t))
    return math.sqrt(max(0., best))


def boxes_overlap(p, r, h, q, s, g):
    """Separating-axis theorem for two oriented finite boxes."""
    delta = q-p
    axes = [r[:, k] for k in range(3)]+[s[:, k] for k in range(3)]
    axes.extend(np.cross(r[:, i], s[:, j]) for i in range(3) for j in range(3))
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            continue
        axis = axis/norm
        radius = float(np.abs(axis@r)@h+np.abs(axis@s)@g)
        if abs(float(axis@delta)) > radius+1e-12:
            return False
    return True


def rotation_valid(rotation):
    return (rotation.shape == (3, 3) and np.isfinite(rotation).all()
            and np.max(np.abs(rotation.T@rotation-np.eye(3))) < 1e-7
            and abs(np.linalg.det(rotation)-1.) < 1e-7)


class SafetyChecker:
    """Conservative continuous clearance check for the configured BS model.

    ``half_interval`` expands each object by its point-speed bound and tightens
    scalar geometric constraints by a Lipschitz bound. A midpoint that passes
    these tests therefore protects the *whole* surrounding interval. This is a
    conservative reject policy, not a globally complete motion planner.
    """
    def __init__(self, mechanical_config, mode, centre, region_side_length):
        self.cfg, self.mode = mechanical_config, mode
        self.centre = np.asarray(centre, float)
        self.half_extents = np.asarray(mechanical_config["panel_size"], float)/2
        self.corner_radius = float(np.linalg.norm(self.half_extents))
        self.region_side_length = float(region_side_length)

    def check(self, state, half_interval=0.):
        p = np.asarray(state["positions"], float)-self.centre
        rotations = np.asarray(state["rotations"], float)
        moving = np.asarray(state.get("moving", np.ones(len(p), bool)), bool)
        cp = np.where(moving, self.cfg["max_center_velocity"], 0.)
        wp = np.where(moving, self.cfg["max_panel_angular_velocity"], 0.)
        margin = (cp+self.corner_radius*wp)*float(half_interval)
        violations = []
        def issue(kind, panels, **extra):
            violations.append({"constraint": kind, "panels": list(panels), **extra})
        if p.shape != (16, 3) or rotations.shape != (16, 3, 3) or not np.isfinite(p).all():
            return {"feasible": False, "violations": [{"constraint": "nonfinite_or_wrong_pose_dimensions"}]}
        for b, r in enumerate(rotations):
            if not rotation_valid(r):
                issue("invalid_SO3", [b])
        if violations:
            return {"feasible": False, "violations": violations}
        displacement = p[None, :, :]-p[:, None, :]
        distances = np.linalg.norm(displacement, axis=-1)
        pair_padding = (cp[:, None]+cp[None, :])*half_interval
        n = rotations[:, :, 0]
        directional = np.einsum("bi,bji->bj", n, displacement)
        directional_padding = (wp[:, None]*(distances+pair_padding)+cp[:, None]+cp[None, :])*half_interval
        diagonal = np.eye(16, dtype=bool)
        minimum = float(np.min(np.where(diagonal, np.inf, distances)))
        bad = np.argwhere((distances-pair_padding < self.cfg["min_center_distance"]-1e-10)&~diagonal)
        for b, j in bad:
            if b < j:
                issue("minimum_centre_distance", [int(b), int(j)], distance=float(distances[b, j]))
        if self.cfg.get("enforce_published_rotation_constraints", True):
            for b, j in np.argwhere((directional+directional_padding > 1e-10)&~diagonal):
                issue("twc2025_other_panel_in_front", [int(b), int(j)], value=float(directional[b, j]))
            main = np.einsum("bi,bi->b", n, p)
            main_padding = (wp*(np.linalg.norm(p, axis=1)+cp*half_interval)+cp)*half_interval
            for b in np.flatnonzero(main-main_padding < -1e-10):
                issue("twc2025_main_structure_in_front", [int(b)], value=float(main[b]))
        if self.mode == "ideal_6d_end_effector":
            for b in np.flatnonzero(np.any(np.abs(p)+cp[:, None]*half_interval > self.region_side_length/2+1e-10, axis=1)):
                issue("canonical_cube_workspace", [int(b)])
        core_radius = self.cfg["main_structure_radius"]
        for b in range(16):
            h = self.half_extents+margin[b]
            core_box_distance = np.linalg.norm(np.maximum(np.abs(rotations[b].T@(-p[b]))-h, 0.))
            if core_box_distance <= core_radius+1e-10:
                issue("panel_main_structure_collision", [b])
            for j in range(b+1, 16):
                if distances[b, j] <= 2*self.corner_radius+margin[b]+margin[j]:
                    if boxes_overlap(p[b], rotations[b], h, p[j], rotations[j], self.half_extents+margin[j]):
                        issue("panel_panel_collision", [b, j])
        if self.mode == "rod_mounted":
            anchors = np.asarray(state["rod_anchors"], float)-self.centre
            lengths = np.asarray([rod["length"] for rod in state["rod_states"]], float)
            directions = (p-anchors)/lengths[:, None]
            if self.cfg.get("enforce_actual_rod_panel_angle", True):
                actual_angles = np.arccos(np.clip(np.einsum("bi,bi->b", directions, n), -1., 1.))
                rod_omega = np.where(moving, self.cfg["max_rod_angular_velocity"], 0.)
                for b in np.flatnonzero(actual_angles < self.cfg["min_rod_panel_angle"]+(rod_omega+wp)*half_interval-1e-10):
                    issue("adapted_actual_rod_panel_angle", [int(b)], angle=float(actual_angles[b]))
            q_norm = np.linalg.norm(p, axis=1)
            if np.any(q_norm < 1e-10):
                issue("panel_at_origin", list(np.flatnonzero(q_norm < 1e-10)))
            else:
                radial = p/q_norm[:, None]
                angular_padding = cp/np.maximum(q_norm-cp*half_interval, 1e-9)*half_interval
                angles = np.arccos(np.clip(radial@radial.T, -1., 1.))
                required = self.cfg["min_rod_pair_angle"]+angular_padding[:, None]+angular_padding[None, :]
                for b, j in np.argwhere((angles < required-1e-10)&~diagonal):
                    if b < j:
                        issue("twc2026_rod_pair_angle", [int(b), int(j)], angle=float(angles[b, j]))
                outward = np.asarray(self.cfg["bs_outward_normal"], float)
                outward /= np.linalg.norm(outward)
                main_angles = np.arccos(np.clip(radial@outward, -1., 1.))
                for b in np.flatnonzero(main_angles < self.cfg["min_rod_main_angle"]+angular_padding-1e-10):
                    issue("twc2026_rod_main_angle", [int(b)], angle=float(main_angles[b]))
                end_angles = np.arccos(np.clip(np.einsum("bi,bi->b", radial, n), -1., 1.))
                for b in np.flatnonzero(end_angles < self.cfg["min_rod_panel_angle"]+angular_padding+wp*half_interval-1e-10):
                    issue("twc2026_rod_panel_angle", [int(b)], angle=float(end_angles[b]))
            for b in np.flatnonzero((lengths < self.cfg["rod_length_range"][0]-1e-10)|(lengths > self.cfg["rod_length_range"][1]+1e-10)):
                issue("rod_length_workspace", [int(b)], length=float(lengths[b]))
            radius = self.cfg["rod_radius"]
            rod_distances = pairwise_segment_distances(anchors, p)
            rod_padding = (cp[:, None]+cp[None, :])*half_interval
            for b, j in np.argwhere((rod_distances <= 2*radius+rod_padding)&~diagonal):
                if b < j:
                    issue("rod_rod_collision", [int(b), int(j)])
            own_ends = p-directions*self.cfg["panel_socket_trim"]
            all_ends = np.broadcast_to(p[:, None, :], (16, 16, 3)).copy()
            all_ends[np.arange(16), np.arange(16)] = own_ends
            segments = all_ends-anchors[:, None, :]
            parameters = np.clip(np.einsum("bji,bji->bj", p[None, :, :]-anchors[:, None, :], segments)
                                 /np.einsum("bji,bji->bj", segments, segments), 0, 1)
            broad_distances = np.linalg.norm(p[None, :, :]-anchors[:, None, :]-parameters[:, :, None]*segments, axis=-1)
            candidate_pairs = np.argwhere(broad_distances <= self.corner_radius+radius+margin[None, :]+cp[:, None]*half_interval)
            for b in range(16):
                # An explicitly sized attachment socket is permitted. The rest
                # of the own rod must clear both its panel and the main core.
                core_start = anchors[b]+directions[b]*self.cfg["main_socket_trim"]
                if point_segment_distance(np.zeros(3), core_start, p[b]) <= core_radius+radius+cp[b]*half_interval:
                    issue("rod_main_structure_collision", [b])
            for b, j in candidate_pairs:
                clearance = segment_box_distance(anchors[b], all_ends[b, j], p[j], rotations[j], self.half_extents+margin[j])
                if clearance <= radius+cp[b]*half_interval:
                    issue("rod_panel_collision", [int(b), int(j)])
        return {"feasible": not violations, "violations": violations,
                "minimum_center_distance": minimum,
                "maximum_other_panel_rotation_value": float(np.max(np.where(diagonal, -np.inf, directional))),
                "continuous_half_interval": float(half_interval),
                "method": "OBB_SAT_capsule_exact_distance_with_speed_Lipschitz_guard"}
