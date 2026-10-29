"""Opt-in C2, rest-to-rest external motion, without altering legacy episodes.

Each waypoint is a physical stop. Quintic time scaling on a straight segment
and on a shortest SO(3) geodesic gives continuous position, velocity and
acceleration at stops, including reversal. This is a bounded kinematic model,
not a road vehicle steering/tyre or human gait model.
"""
import math
import numpy as np
from wiwo_ma.mechanics import euler_to_matrix, rotation_exp, rotation_log


LIMITS = ('max_speed_m_s', 'max_acceleration_m_s2',
          'max_angular_speed_rad_s', 'max_angular_acceleration_rad_s2')


def segment_bounds(a, b):
    dt = b['time_s'] - a['time_s']
    distance = np.linalg.norm(np.asarray(b['position_m']) - a['position_m'])
    angle = np.linalg.norm(rotation_log(euler_to_matrix(b['orientation_deg']) @
                                       euler_to_matrix(a['orientation_deg']).T))
    return dict(zip(LIMITS, (1.875*distance/dt, (10/math.sqrt(3))*distance/dt**2,
                            1.875*angle/dt, (10/math.sqrt(3))*angle/dt**2)))


def validate_motion(entity):
    # No inserted defaults: fingerprints of frozen legacy configurations stay
    # unchanged. New scenes must explicitly opt into this physical definition.
    if 'motion' not in entity:
        return
    spec = entity['motion']
    if spec.get('model') != 'bounded_quintic':
        raise ValueError('Unknown motion model; use bounded_quintic')
    for key in LIMITS:
        value = spec.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
            raise ValueError('motion.'+key+' must be explicitly positive')
    for i, (a, b) in enumerate(zip(entity['trajectory'], entity['trajectory'][1:])):
        for key, bound in segment_bounds(a, b).items():
            if bound > spec[key]*(1+1e-10):
                raise ValueError(f"{entity['id']} segment {i}: {key} requires {bound:.6g}, limit {spec[key]:.6g}")


def bounded_pose(entity, t, *, derivatives=False):
    route = entity['trajectory']
    if not route or t >= route[-1]['time_s']:
        q = route[-1] if route else entity
        result = (np.asarray(q['position_m'], float), euler_to_matrix(q['orientation_deg']), np.zeros(3), np.zeros(3))
        return result+(np.zeros(3), np.zeros(3)) if derivatives else result
    if t < 0:
        raise ValueError('Physical time must be nonnegative')
    for a, b in zip(route, route[1:]):
        if a['time_s'] <= t < b['time_s']:
            dt = b['time_s']-a['time_s']; u = (t-a['time_s'])/dt
            s = u**3*(10+u*(-15+6*u))
            ds = 30*u**2*(1-u)**2/dt
            dds = 60*u*(1-u)*(1-2*u)/dt**2
            delta = np.asarray(b['position_m'])-a['position_m']
            r0 = euler_to_matrix(a['orientation_deg'])
            axis = rotation_log(euler_to_matrix(b['orientation_deg']) @ r0.T)
            result = (np.asarray(a['position_m'])+s*delta, rotation_exp(s*axis)@r0, ds*delta, ds*axis)
            return result+(dds*delta, dds*axis) if derivatives else result
    raise ValueError('Trajectory does not cover time0')


def trajectory_diagnostics(entity):
    """Exact segment bounds and legacy velocity jumps; no CSI claim."""
    route = entity.get('trajectory', [])
    bounded = entity.get('motion', {}).get('model') == 'bounded_quintic'
    jumps = []
    if route and not bounded:
        velocities = [(np.asarray(b['position_m'])-a['position_m'])/(b['time_s']-a['time_s'])
                      for a, b in zip(route, route[1:])]
        angular = [rotation_log(euler_to_matrix(b['orientation_deg'])@euler_to_matrix(a['orientation_deg']).T)/(b['time_s']-a['time_s'])
                   for a, b in zip(route, route[1:])]
        for i in range(len(route)):
            before = velocities[i-1] if i else np.zeros(3)
            after = velocities[i] if i < len(velocities) else np.zeros(3)
            wb = angular[i-1] if i else np.zeros(3); wa = angular[i] if i < len(angular) else np.zeros(3)
            dv, dw = float(np.linalg.norm(after-before)), float(np.linalg.norm(wa-wb))
            if dv > 1e-10 or dw > 1e-10:
                jumps.append(dict(time_s=route[i]['time_s'], velocity_jump_m_s=dv, angular_velocity_jump_rad_s=dw))
    returns = [dict(first=i, last=j) for i in range(len(route)) for j in range(i+2, len(route))
               if np.linalg.norm(np.asarray(route[i]['position_m'])-route[j]['position_m']) < 1e-9]
    return dict(id=entity['id'], model='bounded_quintic_C2_stops' if bounded else 'legacy_piecewise_linear',
                velocity_discontinuities=jumps, repeated_waypoint_positions=returns,
                segment_bounds=[segment_bounds(a, b) for a, b in zip(route, route[1:])] if bounded else [],
                leakage_policy='Same episode, overlapping windows and revisited geometry must remain in the same data split')
