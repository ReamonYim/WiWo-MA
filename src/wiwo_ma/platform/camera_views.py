"""Perspective framing of actual world coordinates; display only."""
import numpy as np


def fit_camera(points,direction,*,aspect=1.2,fill=.86,fov_deg=52.,minimum_distance=.1):
    """Fit every supplied point inside the camera, with a visible margin.

    ``direction`` points from the subject towards the camera. Both horizontal
    and vertical field of view matter, especially beside a docked GUI panel.
    World coordinates and the supplied points are never changed.
    """
    points=np.asarray(points,float);direction=np.asarray(direction,float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Camera framing needs finite world points of shape (N,3)')
    if direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction)<1e-9:
        raise ValueError('Camera direction must be a finite nonzero 3-vector')
    if not np.isfinite([aspect,fill,fov_deg,minimum_distance]).all() or aspect<=0 or not 0<fill<1 or not 1<fov_deg<179 or minimum_distance<=0:
        raise ValueError('Invalid camera aspect, fill, field of view or minimum distance')
    backward=direction/np.linalg.norm(direction)
    right=np.cross(-backward,[0.,0.,1.]);length=np.linalg.norm(right)
    if length<1e-6:raise ValueError('Use a slightly tilted direction for a Z-up camera')
    right/=length;up=np.cross(right,-backward)
    center=(points.min(0)+points.max(0))/2;relative=points-center
    depth=relative@backward;half_height=np.tan(np.deg2rad(fov_deg)/2)*fill
    distance=max(float(np.max(depth+abs(relative@right)/(half_height*aspect))),
                 float(np.max(depth+abs(relative@up)/half_height)),
                 float(np.max(depth))+.05,minimum_distance)
    return dict(position=center+backward*distance,look_at=center,fov_deg=float(fov_deg),
                aspect=float(aspect),fill=float(fill),distance_m=distance)


def padded_bounds(points,padding):
    """Eight corners around physical points, for a readable apparatus view."""
    points=np.asarray(points,float);padding=np.broadcast_to(np.asarray(padding,float),(3,))
    lo=points.min(0)-padding;hi=points.max(0)+padding
    return np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
