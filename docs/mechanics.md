# Mechanical conventions and limits

Coordinates are metres in a Z-up world. Angular velocities are radians per
second. Euler input/display is ZYX yaw–pitch–roll in degrees, with
`R = Rz(yaw) Ry(pitch) Rx(roll)`. Panel boresight is local +X; array rows follow
local Z and columns local Y. Elements remain rigid in panel coordinates.

| Mechanism | Permitted base freedoms |
|---|---|
| `fixed` | No translation or rotation |
| `rail_x`, `rail_y`, `rail_z` | Translation along one BS-local axis |
| `planar_xy` | BS-local X/Y translation |
| `cartesian` | BS-local XYZ translation |
| `gimbal` | SO(3) rotation |
| `gantry_gimbal` | XYZ translation and SO(3) rotation |

Per-panel restrictions may reduce the base freedoms. A fixed-axis hinge is
supported; an arbitrary two-component rotation-vector mask is not a validated
two-joint serial gimbal and is rejected. The model is a bounded rigid
end-effector model, not a universal robot inverse-kinematics or structural
dynamics solver.

Mechanical parameters include workspace minima/maxima, permitted axes,
translation speed/acceleration, angular speed/acceleration, maximum rotation,
panel size and command latency. Targets are checked before execution. Pending
control-tick commands, current motion and snapshots are recorded for recovery.
Collision checks cover the implemented panel/actor geometry; they are not a
certificate of continuous collision clearance against every city object.

The continuous example integrates at 2 ms and samples CSI at 10 ms. A smaller
step may be needed for a new motion regime; generic step-refinement accuracy
is not certified. 100 Hz CSI sampling is not a universal guarantee of unaliased
Doppler, especially for high carriers, faster motion, rotation, and topology
changes. No interpolated sample is labelled as a fresh Full RT solution.
