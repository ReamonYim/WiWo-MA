# Validation scope

Three different statements are kept separate: selected core simulation
evidence, a usable source-software release, and unmeasured physical reality or
untested configurations. SDK numerical agreement is not a field measurement.

## Frozen numerical evidence

The source origin is the independently verified local v1.0.0 archive, SHA256
`dd56af848e72c36b22073e658f61b4fcd9e3154b5ee57c0fedc9f4b05f9ccd97`.
The scientific arrays and machine-bound logs remain outside the public source.
The public packaging change record identifies unchanged numerical modules.
No old 401/3001/801 state was re-solved for this packaging task.

| Evidence | Selected SDK result | Scope retained |
|---|---|---|
| Full16/64TX/K40, 4 s / 401 frames | 20 Passed, 381 Unknown; worst per-UE NMSE 6.4625e-15 | Selected complete paths, vertices, delays and CIR bitwise equal; all rows internally CIR→CSI consistent |
| Small 4 s / 401 frames | 9 Passed, 392 Unknown | Cross-process recovery and complete synchronized replay |
| Original 30 s / 3001 frames | 24 Passed, 2977 Unknown | Remaining physical/trajectory limitations retained |
| Additional eight poses | 8 CSI and 8 complete-path comparisons Passed; worst per-UE NMSE 6.443134924264023e-15 | Includes same-time mechanical controls/treatments and dual polarization |

All use the original per-UE NMSE ≤ 1e-3 gate; weak users are not filtered.
The dual-BS/dual-polarization case has 12 spatial TX locations, 24 polarized TX
ports, 3 UEs, RX counts 2/4/2, `H=[3,4,24,7]`, and 736 valid polarized path
terms. Padding and path multiplicity are preserved.

Mechanical causal comparisons hold time, UE/actor state, seed and radio
configuration fixed. Only the specified panel intervention changes. Restored
port geometry is bitwise equal; restored CSI is within the original numerical
gate, not asserted bitwise identical.

## Configuration matrix

| Configuration | CPU / Web | Native CSI / complete paths | Storage / recovery |
|---|---|---|---|
| Fixed single TX | Passed selected geometry and UI | Passed K3/1TX selected pose | Single-frame storage/replay; arbitrary long window Unknown |
| Fixed multi-element array | Passed selected configuration | Exact standalone variant Unknown | CPU format/restore only |
| Single MA on X rail | Passed selected controls | Passed selected pose | Native single frame; arbitrary trajectory Unknown |
| Independent XYZ MA | Passed selected three-axis controls | Exact variant Unknown | CPU snapshot restore |
| Pure Cartesian or gimbal panel | Passed selected mechanics/UI | Standalone variants Unknown; full16 intervention subsets Passed | Standalone native long-window recovery Unknown |
| One 6DoF rigid panel | Passed selected translation/rotation | Passed 4TX/K3 pose | Single frame and mechanical restore |
| Two panels | Passed selected joint actions | Small 4TX/K2 window selected states Passed; other sizes Unknown | Small401/cross-process restore/replay Passed |
| 16 panels / 64TX / 40UE | Passed qualified trajectory and controls | 20/401 plus selected causal poses Passed | Full401 and recovery after201 Passed |
| Two BS, dual polarization, mixed RX | Passed selected mapping/UI | Selected `[3,4,24,7]` pose/path table Passed | Single frame; multi-frame native recovery Unknown |

Neither array-class support nor a form accepting a value certifies every
parameter combination. New-machine CUDA/OptiX, other browsers, arbitrary
counterfactual history in one solver instance, diffuse/transmitted/diffracted
combinations and measured physical truth remain unverified.

## Software and tests

The frozen local release passed nine scoped software gates. Its full CPU
collection was 131 Passed / 4 optional historical skips. This repository moves
those four artifact-dependent tests into `tests/restricted` instead of passing
them with missing data. The public CPU suite retains the numerical gate,
mechanics, storage, action causality safeguards, GUI delivery regression and
failure-injection checks without requiring those private datasets.

The GUI fix addresses the original cross-control latest-message throttle and
awaits ordered critical callbacks. Permanent tests execute the actual compiled
sender, and browser evidence covers fill/keyboard/same-value/rapid edits,
joint panel actions, explicit no-op status and invalid-number refusal.

GitHub-hosted CI status remains **Unknown/not run** until a real repository and
workflow run exist. Local preparation results are recorded in
`docs/release/LOCAL_VERIFICATION.md`; they must not be displayed as a hosted CI
badge. No new GPU computation is part of this release-preparation task.
