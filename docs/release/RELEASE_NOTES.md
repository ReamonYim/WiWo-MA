# WiWo-MA Platform v1.0.0

Project-owned code is released under MIT. Third-party software and scene data
retain their independent terms. Formal paper/author metadata are not asserted.

WiWo-MA is a physics-grounded interactive simulation platform for movable
antenna systems. It connects bounded rigid-panel mechanics, dynamic UE/actor
geometry, native Sionna RT physical-port channels, durable HDF5 generation and
synchronized browser replay.

This source release contains the platform, lightweight project assets,
configuration examples, CPU regressions, documentation and a real browser
gallery. It installs the pinned Sionna RT / Mitsuba / Dr.Jit / Viser stack as
dependencies. City meshes, compiled dependency sources, GPU grants, raw CSI
datasets and independent reference arrays are excluded.

## Validated scope

- Native 16-panel / 64TX / 40UE continuous four-second generation, 401 frames,
  streaming storage and cross-process recovery after frame 201.
- Twenty independent SDK states Passed; the remaining 381 frames stay Unknown.
  Worst per-UE NMSE in those selected comparisons: 6.4625e-15.
- Eight additional selected pose/path comparisons Passed, including 736 valid
  dual-polarized path terms and same-time mechanical causal interventions.
- CPU mechanical, storage, input-ordering and failure guards; actual browser
  replay and three runnable source examples.

The original numerical evidence is retained from frozen v1.0.0. Preparing the
public copy does not run another GPU campaign or certify new physical states.
The SDK is a numerical reference, not measured real-world ground truth.

## Known limitations

Only listed configurations and sampled states are validated. Arbitrary arrays,
mechanisms, materials and browser/GPU combinations remain unverified. Native
RT is asynchronous, not millisecond real-time. Mechanical preview calculates
no CSI. A 100 Hz sequence is not a universal Doppler guarantee. The original
30-second data retain the documented trajectory and event-attribution limits.
The Web server is intended for trusted local or SSH use.

See README for installation and examples, `docs/validation.md` for coverage,
`THIRD_PARTY_NOTICES.md` for scene/software terms and `data/README.md` for
separate data access. Repository: https://github.com/ReamonYim/WiWo-MA.
No paper, DOI or unconfirmed research author is supplied.
