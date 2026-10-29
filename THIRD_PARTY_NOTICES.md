# Third-party software, scenes and media

Project-owned code is MIT licensed. This does not replace the independent terms
of dependencies, city data, user scenes or datasets. This is a redistribution
inventory, not a legal opinion or a claim of institutional endorsement.

| Component | Source / terms | Release treatment |
|---|---|---|
| Sionna RT 2.2.0 | [Versioned license](https://github.com/NVlabs/sionna-rt/blob/v2.2.0/LICENSE), Apache-2.0 | Install from Python packages; no dependency source copied |
| Mitsuba 3.9.1 | [Upstream license](https://github.com/mitsuba-renderer/mitsuba3/blob/master/LICENSE) | Install from packages; no renderer binaries or source bundled |
| Dr.Jit 1.5.0 | [Upstream license](https://github.com/mitsuba-renderer/drjit/blob/master/LICENSE) | Install from packages; no dependency source bundled |
| Viser 1.1.1 | [Versioned license](https://github.com/viser-project/viser/blob/v1.1.1/LICENSE) | Runtime dependency; generated modified frontend excluded from Git/source archive |
| Munich city data | [Sionna scene documentation](https://nvlabs.github.io/sionna/rt/api/scene.html#sionna.rt.scene.munich), ODbL | Obtain through the installed SDK; no city XML/PLY/Blender models vendored |
| NumPy, SciPy, h5py, Matplotlib, trimesh and other packages | Their upstream distributions | Dependencies only; preserve their independent terms |
| Finite reflector / primitive actor geometry | Project source, MIT | Small project-owned source assets only; no downloaded city meshes |

## Viser frontend adapter

The installed Viser 1.1.1 wheel's `LICENSE` and the versioned repository license
are Apache-2.0. Its package metadata reports MIT; the discrepancy is recorded
rather than silently treating that metadata label as a grant. The actual
versioned license is retained in `assets/platform/viser-ordered-gui-v1/LICENSE-Viser`.

The public repository includes WiWo-MA's input guard and adapter code. The
large compiled upstream frontend and the modified HTML are not distributed.
At runtime the adapter verifies the installed upstream build SHA, makes the
reviewed change, and records generated hashes in Git-ignored files. This
avoids vendoring compiled third-party frontend dependencies and their assets.

## Munich and OpenStreetMap

Sionna's documentation identifies Munich as an OpenStreetMap-derived scene
licensed under ODbL, built with Blender and conversion add-ons. It must not be
treated as Apache-licensed data merely because the ray-tracing code is Apache.
See [OpenStreetMap copyright](https://www.openstreetmap.org/copyright) and the
[OSMF attribution guidelines](https://osmfoundation.org/wiki/Licence/Attribution_Guidelines).

The gallery contains actual rendered platform captures, not copies of the
underlying map database. Each Munich capture and its nearby documentation
carry attribution: **Map data © OpenStreetMap contributors, ODbL; scene supplied
through Sionna RT.** The underlying scene is obtained from the SDK, not this
repository. No raw third-party mesh, texture collection or Blender project is
included. Images are not an NVIDIA, OSMF or other institutional endorsement.

## Data and publication boundaries

Original HDF5, reference arrays, full path receipts, environment snapshots and
GPU approvals/ledgers are not public source assets. Dataset ownership, embedded
metadata privacy and any derived-data terms need a separate review before a
data release. The old 45.2 MiB internal package includes machine-bound evidence
and is **not** approved as a public GitHub Release attachment.

The new source candidate and the data catalog do not grant redistribution
rights for a user's custom scene or recorded data. Assets with unconfirmed
rights are excluded; users must obtain them through authorized sources.
