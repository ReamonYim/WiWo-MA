# Optional real CSI data

No HDF5 or reference array is committed to Git. A separate dataset publication
has not been authorized and no public download URL is invented.

The catalog in `manifests/frozen-examples.json` identifies real frozen datasets
by hash, size/shape and independent validation scope. If you already have an
authorized copy, place a **copy** at `data/demo401/csi.h5`, or pass its path
directly to the reader/replay example:

```bash
python examples/03_read_and_replay.py --dataset /path/to/authorized/csi.h5 --time 2.01
```

The command executes on actual data, not a synthetic CSI fallback. If no data
are supplied it fails clearly; it never creates a pretend native dataset.
Mechanical examples 1 and 2 need no external CSI data.

Raw HDF metadata may contain local execution provenance. Do not upload the
original automatically, strip labels silently, or change CSI to create a
public demo. Dataset exports need a separate documented privacy/license
review and must preserve a link to the immutable source.
