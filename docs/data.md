# Data, provenance and replay

The native HDF5 tensor is `H[time, UE, RX_port, TX_port, subcarrier]`,
`complex64`. Each committed row includes the physical timestamp, configuration,
mechanical state and snapshot, actual antenna world coordinates, UE and actor
state, action history, propagation metadata and a frame hash.

Frequency offsets are stored in `frequency_offsets_hz`. A path CIR coefficient
already includes carrier phase. No receiver noise or automatic gain
normalization is added. Mean `|H|²` is a model channel gain, not calibrated
received power. Heterogeneous RX arrays use `rx_valid`; padding is not an
additional physical antenna.

Rows are streamed through a pending-write/commit protocol. Closed files can be
read independently of the original worker or Web UI. Running non-SWMR writers
are observed through atomic published frames rather than concurrent raw HDF5
reads. Recovery checks committed hashes and must not append a duplicate row.
Arbitrary power loss during an HDF5 internal write is not universally recoverable.

```bash
python scripts/platform/read_saved_CSI.py --dataset data/demo401/csi.h5 --time 2.01
python examples/03_read_and_replay.py --dataset data/demo401/csi.h5 \
  --plot artifacts/analysis/ue0.png
WIWO_GPU=-1 scripts/platform/run replay --dataset data/demo401/csi.h5 --port 8080
```

Use `scripts/platform/run verify --dataset ...` to traverse and check stored row
hashes. This is a storage check, not an independent SDK numerical experiment.
Unknown raw labels remain Unknown unless an independently computed, correctly
bound sidecar verifies that particular row. A CSI-only check does not prove
complete path-set correctness.

The public source repository contains no original HDF5, reference arrays or
GPU ledgers. A local dataset may contain machine-specific provenance; do not
upload one automatically. See [the data catalog](../data/README.md) for the
original hashes and explicit download/publication status. Importing or
copying a dataset must not silently rewrite its CSI or original labels.

For train/test use, split independent trajectories or experiments. Adjacent,
overlapping and mirrored return windows are not independent samples. The
original 30-second sequence has a known mirrored revisit and an instantaneous
trajectory reversal, and 29 events still lack full path attribution.
