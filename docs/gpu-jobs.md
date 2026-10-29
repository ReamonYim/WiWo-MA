# Native CSI jobs, authorization and recovery

This guide does not grant GPU time. CPU preview and HDF5 replay require none.
Before a new native experiment, inspect device availability and estimate
wall-clock time, allocator memory, data/path storage and independent references.
The original Full64 job took hours; no fixed real-time query latency is promised.

## Prepare and freeze — CPU only

```bash
WIWO_GPU=-1 scripts/platform/run plan --config configs/platform/single_ma_bounded.json \
  --job artifacts/platform/jobs/my_sequence --max-wall-s 300 --max-disk-gib 1
```

Preparation validates mechanics, trajectories, scene inputs and source hashes.
Review `plan.json`, `limits.json`, `config.json`, `frozen_sources.json` and any
frozen input/snapshot receipts. A changed configuration, source or current-pose
snapshot requires a new review, not an edited hash receipt.

This lightweight example has one physical TX, one static UE and bounded rail
motion over one second (101 requested CSI frames). Its mechanics and planning
are CPU-checked; it has no new SDK numerical certification. The retained legacy
`reflector.json` includes piecewise-linear external motion and is intentionally
rejected by native-job admission. Use the bounded example for planning; do not
disable the motion guard to make a legacy demonstration launch.

## Explicit resource approval

The resource owner creates a separate JSON for this exact frozen job. A
non-executable example is below; **`approved` is deliberately false**:

```json
{
  "approved": false,
  "scope": "native_platform",
  "gpu": 0,
  "configuration_sha256": ["COPY_FROM_THE_REVIEWED_PLAN"],
  "job_path": "artifacts/platform/jobs/my_sequence",
  "total_wall_s": 300,
  "max_allocator_bytes": 12884901888,
  "max_disk_bytes": 1073741824
}
```

Current-pose jobs also require explicit `initial_snapshot_sha256` binding.
Caps must cover initialization, all native queries, independent references,
synchronization, writing and the required evidence. Authorization JSON and
the adjacent `.ledger.json` stay outside Git. This is an operational guard in
a trusted research environment, not an OS security sandbox.

After real approval, and only on a configured CUDA/OptiX host:

```bash
scripts/platform/run launch --job artifacts/platform/jobs/my_sequence \
  --authorization artifacts/approvals/my_sequence.json
```

The supervisor selects the grant's GPU, enforces resource limits, streams
frames and preserves progress/logs. The browser may disconnect while the
worker continues. It shows actual committed frame count and physical time;
it does not advance fake CSI while a query is pending.

## Recovery and closure

Inspect the recorded failure and worker status before resuming. Reuse committed
frames, verify hashes and remaining same-purpose budget, then request explicit
recovery when needed:

```bash
scripts/platform/run launch --job artifacts/platform/jobs/my_sequence \
  --authorization artifacts/approvals/my_sequence.json --resume
WIWO_GPU=-1 scripts/platform/run audit-closed --job artifacts/platform/jobs/my_sequence
```

Never borrow another experiment's balance, lower the numerical gate, overwrite
a failure or relabel unchecked rows. A successful exit code and complete HDF5
are not sufficient numerical evidence. Fresh SDK reference scenes must use
saved geometry/configuration/seed, never the generated CSI or cached answers.
