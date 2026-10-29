# Local release-preparation verification

This records actual local checks on Linux/Python 3.11. It is not a GitHub
Actions run and does not claim deployment on another machine. No native GPU
computation is used to prepare the public copy.

This is the retained **pre-publication local** record. Current hosted CI status
is available from the repository's actual Actions runs, independently of this table.

| Check | Result |
|---|---|
| Fresh isolated virtual environment and editable installation | Passed; dependency consistency check Passed |
| Public CPU regressions | **136 Passed / 0 Failed / 0 Skipped**, 61.88 s |
| Restricted tests | Four unchanged retained-artifact checks separated; not run without private research artifacts |
| Single-MA example | Passed, 20 mm X displacement at 1 s, 1 physical TX, no CSI computed |
| Full dynamic geometry example | Passed, all 401 states across 4 s, 16 panels / 64TX / 40UE / two actors |
| Saved-CSI reader and plotting example | Passed on actual closed small401, shape `[2,1,4,64]`, complex64; no new solver query |
| Actual browser | Seven real captures, executed translation/rotation checked, displayed CSI checked against HDF5, 401-sample playback reaches 4 s |
| Static checks | Syntax/name lint and byte-compilation checked locally |
| Source-archive clean installation | Passed in another source directory and a second fresh virtual environment; **136 Passed / 0 Failed / 0 Skipped**, 62.75 s |
| Clean-copy actual browser | Passed: complete 401-sample small-sequence playback, pause/seek, refresh at frame 200, displayed geometry/CSI time equality, bit-exact selected CSI and downloaded HDF SHA |
| Public source / index / history scan | Passed scoped pattern/path checks with zero findings; history is empty, with zero committed blobs, not a scanned published history |
| CPU workflow structure and local commands | Passed locally; action revisions are pinned to official upstream commit hashes |
| Native-job planning example | Passed: one TX / one UE, 101 planned frames; no GPU launched |
| GitHub-hosted CI | **Unknown / not run** |
| New GPU queries | **0** |

Private test logs preserve the initial missing-example-config failure and the
first scanner self-pattern false positive. They also preserve an incorrect QA
assumption about a download button and an assertion made before the asynchronous
renderer updated. The browser check now uses the actual documented streaming
endpoint and waits for observed render timestamps, rather than a fixed delay.
No UI, geometry or CSI is forged to satisfy the check. The initial legacy
reflector planning rejection is retained; a separate bounded example passes
the unchanged motion guard. No failing scientific sample or reference was
removed or relabelled.
Private logs and dataset copies are excluded from Git and the source archive.

The read/plot/replay example requires an authorized HDF5 supplied separately.
No public dataset download is approved here. The source candidate's native
propagation modules are compared byte-for-byte with the frozen internal
archive in [the difference inventory](FROZEN_SOURCE_DIFF.json).

Consult [release status](STATUS.md), [validation scope](../validation.md) and
[third-party notices](../../THIRD_PARTY_NOTICES.md) before publication.
