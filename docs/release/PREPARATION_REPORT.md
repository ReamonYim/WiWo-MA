# Local GitHub release preparation

The public workspace is independent of the immutable research tree. The initial
local preparation record below predates publication. The owner subsequently
confirmed MIT and Public ReamonYim/WiWo-MA hosting. Publication status and scope
are maintained in `STATUS.md`; historical local checks are not hosted CI results.

## Included layout

```text
README.md / LICENSE (MIT) / CITATION.md (software/version reference)
CHANGELOG.md / CONTRIBUTING.md / SECURITY.md / THIRD_PARTY_NOTICES.md
pyproject.toml / validated runtime pins / resolved development lock
.github/workflows/cpu.yml
src/wiwo_ma/platform/       native platform and bounded mechanics
src/wiwo_ma/v4/ + v5/       only required collision/display/actor compatibility
scripts/platform/          portable entrypoint and scoped helpers
scripts/release/            link, staged/history scan and deterministic bundle guards
scripts/media/              actual browser gallery capture
configs/platform/          frozen configurations and bounded planning example
examples/                  three runnable mechanics/scene/CSI examples
tests/platform/            136 public CPU tests
tests/restricted/           four retained-artifact tests, not claimed publicly run
docs/                      guides, validation matrix, release checklist and gallery
assets/                    small project geometry and frontend input guard
data/                      metadata catalog only; no HDF5 or download claim
```

No city mesh, dependency source tree, compiled frontend, large research output,
resource approval, raw reference or credential is part of the candidate.
Private local verification, environment and data copies are Git-ignored.

## Local evidence

- Editable installation passed in a fresh environment, then the source archive
  was unpacked and installed in another fresh environment.
- Both public suites passed all 136 tests. Static checks, dependency consistency,
  environment metadata, the three examples and local documentation links passed.
- Seven actual platform captures and one real CSI plot were produced. Complete
  HDF replay, observed render timestamps, refresh and downloaded SHA were checked
  in a real browser. New GPU queries: **zero**.
- Thirty runtime/source modules remain byte-identical to the frozen archive;
  [the file inventory](FROZEN_SOURCE_DIFF.json) explains every extracted-file
  change. No new native numerical certification is claimed by packaging.
- Scoped staged/worktree/history scans found no forbidden data or secret-pattern
  matches. The new repository has no commit history; history coverage is zero
  blobs, explicitly disclosed. Synthetic tests verify detection in nonempty
  temporary Git histories, including a deleted secret fixture.

Private failed preparation attempts are retained. They do not change original
scientific Passed, Failed, Partial or Unknown labels. Original archive and
delivery-manifest hashes are unchanged. Existing research Web services remain
untouched; release QA uses separate temporary ports.

## Publication decisions and remaining scope

The owner approved MIT and Public hosting under ReamonYim/WiWo-MA. Official
GitHub CLI device authentication was completed as that account. Local commit
metadata use its public handle and ID-based noreply address, not an invented
personal name. Formal research authors/DOI remain unspecified in `CITATION.md`.

The earlier local source candidate and its failed preparation attempts remain
retained privately. The final approved source release has a newly generated
SHA256. The staged/history scanner is re-run after the publication commit;
actual hosting/CI/Release results are recorded only once observed.

Munich is treated as ODbL scene data, separately from Sionna's software license.
Rendered screenshots carry attribution; meshes are excluded. The original
45.2 MiB internal archive is not a public attachment. Data distribution remains
separately pending privacy/rights review. See [notices](../../THIRD_PARTY_NOTICES.md).
