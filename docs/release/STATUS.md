# v1.0.0 release scope

WiWo-MA Platform v1.0.0 is distributed under the [MIT license](../../LICENSE)
for project-owned code. The owner explicitly approved MIT and Public hosting
under **ReamonYim/WiWo-MA**. Dependencies and scene data retain their own terms.

Repository: https://github.com/ReamonYim/WiWo-MA

| Gate | Scope and status |
|---|---|
| Frozen numerical and mechanical evidence | Passed only for the recorded states/configurations; see [validation](../validation.md) |
| Propagation-core preservation | Passed byte-for-byte comparison; see [source inventory](FROZEN_SOURCE_DIFF.json) |
| Public CPU suite and clean installation | Passed same-host isolated-environment checks; see [local verification](LOCAL_VERIFICATION.md) |
| Three executable examples and actual gallery | Passed local execution; example 3 needs separately supplied authorized data |
| Source privacy checks | Scoped staged/worktree/reachable-history scanner and visual media review; not a guarantee against every secret |
| Hosted CPU CI | Use the actual repository Actions result; no local result is represented as hosted execution |
| Project license / Public permission | MIT / approved by the owner |
| Research author and DOI metadata | Unspecified; human-readable [citation guidance](../../CITATION.md), no invented author or paper |
| Dataset distribution | Not approved; no raw HDF5, references or download link shipped |
| Other-machine CUDA / arbitrary configurations | Unknown until separately validated |

Historical Full401 remains **20 Passed / 381 Unknown**, small401 remains
**9 Passed / 392 Unknown**, and old3001 remains **24 Passed / 2977 Unknown**.
Packaging does not recertify new states and uses no new GPU queries. Overall
research validity remains **Partial**, independently of software availability.

The original internal archive, HDF5, raw references, historical failed records
and resource ledgers remain outside this source distribution. City meshes,
compiled dependency frontends and credentials are excluded. See [third-party
notices](../../THIRD_PARTY_NOTICES.md) and the [publication procedure](PUBLISHING.md).
