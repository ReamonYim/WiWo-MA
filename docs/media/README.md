# Real platform gallery

These are actual Chromium captures of the source candidate and the frozen
native Full401 dataset. Capturing them performs **no new ray tracing**. The
gallery script reads saved complex CSI and checks selected displayed arrays
against the actual HDF5; it also executes and independently checks the shown
6 mm / 3 mm translation and 0.003 rad rotation in mechanical-preview mode.

| Image | Content and source |
|---|---|
| [Munich overview](01_munich_overview.png) | Full16/64/40 stored geometry, t=0 s |
| [Full64 array](02_full64_array.png) | Actual physical TX geometry in the city |
| [Mechanical controls](03_mechanics_controls.png) | Real target inputs; no CSI solve |
| [Executed panel motion](04_panel_motion.png) | Bounded translation and rotation; no CSI solve |
| [Actor occlusion](05_actor_occlusion.png) | Recorded UE0 LOS ports 64 → 0 → 64, blocked state at 1.68 s |
| [CSI response](06_csi_response.png) | Recorded amplitude/phase, ports and physical timestamp |
| [Complete replay](07_full_sequence_replay.png) | Actual browser playback reaches frame 400 / 4 s |
| [CSI time–frequency plot](08_csi_time_frequency.png) | Example 3 reads all 401 small-sequence samples; UE0/RX0/TX0, unnormalized dimensionless H |

The UI remains Chinese/bilingual; the English [control guide](../ui.md) maps
the real buttons. Headless capture uses software WebGL and does not claim GPU
rendering performance. Footer text identifies data mode, physical time and
map attribution; it is an annotation, not a replacement for the UI output.
Hashes and timestamps are in [gallery.json](gallery.json).

**Map data © OpenStreetMap contributors,
[ODbL](https://www.openstreetmap.org/copyright); Munich scene supplied through
Sionna RT.** The images are rendered views; the underlying city database and
meshes are not included. See [third-party notices](../../THIRD_PARTY_NOTICES.md).
Project code is MIT licensed. Rendered map content retains the attribution and
data-license boundaries described in those notices.

To recapture, install Playwright Chromium, run a dedicated replay service with
an authorized Full401 copy inside the checkout, then execute
`scripts/media/capture_gallery.py --help`. The script changes that dedicated
service's preview/replay state; do not point it at a shared research session.
