# Platform images

## Current Town02 gallery

These images were captured from the running Town02 platform with Chromium
151.0.7922.34 and software WebGL, at 1440 × 960 pixels. They use the strict
five-model ITU scene, 16 panels, 64 physical TX ports and 40 UEs. No native
CSI was computed during capture.

| Image | Captured state |
|---|---|
| [Town02 panorama](town02-panorama.png) | Complete imported map, paused at t = 0 s |
| [Movable antennas](town02-antennas.png) | Antenna panels, physical ports, workspaces and controls before motion |
| [Running platform](town02-platform.png) | Frontend at t = 1.41 s, after four scheduled panel translations |

The [capture record](town02-gallery.json) contains scene and image hashes,
timestamps, browser details and the initial/final panel positions. Capture
verified the four commanded ±20 mm translations, the twelve unchanged panels,
and displayed TX coordinates against the physical state. The temporary service
was stopped after capture; the regular background service was left unchanged.

Town02 views render the user-supplied external scene. Its resource terms remain
separate from the MIT license for project-owned code; raw meshes and textures
are not included in this repository.

The [Town02 overview](town02-map-overview.png) is a projection of the actual
imported PLY vertices using the current five material groups. It is not a
browser screenshot or a ray-tracing result.

To capture the current Town02 frontend, panorama and movable antenna panels:

```bash
.venv-full/bin/python scripts/media/capture_town02.py
```

The script starts an isolated CPU-preview service on a fresh private port pair,
sets its update rate to 2 Hz before running, captures the actual browser,
verifies physical panel motion and displayed TX coordinates, and saves image
hashes and timestamps in `town02-gallery.json` when successful. It does not
change an existing background service or use the historical dataset below.
The interface remains Chinese/bilingual; captions and the main guide are English.

## Historical Munich gallery

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
