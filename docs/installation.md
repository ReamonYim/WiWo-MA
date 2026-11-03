# Installation and environment

For Town02, first obtain the external resource package described in
[scene resources](../assets/README.md), then run `scripts/platform/install` and
`scripts/platform/town02 --port 8080`. The installer creates `.venv-full`
with the runtime and CPU test tools, imports the supplied resources, runs CPU
checks, and installs a browser compatible with the host OS. Browser automation
can also be installed separately. See [Town02 setup](town02.md) for resource paths and scope.

Town02 is required only for Town02 configurations. To install the platform
without that external map or an automation browser, use:

```bash
scripts/platform/install --no-town02 --no-browser
```

The included reflector scene is available without external resources. Use
`python -m wiwo_ma.platform.town02 status --require` to inspect a Town02
installation before starting its preview. A partial import is an error;
committed experiment JSON files alone do not establish map availability.

Use a source checkout and Python 3.11. The reviewed runtime pins Sionna RT
2.2.0, Mitsuba 3.9.1, Dr.Jit 1.5.0, Viser 1.1.1, NumPy 2.4.6 and SciPy 1.17.1.
Torch and the old research engines are not required.

```bash
git clone https://github.com/ReamonYim/WiWo-MA.git
cd WiWo-MA
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pip check
WIWO_GPU=-1 scripts/platform/run doctor
```

`requirements-platform-validated.txt` records the numerical core. The resolved
development environment is in `requirements-dev.lock.txt`. The doctor checks
package metadata and the pinned Viser frontend build without initializing
CUDA. A successful CPU doctor is not a successful GPU compatibility test.

The source-tree layout is intentional: configurations and project-owned assets
remain beside `src`. An editable install is supported; a bare wheel installed
without the source resources is not currently the deployment contract.

## CPU preview and replay

```bash
WIWO_GPU=-1 python examples/01_single_ma.py --port 8080 --config configs/platform/single_ma_bounded.json
```

Open the Web port and forward its adjacent API port over SSH. For example,
forward 8080 and 8081. Use unoccupied ports; do not replace an existing service.
The launcher uses `WIWO_PYTHON`, the active virtual environment, a checked
`.venv-full`, `.venv`, or the available `python3`, in that order. The Town02
launcher selects the checked `.venv-full` unless `WIWO_PYTHON` is explicit.

Viser is installed as a dependency. At first startup, WiWo-MA generates its
reviewed GUI adapter from the installed, SHA-pinned Viser frontend. Generated
HTML is ignored by Git and is not vendored in the public source bundle.
Unknown upstream builds fail closed rather than silently accepting a patch.

## Native NVIDIA / OptiX computation

Install a driver and OptiX library compatible with the target host. No driver
binary or machine-specific library path is distributed. If discovery needs an
explicit path, use:

```bash
export WIWO_OPTIX_LIBRARY=/path/to/the/matching/libnvoptix.so.1
WIWO_GPU=-1 scripts/platform/run doctor
```

Only an independently approved job may initialize the native solver. A new
machine needs its own small device smoke test and independent comparison;
compatibility is not inferred from matching package versions.

## Optional automated browser tests

Playwright 1.63 dropped Ubuntu 20.04 support. On that OS, install the
`browser-focal` extra (1.62.0); on supported newer systems use `browser`
(1.63.0). These extras are separate from the CPU `test` extra.

```bash
python -m pip install -e '.[browser]'  # Ubuntu 22.04+; use [browser-focal] on 20.04
python -m playwright install --with-deps chromium
```

Headless Chromium may also require OS libraries documented by Playwright. A
researcher using their own desktop browser does not need Chromium on the
server. The validated gallery uses Chromium/Linux and software WebGL rendering
for the UI; screenshots do not imply that CUDA ran during capture.

The current Town02 gallery is captured from a separate temporary CPU-preview
service on a fresh private port pair, leaving an existing background service
unchanged. VS Code automatic forwarding ignores this temporary service:

```bash
.venv-full/bin/python scripts/media/capture_town02.py
```

The capture verifies the strict scene, actual panel motion, displayed TX
coordinates and absence of computed CSI. It requires a terminal with local
socket and browser-launch permissions. See the [main guide](../README.md) for
the current screenshots and their scope.

Large software-rendered scenes use the regular Chromium channel in headless
mode, a 1440x960 viewport and a 2 Hz preview update rate. Each screenshot has
a 180-second timeout and one retry at a smaller viewport. Geometry and material
bindings are preserved; dimensions and retry history are recorded in the image
provenance. Override `--width`, `--height` or `--screenshot-timeout` if needed.
