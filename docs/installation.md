# Installation and environment

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
WIWO_GPU=-1 python examples/01_single_ma.py --port 8080
```

Open the Web port and forward its adjacent API port over SSH. For example,
forward 8080 and 8081. Use unoccupied ports; do not replace an existing service.
The launcher uses `WIWO_PYTHON`, the active virtual environment, `.venv`, or
the available `python3`, in that order.

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

```bash
python -m playwright install chromium
```

Headless Chromium may also require OS libraries documented by Playwright. A
researcher using their own desktop browser does not need Chromium on the
server. The validated gallery uses Chromium/Linux and software WebGL rendering
for the UI; screenshots do not imply that CUDA ran during capture.
