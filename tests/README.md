# CPU tests and external-scene integration

The public CPU tests do not require a GPU or a copy of the Town02 city map.

```bash
python -m pytest tests/platform -m 'not town02' -q --strict-markers --strict-config
```

This includes the strict ITU rules, checksum and tamper regressions, service
ownership checks, controller/storage tests, and a generated 25-mesh scene.
`test_town02_resources.py` authors that scene from quadrilaterals and writes
valid XML, PLY, CSV and a source checksum manifest in a temporary directory.
It runs the production importer, material conversion, display mesh reader and
mechanics. No part of the external city map is copied into the fixture.

Tests marked `town02` retain the original assertions about the real 6,085-mesh
scene and 8,635 imported files:

```bash
python -m wiwo_ma.platform.town02 status --require
python -m pytest tests/platform -m town02 --require-town02 -q
```

The session fixture checks actual original/strict XMLs, audit, manifest,
referenced mesh files, and configurations. A completely absent external
import is skipped only when it was not explicitly required. A partial import
fails its prerequisites; it is never converted into a skip. Full content hash
verification still runs in the resource and strict material tests.

Running `pytest tests/platform` collects both groups. Public CI runs the
self-contained group and then reports external-scene integration availability.
On a machine with Town02 imported, `scripts/platform/check --require-town02`
runs all CPU tests plus both complete mechanical examples.

`tests/restricted` is a separate historical-data suite. It needs the retained
research artifacts and is not a substitute for either public CPU tests or a
new native GPU validation run.
