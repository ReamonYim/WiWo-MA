# Town02 resource handling in CPU CI

## Failure and cause

The first Town02 integration commit failed in
`test_actual_variant_has_only_five_builtin_types_and_preserves_original_imports`.
Its availability guard checked a tracked experiment JSON file, which is present
in every checkout. The test then tried to hash `assets/town02/town2_3mat.xml`,
which was not present on the hosted runner.

The XML was not accidentally deleted from Git. `assets/town02/` is deliberately
ignored and absent from the public tree. The local import contains 8,635 files
and 151,049,702 bytes of external resource data. Source code, rules and experiment
configurations are tracked; the original XML/PLY/CSV and derived strict scene
are imported or generated locally. Town02 is required for its own presets, not
for the platform's reflector scene, built-in scenes or self-contained CPU tests.

## Corrected behavior

- `town02 status` inspects the actual resources rather than inferring availability
  from a JSON preset. It reports `not_imported`, `incomplete` or `available`.
- The real-scene tests share a `town02` marker and prerequisite fixture. They
  skip only for an entirely absent, optional import. Missing XML, audit or mesh
  files in a partial import fail. `--require-town02` also makes complete absence
  a failure.
- The material/geometry assertions in the real-scene tests are unchanged.
- Public CI exercises the production import/conversion/display/mechanics path
  using independently authored, valid XML/PLY/CSV fixtures. It separately reports
  full-scene integration availability.
- Source files and hashes are checked before an import changes existing files.
  A damaged source package cannot silently replace the working import.

## Reproduction

```bash
# No external map needed:
python -m pytest tests/platform -m 'not town02' -q

# With the complete external package:
python -m wiwo_ma.platform.town02 import --source /path/to/Sionna_Town02
scripts/platform/check --require-town02
```

The local verification also uses an independent source-only worktree with no
`assets/town02/` directory. Negative checks explicitly require the missing map,
and separately supply a partial import; both must fail with prerequisite errors,
rather than pass or skip. See [scene resources](../assets/README.md) and
[test groups](../tests/README.md).
