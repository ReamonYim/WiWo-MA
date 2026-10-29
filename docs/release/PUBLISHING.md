# Maintainer publication procedure

The owner has approved **MIT** and a **Public ReamonYim/WiWo-MA** repository
for v1.0.0. These decisions do not approve publishing raw datasets, scene meshes,
credentials or historical research logs. Future unrelated data releases need
separate rights/privacy review.

## Checks before a release

```bash
python -m pip install -e '.[test]'
python -m pip check
CUDA_VISIBLE_DEVICES='' WIWO_GPU=-1 python -m pytest tests/platform -q
python -m ruff check src tests examples scripts
python -m compileall -q src examples scripts
python scripts/release/check_links.py
git add .
python scripts/release/audit_public.py --output artifacts/pre-tag-scan.json
python scripts/release/source_bundle.py \
  --output dist/WiWo-MA-v1.0.0-source.tar.gz
```

The scanner reads staged blobs, matching working files and every reachable Git
blob. Review the report and staged paths; deterministic rules are not proof
that every possible secret is absent. Inspect media too. Unpack the exact source
archive in another directory, install as an editable checkout and run CPU checks.
A bare wheel without the source resources is not the deployment contract.

Do not attach the original 45.2 MiB internal archive or raw HDF5. Data, GPU
approvals, dependency binaries and city meshes are excluded from the public
source. Source artifacts and SHA256 files must be rebuilt after any content edit.

## Authentication, commits and upload

Use the official GitHub CLI and authenticate on GitHub's device-login page.
Never put a token in the repository, remote URL, issue or chat. Check that the
actual authenticated account is ReamonYim before writing to this repository.
Workflow pushes need the appropriate workflow permission. The local publication
identity uses the supplied account and its ID-based GitHub noreply address;
this does not assert a research-paper author or affiliation.

```bash
gh auth login --hostname github.com --git-protocol https --web --scopes workflow
gh api user --jq .login
git add .
git commit -m "Release WiWo-MA Platform v1.0.0 under MIT"
# Run the scanner again after committing, including the committed history.
python scripts/release/audit_public.py --output artifacts/committed-scan.json
# Initial creation only; never overwrite an existing unrelated repository.
gh repo create ReamonYim/WiWo-MA --public --source=. --remote=origin
git push -u origin main
```

Check the first real hosted CPU run. Fix any failure openly and re-run; keep
its logs/history. After the intended commit passes, create the tag and Release:

```bash
git tag -a v1.0.0 -m "WiWo-MA Platform v1.0.0: limited validated scope"
git push origin v1.0.0
gh release create v1.0.0 \
  dist/WiWo-MA-v1.0.0-source.tar.gz \
  dist/WiWo-MA-v1.0.0-source.tar.gz.sha256 \
  --verify-tag --title "WiWo-MA Platform v1.0.0" \
  --notes-file docs/release/RELEASE_NOTES.md
```

Verify anonymous access to the actual repository and Release, inspect the
published license and download the uploaded source/checksum to verify hashes.
Report URLs only after the real objects exist. GitHub may generate its own
automatic source archives in addition to the explicit reviewed source asset.

Formal research-author metadata remain unspecified. `CITATION.md` gives a
software/version reference without fabricating an author list or DOI.
