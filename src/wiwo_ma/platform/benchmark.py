"""Append-only hash seal for the existing dataset, not a rewrite or relabel."""
import json
from pathlib import Path
from .config import ROOT
from .dataset_audit import sha_file
from .storage import atomic_json


def seal_csi30s_benchmark(output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    job=ROOT/'artifacts/platform/jobs/munich30s_native_audited_attempt01'
    audit=ROOT/'artifacts/platform/audits/csi30s_comprehensive_attempt02'
    receipt=json.loads((audit/'input_receipt.json').read_text())
    errors=[];files={}
    def bind(path,expected=None):
        digest=sha_file(path);files[str(path.relative_to(ROOT))]=digest
        if expected and digest!=expected:errors.append(str(path.relative_to(ROOT)))
    bind(job/'dataset/csi.h5',receipt['sha256'])
    for name,digest in receipt['auxiliary_files'].items():bind(job/name,digest)
    source=json.loads((job/'source_receipt.json').read_text())
    for row in source['files']:bind(job/'executed_sources'/row['path'],row['sha256'])
    paths=[ROOT/'artifacts/platform/audits/csi30s_sdk16_attempt01/independent_CPU_audit/audit_numeric_label_ledger.json',
        ROOT/'artifacts/platform/audits/csi30s_sdk16_delivery_snapshot01/manifest.json',
        ROOT/'artifacts/platform/audits/csi30s_sdk16_delivery_snapshot01/independent_final_verification.json',
        audit/'software_and_source_provenance.json']
    for path in paths:bind(path)
    labels=json.loads(paths[0].read_text())
    if labels['counts']!={'Passed':24,'Partial':0,'Failed':0,'Unknown':2977}:errors.append('audit_label_counts')
    report=dict(schema='wiwo-immutable-benchmark-hash-seal-v1',status='Failed' if errors else 'Passed',files=files,mismatches=errors,
        original_ledger_counts=dict(Passed=7,Unknown=2994,Failed=0),audit_overlay_counts=labels['counts'],new_GPU_seconds=0,
        protection='Hash-sealed baseline. No data copies, permission changes, rewrites or relabels. Verify hashes before scientific reuse.',
        limitations=['29 low-correlation events still lack full path attribution','Original complete path sets Unknown',
            'All direct links LOS; linear external motion reverses instantaneously','Mirror/revisited geometry must be grouped within one split',
            'SDK near-contact ray offsets produce short-segment reflection direction residuals'])
    out.mkdir(parents=True);atomic_json(out/'benchmark_manifest.json',report)
    if errors:raise ValueError('Existing benchmark identity mismatch: '+str(errors))
    return report
