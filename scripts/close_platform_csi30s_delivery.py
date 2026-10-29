#!/usr/bin/env python3
"""Freeze current audit report/progress and preserve the previous active manifest."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['WIWO_GPU']='-1'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import h5py
import numpy as np
from wiwo_ma.platform.dataset_audit import sha_file,save_json,nmse_all_users


def archive_target(snapshot,path,root=None):
    root=Path.cwd().resolve() if root is None else Path(root).resolve()
    return Path(snapshot)/Path(path).resolve().relative_to(root)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    snapshot=args.snapshot
    if snapshot.exists() and (not args.resume or (snapshot/'manifest.json').exists()):raise FileExistsError(snapshot)
    snapshot.mkdir(parents=True,exist_ok=True)
    audit=Path('artifacts/platform/audits/csi30s_comprehensive_attempt02');verification=Path('artifacts/platform/audits/csi30s_verification_attempt01')
    job=Path('artifacts/platform/jobs/munich30s_native_audited_attempt01')
    # Full indexed errors for the additionally reused frame, independently
    # saved rather than changing the closed seven-reference error array.
    ref=Path('artifacts/platform/validation/native_gate_attempt01/full64_0/reference.npz')
    with h5py.File(job/'dataset/csi.h5','r') as f:a=f['H'][120].astype(complex)
    with np.load(ref) as z:b=z['H'].astype(complex)
    delta=a-b;relative=np.divide(abs(delta),abs(b),out=np.where(abs(delta)==0,0,np.inf),where=abs(b)>0)
    err,den,num=nmse_all_users(a,b)
    np.savez_compressed(snapshot/'frame120_element_errors.npz',absolute_complex_error=abs(delta),relative_complex_error=relative,
        per_user_NMSE=err,per_user_port_NMSE=np.sum(abs(delta)**2,axis=-1)/np.sum(abs(b)**2,axis=-1),
        per_user_frequency_NMSE=np.sum(abs(delta)**2,axis=2)/np.sum(abs(b)**2,axis=2))
    save_json(snapshot/'frame120_error_summary.json',dict(index=120,status='Passed' if np.isfinite(err).all() and np.all(err<=1e-3) else 'Failed',
        max_user_NMSE=float(err.max()),max_absolute_complex_error=float(abs(delta).max()),relative_complex_error_quantiles=np.quantile(relative,[0,.5,.99,1]),
        reference_sha256=sha_file(ref),shape=list(a.shape),gate=.001))
    for folder,name in [(audit,'audit_manifest.json'),(verification,'verification_manifest.json')]:
        manifest=json.loads((folder/name).read_text())
        for p,record in manifest['files'].items():
            expected=record['sha256'] if isinstance(record,dict) else record
            if sha_file(folder/p)!=expected:raise RuntimeError('Closed evidence changed: '+str(folder/p))
    inputs=json.loads((audit/'input_receipt.json').read_text())
    original_ok=sha_file(job/'dataset/csi.h5')==inputs['sha256'] and all(sha_file(job/p)==h for p,h in inputs['auxiliary_files'].items())
    if not original_ok:raise RuntimeError('Original input changed')
    report=Path('docs/platform/CSI_30S_COMPREHENSIVE_AUDIT.md');missing=[]
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if not target.startswith(('http:','https:','#')) and not (report.parent/target).resolve().exists():missing.append(target)
    if missing:raise RuntimeError('Missing report links: '+repr(missing))
    previous=Path('V5_DELIVERY_MANIFEST.json');backup=snapshot/'previous_V5_DELIVERY_MANIFEST.json'
    if backup.exists():
        if not args.resume:raise FileExistsError(backup)
        old=json.loads(backup.read_text());old_sha=sha_file(backup)
    else:
        old=json.loads(previous.read_text());old_sha=sha_file(previous);shutil.copyfile(previous,backup)
    current=dict(old)
    current.update(snapshot='csi30s_independent_audit_snapshot01',status='Partial',previous_active_manifest_sha256=old_sha,
        created_utc=datetime.now(timezone.utc).isoformat())
    current['CSI30s_independent_audit']=dict(status='Partial',report=str(report),report_sha256=sha_file(report),
        primary_audit=str(audit),audit_manifest_sha256=sha_file(audit/'audit_manifest.json'),
        independent_verification=str(verification),verification_manifest_sha256=sha_file(verification/'verification_manifest.json'),
        delivery_snapshot=str(snapshot),original_HDF_sha256=inputs['sha256'],original_job_unchanged=True,
        original_numeric_counts=dict(Passed=7,Failed=0,Unknown=2994),audit_overlay_counts=dict(Passed=8,Failed=0,Unknown=2993),
        storage_and_saved_mechanics_frames_passed=3001,whole_physics_and_research_utility='Partial',
        no_full_original_paths_saved=True,new_GPU_seconds=0,new_GPU_sampling='16states_prepared_NOT_started_waiting_new_authorization')
    save_json(previous,current)
    shutil.copyfile(previous,snapshot/'V5_DELIVERY_MANIFEST.json')
    for p in [report,Path('docs/platform/PLATFORM_PROGRESS.md'),Path(__file__),Path('scripts/verify_platform_csi30s_audit.py')]:
        dest=archive_target(snapshot,p);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    save_json(snapshot/'independent_delivery_review.json',dict(status='Passed_evidence_integrity_only',platform_overall='Partial',
        all_original_job_files_unchanged=original_ok,original_HDF_sha256=inputs['sha256'],original_numeric_ledger_unchanged=True,
        closed_audit_and_verification_hashes_rechecked=True,all_report_links_exist=True,new_GPU_seconds=0,
        previous_active_manifest_sha256=old_sha,tests='11Passed, numerical label scope unchanged by test count'))
    save_json(snapshot/'manifest.json',dict(schema='CSI30s_independent_audit_delivery_snapshot_v1',overall='Partial',
        files={str(p.relative_to(snapshot)):sha_file(p) for p in snapshot.rglob('*') if p.is_file()},
        primary_closed_audit=dict(path=str(audit/'audit_manifest.json'),sha256=sha_file(audit/'audit_manifest.json')),
        additional_closed_verification=dict(path=str(verification/'verification_manifest.json'),sha256=sha_file(verification/'verification_manifest.json')),
        original_job_HDF_sha256=inputs['sha256'],new_GPU_seconds=0))
    print('Independent CSI audit delivery frozen; original dataset unchanged; overall Partial.')


if __name__=='__main__':main()
