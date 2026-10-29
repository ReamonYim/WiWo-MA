"""Compact current delivery and independent on-disk verification.

Retired historical bulky arrays are never silently expected or regenerated.
Snapshot50 remains a historical inventory, not a current storage promise.
"""
import hashlib,json,os,datetime
from pathlib import Path
from .config import ROOT
from .storage import atomic_json


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def publish_closed_long_validation(folder):
    """Publish only CPU-rechecked rows after the native writer has exited."""
    closure=folder/'closure.json';ledger=folder/'label_ledger.json';dataset=folder/'dataset/csi.h5'
    if not closure.exists() or not ledger.exists() or not dataset.exists():return
    import h5py
    checked=json.loads(ledger.read_text());rows=[]
    with h5py.File(dataset,'r') as f:
        for item in checked['frames']:
            i=item['index'];ref=folder/'references'/str(i)/'H.npz'
            if item['status']=='Unknown' or not ref.exists():continue
            rows.append(dict(status=item['status'],physical_timestamp=float(f['timestamps'][i]),frame_sha256=f['frame_sha256'].asstr()[i],
                configuration_sha256=f.attrs['configuration_sha256'],per_user_NMSE=item['per_user_NMSE'],reference_array_path=str(ref.relative_to(ROOT)),reference_array_file_sha256=digest(ref),
                scope='This checked row only; raw recorded Unknown and all unchecked rows unchanged'))
    target=dataset.with_name('independent_validation.json')
    payload=dict(schema='native-frame-validation-sidecar-v1',rows=rows,verifier_sha256=digest(Path(__file__)),CPU_ledger_sha256=digest(ledger))
    if target.exists():
        previous=json.loads(target.read_text())
        if previous.get('rows')==rows:return
        archive=target.with_name('independent_validation_previous_'+digest(target)[:16]+'.json')
        if not archive.exists():archive.write_bytes(target.read_bytes())
    atomic_json(target,payload)


def build(output):
    publish_closed_long_validation(ROOT/'artifacts/platform/jobs/munich30s_native_audited_attempt01')
    out=(ROOT/output).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Snapshot outside project')
    out.mkdir(parents=True,exist_ok=False)
    old=ROOT/'artifacts/v5/m6/delivery_snapshot_attempt50/manifest.json';history=json.loads(old.read_text())
    root_manifest=ROOT/'V5_DELIVERY_MANIFEST.json'
    previous=digest(root_manifest) if root_manifest.exists() else None
    if previous is not None and previous!=digest(old):
        # Preserve a unique previous active manifest, never a redundant copy.
        (out/'previous_unique_manifest.json').write_bytes(root_manifest.read_bytes())
    inventory={}
    for base in ['src/wiwo_ma/platform','tests/platform','scripts/platform','configs/platform','docs/platform']:
        for path in (ROOT/base).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:inventory[str(path.relative_to(ROOT))]=dict(sha256=digest(path),bytes=path.stat().st_size)
    for name in ['README.md','pyproject.toml','docs/v5/V5_PROGRESS.md','scripts/v5/status.py','scripts/v5/build_delivery_snapshot.py']:
        p=ROOT/name;inventory[name]=dict(sha256=digest(p),bytes=p.stat().st_size)
    evidence=[]
    for name in [
        'artifacts/platform/maintenance/cleanup_summary.json',
        'artifacts/platform/maintenance/legacy_retention_20261024.json',
        'artifacts/platform/maintenance/legacy_excerpt_verification.json',
        'artifacts/platform/mechanics30s_attempt01/report.json',
        'artifacts/platform/browser_full_attempt01/report.json',
        'artifacts/platform/browser_full_attempt02/report.json',
        'artifacts/platform/tests/final_cpu.json',
        'artifacts/platform/tests/final_cpu_snapshot52.json',
        'artifacts/platform/validation/native_gate_attempt01/proposal.json',
        'artifacts/platform/validation/native_gate_attempt01/independent_audit.json',
        'artifacts/platform/browser_native_CSI_attempt02/report.json',
        'artifacts/platform/browser_config_attempt01/report.json',
        'artifacts/platform/validation/native_gate_attempt01/execution_source_receipt.json',
        'artifacts/platform/browser_job_attach_attempt01/report.json',
        'artifacts/platform/jobs/munich30s_native_audited_attempt01/proposal.json',
        'artifacts/platform/jobs/munich30s_native_audited_attempt01/source_receipt.json',
        'experiments/platform/native_long30s36h_GPU3_approved_20261025.json',
        'artifacts/platform/browser_live_native_attempt01/report.json',
        'artifacts/platform/jobs/munich30s_native_audited_attempt01/closure.json',
        'artifacts/platform/jobs/munich30s_native_audited_attempt01/label_ledger.json',
        'artifacts/platform/jobs/munich30s_native_audited_attempt01/dataset/independent_validation.json',
        'experiments/platform/native_gate600_GPU3_approved_20261024.ledger.json',
    ]:
        p=ROOT/name
        if p.exists():evidence.append(dict(path=name,sha256=digest(p)))
    cleanup=json.loads((ROOT/'artifacts/platform/maintenance/cleanup_summary.json').read_text())
    plan=ROOT/'artifacts/platform/validation/native_gate_attempt01';gate=json.loads((plan/'state.json').read_text())
    gate_audit=json.loads((plan/'independent_audit.json').read_text()) if (plan/'independent_audit.json').exists() else {}
    receipt=ROOT/'experiments/platform/native_gate600_GPU3_approved_20261024.ledger.json'
    spent=json.loads(receipt.read_text())['spent_wall_s'] if receipt.exists() else 0.
    long_folder=ROOT/'artifacts/platform/jobs/munich30s_native_audited_attempt01'
    long_state=json.loads((long_folder/'state.json').read_text()) if (long_folder/'state.json').exists() else dict(status='not_prepared')
    early=[json.loads((long_folder/'references'/str(i)/'report.json').read_text()).get('status') if (long_folder/'references'/str(i)/'report.json').exists() else 'Unknown' for i in [0,1,2]]
    branch=json.loads((long_folder/'branch_report.json').read_text()) if (long_folder/'branch_report.json').exists() else dict(status='Unknown')
    continuity='Passed_first3_same_instance_and_restore_only' if early==['Passed']*3 and branch['status']=='Passed' else 'Partial_or_Unknown_early_gate_not_closed'
    data=dict(schema='wiwo-native-platform-delivery-v1',snapshot='Snapshot51' if out.name=='delivery_snapshot51' else out.name,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='Partial',
        active_scope='Native Sionna RT MA/6DMA mechanics, browser interaction, continuous complex CSI generation',
        paused_research=['V5 acceleration','DRT/E-DRT comparisons','Source Image DAG','Grouped indices','Actor Frontier','World Model','neural training','long Action Planning'],
        cleanup=cleanup,inventory=inventory,evidence=evidence,
        current_gates=dict(cleanup='Passed_scoped_retirement_with_receipts',CPU_mechanics='Passed_3001_samples_30s_and_restore_endpoint',
            CPU_unit_tests='See_bound_final_CPU_report',browser='Passed_scoped_real_mouse_and_legacy_CSI_replay_attempt02',
            browser_previous_attempt='Failed_timeout_preserved_Stop_callback_fixed',native_GPU_gate=gate,
            native_CUDA_CSI=gate_audit.get('counts','Unknown'),native_complete_path_set='Unknown_not_exported',
            same_instance_continuous_native_CSI=continuity,early_frame_reference_statuses=early,restored_branch_status=branch,new_30s_native_CSI=long_state,overall='Partial'),
        original_SDK_NMSE_threshold=.001,
        native_gate_GPU_seconds=spent,native_gate_outputs=gate_audit.get('native_saved_outputs',0),native_gate_distinct_states=gate_audit.get('distinct_configuration_physical_states',0),
        native_gate_scope='Snapshot_state_checks; same_SDK_instance_long_CSI_not_yet_tested',
        old_evidence=dict(snapshot50=dict(path=str(old.relative_to(ROOT)),sha256=digest(old),scope='Historical_inventory_some_arrays_explicitly_retired'),
            canonical801=history['current_validation']['canonical801'],
            original_DRT_weak_user=dict(status='Failed',NMSE=.0018550368728901295,research='Paused'),
            other_historical_Unknown_Partial_Failed='Retained reports/ledgers; no relabeling; raw retention governed by cleanup receipts'),
        long_job=dict(path='artifacts/platform/jobs/munich30s_native_audited_attempt01',state=long_state,frames=3001,
            H_axes=['time','user','rx_antenna','tx_antenna','subcarrier'],shape=[3001,40,1,64,64],raw_H_bytes=3933470720,
            RT_wall_budget='User_approved_independent36h_GPU3; estimated18-30h_not_a_guarantee; max12GiB/8GiB',
            authorization='experiments/platform/native_long30s36h_GPU3_approved_20261025.json',
            old_6h_plan='artifacts/platform/jobs/munich30s_prepared_attempt01_preserved_superseded_by_actual_timing'),
        previous_active_manifest_sha256=previous,historical_complete_inventory_is_not_current_inventory=True)
    atomic_json(out/'manifest.json',data);atomic_json(root_manifest,data)
    return out


def audit(output):
    out=(ROOT/output).resolve();p=out/'manifest.json';m=json.loads(p.read_text());errors=[];checked=0
    for name,item in m['inventory'].items():
        path=ROOT/name
        if not path.is_file() or digest(path)!=item['sha256']:errors.append(dict(path=name,error='current_source_or_config_hash_mismatch'))
        checked+=1
    for item in m['evidence']:
        if digest(ROOT/item['path'])!=item['sha256']:errors.append(dict(path=item['path'],error='bound_evidence_hash_mismatch'))
    gate=ROOT/'artifacts/platform/validation/native_gate_attempt01';proposal=json.loads((gate/'proposal.json').read_text())
    for name,expected in proposal['source_sha256'].items():
        path=gate/'executed_sources'/name
        if not path.exists() or digest(path)!=expected:errors.append(dict(path=str(path.relative_to(ROOT)),error='native_gate_executed_source_mismatch'))
    numeric=json.loads((gate/'independent_audit.json').read_text())
    for row in numeric['rows']:
        for filename,key in [('native.npz','native_array_file_sha256'),('reference.npz','reference_array_file_sha256')]:
            path=gate/row['case']/filename
            if digest(path)!=row[key]:errors.append(dict(path=str(path.relative_to(ROOT)),error='native_gate_raw_array_binding_mismatch'))
    frozen=json.loads((ROOT/'artifacts/v5/m0/v4_frozen_20261013/manifest.json').read_text())
    for name,expected in frozen['executed_V4_package_sources'].items():
        if digest(ROOT/name)!=expected:errors.append(dict(path=name,error='frozen_V4_source_changed'))
    historical=json.loads((ROOT/m['old_evidence']['snapshot50']['path']).read_text());prefix='artifacts/v5/m2/absolute_actor_continuous_attempt01/full/actor40/'
    canonical=[]
    for name,item in historical['inventory'].items():
        if not name.startswith(prefix):continue
        path=ROOT/name
        if not path.is_file() or digest(path)!=item['sha256']:errors.append(dict(path=name,error='retained_original801_bytes_changed_or_missing'))
        canonical.append(name)
    for item in m['cleanup']['retained']:
        if not (ROOT/item['path']).exists():errors.append(dict(path=item['path'],error='required_retained_dependency_missing'))
    retained_v4=[]
    for item in historical['large_frozen_datasets']:
        if item['path'].startswith('artifacts/v4_release/'):
            actual=(ROOT/item['path']).stat().st_size
            if actual!=item['bytes']:errors.append(dict(path=item['path'],error='retained_long_HDF_size_changed'))
            retained_v4.append(dict(path=item['path'],bytes=actual,check='Size/existence now; original full-byte hash retained, not rehashed this audit'))
    report=dict(status='Passed_scoped_independent_inventory' if not errors else 'Failed',manifest_sha256=digest(p),source_files_checked=checked,
        frozen_V4_source_files_checked=len(frozen['executed_V4_package_sources']),canonical801_files_fully_rehashed=len(canonical),retained_V4=retained_v4,
        errors=errors,new_GPU_seconds=0,new_CSI_labels=0,numerical_accuracy_scope='Not_recomputed_by_inventory_audit; see_bound_raw5Passed; long_and_full_path_unknown',not_an_overall_platform_acceptance=True)
    atomic_json(out/'independent_audit.json',report);atomic_json(out/'state.json',dict(status=report['status'],overall='Partial'));return report
