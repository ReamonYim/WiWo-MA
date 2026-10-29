"""Test the conservative guard's scope: never attribute Ngid or foreign GPUs."""
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'namespace_bound', Path(__file__).resolve().parents[2]/'scripts/platform/monitor_GPU_namespace_bound.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_bound_uses_all_reported_processes_and_only_approved_uuid():
    rows = monitor.parse_rows('100, 4000, GPU-other\n200, 1, GPU-ok\n201, 3, GPU-ok\n')
    result = monitor.bound(rows, 'GPU-ok', 4*1024**2)
    assert result['upper_bound_bytes'] == 3*1024**2
    assert result['stop_owned_worker'] is False


def test_unknown_memory_cannot_be_reported_as_safe():
    rows = monitor.parse_rows('100, [N/A], GPU-ok\n101, 2, GPU-ok\n')
    result = monitor.bound(rows, 'GPU-ok', 1024**2)
    assert result['status'].startswith('Unknown')
    assert result['upper_bound_bytes'] is None
    assert result['stop_owned_worker'] is False


def test_empty_gpu_is_unknown_and_real_overage_trips_conservative_guard():
    assert monitor.bound([], 'GPU-ok', 1024)['status'].startswith('Unknown')
    rows = monitor.parse_rows('100, 4, GPU-ok\n')
    result = monitor.bound(rows, 'GPU-ok', 3*1024**2)
    assert result['stop_owned_worker'] is True


def test_exact_limit_is_allowed_and_no_pid_identity_is_inferred():
    rows = monitor.parse_rows('580245, 4, GPU-ok\n')
    result = monitor.bound(rows, 'GPU-ok', 4*1024**2)
    assert result['stop_owned_worker'] is False
    assert 'owned_NVML_process_identity' not in result
