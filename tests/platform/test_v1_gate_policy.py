import importlib.util
import pytest
from wiwo_ma.platform.config import ROOT


def gate():
    spec=importlib.util.spec_from_file_location('current_gate',ROOT/'scripts/platform/run_current_pose_Web_gate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.require_case_success


@pytest.mark.parametrize('layer',['Partial','Failed','Unknown'])
def test_required_path_gate_stops_even_when_CSI_passed(layer):
    paths=dict(Passed=0,Partial=0,Failed=0,Unknown=0);paths[layer]=1
    with pytest.raises(RuntimeError,match='stop before later GPU'):gate()(dict(path_gate='Required_Passed'),dict(counts=dict(Passed=1,Failed=0,Unknown=0),path_counts=paths))


def test_original_CSI_only_polarized_history_is_not_retroactively_upgraded_or_rejected():
    gate()(dict(path_gate='Unknown_allowed_historical_scope'),dict(counts=dict(Passed=1,Failed=0,Unknown=0),path_counts=dict(Passed=0,Partial=0,Failed=0,Unknown=1)))


def test_new_complete_gate_accepts_only_an_actually_Passed_state():
    gate()(dict(path_gate='Required_Passed'),dict(counts=dict(Passed=1,Failed=0,Unknown=0),path_counts=dict(Passed=1,Partial=0,Failed=0,Unknown=0)))
