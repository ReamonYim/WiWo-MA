import json
import numpy as np
import pytest
from wiwo_ma.platform.config import ROOT
from wiwo_ma.platform.longstudy import nmse,verify_inputs,supervise
from wiwo_ma.platform.controller import Replay


def test_relative_complex_gate_includes_weak_and_zero_reference_users():
    ref=np.array([1e-10+2e-10j,0j,0j],np.complex64).reshape(3,1,1,1);a=ref.copy();a[0]+=1e-11;a[2]=1e-15
    e=nmse(a,ref)
    assert e[0]>.001 and e[1]==0 and np.isinf(e[2])




