"""Regression of an actually observed browser seek crossing a Float64 ULP."""
import numpy as np
import pytest
from wiwo_ma.platform.storage import time_index


def test_decimal_GUI_time_matches_same_mechanics_tick_not_previous_CSI():
    times=np.arange(401)*5*.002
    assert times[201]>2.01
    assert time_index(times,2.01)==201
    for index in range(401):assert time_index(times,index/100)==index


def test_off_grid_times_are_floored_and_invalid_times_rejected():
    times=np.arange(401)*5*.002
    assert time_index(times,2.00999999)==200
    assert time_index(times,2.015)==201
    assert time_index(times,-1)==0 and time_index(times,8)==400
    for bad in (np.nan,np.inf):
        with pytest.raises(ValueError):time_index(times,bad)
    with pytest.raises(ValueError):time_index([],0)
