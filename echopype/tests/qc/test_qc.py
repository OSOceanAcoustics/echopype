import numpy as np
import xarray as xr

from echopype.qc import coerce_increasing_time
from echopype.qc.api import _clean_reversed


def test__clean_reversed():
    arr = np.array([0,1,2,3,4,2,3,4,6,8,10,11,12,13,15,17,19,13,15,17,21])
    arr_fixed = _clean_reversed(arr)
    arr_fixed_diff = np.diff(arr_fixed)

    # fixed array follows monotonically increasing order
    assert np.argwhere(arr_fixed_diff < 0).flatten().size == 0


def test_coerce_increasing_time():
    ds = xr.Dataset(
        data_vars={
            "a": ("time1", np.random.random(21)),
            "b": ("time1", np.random.random(21)),
        },
        coords={"time1": [0,1,2,3,4,2,3,4,6,8,10,11,12,13,15,17,19,13,15,17,21]},
    )

    coerce_increasing_time(ds, "time1")

    # fixed timestamp follows monotonically increasing order
    assert np.argwhere(ds["time1"].data < 0).flatten().size == 0
