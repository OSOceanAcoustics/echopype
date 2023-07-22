import numpy as np
import xarray as xr

from echopype.qc import coerce_increasing_time, exist_reversed_time
from echopype.qc.api import _clean_reversed

import pytest


@pytest.fixture
def ds_time():
    return  xr.Dataset(
        data_vars={"a": ("time", np.arange(36))},
        coords={
            "time": np.array([
                '2021-07-15T22:59:54.328000000', '2021-07-15T22:59:54.598000128',
                '2021-07-15T22:59:54.824999936', '2021-07-15T22:59:55.170999808',
                '2021-07-15T22:59:56.172999680', '2021-07-15T22:59:55.467999744',
                '2021-07-15T22:59:55.737999872', '2021-07-15T22:59:55.966000128',
                '2021-07-15T22:59:56.467999744', '2021-07-15T22:59:56.813000192',
                '2021-07-15T22:59:57.040999936', '2021-07-15T22:59:57.178999808',
                '2021-07-15T22:59:58.178999808', '2021-07-15T22:59:57.821000192',
                '2021-07-15T22:59:58.092000256', '2021-07-15T22:59:58.318999552',
                '2021-07-15T22:59:58.730999808', '2021-07-15T22:59:59.092000256',
                '2021-07-15T22:59:59.170999808', '2021-07-15T23:00:00.170999808',
                '2021-07-15T23:00:01.170999808', '2021-07-15T22:59:59.719000064',
                '2021-07-15T22:59:59.989999616', '2021-07-15T23:00:00.573000192',
                '2021-07-15T23:00:00.843999744', '2021-07-15T23:00:01.071000064',
                '2021-07-15T23:00:02.170999808', '2021-07-15T23:00:03.181000192',
                '2021-07-15T23:00:01.692999680', '2021-07-15T23:00:02.054000128',
                '2021-07-15T23:00:02.592999936', '2021-07-15T23:00:02.864000000',
                '2021-07-15T23:00:03.480999936', '2021-07-15T23:00:04.171999744',
                '2021-07-15T23:00:05.179999744', '2021-07-15T23:00:03.771999744'],
                dtype='datetime64[ns]')
        },
    )



@pytest.mark.parametrize(
    ["win_len", "input_arr", "expected_arr"],
    [
        (
            2,
            np.array([0,1,2,3,4,2,3,4,6,8,10,11,12,13,15,17,19,13,15,17,21], dtype="datetime64[ns]"),
            np.array([0,1,2,3,4,5,6,7,9,11,13,14,15,16,18,20,22,24,26,28,32], dtype="datetime64[ns]")
        ),
        (
            6,
            (np.array([0,1,2,3,4,2,3,4,6,8,10,11,12,13,15,17,19,13,15,17,21])*2).astype("datetime64[ns]"),
            np.array([0,2,4,6,8,10,12,14,18,22,26,28,30,32,36,40,44,47,51,55,63]).astype("datetime64[ns]"),
        ),
    ],
    ids=[
        "win_len2",
        "win_len6"
    ]
)
def test__clean_reversed(win_len, input_arr, expected_arr):
    arr_fixed = _clean_reversed(input_arr, win_len)

    # fixed array follows monotonically increasing order
    arr_fixed_diff = np.diff(arr_fixed)
    assert np.argwhere(arr_fixed_diff < np.timedelta64(0, "ns")).flatten().size == 0

    # new filled value should have diff being the median of local_win_len before reversal
    assert np.all(arr_fixed == expected_arr)


def test_coerce_increasing_time(ds_time):
    # fixed timestamp follows monotonically increasing order
    coerce_increasing_time(ds_time, "time")
    assert np.argwhere(ds_time["time"].diff(dim="time").data < np.timedelta64(0, "ns")).flatten().size == 0


def test_exist_reversed_time(ds_time):
    # data has reversed timestamps to begin with
    assert exist_reversed_time(ds_time, "time") == True

    # after correction there are no reversed timestamps
    coerce_increasing_time(ds_time, "time")
    assert exist_reversed_time(ds_time, "time") == False
