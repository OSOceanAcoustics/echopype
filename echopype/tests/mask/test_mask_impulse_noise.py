import os
from typing import List, Optional, Union, Tuple

import echopype as ep
import echopype.mask
import numpy as np
import pytest
import xarray as xr


"""
In order to run `test_mask_impulse_noise.py` with pytest, follow the steps below:
    1. Locate the `test_data` directory in the current project.
    2. Inside the `test_data` directory, create a new folder named `test_impulse_noise_mask_data`.
    3. Download the necessary test file `JR230-D20091215-T121917.raw` from the provided link.
        Link: [ ftp://ftp.bas.ac.uk/rapidkrill/ ]
        The link was proviced HERE: [ https://github.com/open-ocean-sounding/echopy/tree/master/data ]
After these steps, you should be able to successfully run the tests with pytest.
"""

echopype_path = os.path.abspath("./")
test_data_path = os.path.join(
    echopype_path,
    "echopype",
    "test_data",
    "test_impulse_noise_mask_data",
    "JR230-D20091215-T121917.raw",
)


file_path = test_data_path


def get_sv_dataset(file_path: str) -> xr.DataArray:
    ed = ep.open_raw(file_path, sonar_model="ek60")
    Sv = ep.calibrate.compute_Sv(ed).compute()
    return Sv


source_Sv = get_sv_dataset(file_path)
desired_channel = "GPT 120 kHz 00907203422d 1 ES120-7"


@pytest.mark.parametrize(
    "method,thr,m,n,erode,dilate,median,expected_true_false_counts",
    [
        ("ryan", 10, 5, 1, None, None, None, (2130885, 32419)),
        ("ryan_iterable", 10, 5, (1, 2), None, None, None, (2125144, 38160)),
        ("wang", (-70, -40), None, None, [(3, 3)], [(5, 5), (7, 7)], [(7, 7)], (635732, 1527572)),
    ],
)
def test_get_impulse_noise_mask(
    method, thr, m, n, erode, dilate, median, expected_true_false_counts
):
    mask = echopype.mask.get_impulse_noise_mask(
        source_Sv,
        desired_channel,
        thr=thr,
        m=m,
        n=n,
        erode=erode,
        dilate=dilate,
        median=median,
        method=method,
    )
    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)

    assert true_false_counts == expected_true_false_counts
