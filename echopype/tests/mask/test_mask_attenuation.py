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
    2. Download the necessary test file `JR161-D20061118-T010645.raw` from the provided link.
        Link: [ ftp://ftp.bas.ac.uk/rapidkrill/ ]
        The link was proviced HERE: [ https://github.com/open-ocean-sounding/echopy/tree/master/data ]
After these steps, you should be able to successfully run the tests with pytest.
"""

def get_sv_dataset(file_path: str) -> xr.DataArray:
    ed = ep.open_raw(file_path, sonar_model="ek60")
    Sv = ep.calibrate.compute_Sv(ed).compute()
    return Sv


@pytest.mark.parametrize(
    "mask_type,r0,r1,n,m,thr,start,offset,expected_true_false_counts",
    [
        ("ryan", 180, 280, 30, None, -6, 0, 0, (1613100, 553831)),
    ],
)
def test_get_signal_attenuation_mask(
    mask_type, r0,r1,n,m,thr,start,offset,expected_true_false_counts
):
    echopype_path = os.path.abspath("../")
    test_data_path = os.path.join(
        echopype_path,
        "test_data",
        "JR161-D20061118-T010645.raw",
    )
    file_path = test_data_path
    source_Sv = get_sv_dataset(file_path)
    mask = echopype.mask.get_attenuation_mask(
        source_Sv,
        mask_type=mask_type,
        r0=r0,
        r1=r1,
        thr=thr,
        m=m,
        n=n,
        start=start,
        offset=offset
    )
    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)

    assert true_false_counts == expected_true_false_counts
