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
    2. Download the necessary test file `JR179-D20080410-T150637.raw` from the provided link.
        Link: [ ftp://ftp.bas.ac.uk/rapidkrill/ ]
        The link was proviced HERE: [ https://github.com/open-ocean-sounding/echopy/tree/master/data ]
After these steps, you should be able to successfully run the tests with pytest.
"""



def get_sv_dataset(file_path: str) -> xr.DataArray:
    ed = ep.open_raw(file_path, sonar_model="ek60")
    Sv = ep.calibrate.compute_Sv(ed).compute()
    
    return Sv, ed

@pytest.mark.parametrize(
    "mask_type,expected_true_false_counts",
    [
        ("ariza", (1430880, 736051)),
        ("experimental", (1514853, 652078)),
        ("blackwell", (1748083, 418848)),
        ("blackwell_mod", (1946168, 220763)),
        
    ],
)
def test_get_impulse_noise_mask(
    mask_type,expected_true_false_counts
):
    echopype_path = os.path.abspath("../")
    test_data_path = os.path.join(
        echopype_path,
        "test_data",
        "JR179-D20080410-T150637.raw",
    )
    file_path = test_data_path
    source_Sv, ed = get_sv_dataset(file_path)
    mask = echopype.mask.get_seabed_mask(
        source_Sv,
        mask_type=mask_type,
        theta=ed["Sonar/Beam_group1"]["angle_alongship"].values[0,:,:,0].T,
        phi  =ed["Sonar/Beam_group1"]["angle_athwartship"].values[0,:,:,0].T
    )
    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)

    assert true_false_counts == expected_true_false_counts
