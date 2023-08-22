import os
import subprocess

import echopype as ep
import echopype.mask.mask_shoal
import numpy as np
import xarray as xr

import pytest
from echopype.testing import TEST_DATA_FOLDER


file_name = "JR161-D20061118-T010645.raw"
ftp_main = "ftp://ftp.bas.ac.uk"
ftp_partial_path = "/rapidkrill/ek60/"

test_data_path: str = os.path.join(
    TEST_DATA_FOLDER,
    file_name,
)


def set_up():
    """Gets the test data if it doesn't already exist"""
    if not os.path.exists(TEST_DATA_FOLDER):
        os.mkdir(TEST_DATA_FOLDER)
    if not os.path.exists(test_data_path):
        ftp_file_path = ftp_main + ftp_partial_path + file_name
        subprocess.run(["wget", ftp_file_path, "-O", test_data_path])


def get_sv_dataset(file_path: str) -> xr.DataArray:
    set_up()
    ed = ep.open_raw(file_path, sonar_model="EK60")
    Sv = ep.calibrate.compute_Sv(ed).compute()
    Sv_loc = ep.consolidate.add_location(Sv, ed)
    # Sv_depth = ep.consolidate.add_depth(Sv_loc)
    return Sv_loc


@pytest.mark.parametrize(
    "expected_tf_counts,expected_tf_counts_,test_data_path",
    [((186650, 1980281), (2166931, 0), test_data_path)],
)
def test_get_shoal_mask_weill(expected_tf_counts, expected_tf_counts_, test_data_path):
    source_Sv = get_sv_dataset(test_data_path)
    mask, mask_ = echopype.mask.mask_shoal.weill(source_Sv)

    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)
    assert true_false_counts == expected_tf_counts

    count_true_ = np.count_nonzero(mask_)
    count_false_ = mask.size - count_true_
    true_false_counts_ = (count_true_, count_false_)
    assert true_false_counts_ == expected_tf_counts_
