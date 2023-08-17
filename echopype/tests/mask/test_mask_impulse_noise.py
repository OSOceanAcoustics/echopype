import os
import subprocess

import echopype as ep
import echopype.mask
import numpy as np
import pytest
import xarray as xr

from echopype.testing import TEST_DATA_FOLDER

file_name = "JR230-D20091215-T121917.raw"
ftp_main = "ftp://ftp.bas.ac.uk"
ftp_partial_path = "/rapidkrill/ek60/"

test_data_path: str = os.path.join(
    TEST_DATA_FOLDER,
    file_name,
)


def set_up():
    "Gets the test data if it doesn't already exist"
    if not os.path.exists(TEST_DATA_FOLDER):
        os.mkdir(TEST_DATA_FOLDER)
    if not os.path.exists(test_data_path):
        ftp_file_path = ftp_main + ftp_partial_path + file_name
        subprocess.run(["wget", ftp_file_path, "-O", test_data_path])


def get_sv_dataset(file_path: str) -> xr.DataArray:
    set_up()
    ed = ep.open_raw(file_path, sonar_model="ek60")
    Sv = ep.calibrate.compute_Sv(ed).compute()
    return Sv


source_Sv = get_sv_dataset(test_data_path)
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
