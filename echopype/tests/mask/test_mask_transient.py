import os
import subprocess

import echopype as ep
import echopype.mask
import numpy as np
import pytest
import xarray as xr

from echopype.testing import TEST_DATA_FOLDER


file_name = "JR161-D20061118-T010645.raw"
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


@pytest.mark.parametrize(
    "mask_type,r0,r1,n,roff,thr,excludeabove,operation,expected_true_false_counts",
    [
        ("ryan", None, None, 20, None, 20, 250, 'percentile15', (1941916, 225015)),
        ("fielding", 200, 1000, 20, 250, [2, 0], None, None, (1890033, 276898)),
    ],
)
def test_get_transient_mask(
        mask_type, r0, r1, n, roff, thr, excludeabove, operation, expected_true_false_counts
):
    source_Sv = get_sv_dataset(test_data_path)
    desired_channel = "GPT  38 kHz 009072033fa5 1 ES38"
    mask = echopype.mask.get_transient_noise_mask(
        source_Sv,
        desired_channel,
        mask_type=mask_type,
        r0=r0,
        r1=r1,
        n=n,
        roff=roff,
        thr=thr,
        excludeabove=excludeabove,
        operation=operation
    )
    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)

    assert true_false_counts == expected_true_false_counts
