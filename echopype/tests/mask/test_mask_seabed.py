import os
import subprocess
from typing import Optional

from xarray import Dataset

import echopype as ep
import echopype.mask
import numpy as np
import pytest

from echopype.echodata import EchoData
from echopype.testing import TEST_DATA_FOLDER

file_name = "JR179-D20080410-T150637.raw"
ftp_main = "ftp://ftp.bas.ac.uk"
ftp_partial_path = "/rapidkrill/ek60/"

test_data_path: str = os.path.join(
    TEST_DATA_FOLDER,
    file_name.upper(),
)


def set_up():
    """Gets the test data if it doesn't already exist"""
    if not os.path.exists(TEST_DATA_FOLDER):
        os.mkdir(TEST_DATA_FOLDER)
    if not os.path.exists(test_data_path):
        ftp_file_path = ftp_main + ftp_partial_path + file_name
        print(ftp_file_path)
        print(test_data_path)
        subprocess.run(["wget", ftp_file_path, "-O", test_data_path])


def get_sv_dataset(file_path: str) -> tuple[Dataset, Optional[EchoData]]:
    set_up()
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
def test_mask_seabed(mask_type, expected_true_false_counts):
    source_Sv, ed = get_sv_dataset(test_data_path)
    mask = echopype.mask.get_seabed_mask(
        source_Sv,
        mask_type=mask_type,
        theta=ed["Sonar/Beam_group1"]["angle_alongship"].values[0, :, :, 0].T,
        phi=ed["Sonar/Beam_group1"]["angle_athwartship"].values[0, :, :, 0].T,
    )
    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)

    assert true_false_counts == expected_true_false_counts
