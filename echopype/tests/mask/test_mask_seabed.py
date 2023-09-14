import os
import subprocess
from typing import Optional

from xarray import Dataset

import echopype as ep
from echopype.mask.api import get_seabed_mask
import numpy as np
import pytest

from echopype.echodata import EchoData
from echopype.testing import TEST_DATA_FOLDER
from echopype.consolidate import add_splitbeam_angle
from echopype.tests.conftest import complete_dataset_jr179

DESIRED_CHANNEL = "GPT  38 kHz 009072033fa5 1 ES38"


@pytest.mark.parametrize(
    "desired_channel,mask_type,expected_true_false_counts",
    [
        (DESIRED_CHANNEL, "ariza", (1430880, 736051)),
        (DESIRED_CHANNEL, "experimental", (1514853, 652078)),
        # (DESIRED_CHANNEL, "blackwell", (1748083, 418848)),
        # (DESIRED_CHANNEL, "blackwell_mod", (1946168, 220763)),
    ],
)
def test_mask_seabed(
    complete_dataset_jr179, desired_channel, mask_type, expected_true_false_counts
):
    source_Sv = complete_dataset_jr179
    theta = source_Sv["angle_alongship"].values[0, :, :].T
    phi = source_Sv["angle_athwartship"].values[0, :, :].T

    # mask = get_seabed_mask(source_Sv, desired_channel, mask_type=mask_type, theta=theta, phi=phi)
    mask = get_seabed_mask(source_Sv, desired_channel=desired_channel, mask_type=mask_type)
    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)

    assert true_false_counts == expected_true_false_counts


def test_scratch(complete_dataset_jr179):
    source_Sv = complete_dataset_jr179
    channel_Sv = source_Sv.sel(channel=DESIRED_CHANNEL)
    Sv_old = source_Sv["Sv"].values[0].T
    t1 = channel_Sv["angle_alongship"]
    Sv = channel_Sv["Sv"].values.T
    r = channel_Sv["echo_range"].values[0, 0]
    # print(t1)
    # print(channel_Sv)
    print(Sv.shape)
    print(Sv_old.shape)
    print(r.shape)
