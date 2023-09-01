import os
import subprocess

import echopype.clean
import numpy as np
import pytest
import xarray as xr


# Note: We've removed all the setup and utility functions since they are now in conftest.py


@pytest.mark.parametrize(
    "mask_type,r0,r1,n,roff,thr,excludeabove,operation,expected_true_false_counts",
    [
        ("ryan", None, None, 20, None, 20, 250, "percentile15", (1941916, 225015)),
        ("fielding", 200, 1000, 20, 250, [2, 0], None, None, (1890033, 276898)),
    ],
)
def test_get_transient_mask(
    sv_dataset_jr161,  # Use the specific fixture for the JR161 file
    mask_type,
    r0,
    r1,
    n,
    roff,
    thr,
    excludeabove,
    operation,
    expected_true_false_counts,
):
    source_Sv = sv_dataset_jr161
    desired_channel = "GPT  38 kHz 009072033fa5 1 ES38"
    mask = echopype.clean.get_transient_noise_mask(
        source_Sv,
        desired_channel,
        mask_type=mask_type,
        r0=r0,
        r1=r1,
        n=n,
        roff=roff,
        thr=thr,
        excludeabove=excludeabove,
        operation=operation,
    )
    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)

    assert true_false_counts == expected_true_false_counts
