import pytest

import numpy as np
import echopype.clean
from echopype.clean.transient_noise import RYAN_DEFAULT_PARAMS, FIELDING_DEFAULT_PARAMS

# Note: We've removed all the setup and utility functions since they are now in conftest.py


@pytest.mark.parametrize(
    "method, parameters ,expected_true_false_counts",
    [
        ("ryan", RYAN_DEFAULT_PARAMS, (2115052, 51879)),
        ("fielding", FIELDING_DEFAULT_PARAMS, (2117333, 49598)),
    ],
)
def test_get_transient_mask(
    sv_dataset_jr161,  # Use the specific fixture for the JR161 file
    method,
    parameters,
    expected_true_false_counts,
):
    source_Sv = sv_dataset_jr161
    desired_channel = "GPT  38 kHz 009072033fa5 1 ES38"
    mask = echopype.clean.get_transient_noise_mask(
        source_Sv,
        parameters=parameters,
        desired_channel=desired_channel,
        method=method,
    )
    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)

    assert true_false_counts == expected_true_false_counts
