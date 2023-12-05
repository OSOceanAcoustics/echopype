import numpy as np
import pytest
from echopype.mask.api import get_seabed_mask
from echopype.mask.seabed import (
    ARIZA_DEFAULT_PARAMS,
    ARIZA_SPIKE_DEFAULT_PARAMS,
    ARIZA_EXPERIMENTAL_DEFAULT_PARAMS,
    BLACKWELL_DEFAULT_PARAMS,
    BLACKWELL_MOD_DEFAULT_PARAMS,
)

DESIRED_CHANNEL = "GPT  38 kHz 009072033fa5 1 ES38"


@pytest.mark.parametrize(
    "desired_channel,method,parameters,expected_true_false_counts",
    [
        (DESIRED_CHANNEL, "ariza", ARIZA_DEFAULT_PARAMS, (1531524, 635407)),
        (DESIRED_CHANNEL, "ariza", ARIZA_SPIKE_DEFAULT_PARAMS, (1531514, 635417)),
        (DESIRED_CHANNEL, "ariza", ARIZA_EXPERIMENTAL_DEFAULT_PARAMS, (1524807, 642124)),
        (DESIRED_CHANNEL, "blackwell", BLACKWELL_DEFAULT_PARAMS, (1738163, 428768)),
        (DESIRED_CHANNEL, "blackwell", BLACKWELL_MOD_DEFAULT_PARAMS, (1738163, 428768)),
    ],
)
def test_mask_seabed(
    complete_dataset_jr179, desired_channel, method, parameters, expected_true_false_counts
):
    source_Sv = complete_dataset_jr179
    mask = get_seabed_mask(
        source_Sv, desired_channel=desired_channel, method=method, parameters=parameters
    )
    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)

    assert true_false_counts == expected_true_false_counts
