import numpy as np
import pytest

from echopype.mask.api import get_shoal_mask
from echopype.mask.shoal import WEILL_DEFAULT_PARAMETERS

DESIRED_CHANNEL = "GPT  38 kHz 009072033fa5 1 ES38"


@pytest.mark.parametrize(
    "method, desired_channel,parameters,expected_tf_counts",
    [("will", DESIRED_CHANNEL, WEILL_DEFAULT_PARAMETERS, (101550, 2065381))],
)
def test_get_shoal_mask_weill(
    sv_dataset_jr161, method, desired_channel, parameters, expected_tf_counts
):
    source_Sv = sv_dataset_jr161
    mask = get_shoal_mask(
        source_Sv,
        method=method,
        desired_channel=desired_channel,
        parameters=parameters,
    )

    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)
    assert true_false_counts == expected_tf_counts
