import numpy as np
import pytest

from echopype.mask.api import shoal_weill
from echopype.tests.conftest import sv_dataset_jr161

DESIRED_CHANNEL = "GPT  38 kHz 009072033fa5 1 ES38"


@pytest.mark.parametrize(
    "desired_channel,expected_tf_counts,expected_tf_counts_",
    [(DESIRED_CHANNEL, (186650, 1980281), (2166931, 0))],
)
def test_get_shoal_mask_weill(
    sv_dataset_jr161, desired_channel, expected_tf_counts, expected_tf_counts_
):
    source_Sv = sv_dataset_jr161
    mask, mask_ = shoal_weill(source_Sv, desired_channel)

    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)
    assert true_false_counts == expected_tf_counts

    count_true_ = np.count_nonzero(mask_)
    count_false_ = mask.size - count_true_
    true_false_counts_ = (count_true_, count_false_)
    assert true_false_counts_ == expected_tf_counts_
