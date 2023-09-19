import numpy as np
import pytest
from echopype.mask.api import get_seabed_mask

DESIRED_CHANNEL = "GPT  38 kHz 009072033fa5 1 ES38"


@pytest.mark.parametrize(
    "desired_channel,mask_type,expected_true_false_counts",
    [
        (DESIRED_CHANNEL, "ariza", (1430880, 736051)),
        (DESIRED_CHANNEL, "experimental", (1514853, 652078)),
        (DESIRED_CHANNEL, "blackwell", (1746551, 420380)),
        (DESIRED_CHANNEL, "blackwell_mod", (1945202, 221729)),
    ],
)
def test_mask_seabed(
    complete_dataset_jr179, desired_channel, mask_type, expected_true_false_counts
):
    source_Sv = complete_dataset_jr179
    mask = get_seabed_mask(source_Sv, desired_channel=desired_channel, mask_type=mask_type)
    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)

    assert true_false_counts == expected_true_false_counts
