import pytest
import numpy as np
import echopype.clean

from echopype.clean.signal_attenuation import DEFAULT_RYAN_PARAMS

# commented ariza out since the current interpretation relies on a
# preexisting seabed mask, which is not available in this PR


@pytest.mark.parametrize(
    "method,parameters,expected_true_false_counts",
    [
        ("ryan", DEFAULT_RYAN_PARAMS, (1881950, 284981)),
        # ("ariza", DEFAULT_ARIZA_PARAMS, (39897, 2127034)),
    ],
)
def test_get_signal_attenuation_mask(
    sv_dataset_jr161,
    method,
    parameters,
    expected_true_false_counts,
):
    # source_Sv = get_sv_dataset(test_data_path)
    desired_channel = "GPT  38 kHz 009072033fa5 1 ES38"
    mask = echopype.clean.api.get_attenuation_mask(
        sv_dataset_jr161,
        parameters=parameters,
        method=method,
        desired_channel=desired_channel,
    )
    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)

    assert true_false_counts == expected_true_false_counts
