import pytest
import numpy as np
import echopype.clean


DEFAULT_RYAN_PARAMS = {"r0": 180, "r1": 280, "n": 30, "thr": -6, "start": 0}

# commented ariza out since the current interpretation relies on a
# preexisting seabed mask, which is not available in this PR
# DEFAULT_ARIZA_PARAMS = {"offset": 20, "thr": (-40, -35), "m": 20, "n": 50}


@pytest.mark.parametrize(
    "method,parameters,expected_true_false_counts",
    [
        ("ryan", DEFAULT_RYAN_PARAMS, (1838934, 327997)),
        # ("ariza", DEFAULT_ARIZA_PARAMS, (39897, 2127034)),
    ],
)
def test_get_signal_attenuation_mask(
    sv_dataset_jr161, method, parameters, expected_true_false_counts
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
    print(true_false_counts)
    assert true_false_counts == expected_true_false_counts
