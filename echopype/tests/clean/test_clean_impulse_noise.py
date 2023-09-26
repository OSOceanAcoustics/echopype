import pytest
import numpy as np
import echopype.clean


# Note: We've removed all the setup and utility functions since they are now in conftest.py


@pytest.mark.parametrize(
    "method,thr,m,n,erode,dilate,median,expected_true_false_counts",
    [
        ("ryan", 10, 5, 1, None, None, None, (2130885, 32419)),
        ("ryan_iterable", 10, 5, (1, 2), None, None, None, (2124976, 38328)),
        ("wang", (-70, -40), None, None, [(3, 3)], [(5, 5), (7, 7)], [(7, 7)], (635732, 1527572)),
    ],
)
def test_get_impulse_noise_mask(
    sv_dataset_jr230,  # Use the specific fixture for the JR230 file
    method,
    thr,
    m,
    n,
    erode,
    dilate,
    median,
    expected_true_false_counts,
):
    source_Sv = sv_dataset_jr230
    desired_channel = "GPT 120 kHz 00907203422d 1 ES120-7"
    mask = echopype.clean.get_impulse_noise_mask(
        source_Sv,
        desired_channel,
        thr=thr,
        m=m,
        n=n,
        erode=erode,
        dilate=dilate,
        median=median,
        method=method,
    )
    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)

    assert true_false_counts == expected_true_false_counts
