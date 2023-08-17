import pytest
import numpy as np
import echopype.clean


@pytest.mark.parametrize(
    "mask_type,r0,r1,n,m,thr,start,offset,expected_true_false_counts",
    [
        ("ryan", 180, 280, 30, None, -6, 0, 0, (1613100, 553831)),
    ],
)
def test_get_signal_attenuation_mask(
    sv_dataset_jr161, mask_type, r0, r1, n, m, thr, start, offset, expected_true_false_counts
):
    # source_Sv = get_sv_dataset(test_data_path)
    mask = echopype.clean.api.get_attenuation_mask(
        sv_dataset_jr161,
        mask_type=mask_type,
        r0=r0,
        r1=r1,
        thr=thr,
        m=m,
        n=n,
        start=start,
        offset=offset,
    )
    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)

    assert true_false_counts == expected_true_false_counts
