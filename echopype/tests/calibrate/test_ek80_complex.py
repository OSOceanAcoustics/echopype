import pytest
import numpy as np
import xarray as xr

from echopype.calibrate.ek80_complex import get_vend_filter_EK80


@pytest.fixture
def ek80_path(test_path):
    return test_path['EK80']


def gen_mock_vend(ch_num, filter_len=10, has_nan=False):
    vend = xr.Dataset(
        data_vars={
            "WBT_filter_r": (["channel", "WBT_filter_n"], np.random.rand(ch_num, filter_len)),
            "WBT_filter_i": (["channel", "WBT_filter_n"], np.random.rand(ch_num, filter_len)),
            "WBT_decimation": 6,
            "PC_filter_r": (["channel", "PC_filter_n"], np.random.rand(ch_num, filter_len*2)),
            "PC_filter_i": (["channel", "PC_filter_n"], np.random.rand(ch_num, filter_len*2)),
            "PC_decimation": 1,
        },
        coords={
            "channel": [f"ch_{ch}" for ch in np.arange(ch_num)],
            "WBT_filter_n": np.arange(filter_len),
            "PC_filter_n": np.arange(filter_len*2),
        }
    )
    if has_nan:  # replace some parts of filter coeff with NaN
        if filter_len != 1:
            vend["WBT_filter_r"].data[:, int(filter_len/2):] = np.nan
            vend["WBT_filter_i"].data[:, int(filter_len/2):] = np.nan
            vend["PC_filter_r"].data[:, filter_len:] = np.nan
            vend["PC_filter_i"].data[:, filter_len:] = np.nan
        else:
            raise ValueError("Cannot replace some parts of filter coeff with NaN")
    return vend


@pytest.mark.parametrize(
    ("ch_num", "filter_len", "has_nan"),
    [
        # filter coeff are of the same length for all channels
        (2, 10, False),
        # filter coeff are of different lengths across channels, so some parts are NaN-padded
        (2, 10, True),
        # filter coeff is of length=1
        (2, 1, False),
    ],
    ids=[
        "filter_coeff_filled",
        "filter_coeff_has_nan",
        "filter_coeff_len_1",
    ]
)
def test_get_vend_filter_EK80(ch_num, filter_len, has_nan):
    vend = gen_mock_vend(ch_num, filter_len, has_nan)
    
    for ch in [f"ch_{ch}" for ch in np.arange(ch_num)]:
        for filter_name in ["WBT", "PC"]:
            var_imag = f"{filter_name}_filter_i"
            var_real = f"{filter_name}_filter_r"
            var_df = f"{filter_name}_decimation"
            sel_vend = vend.sel(channel=ch)
            
            assert np.all(
                (sel_vend[var_real] + 1j * sel_vend[var_imag]).dropna(dim=f"{filter_name}_filter_n").values
                == get_vend_filter_EK80(vend, channel_id=ch, filter_name=filter_name, param_type="coeff")
            )

            assert sel_vend[var_df].values == get_vend_filter_EK80(
                vend, channel_id=ch, filter_name=filter_name, param_type="decimation"
            )
