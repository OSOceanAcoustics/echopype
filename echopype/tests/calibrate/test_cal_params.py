import pytest

import numpy as np
import xarray as xr

from echopype.calibrate.cal_params import _get_interp_da


@pytest.fixture
def freq_center():
    return xr.DataArray([25, 55], dims=["channel"], coords={"channel": ["chA", "chB"]})


# check for output dimension: channel as coordinate and input values are the output da values
def test_param_dict2da():
    pass


# sonar_model: EK or AZFP
# input dict:
#   - contain extra param: should come out with only those defined in CAL_PARAMS
#   - contain missing param: missing ones (wrt CAL_PARAMS) should be empty
# input params:
#   - scalar: no change
#   - xr.DataArray without channel coorindate: fail with value error
#   - xr.DataArray with channel cooridinate not identical to argin channel: fail with value error
#       - argin channel: list or xr.DataArray
#   - argin channel is not a list nor an xr.DataArray: fail with value error
def test_check_user_cal_dict():
    pass


# da_param: None
#   - alternative is const: output is xr.DataArray with all const
#   - alternative is xr.DataArray: output selected with the right channel
# THIS CASE DOES NOT EXIST: da_param: xr.DataArray without freq-dependent values/coorindates
#   - output selected with right channels
# da_param: xr.DataArray with freq-dependent values/coordinates
#   - output interpolated with the right values
@pytest.mark.parametrize(
    ("da_param", "alternative", "da_output"),
    [
        (None, 1, xr.DataArray([1, 1], dims=["channel"], coords={"channel": ["chA", "chB"]})),
        (
            None,
            xr.DataArray([1, 1, 2], dims=["channel"], coords={"channel": ["chA", "chB", "chC"]}),
            xr.DataArray([1, 1], dims=["channel"], coords={"channel": ["chA", "chB"]})
        ),
        (
            xr.DataArray(
                np.array([[1, 2, 3, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, 4, 5, 6],
                          [np.nan, 2, 3, 4, np.nan, np.nan]]),
                dims=["cal_channel_id", "cal_frequency"],
                coords={"cal_channel_id": ["chA", "chB", "chC"],
                        "cal_frequency": [10, 20, 30, 40, 50, 60]},
            ),
            None,
            xr.DataArray([2.5, 5.5], dims=["channel"], coords={"channel": ["chA", "chB"]}),
        ),
    ],
    ids=[
        "in_None_alt_const",
        "in_None_alt_da",
        "in_da_out_interp",
    ]
)
def test_get_interp_da(freq_center, da_param, alternative, da_output):
    da_interp = _get_interp_da(da_param, freq_center, alternative)
    assert da_interp.identical(da_output)
