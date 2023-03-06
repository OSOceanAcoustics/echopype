import pytest

import numpy as np
import xarray as xr

from echopype.calibrate.cal_params import CAL_PARAMS, param2da, sanitize_user_cal_dict, _get_interp_da


@pytest.fixture
def freq_center():
    return xr.DataArray([[25, 55]], dims=["ping_time", "channel"], coords={"channel": ["chA", "chB"], "ping_time": [1]})


@pytest.mark.parametrize(
    ("p_val", "channel", "da_output"),
    [
        # input p_val a scalar, input channel a list
        (1, ["chA", "chB"], xr.DataArray([1, 1], dims=["channel"], coords={"channel": ["chA", "chB"]})),
        # input p_val a list, input channel an xr.DataArray
        (
            [1, 2],
            xr.DataArray(["chA", "chB"], dims=["channel"], coords={"channel": ["chA", "chB"]}),
            xr.DataArray([1, 2], dims=["channel"], coords={"channel": ["chA", "chB"]})
        ),
        # input p_val a list with the wrong length: this should fail
        pytest.param(
            [1, 2, 3], ["chA", "chB"], None,
            marks=pytest.mark.xfail(strict=True, reason="Fail since lengths of p_val and channel are not identical")
        ),
    ],
    ids=[
        "in_p_val_scalar_channel_list",
        "in_p_val_list_channel_xrda",
        "in_p_val_list_wrong_length",
    ]
)
def test_param2da(p_val, channel, da_output):
    da_assembled = param2da(p_val, channel)
    assert da_assembled.identical(da_output)


@pytest.mark.parametrize(
    ("sonar_type", "user_dict", "channel", "out_dict"),
    [
        # sonar_type only allows EK or AZFP
        pytest.param(
            "XYZ", None, None, None,
            marks=pytest.mark.xfail(strict=True, reason="Fail since sonar_type is not 'EK' nor 'AZFP'")
        ),
        # input channel
        #   - is not a list nor an xr.DataArray: fail with value error
        pytest.param(
            "EK", 1, None, None,
            marks=pytest.mark.xfail(strict=True, reason="Fail since channel has to be either a list or an xr.DataArray"),
        ),
        # input param dict
        #   - contains extra param: should come out with only those defined in CAL_PARAMS
        #   - contains missing param: missing ones (wrt CAL_PARAMS) should be empty
        pytest.param("EK", {"extra_param": 1}, ["chA", "chB"], dict.fromkeys(CAL_PARAMS["EK"])),
        # input param:
        #   - is xr.DataArray without channel coorindate: fail with value error
        pytest.param(
            "EK",
            {"sa_correction": xr.DataArray([1, 1], dims=["some_coords"], coords={"some_coords": ["A", "B"]})},
            ["chA", "chB"], None,
            marks=pytest.mark.xfail(strict=True, reason="input sa_correction does not contain a 'channel' coordinate"),
        ),
        # input individual param:
        #   - with channel cooridinate but not identical to argin channel: fail with value error
        pytest.param(
            "EK",
            {"sa_correction": xr.DataArray([1, 1], dims=["channel"], coords={"channel": ["chA", "B"]})},
            ["chA", "chB"], None,
            marks=pytest.mark.xfail(strict=True,
                reason="input sa_correction contains a 'channel' coordinate but it is not identical with input channel"),
        ),
        # input individual param:
        #   - with channel cooridinate identical to argin channel: should pass
        pytest.param(
            "EK",
            {"sa_correction": xr.DataArray([1, 1], dims=["channel"], coords={"channel": ["chA", "chB"]})},
            ["chA", "chB"],
            dict(dict.fromkeys(CAL_PARAMS["EK"]),
                **{"sa_correction": xr.DataArray([1, 1], dims=["channel"], coords={"channel": ["chA", "chB"]})}),
        ),
        # input individual param:
        #   - a scalar needing to be organized to xr.DataArray at output via param2da: should pass
        pytest.param(
            "EK",
            {"sa_correction": 1},
            ["chA", "chB"],
            dict(dict.fromkeys(CAL_PARAMS["EK"]),
                **{"sa_correction": xr.DataArray([1, 1], dims=["channel"], coords={"channel": ["chA", "chB"]})}),
        ),
        # input individual param:
        #   - a list needing to be organized to xr.DataArray at output via param2da: should pass
        pytest.param(
            "EK",
            {"sa_correction": [1, 2]},
            ["chA", "chB"],
            dict(dict.fromkeys(CAL_PARAMS["EK"]),
                **{"sa_correction": xr.DataArray([1, 2], dims=["channel"], coords={"channel": ["chA", "chB"]})}),
        ),    
        # input individual param:
        #   - a list with wrong length (ie not identical to channel): fail with value error
        pytest.param(
            "EK", {"sa_correction": [1, 2, 3]}, ["chA", "chB"], None,
            marks=pytest.mark.xfail(strict=True,
                reason="input sa_correction contains a list of wrong length that does not match that of channel"),
        ),
    ],
    ids=[
        "sonar_type_invalid",
        "channel_invalid",
        "in_extra_param",
        "in_da_no_channel_coord",
        "in_da_channel_not_identical",
        "in_da_channel_identical",
        "in_scalar",
        "in_list",
        "in_list_wrong_length",
    ],
)
def test_sanitize_user_cal_dict(sonar_type, user_dict, channel, out_dict):
    sanitized_dict = sanitize_user_cal_dict(sonar_type, user_dict, channel)
    assert isinstance(sanitized_dict, dict)
    assert len(sanitized_dict) == len(out_dict)
    for p_name, p_val in sanitized_dict.items():
        if isinstance(p_val, xr.DataArray):
            assert p_val.identical(out_dict[p_name])
        else:
            assert p_val == out_dict[p_name]


@pytest.mark.parametrize(
    ("da_param", "alternative", "da_output"),
    [
        # da_param: alternative is const: output is xr.DataArray with all const
        (
            None,
            1,
            xr.DataArray([[1], [1]], dims=["channel", "ping_time"], coords={"channel": ["chA", "chB"], "ping_time": [1]})
        ),
        # da_param: alternative is xr.DataArray: output selected with the right channel
        (
            None,
            xr.DataArray([1, 1, 2], dims=["channel"], coords={"channel": ["chA", "chB", "chC"]}),
            xr.DataArray([[1], [1]], dims=["channel", "ping_time"], coords={"channel": ["chA", "chB"], "ping_time": [1]})
        ),
        # da_param: xr.DataArray with freq-dependent values/coordinates
        #   - output should be interpolated with the right values
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
            xr.DataArray([[2.5], [5.5]], dims=["channel", "ping_time"], coords={"ping_time": [1], "channel": ["chA", "chB"]}),
        ),
        # da_param: xr.DataArray with only one channel having freq-dependent values/coordinates
        #   - that single channel should be interpolated with the right value
        #   - other channels will use alternative
        #   - alternative could be of the following form:
        #       - scalar
        (
            xr.DataArray(
                np.array([[np.nan, np.nan, np.nan, 4, 5, 6]]),
                dims=["cal_channel_id", "cal_frequency"],
                coords={"cal_channel_id": ["chB"],
                        "cal_frequency": [10, 20, 30, 40, 50, 60]},
            ),
            75,
            xr.DataArray(
                [[75], [5.5]],
                dims=["channel", "ping_time"],
                coords={"ping_time": [1], "channel": ["chA", "chB"]}
            ),
        ),
        #       - xr.DataArray with coordinates channel, ping_time, beam
        (
            xr.DataArray(
                np.array([[np.nan, np.nan, np.nan, 4, 5, 6]]),
                dims=["cal_channel_id", "cal_frequency"],
                coords={"cal_channel_id": ["chB"],
                        "cal_frequency": [10, 20, 30, 40, 50, 60]},
            ),
            xr.DataArray(
                np.array([[[100, 200]]] * 4),
                dims=["beam", "ping_time", "channel"],
                coords={"beam": [0, 1, 2, 3], "ping_time": [1], "channel": ["chA", "chB"]},
            ),
            xr.DataArray(
                [[100], [5.5]],
                dims=["channel", "ping_time"],
                coords={"ping_time": [1], "channel": ["chA", "chB"]}
            ),
        ),
        #       - xr.DataArray with coordinates channel, ping_time
        (
            xr.DataArray(
                np.array([[np.nan, np.nan, np.nan, 4, 5, 6]]),
                dims=["cal_channel_id", "cal_frequency"],
                coords={"cal_channel_id": ["chB"],
                        "cal_frequency": [10, 20, 30, 40, 50, 60]},
            ),
            xr.DataArray(
                np.array([[100], [200]]),
                dims=["channel", "ping_time"],
                coords={"ping_time": [1], "channel": ["chA", "chB"]},
            ),
            xr.DataArray(
                [[100], [5.5]],
                dims=["channel", "ping_time"],
                coords={"ping_time": [1], "channel": ["chA", "chB"]}
            ),
        # TODO: cases where freq_center does not have the ping_time dimension
        #       this is the case for CW data since freq_center = beam["frequency_nominal"]
        #       this was caught by the file in test_compute_Sv_ek80_CW_complex()
        # TODO: cases where freq_center contains only a single frequency
        #       in this case had to use freq_center.sel(channel=ch_id).size because
        #       len(freq_center.sel(channel=ch_id)) is an invalid statement
        #       this was caught by the file in test_compute_Sv_ek80_CW_power_BB_complex()
        ),
    ],
    ids=[
        "in_None_alt_const",
        "in_None_alt_da",
        "in_da_all_channel_out_interp",
        "in_da_some_channel_alt_scalar",
        "in_da_some_channel_alt_da3coords",  # channel, ping_time, beam:
        "in_da_some_channel_alt_da2coords",  # channel, ping_time
    ]
)
def test_get_interp_da(freq_center, da_param, alternative, da_output):
    da_interp = _get_interp_da(da_param, freq_center, alternative)
    assert da_interp.identical(da_output)
