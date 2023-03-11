import pytest

import numpy as np
import xarray as xr

from echopype.calibrate.cal_params import (
    CAL_PARAMS, param2da, sanitize_user_cal_dict, _get_interp_da,
    get_cal_params_AZFP, get_cal_params_EK, get_vend_cal_params_power
)


@pytest.fixture
def freq_center():
    return xr.DataArray(
        [[25, 55]],
        dims=["ping_time", "channel"],
        coords={"channel": ["chA", "chB"], "ping_time": [1]}
    )


@pytest.fixture
def vend_AZFP():
    """
    A mock AZFP Vendor_specific group for cal testing.
    """
    da = xr.DataArray([10, 20], dims=["channel"], coords={"channel": ["chA", "chB"]})
    vend = xr.Dataset()
    for p_name in CAL_PARAMS["AZFP"]:
        if p_name != "equivalent_beam_angle":
            da.name = p_name
            vend[p_name] = da
    return vend


@pytest.fixture
def beam_AZFP():
    """
    A mock AZFP Sonar/Beam_group1 group for cal testing.
    """
    beam = xr.Dataset()
    beam["equivalent_beam_angle"] = xr.DataArray(
        [[[10, 20]]],
        dims=["ping_time", "beam", "channel"],
        coords={"channel": ["chA", "chB"], "ping_time": [1], "beam": [1]},
    )
    return beam.transpose("channel", "ping_time", "beam")


@pytest.fixture
def vend_EK():
    """
    A mock EK Sonar/Beam_groupX group for cal testing.
    """
    vend = xr.Dataset()
    for p_name in ["sa_correction", "gain_correction"]:
        vend[p_name] = xr.DataArray(
            np.array([[10, 20, 30, 40], [110, 120, 130, 140]]),
            dims=["channel", "pulse_length_bin"],
            coords={"channel": ["chA", "chB"], "pulse_length_bin": [0, 1, 2, 3]},
        )
    vend["pulse_length"] = xr.DataArray(
            np.array([[64, 128, 256, 512], [128, 256, 512, 1024]]),
            coords={"channel": vend["channel"], "pulse_length_bin": vend["pulse_length_bin"]}
    )
    vend["impedance_receive"] = xr.DataArray(
        [1000, 2000], coords={"channel": vend["channel"]}
    )
    vend["transceiver_type"] = xr.DataArray(
        ["WBT", "WBT"], coords={"channel": vend["channel"]}
    )
    return vend


@pytest.fixture
def beam_EK():
    """
    A mock EK Sonar/Beam_groupX group for cal testing.
    """
    beam = xr.Dataset()
    for p_name in [
        "equivalent_beam_angle",
        "angle_offset_alongship", "angle_offset_athwartship",
        "angle_sensitivity_alongship", "angle_sensitivity_athwartship",
        "beamwidth_twoway_alongship", "beamwidth_twoway_athwartship"
    ]:
        beam[p_name] = xr.DataArray(
            np.array([[[123, 123, 123, 123], [456, 456, 456, 456]]]),
            dims=["ping_time", "channel", "beam"],
            coords={"channel": ["chA", "chB"], "ping_time": [1], "beam": [1, 2, 3, 4]},
        )
    beam["frequency_nominal"] = xr.DataArray([25, 55], dims=["channel"], coords={"channel": ["chA", "chB"]})
    return beam.transpose("channel", "ping_time", "beam")


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
            "EK80", 1, None, None,
            marks=pytest.mark.xfail(strict=True, reason="Fail since channel has to be either a list or an xr.DataArray"),
        ),
        # TODO: input channel has different order than those in the inarg channel
        # input param dict
        #   - contains extra param: should come out with only those defined in CAL_PARAMS
        #   - contains missing param: missing ones (wrt CAL_PARAMS) should be empty
        pytest.param("EK80", {"extra_param": 1}, ["chA", "chB"], dict.fromkeys(CAL_PARAMS["EK80"])),
        # input param:
        #   - is xr.DataArray without channel coorindate: fail with value error
        pytest.param(
            "EK80",
            {"sa_correction": xr.DataArray([1, 1], dims=["some_coords"], coords={"some_coords": ["A", "B"]})},
            ["chA", "chB"], None,
            marks=pytest.mark.xfail(strict=True, reason="input sa_correction does not contain a 'channel' coordinate"),
        ),
        # input individual param:
        #   - with channel cooridinate but not identical to argin channel: fail with value error
        pytest.param(
            "EK80",
            {"sa_correction": xr.DataArray([1, 1], dims=["channel"], coords={"channel": ["chA", "B"]})},
            ["chA", "chB"], None,
            marks=pytest.mark.xfail(strict=True,
                reason="input sa_correction contains a 'channel' coordinate but it is not identical with input channel"),
        ),
        # input individual param:
        #   - with channel cooridinate identical to argin channel: should pass
        pytest.param(
            "EK80",
            {"sa_correction": xr.DataArray([1, 1], dims=["channel"], coords={"channel": ["chA", "chB"]})},
            ["chA", "chB"],
            dict(dict.fromkeys(CAL_PARAMS["EK80"]),
                **{"sa_correction": xr.DataArray([1, 1], dims=["channel"], coords={"channel": ["chA", "chB"]})}),
        ),
        # input individual param:
        #   - a scalar needing to be organized to xr.DataArray at output via param2da: should pass
        pytest.param(
            "EK80",
            {"sa_correction": 1},
            ["chA", "chB"],
            dict(dict.fromkeys(CAL_PARAMS["EK80"]),
                **{"sa_correction": xr.DataArray([1, 1], dims=["channel"], coords={"channel": ["chA", "chB"]})}),
        ),
        # input individual param:
        #   - a list needing to be organized to xr.DataArray at output via param2da: should pass
        pytest.param(
            "EK80",
            {"sa_correction": [1, 2]},
            ["chA", "chB"],
            dict(dict.fromkeys(CAL_PARAMS["EK80"]),
                **{"sa_correction": xr.DataArray([1, 2], dims=["channel"], coords={"channel": ["chA", "chB"]})}),
        ),    
        # input individual param:
        #   - a list with wrong length (ie not identical to channel): fail with value error
        pytest.param(
            "EK80", {"sa_correction": [1, 2, 3]}, ["chA", "chB"], None,
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
        "in_da_some_channel_alt_da3coords",  # channel, ping_time, beam
        "in_da_some_channel_alt_da2coords",  # channel, ping_time
    ]
)
def test_get_interp_da(freq_center, da_param, alternative, da_output):
    da_interp = _get_interp_da(da_param, freq_center, alternative)
    assert da_interp.identical(da_output)


@pytest.mark.parametrize(
    ("user_dict", "out_dict"),
    [
        # input param is a scalar
        (
            {"EL": 1, "equivalent_beam_angle": 2},
            dict(
                {p_name: xr.DataArray([10, 20], dims=["channel"], coords={"channel": ["chA", "chB"]}) for p_name in CAL_PARAMS["AZFP"]},
                **{
                    "EL": xr.DataArray([1, 1], dims=["channel"], coords={"channel": ["chA", "chB"]}),
                    "equivalent_beam_angle": xr.DataArray([2, 2], dims=["channel"], coords={"channel": ["chA", "chB"]}),
                }
            ),
        ),
        # input param is a list
        (
            {"EL": [1, 2], "equivalent_beam_angle": [3, 4]},
            dict(
                {p_name: xr.DataArray([10, 20], dims=["channel"], coords={"channel": ["chA", "chB"]}) for p_name in CAL_PARAMS["AZFP"]},
                **{
                    "EL": xr.DataArray([1, 2], dims=["channel"], coords={"channel": ["chA", "chB"]}),
                    "equivalent_beam_angle": xr.DataArray([3, 4], dims=["channel"], coords={"channel": ["chA", "chB"]}),
                }
            ),
        ),
        # input param is a list of wrong length: this should fail
        pytest.param(
            {"EL": [1, 2, 3], "equivalent_beam_angle": [3, 4]}, None,
            marks=pytest.mark.xfail(strict=True, reason="Fail since lengths of input list and channel are not identical"),
        ),
        # input param is an xr.DataArray with coordinate 'channel'
        (
            {
                "EL": xr.DataArray([1, 2], dims=["channel"], coords={"channel": ["chA", "chB"]}),
                "equivalent_beam_angle": xr.DataArray([3, 4], dims=["channel"], coords={"channel": ["chA", "chB"]}),
            },
            dict(
                {p_name: xr.DataArray([10, 20], dims=["channel"], coords={"channel": ["chA", "chB"]}) for p_name in CAL_PARAMS["AZFP"]},
                **{
                    "EL": xr.DataArray([1, 2], dims=["channel"], coords={"channel": ["chA", "chB"]}),
                    "equivalent_beam_angle": xr.DataArray([3, 4], dims=["channel"], coords={"channel": ["chA", "chB"]}),
                }
            ),
        ),
        # input param is an xr.DataArray with coordinate 'channel' but wrong length: this should fail
        pytest.param(
            {
                "EL": xr.DataArray([1, 2, 3], dims=["channel"], coords={"channel": ["chA", "chB", "chC"]}),
                "equivalent_beam_angle": xr.DataArray([3, 4, 5], dims=["channel"], coords={"channel": ["chA", "chB", "chC"]}),
            }, None,
            marks=pytest.mark.xfail(strict=True, reason="Fail since lengths of input data array channel and data channel are not identical"),
        ),
    ],
    ids=[
        "in_scalar",
        "in_list",
        "in_list_wrong_length",
        "in_da",
        "in_da_wrong_length",
    ]
)
def test_get_cal_params_AZFP(beam_AZFP, vend_AZFP, user_dict, out_dict):
    cal_dict = get_cal_params_AZFP(beam=beam_AZFP, vend=vend_AZFP, user_dict=user_dict)
    for p_name, p_val in cal_dict.items():
        # remove name for all da
        p_val.name = None
        out_val = out_dict[p_name]
        out_val.name = None
        assert p_val.identical(out_val)


# The test above 'test_get_cal_params_AZFP' covers the cases where user input param
# is one of the following: a scalar, list, and xr.DataArray of coords/dims ('channel')
# Here we only test for the following new cases:
#   - where all params are input by user
#   - input xr.DataArray has coords/dims (cal_channel_id, cal_frequency)
@pytest.mark.parametrize(
    ("user_dict", "out_dict"),
    [
        # input xr.DataArray has coords/dims (cal_channel_id, cal_frequency)
        (
            {
                "gain_correction": xr.DataArray(
                    np.array([[1, 2, 3, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, 4, 5, 6]]),
                    dims=["cal_channel_id", "cal_frequency"],
                    coords={"cal_channel_id": ["chA", "chB"],
                            "cal_frequency": [10, 20, 30, 40, 50, 60]},
                ),
                # add sa_correction here to bypass things going into get_vend_cal_params_power
                "sa_correction": xr.DataArray(
                    np.array([111, 222]), dims=["channel"], coords={"channel": ["chA", "chB"]},
                )
            },
            dict(
                {
                    p_name: xr.DataArray(
                        [[123], [456]],
                        dims=["channel", "ping_time"],
                        coords={"channel": ["chA", "chB"], "ping_time": [1]},
                    )
                    for p_name in CAL_PARAMS["EK80"]
                },
                **{
                    "gain_correction": xr.DataArray(
                        [[2.5], [5.5]],
                        dims=["channel", "ping_time"],
                        coords={"ping_time": [1], "channel": ["chA", "chB"]},
                    ),
                    "sa_correction": xr.DataArray(
                        np.array([111, 222]), dims=["channel"],
                        coords={"channel": ["chA", "chB"]}
                    ),
                    "impedance_transmit": xr.DataArray(
                        np.array([[75], [75]]), dims=["channel", "ping_time"],
                        coords={"channel": ["chA", "chB"], "ping_time": [1]}
                    ),
                    "impedance_receive": xr.DataArray(
                        np.array([1000, 2000]), dims=["channel"],
                        coords={"channel": ["chA", "chB"]}
                    ),
                    "receiver_sampling_frequency": xr.DataArray(
                        np.array([1500000, 1500000]), dims=["channel"],
                        coords={"channel": ["chA", "chB"]}
                    ),
                },
            ),
        ),
        pytest.param(
            {
                "gain_correction": xr.DataArray(
                    np.array([[1, 2, 3, np.nan], [np.nan, 4, 5, 6], [np.nan, 2, 3, np.nan]]),
                    dims=["cal_channel_id", "cal_frequency"],
                    coords={"cal_channel_id": ["chA", "chB", "chC"],
                            "cal_frequency": [10, 20, 30, 40]},
                ),
            },
            None,
            marks=pytest.mark.xfail(strict=True, reason="Fail since cal_channel_id in input param does not match channel of data"),
        ),
    ],
    ids=[
        "in_da_freq_dep",
        "in_da_freq_dep_channel_mismatch",
    ]
)
def test_get_cal_params_EK80_BB(beam_EK, vend_EK, freq_center, user_dict, out_dict):
    cal_dict = get_cal_params_EK(
        waveform_mode="BB", freq_center=freq_center, beam=beam_EK, vend=vend_EK, user_dict=user_dict
    )
    for p_name, p_val in cal_dict.items():
        # remove name for all da
        p_val.name = None
        out_val = out_dict[p_name]
        out_val.name = None
        assert p_val.identical(out_dict[p_name])


@pytest.mark.parametrize(
    ("user_dict", "out_dict"),
    [
        # cal_params should not contain:
        #   impedance_transmit, impedance_receive, receiver_sampling_frequency
        (
            {
                # add sa_correction here to bypass things going into get_vend_cal_params_power
                "gain_correction": xr.DataArray(
                    [555, 777], dims=["channel"], coords={"channel": ["chA", "chB"]},
                ),
                # add sa_correction here to bypass things going into get_vend_cal_params_power
                "sa_correction": xr.DataArray(
                    [111, 222], dims=["channel"], coords={"channel": ["chA", "chB"]},
                )
            },
            dict(
                {
                    p_name: xr.DataArray(
                        [[123], [456]],
                        dims=["channel", "ping_time"],
                        coords={"channel": ["chA", "chB"], "ping_time": [1]},
                    )
                    for p_name in [
                        "sa_correction", "gain_correction", "equivalent_beam_angle",
                        "angle_offset_alongship", "angle_offset_athwartship",
                        "angle_sensitivity_alongship", "angle_sensitivity_athwartship",
                        "beamwidth_alongship", "beamwidth_athwartship",
                    ]
                },
                **{
                    "gain_correction": xr.DataArray(
                        [555, 777], dims=["channel"], coords={"channel": ["chA", "chB"]},
                    ),
                    "sa_correction": xr.DataArray(
                        [111, 222], dims=["channel"], coords={"channel": ["chA", "chB"]}
                    ),
                },
            ),
        ),
    ],
    ids=[
        "in_da",
    ]
)
def test_get_cal_params_EK60(beam_EK, vend_EK, freq_center, user_dict, out_dict):
    # Remove some variables from Vendor group to mimic EK60 data
    vend_EK = vend_EK.drop("impedance_receive").drop("transceiver_type")
    cal_dict = get_cal_params_EK(
        waveform_mode="CW", freq_center=freq_center,
        beam=beam_EK, vend=vend_EK,
        user_dict=user_dict, sonar_type="EK60"
    )
    for p_name, p_val in cal_dict.items():
        # remove name for all da
        p_val.name = None
        out_val = out_dict[p_name]
        out_val.name = None
        assert p_val.identical(out_dict[p_name])


@pytest.mark.parametrize(
    ("param", "beam", "da_output"),
    [
        # no NaN entry in transmit_duration_nominal
        (
            "sa_correction",
            xr.DataArray(
                np.array([[64, 256, 128, 512], [512, 1024, 256, 128]]).T,
                dims=["ping_time", "channel"],
                coords={"ping_time": [1, 2, 3, 4], "channel": ["chA", "chB"]},
                name="transmit_duration_nominal",
            ).to_dataset(),
            xr.DataArray(
                np.array([[10, 30, 20, 40], [130, 140, 120, 110]]).T,
                dims=["ping_time", "channel"],
                coords={"ping_time": [1, 2, 3, 4], "channel": ["chA", "chB"]},
                name="sa_correction",
            ),
        ),
        # with NaN entry in transmit_duration_nominal
        (
            "sa_correction",
            xr.DataArray(
                np.array([[64, np.nan, 128, 512], [512, 1024, 256, np.nan]]).T,
                dims=["ping_time", "channel"],
                coords={"ping_time": [1, 2, 3, 4], "channel": ["chA", "chB"]},
                name="transmit_duration_nominal",
            ).to_dataset(),
            xr.DataArray(
                np.array([[10, np.nan, 20, 40], [130, 140, 120, np.nan]]).T,
                dims=["ping_time", "channel"],
                coords={"ping_time": [1, 2, 3, 4], "channel": ["chA", "chB"]},
                name="sa_correction",
            ),
        ),
    ],
    ids=[
        "in_no_nan",
        "in_with_nan",
    ]
)
def test_get_vend_cal_params_power(vend_EK, beam, param, da_output):
    da_param = get_vend_cal_params_power(beam, vend_EK, param)
    assert da_param.identical(da_output)
