import pytest

import numpy as np
import xarray as xr

import echopype as ep
from echopype.calibrate.env_params import (
    harmonize_env_param_time,
    sanitize_user_env_dict,
    ENV_PARAMS,
    get_env_params_AZFP
)


@pytest.fixture
def azfp_path(test_path):
    return test_path['AZFP']


@pytest.fixture
def ek60_path(test_path):
    return test_path['EK60']


@pytest.fixture
def ek80_cal_path(test_path):
    return test_path['EK80_CAL']


def test_harmonize_env_param_time():
    # Scalar
    p = 10.05
    assert harmonize_env_param_time(p=p) == 10.05

    # time1 length=1, should return length=1 numpy array
    p = xr.DataArray(
        data=[1],
        coords={
            "time1": np.array(["2017-06-20T01:00:00"], dtype="datetime64[ns]")
        },
        dims=["time1"]
    )
    assert harmonize_env_param_time(p=p) == 1

    # time1 length>1, interpolate to tareget ping_time
    p = xr.DataArray(
        data=np.array([0, 1]),
        coords={
            "time1": np.arange("2017-06-20T01:00:00", "2017-06-20T01:00:31", np.timedelta64(30, "s"), dtype="datetime64[ns]")
        },
        dims=["time1"]
    )
    # ping_time target is identical to time1
    ping_time_target = p["time1"].rename({"time1": "ping_time"})
    p_new = harmonize_env_param_time(p=p, ping_time=ping_time_target)
    assert (p_new["ping_time"] == ping_time_target).all()
    assert (p_new.data == p.data).all()
    # ping_time target requires actual interpolation
    ping_time_target = xr.DataArray(
        data=[1],
        coords={
            "ping_time": np.array(["2017-06-20T01:00:15"], dtype="datetime64[ns]")
        },
        dims=["ping_time"]
    )
    p_new = harmonize_env_param_time(p=p, ping_time=ping_time_target["ping_time"])
    assert p_new["ping_time"] == ping_time_target["ping_time"]
    assert p_new.data == 0.5


@pytest.mark.parametrize(
    ("user_dict", "channel", "out_dict"),
    [
        # dict all scalars, channel a list, output should be all scalars
        #   - this behavior departs from sanitize_user_cal_dict, which will make scalars into xr.DataArray
        (
            {"temperature": 10, "salinity": 20},
            ["chA", "chB"],
            dict(
                dict.fromkeys(ENV_PARAMS), **{"temperature": 10, "salinity": 20}
            )
        ),
        # dict has xr.DataArray, channel a list with matching values with those in dict
        (
            {"temperature": 10, "sound_absorption": xr.DataArray([10, 20], coords={"channel": ["chA", "chB"]})},
            ["chA", "chB"],
            dict(
                dict.fromkeys(ENV_PARAMS),
                **{"temperature": 10, "sound_absorption": xr.DataArray([10, 20], coords={"channel": ["chA", "chB"]})}
            )
        ),
        # dict has xr.DataArray, channel a list with non-matching values with those in dict: XFAIL
        # dict has xr.DataArray, channel a xr.DataArray
        # dict has sound_absorption as a scalar: XFAIL
    ],
    ids=[
        "in_scalar_channel_list_out_scalar",
        "in_da_channel_list_out_da",
    ]
)
def test_sanitize_user_env_dict(user_dict, channel, out_dict):
    """
    Only test the case where the input sound_absorption is not an xr.DataArray nor a list,
    since other cases are tested under test_cal_params::test_sanitize_user_cal_dict
    """
    env_dict = sanitize_user_env_dict(user_dict, channel)
    for p, v in env_dict.items():
        if isinstance(v, xr.DataArray):
            assert v.identical(out_dict[p])
        else:
            assert v == out_dict[p]


@pytest.mark.parametrize(
    ("env_ext", "out_dict"),
    [
        # pH should not exist in the output Sv dataset, formula sources should both be AZFP
        (
            {"temperature": 10, "salinity": 20, "pressure": 100, "pH": 8.1},
            dict(
                dict.fromkeys(ENV_PARAMS), **{"temperature": 10, "salinity": 20, "pressure": 100}
            )
        ),
        # not including salinity or pressure: XFAIL
        pytest.param(
            {"temperature": 10, "pressure": 100, "pH": 8.1}, None,
            marks=pytest.mark.xfail(strict=True, reason="Fail since cal_channel_id in input param does not match channel of data"),
        ),
    ],
    ids=[
        "default",
        "no_salinity",
    ]
)
def test_get_env_params_AZFP(azfp_path, env_ext, out_dict):
    azfp_01a_path = str(azfp_path.joinpath('17082117.01A'))
    azfp_xml_path = str(azfp_path.joinpath('17041823.XML'))
    ed = ep.open_raw(azfp_01a_path, sonar_model='AZFP', xml_path=azfp_xml_path)

    env_dict = get_env_params_AZFP(echodata=ed, user_dict=env_ext)

    out_dict = dict(
        out_dict,
        **{
            "sound_speed": ep.utils.uwa.calc_sound_speed(
                temperature=env_dict["temperature"],
                salinity=env_dict["salinity"],
                pressure=env_dict["pressure"],
                formula_source="AZFP"
            ),
            "sound_absorption": ep.utils.uwa.calc_absorption(
                frequency=ed["Sonar/Beam_group1"]["frequency_nominal"],
                temperature=env_dict["temperature"],
                salinity=env_dict["salinity"],
                pressure=env_dict["pressure"],
                formula_source="AZFP",
            ),
            "formula_sound_speed": "AZFP",
            "formula_absorption": "AZFP",
        }
    )

    assert "pH" not in env_dict
    assert env_dict["formula_absorption"] == "AZFP"
    assert env_dict["formula_sound_speed"] == "AZFP"

    for p, v in env_dict.items():
        if isinstance(v, xr.DataArray):
            assert v.identical(out_dict[p])
        else:
            assert v == out_dict[p]


# TODO: unit test for get_env_params_AZFP/EK60/EK80
# - make sure the combination is correctly passed in
# - make sure the sound speed and absorption are correctly calculated
