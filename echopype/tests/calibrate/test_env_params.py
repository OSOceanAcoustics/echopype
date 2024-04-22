import pytest

import dask.array
import numpy as np
import xarray as xr
import pandas as pd

import echopype as ep
from echopype.calibrate.env_params import (
    harmonize_env_param_time,
    sanitize_user_env_dict,
    ENV_PARAMS,
    get_env_params_AZFP,
    get_env_params_EK,
)


@pytest.fixture
def azfp_path(test_path):
    return test_path["AZFP"]


@pytest.fixture
def ek60_path(test_path):
    return test_path["EK60"]


@pytest.fixture
def ek80_cal_path(test_path):
    return test_path["EK80_CAL"]


def test_harmonize_env_param_time():
    # Scalar
    p = 10.05
    assert harmonize_env_param_time(p=p) == 10.05

    # time1 length=1, should return length=1 numpy array
    p = xr.DataArray(
        data=[2],
        coords={"time1": np.array(["2017-06-20T01:00:00"], dtype="datetime64[ns]")},
        dims=["time1"],
    )
    assert harmonize_env_param_time(p=p) == 2

    # time1 length>1
    p = xr.DataArray(
        data=np.array([0, 1, 2]),
        coords={
            "time1": np.arange(
                "2017-06-20T01:00:00",
                "2017-06-20T01:01:30",
                np.timedelta64(30, "s"),
                dtype="datetime64[ns]",
            )
        },
        dims=["time1"],
    )
    # Ensure that ping_time cannot be None when time1 length > 1
    with pytest.raises(ValueError):
        harmonize_env_param_time(p=p, ping_time=None)

    # ping_time target is identical to time1
    ping_time_target = p["time1"].rename({"time1": "ping_time"})
    p_new = harmonize_env_param_time(p=p, ping_time=ping_time_target)
    assert (p_new["ping_time"] == ping_time_target).all()
    assert (p_new.data == p.data).all()

    # ping_time target requires actual interpolation
    ping_time_target = xr.DataArray(
        data=[1, 2],
        coords={
            "ping_time": np.array(
                ["2017-06-20T01:00:15", "2017-06-20T01:00:30"], dtype="datetime64[ns]"
            )
        },
        dims=["ping_time"],
    )

    ping_time_target = xr.DataArray(
        data=[1],
        coords={
            "ping_time": np.array(
                [
                    "2017-06-20T01:00:15",
                ],
                dtype="datetime64[ns]",
            )
        },
        dims=["ping_time"],
    )

    p_new = harmonize_env_param_time(p=p, ping_time=ping_time_target["ping_time"])
    assert (p_new["ping_time"] == ping_time_target["ping_time"]).all()
    assert p_new.data == 0.5

    # create a larger p with dask arrays under the hood
    time1 = np.arange(
        "2017-06-20T01:00:00",
        "2017-06-22T01:00:31",
        np.timedelta64(30, "s"),
        dtype="datetime64[ns]",
    )
    p = xr.DataArray(
        data=np.arange(len(time1)),
        coords={"time1": time1},
        dims=["time1"],
    ).chunk({"time1": 1000})

    # ping_time target is identical to time1
    ping_time_target = p["time1"].rename({"time1": "ping_time"})
    p_new = harmonize_env_param_time(p=p, ping_time=ping_time_target)
    assert (p_new["ping_time"] == ping_time_target).all()
    assert isinstance(p_new.data, dask.array.Array)
    # np.array_equal() computes dask array under the hood
    assert np.array_equal(p_new.data, p.data)

    # ping_time target requires actual interpolation
    ping_time_target = xr.DataArray(
        data=[1, 2],
        coords={
            "ping_time": np.array(
                ["2017-06-20T01:00:15", "2017-06-21T01:00:15"], dtype="datetime64[ns]"
            )
        },
        dims=["ping_time"],
    )

    p_new = harmonize_env_param_time(p=p, ping_time=ping_time_target["ping_time"])
    assert np.array_equal(p_new["ping_time"], ping_time_target["ping_time"])

    assert isinstance(p_new.data, dask.array.Array)
    # .all computes dask array under the hood
    assert (p_new.data == [0.5, 2880.5]).all()

@pytest.mark.unit
def test_harmonize_env_param_time_only_one_non_NaN_along_time1():
    # Create data array with time1 dimension with only 1 non-NaN value
    data = np.array([1, np.nan, np.nan])
    time = pd.date_range(start='2024-01-01', periods=3, freq='D')
    da = xr.DataArray(data, dims='time1', coords={'time1': time})

    # Check that output da has only 1 value and no dimension
    output_da = harmonize_env_param_time(da, None)
    assert output_da == 1, "Output data array should just be 1"
    assert 'time1' not in output_da.dims, "```harmonize_env_param_time``` should have dropped 'time1' dimension"

@pytest.mark.parametrize(
    ("user_dict", "channel", "out_dict"),
    [
        # dict all scalars, channel a list, output should be all scalars
        #   - this behavior departs from sanitize_user_cal_dict, which will make scalars into xr.DataArray
        (
            {"temperature": 10, "salinity": 20},
            ["chA", "chB"],
            dict(dict.fromkeys(ENV_PARAMS), **{"temperature": 10, "salinity": 20}),
        ),
        # dict has xr.DataArray, channel a list with matching values with those in dict
        (
            {
                "temperature": 10,
                "sound_absorption": xr.DataArray([10, 20], coords={"channel": ["chA", "chB"]}),
            },
            ["chA", "chB"],
            dict(
                dict.fromkeys(ENV_PARAMS),
                **{
                    "temperature": 10,
                    "sound_absorption": xr.DataArray([10, 20], coords={"channel": ["chA", "chB"]}),
                }
            ),
        ),
        # dict has xr.DataArray, channel a list with non-matching values with those in dict: XFAIL
        pytest.param(
            {
                "temperature": 10,
                "sound_absorption": xr.DataArray([10, 20], coords={"channel": ["chA", "chB"]}),
            },
            ["chA", "chC"],
            None,
            marks=pytest.mark.xfail(
                strict=True,
                reason="channel coordinate in param xr.DataArray mismatches that in the channel list",
            ),
        ),
        # dict has xr.DataArray, channel a xr.DataArray
        (
            {
                "temperature": 10,
                "sound_absorption": xr.DataArray([10, 20], coords={"channel": ["chA", "chB"]}),
            },
            xr.DataArray(["chA", "chB"], coords={"channel": ["chA", "chB"]}),
            dict(
                dict.fromkeys(ENV_PARAMS),
                **{
                    "temperature": 10,
                    "sound_absorption": xr.DataArray([10, 20], coords={"channel": ["chA", "chB"]}),
                }
            ),
        ),
        # dict has sound_absorption as a scalar: XFAIL
        pytest.param(
            {"temperature": 10, "sound_absorption": 0.02},
            ["chA", "chB"],
            None,
            marks=pytest.mark.xfail(
                strict=True, reason="sound_absorption should be a list or an xr.DataArray"
            ),
        ),
    ],
    ids=[
        "in_scalar_channel_list_out_scalar",
        "in_da_channel_list_out_da",
        "in_da_channel_list_mismatch",
        "in_da_channel_da",
        "in_absorption_scalae",
    ],
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
            dict(dict.fromkeys(ENV_PARAMS), **{"temperature": 10, "salinity": 20, "pressure": 100}),
        ),
        # not including salinity or pressure: XFAIL
        pytest.param(
            {"temperature": 10, "pressure": 100, "pH": 8.1},
            None,
            marks=pytest.mark.xfail(
                strict=True,
                reason="Fail since cal_channel_id in input param does not match channel of data",
            ),
        ),
    ],
    ids=[
        "default",
        "no_salinity",
    ],
)
def test_get_env_params_AZFP(azfp_path, env_ext, out_dict):
    azfp_01a_path = str(azfp_path.joinpath("17082117.01A"))
    azfp_xml_path = str(azfp_path.joinpath("17041823.XML"))
    ed = ep.open_raw(azfp_01a_path, sonar_model="AZFP", xml_path=azfp_xml_path)

    env_dict = get_env_params_AZFP(echodata=ed, user_dict=env_ext)

    out_dict = dict(
        out_dict,
        **{
            "sound_speed": ep.utils.uwa.calc_sound_speed(
                temperature=env_dict["temperature"],
                salinity=env_dict["salinity"],
                pressure=env_dict["pressure"],
                formula_source="AZFP",
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


@pytest.mark.parametrize(
    ("env_ext", "ref_formula_sound_speed", "ref_formula_absorption"),
    [
        # T, S, P, pH all exist so will trigger calculation, check default formula sources
        (
            {"temperature": 10, "salinity": 30, "pressure": 100, "pH": 8.1},
            "Mackenzie",
            "FG",
        ),
        # T, S, P, pH all exist, will calculate; has absorption formula passed in, check using the correct formula
        (
            {
                "temperature": 10,
                "salinity": 30,
                "pressure": 100,
                "pH": 8.1,
                "formula_absorption": "AM",
            },
            "Mackenzie",
            "AM",
        ),
    ],
    ids=[
        "calc_no_formula",
        "calc_with_formula",
    ],
)
def test_get_env_params_EK60_calculate(
    ek60_path, env_ext, ref_formula_sound_speed, ref_formula_absorption
):
    ed = ep.open_raw(
        ek60_path / "ncei-wcsd" / "Summer2017-D20170620-T011027.raw", sonar_model="EK60"
    )

    env_dict = get_env_params_EK(
        sonar_type="EK60",
        beam=ed["Sonar/Beam_group1"],
        env=ed["Environment"],
        user_dict=env_ext,
    )

    # Check formula sources
    assert env_dict["formula_sound_speed"] == ref_formula_sound_speed
    assert env_dict["formula_absorption"] == ref_formula_absorption

    # Check computation results
    sound_speed_ref = ep.utils.uwa.calc_sound_speed(
        temperature=env_ext["temperature"],
        salinity=env_ext["salinity"],
        pressure=env_ext["pressure"],
        formula_source=ref_formula_sound_speed,
    )
    sound_speed_ref = ep.calibrate.env_params.harmonize_env_param_time(
        sound_speed_ref, ping_time=ed["Sonar/Beam_group1"]["ping_time"]
    )
    absorption_ref = ep.utils.uwa.calc_absorption(
        frequency=ed["Sonar/Beam_group1"]["frequency_nominal"],
        temperature=env_ext["temperature"],
        salinity=env_ext["salinity"],
        pressure=env_ext["pressure"],
        pH=env_ext["pH"],
        sound_speed=sound_speed_ref,
        formula_source=ref_formula_absorption,
    )
    absorption_ref = ep.calibrate.env_params.harmonize_env_param_time(
        absorption_ref, ping_time=ed["Sonar/Beam_group1"]["ping_time"]
    )

    assert env_dict["sound_speed"] == sound_speed_ref
    assert env_dict["sound_absorption"].identical(absorption_ref)


def test_get_env_params_EK60_from_data(ek60_path):
    """
    If one of T, S, P, pH does not exist, use values from data file
    """
    ed = ep.open_raw(
        ek60_path / "ncei-wcsd" / "Summer2017-D20170620-T011027.raw", sonar_model="EK60"
    )

    env_dict = get_env_params_EK(
        sonar_type="EK60",
        beam=ed["Sonar/Beam_group1"],
        env=ed["Environment"],
        user_dict={"temperature": 10},
    )

    # Check default formula sources
    assert "formula_sound_speed" not in env_dict
    assert "formula_absorption" not in env_dict

    # Check params from data file: need to make time1 --> ping_time
    ref_sound_speed = ed["Environment"]["sound_speed_indicative"].copy()
    ref_sound_speed.coords["ping_time"] = ref_sound_speed["time1"]
    ref_sound_speed = ref_sound_speed.swap_dims({"time1": "ping_time"}).drop_vars("time1")
    assert env_dict["sound_speed"].identical(ref_sound_speed)

    ref_absorption = ed["Environment"]["absorption_indicative"].copy()
    ref_absorption.coords["ping_time"] = ref_absorption["time1"]
    ref_absorption = ref_absorption.swap_dims({"time1": "ping_time"}).drop_vars("time1")
    assert env_dict["sound_absorption"].identical(ref_absorption)


@pytest.mark.parametrize(
    ("env_ext", "ref_formula_sound_speed", "ref_formula_absorption"),
    [
        # T, S, P, pH all exist, check default formula sources
        (
            {"temperature": 10, "salinity": 30, "pressure": 100, "pH": 8.1},
            "Mackenzie",
            "FG",
        ),
        # T, S, P, pH all exist; has absorption formula passed in, check using the correct formula
        (
            {
                "temperature": 10,
                "salinity": 30,
                "pressure": 100,
                "pH": 8.1,
                "formula_absorption": "AM",
            },
            "Mackenzie",
            "AM",
        ),
    ],
    ids=[
        "calc_no_formula",
        "calc_with_formula",
    ],
)
def test_get_env_params_EK80_calculate(
    ek80_cal_path, env_ext, ref_formula_sound_speed, ref_formula_absorption
):
    ed = ep.open_raw(ek80_cal_path / "2018115-D20181213-T094600.raw", sonar_model="EK80")

    env_dict = get_env_params_EK(
        sonar_type="EK60",
        beam=ed["Sonar/Beam_group1"],
        env=ed["Environment"],
        user_dict=env_ext,
    )

    # Check formula sources
    assert env_dict["formula_sound_speed"] == ref_formula_sound_speed
    assert env_dict["formula_absorption"] == ref_formula_absorption

    # Check computation results
    sound_speed_ref = ep.utils.uwa.calc_sound_speed(
        temperature=env_ext["temperature"],
        salinity=env_ext["salinity"],
        pressure=env_ext["pressure"],
        formula_source=ref_formula_sound_speed,
    )
    sound_speed_ref = ep.calibrate.env_params.harmonize_env_param_time(
        sound_speed_ref, ping_time=ed["Sonar/Beam_group1"]["ping_time"]
    )
    absorption_ref = ep.utils.uwa.calc_absorption(
        frequency=ed["Sonar/Beam_group1"]["frequency_nominal"],
        temperature=env_ext["temperature"],
        salinity=env_ext["salinity"],
        pressure=env_ext["pressure"],
        pH=env_ext["pH"],
        sound_speed=sound_speed_ref,
        formula_source=ref_formula_absorption,
    )
    absorption_ref = ep.calibrate.env_params.harmonize_env_param_time(
        absorption_ref, ping_time=ed["Sonar/Beam_group1"]["ping_time"]
    )

    assert env_dict["sound_speed"] == sound_speed_ref
    assert env_dict["sound_absorption"].identical(absorption_ref)


@pytest.mark.parametrize(
    ("env_ext", "ref_formula_sound_speed", "ref_formula_absorption"),
    [
        # Only T exists, so use S, P, pH from data;
        # check default formula sources
        (
            {"temperature": 10},
            "Mackenzie",
            "FG",
        ),
        # Only T exists, so use S, P, pH from data;
        # has absorption formula passed in, check using the correct formula
        (
            {"temperature": 10, "formula_absorption": "AM"},
            "Mackenzie",
            "AM",
        ),
    ],
    ids=[
        "calc_no_formula",
        "calc_with_formula",
    ],
)
def test_get_env_params_EK80_from_data(
    ek80_cal_path, env_ext, ref_formula_sound_speed, ref_formula_absorption
):
    ed = ep.open_raw(ek80_cal_path / "2018115-D20181213-T094600.raw", sonar_model="EK80")

    env_dict = get_env_params_EK(
        sonar_type="EK80",
        beam=ed["Sonar/Beam_group1"],
        env=ed["Environment"],
        user_dict=env_ext,
        # technically should use center freq, use frequency nominal here for convenience
        freq=ed["Sonar/Beam_group1"]["frequency_nominal"],
    )

    # Check formula sources
    assert "formula_sound_speed" not in env_dict
    assert env_dict["formula_absorption"] == ref_formula_absorption

    # Check computation results
    # Use sound speed from data when T, S, P, pH are not all provided
    sound_speed_ref = ed["Environment"]["sound_speed_indicative"]
    # Always compute absorption for EK80
    absorption_ref = ep.utils.uwa.calc_absorption(
        frequency=ed["Sonar/Beam_group1"]["frequency_nominal"],
        temperature=env_ext["temperature"],  # use user-provided value if exists
        salinity=ed["Environment"]["salinity"],
        pressure=ed["Environment"]["depth"],
        pH=ed["Environment"]["acidity"],
        sound_speed=sound_speed_ref,
        formula_source=ref_formula_absorption,
    )
    absorption_ref = ep.calibrate.env_params.harmonize_env_param_time(
        absorption_ref, ping_time=ed["Sonar/Beam_group1"]["ping_time"]
    )

    assert env_dict["sound_speed"] == sound_speed_ref
    assert env_dict["sound_absorption"].identical(absorption_ref)
