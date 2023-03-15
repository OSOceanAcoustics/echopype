import pytest

import numpy as np
import xarray as xr

import echopype as ep


@pytest.fixture
def azfp_path(test_path):
    return test_path['AZFP']


@pytest.fixture
def ek60_path(test_path):
    return test_path['EK60']


@pytest.fixture
def ek80_cal_path(test_path):
    return test_path['EK80_CAL']


def test_env_params_intake_AZFP(azfp_path):
    """
    Test env param intake for AZFP calibration.
    """
    azfp_01a_path = str(azfp_path.joinpath('17082117.01A'))
    azfp_xml_path = str(azfp_path.joinpath('17041823.XML'))
    ed = ep.open_raw(azfp_01a_path, sonar_model='AZFP', xml_path=azfp_xml_path)

    # Assemble external env param
    env_ext = {"salinity": 30, "pressure": 100}

    # Manually go through env params intake
    env_params_manual = ep.calibrate.env_params.get_env_params_AZFP(echodata=ed, user_dict=env_ext)
    for p in env_params_manual.keys():
        env_params_manual[p] = ep.calibrate.env_params.harmonize_env_param_time(
            env_params_manual[p], ping_time=ed["Sonar/Beam_group1"]["ping_time"]
        )
    env_params_manual["sound_speed"].name = "sound_speed"
    env_params_manual["sound_absorption"].name = "sound_absorption"

    # Check against the final env params in the calibration output
    ds_Sv = ep.calibrate.compute_Sv(ed, env_params=env_ext)
    assert ds_Sv["formula_sound_speed"] == "AZFP"
    assert ds_Sv["formula_absorption"] == "AZFP"
    assert ds_Sv["sound_speed"].identical(env_params_manual["sound_speed"])
    assert ds_Sv["sound_absorption"].identical(env_params_manual["sound_absorption"])


def test_env_params_intake_EK60_with_input(ek60_path):
    """
    Test env param intake for EK60 calibration.
    """
    ed = ep.open_raw(ek60_path / "ncei-wcsd" / "Summer2017-D20170620-T011027.raw", sonar_model="EK60")

    # Assemble external env param
    env_ext = {"temperature": 10, "salinity": 30, "pressure": 100, "pH": 8.1}

    # Manually go through env params intake
    env_params_manual = ep.calibrate.env_params.get_env_params_EK(
        sonar_type="EK60",
        beam=ed["Sonar/Beam_group1"],
        env=ed["Environment"],
        user_dict=env_ext
    )
    for p in env_params_manual.keys():
        env_params_manual[p] = ep.calibrate.env_params.harmonize_env_param_time(
            env_params_manual[p], ping_time=ed["Sonar/Beam_group1"]["ping_time"]
        )

    # Check against the final env params in the calibration output
    ds_Sv = ep.calibrate.compute_Sv(ed, env_params=env_ext)
    assert ds_Sv["formula_sound_speed"] == "Mackenzie"
    assert ds_Sv["formula_absorption"] == "FG"
    assert ds_Sv["sound_speed"].values == env_params_manual["sound_speed"]
    assert np.all(ds_Sv["sound_absorption"].values == env_params_manual["sound_absorption"].values)


def test_env_params_intake_EK60_no_input(ek60_path):
    """
    Test default env param extraction for EK60 calibration.
    """
    ed = ep.open_raw(ek60_path / "ncei-wcsd" / "Summer2017-D20170620-T011027.raw", sonar_model="EK60")
    ds_Sv = ep.calibrate.compute_Sv(ed)
    assert np.all(ds_Sv["sound_speed"].values == ed["Environment"]["sound_speed_indicative"].values)
    assert np.all(ds_Sv["sound_absorption"].values == ed["Environment"]["absorption_indicative"].values)


def test_env_params_intake_EK80_no_input(ek80_cal_path):
    """
    Test default env param extraction for EK80 calibration.
    """
    ed = ep.open_raw(ek80_cal_path / "2018115-D20181213-T094600.raw", sonar_model="EK80")
    ds_Sv = ep.calibrate.compute_Sv(ed, waveform_mode="BB", encode_mode="complex")

    # Use sound speed stored in Environment group
    assert ds_Sv["sound_speed"].values == ed["Environment"]["sound_speed_indicative"].values

    # Manually compute absorption
    cal_obj = ep.calibrate.calibrate_ek.CalibrateEK80(
        echodata=ed, waveform_mode="BB", encode_mode="complex", cal_params=None, env_params=None
    )
    absorption_ref = ep.utils.uwa.calc_absorption(
        frequency=cal_obj.freq_center,
        temperature=ed["Environment"]["temperature"],
        salinity=ed["Environment"]["salinity"],
        pressure=ed["Environment"]["depth"],
        pH=ed["Environment"]["acidity"],
        sound_speed=ed["Environment"]["sound_speed_indicative"],
        formula_source="FG",
    )
    absorption_ref = ep.calibrate.env_params.harmonize_env_param_time(
        absorption_ref, ping_time=ed["Sonar/Beam_group1"]["ping_time"]
    )
    assert np.all(cal_obj.env_params["sound_absorption"].values == absorption_ref.values)    
    assert np.all(ds_Sv["sound_absorption"].values == absorption_ref.values)


# TODO: Consolidate this and the one with EK60 with input
def test_env_params_intake_EK80_with_input_scalar(ek80_cal_path):
    """
    Test default env param extraction for EK80 calibration.
    """
    ed = ep.open_raw(ek80_cal_path / "2018115-D20181213-T094600.raw", sonar_model="EK80")

    # Assemble external env param
    env_ext = {"temperature": 10, "salinity": 30, "pressure": 100, "pH": 8.1}

    ds_Sv = ep.calibrate.compute_Sv(ed, waveform_mode="CW", encode_mode="complex", env_params=env_ext)

    # Manually compute absorption
    cal_obj = ep.calibrate.calibrate_ek.CalibrateEK80(
        echodata=ed, waveform_mode="CW", encode_mode="complex", cal_params=None, env_params=env_ext
    )
    sound_speed_ref = ep.utils.uwa.calc_sound_speed(
        temperature=env_ext["temperature"],
        salinity=env_ext["salinity"],
        pressure=env_ext["pressure"],
        formula_source="Mackenzie",
    )
    sound_speed_ref = ep.calibrate.env_params.harmonize_env_param_time(
        sound_speed_ref, ping_time=ed["Sonar/Beam_group1"]["ping_time"]
    )
    absorption_ref = ep.utils.uwa.calc_absorption(
        frequency=cal_obj.freq_center,
        temperature=env_ext["temperature"],
        salinity=env_ext["salinity"],
        pressure=env_ext["pressure"],
        pH=env_ext["pH"],
        sound_speed=sound_speed_ref,
        formula_source="FG",
    )
    absorption_ref = ep.calibrate.env_params.harmonize_env_param_time(
        absorption_ref, ping_time=ed["Sonar/Beam_group1"]["ping_time"]
    )

    assert np.all(cal_obj.env_params["sound_speed"] == sound_speed_ref)
    assert np.all(ds_Sv["sound_speed"] == sound_speed_ref)

    assert np.all(cal_obj.env_params["sound_absorption"].values == absorption_ref.values)
    assert np.all(ds_Sv["sound_absorption"].values == absorption_ref.values)


def test_env_params_intake_EK80_with_input_da(ek80_cal_path):
    """
    Test default env param extraction for EK80 calibration.
    """
    ed = ep.open_raw(ek80_cal_path / "2018115-D20181213-T094600.raw", sonar_model="EK80")

    # Assemble external env param
    env_ext = {
        "temperature": 10, "salinity": 30, "sound_speed": 1498,
        "sound_absorption": xr.DataArray(
            np.arange(1, 5) * 0.01,
            coords={"channel": ['WBT 714581-15 ES18', 'WBT 714583-15 ES120-7C',
                                'WBT 714597-15 ES333-7C', 'WBT 714605-15 ES200-7C']})
    }

    ds_Sv = ep.calibrate.compute_Sv(ed, waveform_mode="CW", encode_mode="complex", env_params=env_ext)

    assert env_ext["sound_speed"] == ds_Sv["sound_speed"].values
    assert np.all(env_ext["sound_absorption"].values == ds_Sv["sound_absorption"].values)
    for p_name in ["temperature", "salinity", "pressure", "pH"]:
        assert p_name not in ds_Sv
