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
    assert ds_Sv["formula_source_sound_speed"] == "AZFP"
    assert ds_Sv["formula_source_absorption"] == "AZFP"
    assert ds_Sv["sound_speed"].identical(env_params_manual["sound_speed"])
    assert ds_Sv["sound_absorption"].identical(env_params_manual["sound_absorption"])


def test_env_params_intake_EK60(ek60_path):
    """
    Test env param intake for EK60 calibration.
    """
    ed = ep.open_raw(ek60_path / "ncei-wcsd" / "Summer2017-D20170620-T011027.raw", sonar_model="EK60")

    # Assemble external env param
    env_ext = {"temperature": 10, "salinity": 30, "pressure": 100}

    # Manually go through env params intake
    env_params_manual = ep.calibrate.env_params.get_env_params_EK60(echodata=ed, user_dict=env_ext)
    for p in env_params_manual.keys():
        env_params_manual[p] = ep.calibrate.env_params.harmonize_env_param_time(
            env_params_manual[p], ping_time=ed["Sonar/Beam_group1"]["ping_time"]
        )

    # Check against the final env params in the calibration output
    ds_Sv = ep.calibrate.compute_Sv(ed, env_params=env_ext)
    assert ds_Sv["formula_source_sound_speed"] == "Mackenzie"
    assert ds_Sv["formula_source_absorption"] == "FG"
    assert ds_Sv["sound_speed"].values == env_params_manual["sound_speed"]
    assert np.all(ds_Sv["sound_absorption"].values == env_params_manual["sound_absorption"].values)
    



# TODO: add tests for EK80 BB/CW complex and CW power
