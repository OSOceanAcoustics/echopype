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
    pass


def test_env_params_intake_EK60(ek60_path):
    """
    Test env param intake for EK60 calibration.
    """
    ed = ep.open_raw(ek60_path / "ncei-wcsd" / "Summer2017-D20170620-T011027.raw", sonar_model="EK60")

    # Assemble external env param
    env_ext = {"temperature": 10, "salinity": 30, "pressure": 100}

    # Manually go through cal params intake
    env_params_manual = ep.calibrate.env_params.get_env_params_EK60(echodata=ed, user_env_dict=env_ext)
    for p in env_params_manual.keys():
        env_params_manual[p] = ep.calibrate.env_params.harmonize_env_param_time(
            env_params_manual[p], ping_time=ed["Sonar/Beam_group1"]["ping_time"]
        )

    # Check against the final env params in the calibration output
    ds_Sv = ep.calibrate.compute_Sv(ed, env_params=env_ext)
    assert ds_Sv["sound_speed"].values == env_params_manual["sound_speed"]
    assert np.all(ds_Sv["sound_absorption"].values == env_params_manual["sound_absorption"].values)
    



# TODO: add tests for EK80 BB/CW complex and CW power
