import pytest

import numpy as np
import xarray as xr

import echopype as ep
from echopype.calibrate.ecs import ECSParser, ecs_ev2ep, ecs_ds2dict, conform_channel_order


# @pytest.fixture
# def azfp_path(test_path):
#     return test_path['AZFP']


@pytest.fixture
def ek60_path(test_path):
    return test_path['EK60']


@pytest.fixture
def ecs_path(test_path):
    return test_path['ECS']


# @pytest.fixture
# def ek80_path(test_path):
#     return test_path['EK80']


# @pytest.fixture
# def ek80_cal_path(test_path):
#     return test_path['EK80_CAL']


# @pytest.fixture
# def ek80_ext_path(test_path):
#     return test_path['EK80_EXT']



def test_ecs_intake_ek60(ek60_path, ecs_path):
    # get EchoData object that has the water_level variable under platform and compute Sv of it
    ed = ep.open_raw(ek60_path / "ncei-wcsd" / "Summer2017-D20170620-T011027.raw", "EK60")
    ecs_file = ecs_path / "Summer2017_JuneCal_3freq_mod.ecs"
    ds_Sv = ep.calibrate.compute_Sv(ed, ecs_file=ecs_file)

    # Parse ECS separately
    ecs = ECSParser(ecs_file)
    ecs.parse()
    ecs_dict = ecs.get_cal_params()  # apply ECS hierarchy
    ds_cal_tmp, ds_env_tmp = ecs_ev2ep(ecs_dict, "EK60")
    ds_env = ecs_ds2dict(conform_channel_order(ds_env_tmp, ed["Sonar/Beam_group1"]["frequency_nominal"]))
    ds_cal = ecs_ds2dict(conform_channel_order(ds_cal_tmp, ed["Sonar/Beam_group1"]["frequency_nominal"]))

    # Check if the final stored params (which are those used in calibration operations)
    # are those parsed from ECS
    for p_name in ["sound_speed", "sound_absorption"]:
        assert "ping_time" not in ds_Sv[p_name]  # only if pull from data will params have time coord
        assert ds_Sv[p_name].identical(ds_env[p_name])

    for p_name in [
            "sa_correction", "gain_correction", "equivalent_beam_angle",
            "beamwidth_alongship", "beamwidth_athwartship",
            "angle_offset_alongship", "angle_offset_athwartship",
            "angle_sensitivity_alongship", "angle_sensitivity_athwartship"
        ]:
        assert ds_Sv[p_name].identical(ds_cal[p_name])
