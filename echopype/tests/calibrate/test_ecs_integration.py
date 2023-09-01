import pytest

import numpy as np
import xarray as xr

import echopype as ep
from echopype.calibrate.ecs import ECSParser, ecs_ev2ep, ecs_ds2dict, conform_channel_order
from echopype.calibrate.env_params import get_env_params_EK
from echopype.calibrate.cal_params import get_cal_params_EK


# @pytest.fixture
# def azfp_path(test_path):
#     return test_path['AZFP']


@pytest.fixture
def ek60_path(test_path):
    return test_path['EK60']


@pytest.fixture
def ecs_path(test_path):
    return test_path['ECS']


@pytest.fixture
def ek80_path(test_path):
    return test_path['EK80']


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
    ds_env_tmp, ds_cal_tmp, _ = ecs_ev2ep(ecs_dict, "EK60")
    env_params = ecs_ds2dict(conform_channel_order(ds_env_tmp, ed["Sonar/Beam_group1"]["frequency_nominal"]))
    cal_params = ecs_ds2dict(conform_channel_order(ds_cal_tmp, ed["Sonar/Beam_group1"]["frequency_nominal"]))

    # Check if the final stored params (which are those used in calibration operations)
    # are those parsed from ECS
    for p_name in ["sound_speed", "sound_absorption"]:
        assert "ping_time" not in ds_Sv[p_name]  # only if pull from data will params have time coord
        assert ds_Sv[p_name].identical(env_params[p_name])

    for p_name in [
            "sa_correction", "gain_correction", "equivalent_beam_angle",
            "beamwidth_alongship", "beamwidth_athwartship",
            "angle_offset_alongship", "angle_offset_athwartship",
            "angle_sensitivity_alongship", "angle_sensitivity_athwartship"
        ]:
        assert ds_Sv[p_name].identical(cal_params[p_name])


# EK80 CW power
def test_ecs_intake_ek80_CW_power(ek80_path, ecs_path):
    # get EchoData object that has the water_level variable under platform and compute Sv of it
    ed = ep.open_raw(ek80_path / "Summer2018--D20180905-T033113.raw", sonar_model="EK80")
    ecs_file = ecs_path / "Simrad_EK80_ES80_WBAT_EKAuto_Kongsberg_EA640_nohash.ecs"
    ds_Sv = ep.calibrate.compute_Sv(ed, ecs_file=ecs_file, waveform_mode="CW", encode_mode="power")

    # Parse ECS separately
    ecs = ECSParser(ecs_file)
    ecs.parse()
    ecs_dict = ecs.get_cal_params()  # apply ECS hierarchy
    ds_env, ds_cal_NB, ds_cal_BB = ecs_ev2ep(ecs_dict, "EK80")
    beam = ed["Sonar/Beam_group2"]
    chan_sel = ["WBT 743366-15 ES38B_ES", "WBT 743367-15 ES18_ES"]
    ecs_env_params = ecs_ds2dict(
        conform_channel_order(ds_env, beam["frequency_nominal"].sel(channel=chan_sel))
    )
    ecs_cal_params = ecs_ds2dict(
        conform_channel_order(ds_cal_NB, beam["frequency_nominal"].sel(channel=chan_sel))
    )
    ecs_cal_tmp_BB_conform_freq = conform_channel_order(
        ds_cal_BB, beam["frequency_nominal"].sel(channel=chan_sel)
    )
    assert ecs_cal_tmp_BB_conform_freq is None

    # Assimilate to standard env_params and cal_params in cal object
    assimilated_env_params = get_env_params_EK(
        sonar_type="EK80",
        beam=beam,
        env=ed["Environment"],
        user_dict=ecs_env_params,
        freq=beam["frequency_nominal"].sel(channel=chan_sel),
    )

    # Check the final stored params (which are those used in calibration operations)
    # For those pulled from ECS
    for p_name in ["sound_speed", "temperature", "salinity", "pressure", "pH"]:
        assert ds_Sv[p_name].identical(ecs_env_params[p_name])
    for p_name in [
            "sa_correction", "gain_correction", "equivalent_beam_angle",
            "beamwidth_alongship", "beamwidth_athwartship",
            "angle_offset_alongship", "angle_offset_athwartship",
            "angle_sensitivity_alongship", "angle_sensitivity_athwartship"
        ]:
        assert ds_Sv[p_name].identical(ecs_cal_params[p_name])
    # For those computed from values in ECS file
    assert np.all(ds_Sv["sound_absorption"].values == assimilated_env_params["sound_absorption"].values)

    # TODO: remove params that are only relevant to EK80 complex sample cals
    #       `impedance_transducer`, `impedance_transceiver`, `receiver_sampling_frequency`
    # for p_name in ["impedance_transducer", "impedance_transceiver", "receiver_sampling_frequency"]:
    #     assert p_name not in ds_Sv


# EK80 BB complex
def test_ecs_intake_ek80_BB_complex(ek80_path, ecs_path):
    # get EchoData object that has the water_level variable under platform and compute Sv of it
    ed = ep.open_raw(ek80_path / "Summer2018--D20180905-T033113.raw", sonar_model="EK80")
    ecs_file = ecs_path / "Simrad_EK80_ES80_WBAT_EKAuto_Kongsberg_EA640_nohash.ecs"
    ds_Sv = ep.calibrate.compute_Sv(ed, ecs_file=ecs_file, waveform_mode="BB", encode_mode="complex")

    # Parse ECS separately
    ecs = ECSParser(ecs_file)
    ecs.parse()
    ecs_dict = ecs.get_cal_params()  # apply ECS hierarchy
    ecs_env, ecs_cal_NB, ecs_cal_BB = ecs_ev2ep(ecs_dict, "EK80")

    # chan_sel = ['WBT 545612-15 ES200-7C_ES', 'WBT 549762-15 ES70-7C_ES', 'WBT 743869-15 ES120-7C_ES']
    ecs_env = conform_channel_order(ecs_env, ds_Sv["frequency_nominal"])
    ecs_cal_NB = conform_channel_order(ecs_cal_NB, ds_Sv["frequency_nominal"])
    ecs_cal_BB = conform_channel_order(ecs_cal_BB, ds_Sv["frequency_nominal"])

    # Check the final stored params (which are those used in calibration operations)
    # For those pulled from ECS
    for p_name in ["sound_speed", "temperature", "salinity", "pressure", "pH"]:
        assert ds_Sv[p_name].identical(ecs_env[p_name])

    for p_name in ["sa_correction", "receiver_sampling_frequency"]:
        assert ds_Sv[p_name].identical(ecs_cal_NB[p_name])

    # Check interpolation was done correctly
    beam = ed["Sonar/Beam_group1"]
    chan_w_BB_param = "WBT 549762-15 ES70-7C_ES"
    freq_center = (
        (beam["transmit_frequency_start"] + beam["transmit_frequency_stop"]) / 2
    ).sel(channel=chan_w_BB_param).drop_vars(["channel"])

    for p_name in [
        "gain_correction", "angle_offset_alongship", "angle_offset_athwartship",
        "beamwidth_alongship", "beamwidth_athwartship",
    ]:
        assert ds_Sv[p_name].sel(channel=chan_w_BB_param).drop_vars("channel").identical(
            ecs_cal_BB[p_name].interp(cal_frequency=freq_center)
            .squeeze().drop_vars(["channel", "cal_frequency"])
        )

    # TODO: remove params that are only relevant to EK80 CW cals `sa_correction`
    # assert "sa_correction" not in ds_Sv