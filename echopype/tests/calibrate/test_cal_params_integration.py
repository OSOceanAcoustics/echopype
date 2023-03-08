import pytest

import numpy as np
import xarray as xr

import echopype as ep


@pytest.fixture
def ek60_path(test_path):
    return test_path['EK60']


@pytest.fixture
def ek80_cal_path(test_path):
    return test_path['EK80_CAL']


def test_cal_params_intake_EK60(ek60_path):
    """
    Test cal param intake for EK60 calibration.
    """
    ed = ep.open_raw(ek60_path / "ncei-wcsd" / "Summer2017-D20170620-T011027.raw", sonar_model="EK60")

    # Assemble external cal param
    chan = ed["Sonar/Beam_group1"]["channel"]
    gain_ext = xr.DataArray([100, 200, 300], dims=["channel"], coords={"channel": chan}, name="gain_correction")

    # Manually go through cal params intake
    cal_params_manual = ep.calibrate.cal_params.get_cal_params_EK(
        waveform_mode="CW",
        freq_center=ed["Sonar/Beam_group1"]["frequency_nominal"],
        beam=ed["Sonar/Beam_group1"],
        vend=ed["Vendor_specific"],
        user_dict={"gain_correction": gain_ext},
        sonar_type="EK60",
    )

    # Manually add cal params in Vendor group and construct cal object
    ed["Vendor_specific"]["gain_correction"].data[0, 1] = gain_ext.data[0]  # GPT  18 kHz 009072058c8d 1-1 ES18-11
    ed["Vendor_specific"]["gain_correction"].data[1, 2] = gain_ext.data[1]  # GPT  38 kHz 009072058146 2-1 ES38B
    ed["Vendor_specific"]["gain_correction"].data[2, 4] = gain_ext.data[2]  # GPT 120 kHz 00907205a6d0 4-1 ES120-7C
    cal_obj = ep.calibrate.calibrate_ek.CalibrateEK60(echodata=ed, env_params=None, cal_params=None)

    # Check cal params ingested from both ways
    assert cal_obj.cal_params["gain_correction"].isel(ping_time=0).drop("ping_time").identical(cal_params_manual["gain_correction"])

    # Check against the final cal params in the calibration output
    ds_Sv = ep.calibrate.compute_Sv(ed, cal_params={"gain_correction": gain_ext})
    assert ds_Sv["gain_correction"].identical(cal_params_manual["gain_correction"])


def test_cal_params_intake_EK80_BB_complex(ek80_cal_path):
    """
    Test frequency-dependent cal param intake for EK80 BB complex calibration.
    """
    ed = ep.open_raw(ek80_cal_path / "2018115-D20181213-T094600.raw", sonar_model="EK80")

    # BB channels
    chan_sel = ["WBT 714590-15 ES70-7C", "WBT 714596-15 ES38-7"]

    # Assemble external freq-dependent cal param
    len_cal_frequency = ed["Vendor_specific"]["cal_frequency"].size
    gain_freq_dep = xr.DataArray(
        np.array([np.arange(len_cal_frequency), (np.arange(len_cal_frequency) + 1000)[::-1]]),
        dims=["cal_channel_id", "cal_frequency"],
        coords={
            "cal_channel_id": chan_sel,
            "cal_frequency": ed["Vendor_specific"]["cal_frequency"],
        },
    )

    # Manually go through cal params intake
    beam = ed["Sonar/Beam_group1"].sel(channel=chan_sel)
    vend = ed["Vendor_specific"].sel(channel=chan_sel)
    freq_center = (
        (beam["frequency_start"] + beam["frequency_end"])
        .sel(channel=chan_sel).isel(beam=0).drop("beam") / 2
    )
    cal_params_manual = ep.calibrate.cal_params.get_cal_params_EK(
        "BB", freq_center, beam, vend, {"gain_correction": gain_freq_dep}
    )    

    # Manually add freq-dependent cal params in Vendor group
    # and construct cal object
    ed["Vendor_specific"]["gain"].data[1, :] = gain_freq_dep[0, :]  # WBT 714590-15 ES70-7C
    ed["Vendor_specific"]["gain"].data[2, :] = gain_freq_dep[1, :]  # WBT 714596-15 ES38-7
    cal_obj = ep.calibrate.calibrate_ek.CalibrateEK80(
        echodata=ed, waveform_mode="BB", encode_mode="complex", cal_params=None, env_params=None
    )

    # Check cal params ingested from both ways
    assert cal_obj.cal_params["gain_correction"].identical(cal_params_manual["gain_correction"])

    # Check against the final cal params in the calibration output
    ds_Sv = ep.calibrate.compute_Sv(
        ed, waveform_mode="BB", encode_mode="complex", cal_params={"gain_correction": gain_freq_dep}
    )
    cal_params_manual["gain_correction"].name = "gain_correction"
    assert ds_Sv["gain_correction"].identical(cal_params_manual["gain_correction"])


def test_cal_params_intake_EK80_CW_complex(ek80_cal_path):
    """
    Test frequency-dependent cal param intake for EK80 CW complex calibration.
    """
    ed = ep.open_raw(ek80_cal_path / "2018115-D20181213-T094600.raw", sonar_model="EK80")

    # CW channels
    chan_sel = ["WBT 714581-15 ES18", "WBT 714583-15 ES120-7C",
                "WBT 714597-15 ES333-7C", "WBT 714605-15 ES200-7C"]

    # Assemble external freq-dependent cal param
    len_cal_frequency = ed["Vendor_specific"]["cal_frequency"].size
    gain_freq_dep = xr.DataArray(
        np.array([
            np.arange(len_cal_frequency),
            (np.arange(len_cal_frequency) + 1000)[::-1],
            (np.arange(len_cal_frequency) + 2000)[::-1],
            (np.arange(len_cal_frequency) + 3000)[::-1],
        ]),
        dims=["cal_channel_id", "cal_frequency"],
        coords={
            "cal_channel_id": chan_sel,
            "cal_frequency": ed["Vendor_specific"]["cal_frequency"],
        },
    )

    # Manually go through cal params intake
    beam = ed["Sonar/Beam_group1"].sel(channel=chan_sel)
    vend = ed["Vendor_specific"].sel(channel=chan_sel)
    freq_center = beam["frequency_nominal"].sel(channel=chan_sel)
    cal_params_manual = ep.calibrate.cal_params.get_cal_params_EK(
        "CW", freq_center, beam, vend, {"gain_correction": gain_freq_dep}
    )
    cal_params_manual["gain_correction"].name = "gain_correction"

    # Manually add cal params in Vendor group construct cal object
    ed["Vendor_specific"]["gain_correction"].data[0, 1] = cal_params_manual["gain_correction"].data[0]  # WBT 714581-15 ES18
    ed["Vendor_specific"]["gain_correction"].data[1, 4] = cal_params_manual["gain_correction"].data[1]  # WBT 714583-15 ES120-7C
    ed["Vendor_specific"]["gain_correction"].data[4, 4] = cal_params_manual["gain_correction"].data[2]  # WBT 714597-15 ES333-7C
    ed["Vendor_specific"]["gain_correction"].data[5, 4] = cal_params_manual["gain_correction"].data[3]  # WBT 714605-15 ES200-7C
    cal_obj = ep.calibrate.calibrate_ek.CalibrateEK80(
        echodata=ed, waveform_mode="CW", encode_mode="complex", cal_params=None, env_params=None
    )

    # Check cal params ingested from both ways
    assert cal_obj.cal_params["gain_correction"].isel(ping_time=0).drop("ping_time").identical(cal_params_manual["gain_correction"])

    # Check against the final cal params in the calibration output
    ds_Sv = ep.calibrate.compute_Sv(
        ed, waveform_mode="CW", encode_mode="complex", cal_params={"gain_correction": gain_freq_dep}
    )
    assert ds_Sv["gain_correction"].identical(cal_params_manual["gain_correction"])
