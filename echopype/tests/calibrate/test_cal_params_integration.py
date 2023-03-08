import pytest

import numpy as np
import xarray as xr

import echopype as ep


@pytest.fixture
def ek80_cal_path(test_path):
    return test_path['EK80_CAL']


def test_cal_params_intake_EK80_BB_complex(ek80_cal_path):
    """
    Tests calibration for file containing both BB and CW mode data
    with both encoded as complex samples.
    """
    ed = ep.open_raw(ek80_cal_path / "2018115-D20181213-T094600.raw", sonar_model="EK80")

    # Assemble external freq-dependent cal param
    len_cal_frequency = ed["Vendor_specific"]["cal_frequency"].size
    gain_freq_dep = xr.DataArray(
        np.array([np.arange(len_cal_frequency), (np.arange(len_cal_frequency) + 1000)[::-1]]),
        dims=["cal_channel_id", "cal_frequency"],
        coords={
            "cal_channel_id": ["WBT 714590-15 ES70-7C", "WBT 714596-15 ES38-7"],
            "cal_frequency": ed["Vendor_specific"]["cal_frequency"],
        },
    )

    # Manually go through cal params intake
    chan_sel = ["WBT 714590-15 ES70-7C", "WBT 714596-15 ES38-7"]
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
