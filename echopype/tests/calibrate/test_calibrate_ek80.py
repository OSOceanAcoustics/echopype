import pytest
import numpy as np
import pandas as pd
import pickle
import xarray as xr

import echopype as ep


@pytest.fixture
def ek80_path(test_path):
    return test_path["EK80"]


@pytest.fixture
def ek80_cal_path(test_path):
    return test_path["EK80_CAL"]


@pytest.fixture
def ek80_ext_path(test_path):
    return test_path["EK80_EXT"]


def test_ek80_transmit_chirp(ek80_cal_path, ek80_ext_path):
    """
    Test transmit chirp reconstruction against Andersen et al. 2021/pyEcholab implementation
    """
    ek80_raw_path = (
        ek80_cal_path / "2018115-D20181213-T094600.raw"
    )  # rx impedance / rx fs / tcvr type
    ed = ep.open_raw(ek80_raw_path, sonar_model="EK80")

    # Calibration object detail
    waveform_mode = "BB"
    encode_mode = "complex"
    cal_obj = ep.calibrate.calibrate_ek.CalibrateEK80(
        echodata=ed,
        waveform_mode=waveform_mode,
        encode_mode=encode_mode,
        env_params=None,
        cal_params=None,
    )
    fs = cal_obj.cal_params["receiver_sampling_frequency"]
    filter_coeff = ep.calibrate.ek80_complex.get_filter_coeff(
        ed["Vendor_specific"].sel(channel=cal_obj.chan_sel)
    )
    tx, tx_time = ep.calibrate.ek80_complex.get_transmit_signal(
        ed["Sonar/Beam_group1"].sel(channel=cal_obj.chan_sel), filter_coeff, waveform_mode, fs
    )
    tau_effective = ep.calibrate.ek80_complex.get_tau_effective(
        ytx_dict=tx,
        fs_deci_dict={k: 1 / np.diff(v[:2]) for (k, v) in tx_time.items()},  # decimated fs
        waveform_mode=cal_obj.waveform_mode,
        channel=cal_obj.chan_sel,
        ping_time=cal_obj.echodata["Sonar/Beam_group1"]["ping_time"],
    )

    # Load pyEcholab object: channel WBT 714590-15 ES70-7C
    with open(ek80_ext_path / "pyecholab/pyel_BB_calibration.pickle", "rb") as handle:
        pyecholab_BB = pickle.load(handle)

    # Compare first ping since all params identical
    ch_sel = "WBT 714590-15 ES70-7C"
    # receive sampling frequency
    assert pyecholab_BB["rx_sample_frequency"][0] == fs.sel(channel=ch_sel)
    # WBT filter
    assert np.all(pyecholab_BB["filters"][1]["coefficients"] == filter_coeff[ch_sel]["wbt_fil"])
    assert np.all(
        pyecholab_BB["filters"][1]["decimation_factor"] == filter_coeff[ch_sel]["wbt_decifac"]
    )
    # PC filter
    assert np.all(pyecholab_BB["filters"][2]["coefficients"] == filter_coeff[ch_sel]["pc_fil"])
    assert np.all(
        pyecholab_BB["filters"][2]["decimation_factor"] == filter_coeff[ch_sel]["pc_decifac"]
    )
    # transmit signal
    assert np.allclose(pyecholab_BB["_tx_signal"][0], tx[ch_sel])
    # tau effective
    # use np.isclose for now since difference is 2.997176e-5 (pyecholab) and 2.99717595e-05 (echopype)
    # will see if it causes downstream major differences
    assert np.isclose(tau_effective.sel(channel=ch_sel).data, pyecholab_BB["_tau_eff"][0])


def test_ek80_BB_params(ek80_cal_path, ek80_ext_path):
    """
    Test power from pulse compressed BB data
    """
    ek80_raw_path = (
        ek80_cal_path / "2018115-D20181213-T094600.raw"
    )  # rx impedance / rx fs / tcvr type
    ed = ep.open_raw(ek80_raw_path, sonar_model="EK80")

    # Calibration object detail
    waveform_mode = "BB"
    encode_mode = "complex"

    cal_obj = ep.calibrate.calibrate_ek.CalibrateEK80(
        echodata=ed,
        waveform_mode=waveform_mode,
        encode_mode=encode_mode,
        env_params={"formula_absorption": "FG"},
        cal_params=None,
    )

    z_er = cal_obj.cal_params["impedance_transceiver"]
    z_et = cal_obj.cal_params["impedance_transducer"]
    # B_theta_phi_m = cal_obj._get_B_theta_phi_m()
    params_BB_map = {
        # param name mapping: echopype (ep) : pyecholab (pyel)
        "angle_offset_alongship": "angle_offset_alongship",
        "angle_offset_athwartship": "angle_offset_athwartship",
        "beamwidth_alongship": "beam_width_alongship",
        "beamwidth_athwartship": "beam_width_athwartship",
        "gain_correction": "gain",  # this is *before* B_theta_phi_m BB correction
    }

    # Load pyEcholab object: channel WBT 714590-15 ES70-7C
    with open(ek80_ext_path / "pyecholab/pyel_BB_calibration.pickle", "rb") as handle:
        pyel_BB_cal = pickle.load(handle)
    with open(ek80_ext_path / "pyecholab/pyel_BB_raw_data.pickle", "rb") as handle:
        pyel_BB_raw = pickle.load(handle)

    ch_sel = "WBT 714590-15 ES70-7C"

    # pyecholab calibration object
    # TODO: need to check B_theta_phi_m values
    assert pyel_BB_cal["impedance"] == z_er.sel(channel=ch_sel)
    for p_ep, p_pyel in params_BB_map.items():  # all interpolated BB params
        assert np.isclose(
            pyel_BB_cal[p_pyel][0], cal_obj.cal_params[p_ep].sel(channel=ch_sel).isel(ping_time=0)
        )
    assert pyel_BB_cal["sa_correction"][0] == cal_obj.cal_params["sa_correction"].sel(
        channel=ch_sel
    ).isel(ping_time=0)
    assert pyel_BB_cal["sound_speed"] == cal_obj.env_params["sound_speed"]
    assert np.isclose(
        pyel_BB_cal["absorption_coefficient"][0],
        cal_obj.env_params["sound_absorption"].sel(channel=ch_sel).isel(ping_time=0),
    )

    # pyecholab raw_data object
    assert pyel_BB_raw["ZTRANSDUCER"] == z_et.sel(channel=ch_sel).isel(ping_time=0)
    assert pyel_BB_raw["transmit_power"][0] == ed["Sonar/Beam_group1"]["transmit_power"].sel(
        channel=ch_sel
    ).isel(ping_time=0)
    assert pyel_BB_raw["transceiver_type"] == ed["Vendor_specific"]["transceiver_type"].sel(
        channel=ch_sel
    )


def test_ek80_BB_range(ek80_cal_path, ek80_ext_path):
    ek80_raw_path = (
        ek80_cal_path / "2018115-D20181213-T094600.raw"
    )  # rx impedance / rx fs / tcvr type
    ed = ep.open_raw(ek80_raw_path, sonar_model="EK80")

    # Calibration object
    waveform_mode = "BB"
    encode_mode = "complex"
    cal_obj = ep.calibrate.calibrate_ek.CalibrateEK80(
        echodata=ed,
        waveform_mode=waveform_mode,
        encode_mode=encode_mode,
        env_params={"formula_absorption": "FG"},
        cal_params=None,
    )

    ch_sel = "WBT 714590-15 ES70-7C"

    # Load pyecholab pickle
    with open(ek80_ext_path / "pyecholab/pyel_BB_p_data.pickle", "rb") as handle:
        pyel_BB_p_data = pickle.load(handle)

    # Assert
    ep_vals = cal_obj.range_meter.sel(channel=ch_sel).isel(ping_time=0).data
    pyel_vals = pyel_BB_p_data["range"]
    assert np.allclose(pyel_vals, ep_vals)

@pytest.mark.parametrize(
    ("raw_data_path,raw_file_name,pyecholab_data_path,pyecholab_file_path, dask_array"),
    [
        (
            "ek80_cal_path",
            "2018115-D20181213-T094600.raw",
            "ek80_ext_path",
            "pyecholab/pyel_BB_p_data.pickle",
            False,
        ),
        (
            "ek80_cal_path",
            "2018115-D20181213-T094600.raw",
            "ek80_ext_path",
            "pyecholab/pyel_BB_p_data.pickle",
            True,
        ),
    ],
)
def test_ek80_BB_power_from_complex(
    raw_data_path,
    raw_file_name,
    pyecholab_data_path,
    pyecholab_file_path,
    dask_array,
    request,
):
    raw_data_path = request.getfixturevalue(raw_data_path)
    ek80_raw_path = raw_data_path / raw_file_name  # rx impedance / rx fs / tcvr type

    if dask_array:
        ed = ep.open_raw(ek80_raw_path, sonar_model="EK80", use_swap=True)
    else:
        ed = ep.open_raw(ek80_raw_path, sonar_model="EK80")

    # Calibration object
    waveform_mode = "BB"
    encode_mode = "complex"
    cal_obj = ep.calibrate.calibrate_ek.CalibrateEK80(
        echodata=ed,
        waveform_mode=waveform_mode,
        encode_mode=encode_mode,
        env_params={"formula_absorption": "FG"},
        cal_params=None,
    )

    # Params needed
    beam = cal_obj.echodata[cal_obj.ed_beam_group].sel(channel=cal_obj.chan_sel)
    z_er = cal_obj.cal_params["impedance_transceiver"]
    z_et = cal_obj.cal_params["impedance_transducer"]
    fs = cal_obj.cal_params["receiver_sampling_frequency"]
    filter_coeff = ep.calibrate.ek80_complex.get_filter_coeff(
        ed["Vendor_specific"].sel(channel=cal_obj.chan_sel)
    )
    tx, _ = ep.calibrate.ek80_complex.get_transmit_signal(beam, filter_coeff, waveform_mode, fs)

    # Get power from complex samples
    prx = cal_obj._get_power_from_complex(beam=beam, chirp=tx, z_et=z_et, z_er=z_er).compute()

    ch_sel = "WBT 714590-15 ES70-7C"

    # Load pyecholab pickle
    pyecholab_data_path = request.getfixturevalue(pyecholab_data_path)
    with open(pyecholab_data_path / pyecholab_file_path, "rb") as handle:
        pyel_BB_p_data = pickle.load(handle)

    # Power: only compare non-Nan, non-Inf values
    pyel_vals = pyel_BB_p_data["power"]
    ep_vals = 10 * np.log10(prx.sel(channel=ch_sel).data)
    assert pyel_vals.shape == ep_vals.shape
    idx_to_cmp = ~(
        np.isinf(pyel_vals) | np.isnan(pyel_vals) | np.isinf(ep_vals) | np.isnan(ep_vals)
    )
    assert np.allclose(pyel_vals[idx_to_cmp], ep_vals[idx_to_cmp])

@pytest.mark.parametrize(
    ("raw_data_path,raw_file_name,pyecholab_data_path,pyecholab_file_path, dask_array"),
    [
        (
            "ek80_cal_path",
            "2018115-D20181213-T094600.raw",
            "ek80_ext_path",
            "pyecholab/pyel_BB_p_data.pickle",
            False,
        ),
        (
            "ek80_cal_path",
            "2018115-D20181213-T094600.raw",
            "ek80_ext_path",
            "pyecholab/pyel_BB_p_data.pickle",
            True,
        ),
    ],
)
def test_ek80_BB_power_compute_Sv(
    raw_data_path,
    raw_file_name,
    pyecholab_data_path,
    pyecholab_file_path,
    dask_array,
    request,
):
    raw_data_path = request.getfixturevalue(raw_data_path)
    ek80_raw_path = raw_data_path / raw_file_name  # rx impedance / rx fs / tcvr type

    if dask_array:
        ed = ep.open_raw(ek80_raw_path, sonar_model="EK80", use_swap=True)
    else:
        ed = ep.open_raw(ek80_raw_path, sonar_model="EK80")

    # Calibration object
    waveform_mode = "BB"
    encode_mode = "complex"

    ch_sel = "WBT 714590-15 ES70-7C"

    # Load pyecholab pickle
    pyecholab_data_path = request.getfixturevalue(pyecholab_data_path)
    with open(pyecholab_data_path / pyecholab_file_path, "rb") as handle:
        pyel_BB_p_data = pickle.load(handle)

    # Sv: only compare non-Nan, non-Inf values
    # comparing for only the last values now until fixing the range computation
    ds_Sv = ep.calibrate.compute_Sv(
        ed,
        waveform_mode=waveform_mode,
        encode_mode=encode_mode,
    )
    pyel_vals = pyel_BB_p_data["sv_data"]
    if dask_array:
        ep_vals = ds_Sv["Sv"].sel(channel=ch_sel).squeeze().data.compute()	
    else:	
        ep_vals = ds_Sv["Sv"].sel(channel=ch_sel).squeeze().data

    assert pyel_vals.shape == ep_vals.shape
    idx_to_cmp = ~(
        np.isinf(pyel_vals) | np.isnan(pyel_vals) | np.isinf(ep_vals) | np.isnan(ep_vals)
    )
    assert np.allclose(pyel_vals[idx_to_cmp], ep_vals[idx_to_cmp])


def test_ek80_BB_power_echoview(ek80_path):
    """Compare pulse compressed outputs from echopype and csv exported from EchoView.

    Unresolved: the difference is large and it is not clear why.
    """
    ek80_raw_path = str(ek80_path.joinpath("D20170912-T234910.raw"))
    ek80_bb_pc_test_path = str(
        ek80_path.joinpath("from_echoview", "70 kHz pulse-compressed power.complex.csv")
    )

    echodata = ep.open_raw(ek80_raw_path, sonar_model="EK80")

    # Create a CalibrateEK80 object to perform pulse compression
    cal_obj = ep.calibrate.calibrate_ek.CalibrateEK80(
        echodata,
        env_params=None,
        cal_params=None,
        waveform_mode="BB",
        encode_mode="complex",
    )
    beam = echodata["Sonar/Beam_group1"].sel(channel=cal_obj.chan_sel)

    coeff = ep.calibrate.ek80_complex.get_filter_coeff(
        echodata["Vendor_specific"].sel(channel=cal_obj.chan_sel)
    )
    chirp, _ = ep.calibrate.ek80_complex.get_transmit_signal(
        beam,
        coeff,
        "BB",
        cal_obj.cal_params["receiver_sampling_frequency"],
    )

    pc = ep.calibrate.ek80_complex.compress_pulse(
        backscatter=beam["backscatter_r"] + 1j * beam["backscatter_i"],
        chirp=chirp,
    ).compute()
    pc = pc / ep.calibrate.ek80_complex.get_norm_fac(chirp)  # normalization for each channel
    pc_mean = pc.sel(channel="WBT 549762-15 ES70-7C").mean(dim="beam").dropna("range_sample")

    # Read EchoView pc raw power output
    df = pd.read_csv(ek80_bb_pc_test_path, header=None, skiprows=[0])
    df_header = pd.read_csv(ek80_bb_pc_test_path, header=0, usecols=range(14), nrows=0)
    df = df.rename(columns={cc: vv for cc, vv in zip(df.columns, df_header.columns.values)})
    df.columns = df.columns.str.strip()
    df_real = df.loc[df["Component"] == " Real", :].iloc[:, 14:]  # values start at column 15

    # Skip an initial chunk of samples due to unknown larger difference
    # this difference is also documented in pyecholab tests
    # Below only compare the first ping
    ev_vals = df_real.values[:, :]
    ep_vals = pc_mean.values.real[:, :]
    assert np.allclose(ev_vals[:, 69:], ep_vals[:, 69:], atol=1e-4)
    assert np.allclose(ev_vals[:, 90:], ep_vals[:, 90:], atol=1e-5)

def test_ek80_CW_complex_Sv_receiver_sampling_freq(ek80_path):
    ek80_raw_path = str(ek80_path.joinpath("D20230804-T083032.raw"))
    ed = ep.open_raw(ek80_raw_path, sonar_model="EK80")
    # Parsed receiver_sampling_frequency is 0
    assert ed["Vendor_specific"]["receiver_sampling_frequency"] == 0
    # Calibration object
    waveform_mode = "CW"
    encode_mode = "complex"
    ds_Sv = ep.calibrate.compute_Sv(
        ed, waveform_mode=waveform_mode, encode_mode=encode_mode
    )

    # receiver_sampling_frequency substituted with default value in compute_Sv
    assert ds_Sv["receiver_sampling_frequency"] is not None
    assert np.allclose(ds_Sv["receiver_sampling_frequency"].data, 1500000)


@pytest.mark.parametrize(
    ("raw_data_path,target_channel_ping_pattern"),
    [
        (
            "NYOS2105-D20210525-T213648.raw",
            # (3,4), 2, 1, (3,4), 2, 1, etc...
            np.array(
                [
                    [np.nan, np.nan,  1., np.nan, np.nan,  1., np.nan, np.nan,  1., np.nan],
                    [np.nan,  1., np.nan, np.nan,  1., np.nan, np.nan,  1., np.nan, np.nan],
                    [ 1., np.nan, np.nan,  1., np.nan, np.nan,  1., np.nan, np.nan,  1.],
                    [ 1., np.nan, np.nan,  1., np.nan, np.nan,  1., np.nan, np.nan,  1.]
                ]
            ),
        ),
        (
            "DRIX08-D20231003-T120051.raw",
            # 2, 1, 2, 1, 2, 1, etc...
            np.array(
                [
                    [np.nan,  1., np.nan, 1,  np.nan, 1, np.nan,  1., np.nan, 1],
                    [1, np.nan,  1., np.nan, 1,  np.nan, 1, np.nan,  1., np.nan],
                ]
            ),
        ),
    ],
)
@pytest.mark.integration
def test_ek80_BB_complex_multiplex_NaNs_and_non_NaNs(raw_data_path, target_channel_ping_pattern):
    # Extract bb complex multiplex EK80 data
    ed = ep.open_raw(f"echopype/test_data/ek80_bb_complex_multiplex/{raw_data_path}", sonar_model="EK80")

    # Compute Sv
    ds_Sv = ep.calibrate.compute_Sv(ed,waveform_mode='BB',encode_mode='complex')
    
    # Extract mask for both real backscatter and calibrated Sv:
    
    # EchoData Real (component) Backscatter mask should be 1 if all values along 
    # a specific range sample and beam dimension are non-NaN; else NaN.
    ed_backscatter_r_mask = xr.where(
        ~np.isnan(ed["Sonar/Beam_group1"]["backscatter_r"]
    ).all(dim=['range_sample', "beam"]), 1, np.nan)
    # Calibrated Sv mask should be 1 if all values along 
    # a specific range sample are non-NaN; else NaN.
    calibrated_Sv_mask = xr.where(
        ~np.isnan(ds_Sv["Sv"]
    ).all(dim=['range_sample']), 1, np.nan)

    # Check that they are equal
    assert np.array_equal(
        ed_backscatter_r_mask.data,
        calibrated_Sv_mask.data,
        equal_nan=True
    )

    # Check that a slice of calibrated Sv mask is equal to the following array
    # with the target channel ping pattern:
    assert np.array_equal(
        calibrated_Sv_mask.isel(ping_time=slice(0,10)).data,
        target_channel_ping_pattern,
        equal_nan=True
    )
