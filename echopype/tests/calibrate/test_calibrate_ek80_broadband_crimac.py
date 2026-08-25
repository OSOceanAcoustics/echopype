import numpy as np
import pytest
import xarray as xr

import echopype as ep

pytestmark = pytest.mark.integration

CRIMAC_CHANNEL = "WBT 747022-15 ES120-7CD_ES"
CRIMAC_PING_INDEX = 509


@pytest.fixture(scope="module")
def ts_spectrum_example_path(test_path):
    return test_path["TS_SPECTRUM_EXAMPLE"]


@pytest.fixture(scope="module")
def ts_raw_path(ts_spectrum_example_path):
    return ts_spectrum_example_path / "IMR-D20211215-T143432-TSf.raw"


@pytest.fixture(scope="module")
def ts_ref(ts_spectrum_example_path):
    return np.load(
        ts_spectrum_example_path / "crimac_tsf_reference_outputs.npz",
        allow_pickle=True,
    )


@pytest.fixture(scope="module")
def ts_echodata(ts_raw_path):
    return ep.open_raw(ts_raw_path, sonar_model="EK80")


def _target_locations_from_crimac(ed, ref, channel, ping_index):
    ping_time = ed["Sonar/Beam_group1"]["ping_time"].isel(ping_time=ping_index).values

    return xr.Dataset(
        data_vars={
            "target_range": ("target_id", [float(ref["r_t"])]),
            "angle_alongship": ("target_id", [float(ref["theta_t"])]),
            "angle_athwartship": ("target_id", [float(ref["phi_t"])]),
            "target_range_min": ("target_id", [float(ref["dum_r"][0])]),
            "target_range_max": ("target_id", [float(ref["dum_r"][-1])]),
        },
        coords={
            "target_id": [0],
            "ping_time": ("target_id", [ping_time]),
            "channel": ("target_id", [channel]),
        },
    )


def test_compute_sp_fm_complex_runs(ts_echodata):
    ds = ep.calibrate.compute_Sp(
        ts_echodata,
        waveform_mode="FM",
        encode_mode="complex",
    )

    assert "Sp" in ds
    assert set(("channel", "ping_time", "range_sample")).issubset(ds["Sp"].dims)
    assert np.isfinite(ds["Sp"]).any()


def test_frequency_dependent_absorption_matches_crimac(ts_echodata, ts_ref):
    cal_obj = ep.calibrate.calibrate_ek.CalibrateEK80(
        echodata=ts_echodata,
        waveform_mode="BB", #TODO change to FM after deprecation of BB
        encode_mode="complex",
        env_params=None,
        cal_params=None,
    )

    sound_speed = float(cal_obj.env_params["sound_speed"])

    absorption_f = cal_obj._compute_absorption_f(
        frequency=ts_ref["f_m"],
        channel=CRIMAC_CHANNEL,
        ping_idx=CRIMAC_PING_INDEX,
        sound_speed=sound_speed,
    )

    np.testing.assert_allclose(
        absorption_f,
        ts_ref["alpha_m"],
        atol=1e-8,
        rtol=0.0,
    )


def test_compute_ts_spectrum_matches_crimac(ts_echodata, ts_ref):
    point_locations = _target_locations_from_crimac(
        ed=ts_echodata,
        ref=ts_ref,
        channel=CRIMAC_CHANNEL,
        ping_index=CRIMAC_PING_INDEX,
    )

    ds = ep.calibrate.compute_TS_spectrum(
        ts_echodata,
        waveform_mode="FM",
        encode_mode="complex",
        point_locations=point_locations,
        n_f_points=ts_ref["f_m"].size,
    )

    ts = (
        ds["TS_spectrum"]
        .sel(channel=CRIMAC_CHANNEL)
        .isel(target_id=0)
        .values
    )

    assert ts.shape == ts_ref["TS_m"].shape
    np.testing.assert_allclose(ts, ts_ref["TS_m"], atol=0.5, rtol=0.0)


def test_compute_Sv_spectrum_not_implemented(ts_echodata):
    """
    Test that compute_Sv_spectrum raises NotImplementedError
    for EK80 broadband complex data, since this is not currently implemented.
    """
    with pytest.raises(NotImplementedError):
        ep.calibrate.compute_Sv_spectrum(
            ts_echodata,
            waveform_mode="FM",
            encode_mode="complex",
        )
        
@pytest.mark.parametrize("window", [None, "boxcar", "hann", "hamming", ("tukey", 0.25)])
def test_compute_ts_spectrum_accepts_scipy_windows(ts_echodata, ts_ref, window):
    point_locations = _target_locations_from_crimac(
        ed=ts_echodata,
        ref=ts_ref,
        channel=CRIMAC_CHANNEL,
        ping_index=CRIMAC_PING_INDEX,
    )

    ds = ep.calibrate.compute_TS_spectrum(
        ts_echodata,
        waveform_mode="FM",
        encode_mode="complex",
        point_locations=point_locations,
        n_f_points=ts_ref["f_m"].size,
        window=window,
    )

    assert "TS_spectrum" in ds
    assert np.isfinite(ds["TS_spectrum"]).any()
    
def test_compute_ts_spectrum_none_window_matches_boxcar(ts_echodata, ts_ref):
    point_locations = _target_locations_from_crimac(
        ed=ts_echodata,
        ref=ts_ref,
        channel=CRIMAC_CHANNEL,
        ping_index=CRIMAC_PING_INDEX,
    )

    kwargs = dict(
        echodata=ts_echodata,
        waveform_mode="FM",
        encode_mode="complex",
        point_locations=point_locations,
        n_f_points=ts_ref["f_m"].size,
    )

    ds_none = ep.calibrate.compute_TS_spectrum(**kwargs, window=None)
    ds_boxcar = ep.calibrate.compute_TS_spectrum(**kwargs, window="boxcar")

    xr.testing.assert_allclose(ds_none["TS_spectrum"], ds_boxcar["TS_spectrum"])
    
    
def test_compute_ts_spectrum_explicit_range_ignores_split_front(ts_echodata, ts_ref):
    point_locations = _target_locations_from_crimac(
        ed=ts_echodata,
        ref=ts_ref,
        channel=CRIMAC_CHANNEL,
        ping_index=CRIMAC_PING_INDEX,
    )

    kwargs = dict(
        echodata=ts_echodata,
        waveform_mode="FM",
        encode_mode="complex",
        point_locations=point_locations,
        n_f_points=ts_ref["f_m"].size,
    )

    ds_025 = ep.calibrate.compute_TS_spectrum(**kwargs, split_front=0.25)
    ds_075 = ep.calibrate.compute_TS_spectrum(**kwargs, split_front=0.75)

    xr.testing.assert_allclose(ds_025["TS_spectrum"], ds_075["TS_spectrum"])
    
def test_compute_ts_spectrum_target_range_only_uses_split_front(ts_echodata, ts_ref):
    point_locations = _target_locations_from_crimac(
        ed=ts_echodata,
        ref=ts_ref,
        channel=CRIMAC_CHANNEL,
        ping_index=CRIMAC_PING_INDEX,
    ).drop_vars(["target_range_min", "target_range_max"])

    kwargs = dict(
        echodata=ts_echodata,
        waveform_mode="FM",
        encode_mode="complex",
        point_locations=point_locations,
        n_f_points=ts_ref["f_m"].size,
    )

    ds_025 = ep.calibrate.compute_TS_spectrum(**kwargs, split_front=0.25)
    ds_075 = ep.calibrate.compute_TS_spectrum(**kwargs, split_front=0.75)

    assert not np.allclose(
        ds_025["TS_spectrum"].values,
        ds_075["TS_spectrum"].values,
        equal_nan=True,
    )