import pytest

import numpy as np
import xarray as xr

from echopype.calibrate.range import compute_range_EK, range_mod_TVG_EK

pytestmark = pytest.mark.unit

SOUND_SPEED = 1500.0
SAMPLE_INTERVAL = 1e-4
TRANSMIT_DURATION = 1e-3


def _mock_beam(channel, n_ping=2, n_range_sample=4, with_beam_dim=False):
    """Minimal Beam_group dataset holding the variables range computation needs."""
    dims = ["channel", "ping_time", "range_sample"]
    coords = {
        "channel": channel,
        "ping_time": np.arange(n_ping),
        "range_sample": np.arange(n_range_sample),
    }

    backscatter_r = xr.DataArray(
        np.ones((len(channel), n_ping, n_range_sample)), dims=dims, coords=coords
    )
    if with_beam_dim:
        # beam is a coordinate in EK80 beam groups, not a bare dimension
        backscatter_r = backscatter_r.expand_dims({"beam": ["1", "2"]}, axis=-1)

    # both vary by channel and ping in real data
    sample_interval = np.full((len(channel), n_ping), SAMPLE_INTERVAL)
    transmit_duration = np.full((len(channel), n_ping), TRANSMIT_DURATION)

    return xr.Dataset(
        {
            "backscatter_r": backscatter_r,
            "sample_interval": (["channel", "ping_time"], sample_interval),
            "transmit_duration_nominal": (["channel", "ping_time"], transmit_duration),
        },
        coords=coords,
    )


def _mock_tvg_inputs(transceiver_type):
    """Beam and Vendor_specific datasets plus the unmodified range."""
    channel = [f"ch{idx}" for idx in range(len(transceiver_type))]
    beam = _mock_beam(channel)
    vend = xr.Dataset(
        {"transceiver_type": ("channel", transceiver_type)},
        coords={"channel": channel},
    )
    range_meter = compute_range_EK("EK60", beam, {"sound_speed": SOUND_SPEED})
    return beam, vend, range_meter


def test_compute_range_EK_values():
    beam = _mock_beam(["ch1", "ch2"])
    echo_range = compute_range_EK("EK60", beam, {"sound_speed": SOUND_SPEED})

    expected = beam["range_sample"].data * SAMPLE_INTERVAL * SOUND_SPEED / 2
    for ch in beam["channel"].data:
        for ping in beam["ping_time"].data:
            assert np.allclose(echo_range.sel(channel=ch, ping_time=ping).data, expected)

    assert echo_range.name == "echo_range"
    assert echo_range.dims == ("channel", "ping_time", "range_sample")


def test_compute_range_EK_sound_speed_scaling():
    beam = _mock_beam(["ch1"])
    fast = compute_range_EK("EK60", beam, {"sound_speed": SOUND_SPEED})
    slow = compute_range_EK("EK60", beam, {"sound_speed": SOUND_SPEED / 2})
    assert np.allclose(slow.data * 2, fast.data)


def test_compute_range_EK_nan_backscatter():
    beam = _mock_beam(["ch1"])
    beam["backscatter_r"][dict(channel=0, ping_time=0, range_sample=slice(2, None))] = np.nan
    echo_range = compute_range_EK("EK60", beam, {"sound_speed": SOUND_SPEED})

    assert np.all(np.isnan(echo_range.isel(channel=0, ping_time=0, range_sample=slice(2, None))))
    assert not np.any(np.isnan(echo_range.isel(channel=0, ping_time=0, range_sample=slice(0, 2))))


def test_compute_range_EK_drops_beam_dim():
    beam = _mock_beam(["ch1"], with_beam_dim=True)
    echo_range = compute_range_EK("EK80", beam, {"sound_speed": SOUND_SPEED})
    assert "beam" not in echo_range.dims


def test_compute_range_EK_unsupported_sonar_model():
    beam = _mock_beam(["ch1"])
    with pytest.raises(ValueError, match="is not supported"):
        compute_range_EK("AZFP", beam, {"sound_speed": SOUND_SPEED})


def test_compute_range_EK_missing_sound_speed():
    beam = _mock_beam(["ch1"])
    with pytest.raises(RuntimeError, match="sounds_speed not included"):
        compute_range_EK("EK60", beam, {})


def test_range_mod_TVG_EK_ex60():
    beam, vend, range_meter = _mock_tvg_inputs(["GPT"])
    modified = range_mod_TVG_EK(
        "EK60", beam, vend, range_meter.copy(deep=True), xr.DataArray(SOUND_SPEED)
    )

    # Ex60 hardware: 2-sample shift at the beginning
    expected_shift = 2 * SAMPLE_INTERVAL * SOUND_SPEED / 2
    assert np.allclose(modified.data, range_meter.data - expected_shift)


def test_range_mod_TVG_EK_ex80():
    beam, vend, range_meter = _mock_tvg_inputs(["WBT", "WBT"])
    modified = range_mod_TVG_EK(
        "EK80", beam, vend, range_meter.copy(deep=True), xr.DataArray(SOUND_SPEED)
    )

    # Ex80 hardware: shift by sound_speed * transmit_duration_nominal / 4
    expected_shift = SOUND_SPEED * TRANSMIT_DURATION / 4
    assert np.allclose(modified.data, range_meter.data - expected_shift)


def test_range_mod_TVG_EK_ex80_mixed_wbt_gpt():
    beam, vend, range_meter = _mock_tvg_inputs(["GPT", "WBT"])
    modified = range_mod_TVG_EK(
        "EK80", beam, vend, range_meter.copy(deep=True), xr.DataArray(SOUND_SPEED)
    )

    ex60_shift = 2 * SAMPLE_INTERVAL * SOUND_SPEED / 2
    ex80_shift = SOUND_SPEED * TRANSMIT_DURATION / 4

    # the WBT channel gets the Ex80 correction
    assert np.allclose(
        modified.sel(channel="ch1").data, range_meter.sel(channel="ch1").data - ex80_shift
    )

    # the GPT channel gets the Ex60 correction on top of the Ex80 one already
    # applied to every channel, so both are subtracted
    assert np.allclose(
        modified.sel(channel="ch0").data,
        range_meter.sel(channel="ch0").data - ex80_shift - ex60_shift,
    )
