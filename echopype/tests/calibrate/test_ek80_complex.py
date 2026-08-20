import pytest
import numpy as np
import xarray as xr

from echopype.calibrate.ek80_complex import (
    get_vend_filter_EK80,
    _get_average_signal,
    _compute_power_from_complex_signal,
    _compute_ts_spectrum_power,
    _align_autocorrelation,
    _compute_ts_spectrum,
    _compute_ts_spectrum_calibrated,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def ek80_path(test_path):
    return test_path['EK80']


def gen_mock_vend(ch_num, filter_len=10, has_nan=False):
    vend = xr.Dataset(
        data_vars={
            "WBT_coeffs_real": (["channel", "WBT_filter_n"], np.random.rand(ch_num, filter_len)),
            "WBT_coeffs_imag": (["channel", "WBT_filter_n"], np.random.rand(ch_num, filter_len)),
            "WBT_deci_fac": 6,
            "PC_coeffs_real": (["channel", "PC_filter_n"], np.random.rand(ch_num, filter_len*2)),
            "PC_coeffs_imag": (["channel", "PC_filter_n"], np.random.rand(ch_num, filter_len*2)),
            "PC_deci_fac": 1,
        },
        coords={
            "channel": [f"ch_{ch}" for ch in np.arange(ch_num)],
            "WBT_filter_n": np.arange(filter_len),
            "PC_filter_n": np.arange(filter_len*2),
        }
    )
    if has_nan:  # replace some parts of filter coeff with NaN
        if filter_len != 1:
            vend["WBT_coeffs_real"].data[:, int(filter_len/2):] = np.nan
            vend["WBT_coeffs_imag"].data[:, int(filter_len/2):] = np.nan
            vend["PC_coeffs_real"].data[:, filter_len:] = np.nan
            vend["PC_coeffs_imag"].data[:, filter_len:] = np.nan
        else:
            raise ValueError("Cannot replace some parts of filter coeff with NaN")
    return vend


@pytest.mark.parametrize(
    ("ch_num", "filter_len", "has_nan"),
    [
        # filter coeff are of the same length for all channels
        (2, 10, False),
        # filter coeff are of different lengths across channels, so some parts are NaN-padded
        (2, 10, True),
        # filter coeff is of length=1
        (2, 1, False),
    ],
    ids=[
        "filter_coeff_filled",
        "filter_coeff_has_nan",
        "filter_coeff_len_1",
    ]
)
def test_get_vend_filter_EK80(ch_num, filter_len, has_nan):
    vend = gen_mock_vend(ch_num, filter_len, has_nan)
    
    for ch in [f"ch_{ch}" for ch in np.arange(ch_num)]:
        for filter_name in ["WBT", "PC"]:
            var_imag = f"{filter_name}_coeffs_imag"
            var_real = f"{filter_name}_coeffs_real"
            var_df = f"{filter_name}_deci_fac"
            sel_vend = vend.sel(channel=ch)

            assert np.all(
                (sel_vend[var_real] + 1j * sel_vend[var_imag]).dropna(dim=f"{filter_name}_filter_n").values  # noqa: E501
                == get_vend_filter_EK80(vend, channel_id=ch, filter_name=filter_name, param_type="coeff")  # noqa: E501
            )

            assert sel_vend[var_df].values == get_vend_filter_EK80(
                vend, channel_id=ch, filter_name=filter_name, param_type="decimation"
            )

def test_get_average_signal():
    signal = xr.DataArray(
        np.array([[1 + 1j, 3 + 3j], [5 + 5j, 7 + 7j]]),
        dims=("range_sample", "beam"),
        coords={"beam": [0, 1]},
    )

    out = _get_average_signal(signal)
    expected = signal.mean(dim="beam")

    xr.testing.assert_allclose(out, expected)
    assert out.name == "average_signal"


def test_compute_power_from_complex_signal():
    signal = xr.DataArray(
        np.array([[1 + 1j, 3 + 3j]]),
        dims=("range_sample", "beam"),
        coords={"beam": [0, 1]},
    )

    z_et = 75.0
    z_er = 5400.0
    n_beams = signal["beam"].size
    avg = signal.mean(dim="beam")

    expected = (
        n_beams
        * np.abs(avg) ** 2
        / (2 * np.sqrt(2)) ** 2
        * (np.abs(z_er + z_et) / np.abs(z_er)) ** 2
        / np.abs(z_et)
    )

    out = _compute_power_from_complex_signal(signal, z_et=z_et, z_er=z_er)

    xr.testing.assert_allclose(out, expected)
    assert out.name == "received_power"


def test_align_autocorrelation():
    mf_auto = np.array([0, 0, 1, 0.5, 0.25, 0.1])
    pc_target = np.array([0.2, 1.0, 0.3])

    out = _align_autocorrelation(mf_auto=mf_auto, pc_target=pc_target)

    np.testing.assert_allclose(out, np.array([0, 1, 0.5]))


def test_compute_ts_spectrum():
    pc_target = np.array([1.0, 2.0, 1.0])
    mf_auto_red = np.array([1.0, 1.0, 1.0])
    frequency = np.array([0.0, 1.0, 2.0])
    fs_dec = 8.0

    y_pc, y_mf, y_norm = _compute_ts_spectrum(
        pc_target=pc_target,
        mf_auto_red=mf_auto_red,
        NFFT=8,
        frequency=frequency,
        fs_dec=fs_dec,
    )

    assert y_pc.shape == frequency.shape
    assert y_mf.shape == frequency.shape
    assert y_norm.shape == frequency.shape
    np.testing.assert_allclose(y_norm, y_pc / y_mf)


def test_compute_ts_spectrum_calibrated():
    power_spectrum = np.array([1e-12, 2e-12])
    frequency = np.array([90000.0, 100000.0])
    sound_speed = 1500.0
    target_range = 10.0
    absorption_f = np.array([0.03, 0.04])
    transmit_power = 1000.0
    gain_f = np.array([100.0, 120.0])

    out = _compute_ts_spectrum_calibrated(
        power_spectrum=power_spectrum,
        target_range=target_range,
        frequency=frequency,
        sound_speed=sound_speed,
        absorption_f=absorption_f,
        transmit_power=transmit_power,
        gain_f=gain_f,
    )

    wavelength = sound_speed / frequency
    expected = (
        10 * np.log10(power_spectrum)
        + 40 * np.log10(target_range)
        + 2 * absorption_f * target_range
        - 10 * np.log10(
            transmit_power * wavelength**2 * gain_f**2 / (16 * np.pi**2)
        )
    )

    np.testing.assert_allclose(out, expected)