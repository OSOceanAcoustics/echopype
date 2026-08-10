import numpy as np
import echopype as ep
import xarray as xr
import pickle
import pytest

pytestmark = pytest.mark.integration

@pytest.fixture(scope="module")
def ek80_path(test_path):
    return test_path["EK80"]


@pytest.fixture(scope="module")
def ek80_ext_path(test_path):
    return test_path["EK80_EXT"]


# --- Comparison settings
_SKIP_RS = 50

_SV_ATOL_DB = {
    ("power", "matecho"): 5e-3,
    ("power", "pyecholab"): 5.5e-3,
    ("complex", "matecho"): 5.5e-3,
    ("complex", "echoview"): 2e-3,
}

_TAU_RTOL = {
    ("power", "matecho"): 1.15e-3,
    ("power", "pyecholab"): 1.25e-3,
    ("complex", "matecho"): 1.25e-3,
}

# --- DATASET FIXTURES

@pytest.fixture(scope="module")
def cw_power_ds(ek80_path):
    raw = ek80_path / "ncei-wcsd/SH2306/Hake-D20230811-T165727.raw"
    ed = ep.open_raw(raw, sonar_model="EK80")
    return ep.calibrate.compute_Sv(
        ed,
        waveform_mode="CW",
        encode_mode="power",
    )


@pytest.fixture(scope="module")
def cw_complex_ds(ek80_path):
    zarr_path = ek80_path / "RV_Svea/ci_subset_D20230626-T235835.zarr"
    ed = ep.open_converted(zarr_path, chunks={})
    return ep.calibrate.compute_Sv(
        ed,
        waveform_mode="CW",
        encode_mode="complex",
    )


# --- Helpers POWER references


def _finite_mask(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a)


def _assert_svs_close(
    ep_sv: xr.DataArray,
    ref_sv: xr.DataArray,
    rtol: float = 0.0,
    atol_db: float = 1e-2,
    skip_range_samples: int = 0,
):

    assert ep_sv.dims == ref_sv.dims
    for d in ep_sv.dims:
        assert ep_sv.sizes[d] == ref_sv.sizes[d]

    ep_vals = np.asarray(ep_sv.data)
    ref_vals = np.asarray(ref_sv.data)

    if skip_range_samples > 0:
        ep_vals = ep_vals[..., skip_range_samples:]
        ref_vals = ref_vals[..., skip_range_samples:]

    m = _finite_mask(ep_vals) & _finite_mask(ref_vals)
    assert m.any(), "No overlapping finite Sv values to compare."

    assert np.allclose(ep_vals[m], ref_vals[m], rtol=rtol, atol=atol_db)


def _load_power_ref(ek80_ext_path, pkg_name: str) -> dict:

    subdir, fname = {
        "matecho": ("matecho", "power_refs_Hake_matecho.pkl"),
        "pyecholab": ("pyecholab", "power_refs_Hake_pyecholab.pkl"),
    }[pkg_name]

    p = ek80_ext_path / subdir / fname

    with open(p, "rb") as f:
        return pickle.load(f)


def _tau_ep_to_freq_vector(ds: xr.Dataset) -> xr.DataArray:

    tau = ds["tau_effective"]
    freq_khz = (ds["frequency_nominal"] / 1000).astype(int)

    return (
        tau.assign_coords(freq_khz=freq_khz)
        .swap_dims({"channel": "freq_khz"})
        .sortby("freq_khz")
    )


def _ref_tau_to_da(ref: dict) -> xr.DataArray:

    return xr.DataArray(
        np.asarray(ref["tau_effective"]),
        dims=("freq_khz",),
        coords={"freq_khz": np.asarray(ref["freq_khz"], dtype=int)},
        name="tau_effective",
    )


def _pad_or_crop_2d(a, n_ping: int, n_range: int, fill=np.nan) -> np.ndarray:

    a = np.asarray(a)

    out = np.full((n_ping, n_range), fill, dtype=float)

    p = min(n_ping, a.shape[0])
    r = min(n_range, a.shape[1])

    out[:p, :r] = a[:p, :r]

    return out


def _ref_sv_to_da(ref: dict, ds_ep: xr.Dataset) -> xr.DataArray:

    sv_ep = ds_ep["Sv"]

    freq_khz = (ds_ep["frequency_nominal"] / 1000).astype(int)
    freq_khz_vals = [int(x) for x in freq_khz.values.tolist()]

    n_ping = sv_ep.sizes["ping_time"]
    n_range = sv_ep.sizes["range_sample"]

    mats = []

    for f in freq_khz_vals:
        mats.append(_pad_or_crop_2d(np.asarray(ref["Sv"][f]), n_ping, n_range))

    stacked = np.stack(mats, axis=0)

    return xr.DataArray(
        stacked,
        dims=("channel", "ping_time", "range_sample"),
        coords={
            "channel": sv_ep["channel"].values,
            "ping_time": sv_ep["ping_time"].values,
            "range_sample": sv_ep["range_sample"].values,
        },
        name="Sv",
    )


# --- POWER SAMPLES CW

@pytest.mark.parametrize(
    "pkg_name, tau_rtol",
    [
        ("matecho", _TAU_RTOL[("power", "matecho")]),
        ("pyecholab", _TAU_RTOL[("power", "pyecholab")]),
    ],
)
def test_ek80_CW_power_tau_effective_matches_reference(
    cw_power_ds, ek80_ext_path, pkg_name, tau_rtol
):
    """Check that tau_effective from echopype CW power calibration
    matches the package reference values."""

    ds = cw_power_ds
    tau_ep = _tau_ep_to_freq_vector(ds)

    ref = _load_power_ref(ek80_ext_path, pkg_name)
    tau_ref = _ref_tau_to_da(ref)

    assert np.array_equal(tau_ep["freq_khz"].values, tau_ref["freq_khz"].values)

    np.testing.assert_allclose(
        tau_ep.data,
        tau_ref.data,
        rtol=tau_rtol,
        atol=0.0,
    )

@pytest.mark.parametrize(
    "pkg_name, sv_atol_db",
    [
        ("matecho", _SV_ATOL_DB[("power", "matecho")]),
        ("pyecholab", _SV_ATOL_DB[("power", "pyecholab")]),
    ],
)
def test_ek80_CW_power_Sv_matches_reference(
    cw_power_ds, ek80_ext_path, pkg_name, sv_atol_db
):
    """Verify that calibrated Sv from echopype CW power mode
    matches the package reference outputs."""

    ds = cw_power_ds
    sv_ep = ds["Sv"]

    ref = _load_power_ref(ek80_ext_path, pkg_name)
    sv_ref = _ref_sv_to_da(ref, ds)

    _assert_svs_close(
        sv_ep,
        sv_ref,
        rtol=0.0,
        atol_db=sv_atol_db,
        skip_range_samples=_SKIP_RS,
    )

# --- COMPLEX

def _load_complex_ref(ek80_ext_path, pkg_name: str) -> dict:

    subdir, fname = {
        "matecho": ("matecho", "complex_refs_RV-Svea_matecho_38kHz.pkl"),
        "echoview": ("echoview", "complex_refs_RV-Svea_echoview_38kHz.pkl"),
    }[pkg_name]

    p = ek80_ext_path / subdir / fname

    with open(p, "rb") as f:
        return pickle.load(f)


def _select_ep_complex_38(ds: xr.Dataset, n_range: int | None = None) -> xr.Dataset:

    channel = ds["channel"].astype(str)
    mask = channel.str.contains("ES38")

    if int(mask.sum()) == 1:
        ds38 = ds.isel(channel=mask)
    elif ds.sizes["channel"] == 1:
        ds38 = ds.isel(channel=[0])
    else:
        raise AssertionError(f"Expected one ES38 channel, got {channel.values}")

    ds38 = ds38.squeeze("channel", drop=True)

    if n_range is not None:
        ds38 = ds38.isel(range_sample=slice(0, n_range))

    return ds38


def _ref_complex_var_to_da(ref: dict, var_name: str, ds_ep_38: xr.Dataset) -> xr.DataArray:

    arr = np.asarray(ref[var_name])

    return xr.DataArray(
        arr,
        dims=("ping_time", "range_sample"),
        coords={
            "ping_time": ds_ep_38["ping_time"].values,
            "range_sample": ds_ep_38["range_sample"].values,
        },
        name=var_name,
    )


def _assert_2d_close(ep_da, ref_da, rtol=0.0, atol=1e-2, skip_range_samples=0):
    ep_vals = np.asarray(ep_da.data)
    ref_vals = np.asarray(ref_da.data)

    if skip_range_samples > 0:
        ep_vals = ep_vals[:, skip_range_samples:]
        ref_vals = ref_vals[:, skip_range_samples:]

    m = np.isfinite(ep_vals) & np.isfinite(ref_vals)
    assert m.any()

    assert np.allclose(ep_vals[m], ref_vals[m], rtol=rtol, atol=atol)

def test_ek80_CW_complex_tau_effective_matches_matecho(cw_complex_ds, ek80_ext_path):
    """Check that tau_effective from echopype CW complex
    calibration matches Matecho reference."""

    ds = cw_complex_ds

    ds38 = _select_ep_complex_38(ds)

    tau_ep = float(ds38["tau_effective"].values)

    ref = _load_complex_ref(ek80_ext_path, "matecho")
    tau_ref = float(ref["tau_effective"])

    np.testing.assert_allclose(
        tau_ep,
        tau_ref,
        rtol=_TAU_RTOL[("complex", "matecho")],
        atol=0.0,
    )


@pytest.mark.parametrize(
    "pkg_name, sv_atol_db",
    [
        ("matecho", _SV_ATOL_DB[("complex", "matecho")]),
        ("echoview", _SV_ATOL_DB[("complex", "echoview")]),
    ],
)
def test_ek80_CW_complex_Sv_matches_reference(
    cw_complex_ds, ek80_ext_path, pkg_name, sv_atol_db
):
    """Verify that Sv from echopype CW complex mode
    agrees with the package reference Sv."""

    ds = cw_complex_ds

    ref = _load_complex_ref(ek80_ext_path, pkg_name)
    n_range = ref["Sv"].shape[1]

    ds38 = _select_ep_complex_38(ds, n_range=n_range)

    sv_ep = ds38["Sv"]
    sv_ref = _ref_complex_var_to_da(ref, "Sv", ds38)

    _assert_2d_close(
        sv_ep,
        sv_ref,
        rtol=0.0,
        atol=sv_atol_db,
        skip_range_samples=_SKIP_RS,
    )