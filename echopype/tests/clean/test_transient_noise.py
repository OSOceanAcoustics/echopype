import numpy as np
import xarray as xr
import echopype as ep
import pytest

# ---------- Fixtures

@pytest.fixture(scope="module")
def ds_small():
    """Open raw, calibrate to Sv, add depth, and take a small deterministic slice."""
    ed = ep.open_raw(
        "echopype/test_data/ek60/from_echopy/JR230-D20091215-T121917.raw",
        sonar_model="EK60",
    )
    ds_Sv = ep.calibrate.compute_Sv(ed)
    ds_Sv = ep.consolidate.add_depth(ds_Sv)

    # could return a smaller object
    return ds_Sv

# don t know if useful at the moment with code implementation
@pytest.fixture(params=[False, True], ids=["unchunked", "chunked"])
def ds_small_chunked(ds_small, request):
    """Parametrize chunking."""
    return ds_small.chunk("auto") if request.param else ds_small


# ---------- Dispatcher tests

@pytest.mark.unit
def test_dispatcher_rejects_unsupported_method(ds_small):
    with pytest.raises(ValueError, match="Unsupported transient noise removal method"):
        ep.clean.detect_transient(
            ds_small, method="not_a_method", params={}
        )


@pytest.mark.unit
@pytest.mark.parametrize("method,expected_name", [
    ("fielding", "fielding_mask_valid"),
    ("matecho",  "matecho_mask_valid"),
])
def test_dispatcher_returns_named_boolean_mask(ds_small, method, expected_name):
    params = {
        # generic minimal args that both methods accept
        "range_var": "depth",
    }
    mask = ep.clean.detect_transient(ds_small, method=method, params=params)
    assert isinstance(mask, xr.DataArray)
    assert mask.dtype == bool
    assert mask.name == expected_name
    # Dims and coords should match Sv exactly
    Sv = ds_small["Sv"]
    assert tuple(mask.dims) == tuple(Sv.dims)
    for dim in Sv.dims:
        assert Sv[dim].equals(mask[dim])


# ---------- Fielding method tests 

@pytest.mark.integration
def test_fielding_dimensions_and_determinism(ds_small_chunked):
    params = dict(
        range_var="depth",
        r0=900, r1=1000, n=30, thr=(3, 1), roff=20, jumps=5, maxts=-35, start=0,
    )
    m1 = ep.clean.detect_transient(ds_small_chunked, "fielding", params)
    m2 = ep.clean.detect_transient(ds_small_chunked, "fielding", params)
    # Dims equal Sv
    Sv = ds_small_chunked["Sv"]
    assert tuple(m1.dims) == tuple(Sv.dims)
    # Deterministic
    xr.testing.assert_identical(m1, m2)


@pytest.mark.integration
def test_fielding_invalid_inputs_raise(ds_small):
    # missing var_name
    bad_ds = ds_small.drop_vars("Sv")
    with pytest.raises(ValueError):
        ep.clean.detect_transient(bad_ds, "fielding", dict(range_var="depth"))
    # missing range_var
    bad_ds2 = ds_small.drop_vars("depth")
    with pytest.raises(ValueError):
        ep.clean.detect_transient(bad_ds2, "fielding", dict(range_var="depth"))


# ---------- Matecho method tests

@pytest.mark.integration
def test_matecho_dimensions_and_determinism(ds_small_chunked):
    params = dict(
        range_var="depth",
        start_depth=220,
        window_meter=450,
        window_ping=100,
        percentile=25,
        delta_db=12,
        extend_ping=0,
        min_window=20,
    )
    m1 = ep.clean.detect_transient(ds_small_chunked, "matecho", params)
    m2 = ep.clean.detect_transient(ds_small_chunked, "matecho", params)
    Sv = ds_small_chunked["Sv"]
    assert tuple(m1.dims) == tuple(Sv.dims)
    xr.testing.assert_identical(m1, m2)


@pytest.mark.integration
def test_matecho_threshold_monotonicity(ds_small):
    """
    Increasing delta_db should make the detector more permissive:
    i.e., fewer columns flagged as transient → more True (valid) in the mask.
    """
    base_params = dict(
        range_var="depth",
        start_depth=220,
        window_meter=450,
        window_ping=100,
        percentile=25,
        extend_ping=0,
        min_window=20,
    )
    m_low  = ep.clean.detect_transient(ds_small, "matecho", dict(delta_db=8,  **base_params))
    m_high = ep.clean.detect_transient(ds_small, "matecho", dict(delta_db=16, **base_params))

    # Masks are True=VALID. A higher threshold should yield >= number of True.
    valid_low  = np.count_nonzero(m_low.values)
    valid_high = np.count_nonzero(m_high.values)
    assert valid_high >= valid_low


@pytest.mark.integration
def test_matecho_bottom_var_optional(ds_small):
    """
    Matecho accepts missing/NaN bottom and should still run.
    If a (shallow) bottom is provided, it must not crash and must keep dims.
    """
    # Run without bottom (default path in your wrapper)
    m_nob = ep.clean.detect_transient(
        ds_small, "matecho",
        dict(range_var="depth", start_depth=220, window_meter=450, window_ping=50, delta_db=12)
    )
    Sv = ds_small["Sv"]
    assert tuple(m_nob.dims) == tuple(Sv.dims)

    # Provide a synthetic shallow bottom to exercise the code path
    shallow = xr.DataArray(
        np.full(ds_small.dims["ping_time"], 300.0),
        dims=["ping_time"],
        coords=[ds_small["ping_time"]],
        name="bottom",
    )
    ds_btm = ds_small.assign(bottom_var=shallow)
    
    # Your current wrapper ignores bottom_var argument; here we at least ensure it doesn't explode
    m_btm = ep.clean.detect_transient(
        ds_btm, "matecho",
        dict(range_var="depth", start_depth=220, window_meter=450, window_ping=50, delta_db=12)
    )
    assert tuple(m_btm.dims) == tuple(Sv.dims)


# ---------- Cross-method consistency

@pytest.mark.integration
def test_methods_return_boolean_and_same_shape(ds_small_chunked):
    params_fielding = dict(range_var="depth")
    params_matecho  = dict(range_var="depth")
    mf = ep.clean.detect_transient(ds_small_chunked, "fielding", params_fielding)
    mm = ep.clean.detect_transient(ds_small_chunked, "matecho",  params_matecho)
    Sv = ds_small_chunked["Sv"]
    assert mf.dtype == bool and mm.dtype == bool
    assert tuple(mf.dims) == tuple(Sv.dims) == tuple(mm.dims)
