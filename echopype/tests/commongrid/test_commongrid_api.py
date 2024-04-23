import pytest

import numpy as np
import pandas as pd
from flox.xarray import xarray_reduce
import xarray as xr
import echopype as ep
from echopype.consolidate import add_location, add_depth
from echopype.commongrid.utils import (
    _parse_x_bin,
    _groupby_x_along_channels,
    get_distance_from_latlon,
    compute_raw_NASC
)
from echopype.tests.commongrid.conftest import get_NASC_echoview


# Utilities Tests
@pytest.mark.parametrize(
    ["x_bin", "x_label", "expected_result"],
    [
        # Success
        ("10m", "range_bin", 10.0),
        ("0.2m", "range_bin", 0.2),
        ("0.5nmi", "dist_bin", 0.5),
        # Errored
        (10, "range_bin", TypeError),
        ("10km", "range_bin", ValueError),
        ("10", "range_bin", ValueError),
        ("10m", "invalid_label", KeyError),
    ],
)
def test__parse_x_bin(x_bin, x_label, expected_result):
    if x_label == "invalid_label":
        expected_error_msg = r"x_label must be one of"
    elif isinstance(x_bin, int):
        expected_error_msg = r"must be a string"
    elif x_bin in ["10km", "10"]:
        expected_error_msg = r"must be in"

    if not isinstance(expected_result, float):
        with pytest.raises(expected_result, match=expected_error_msg):
            ep.commongrid.api._parse_x_bin(x_bin, x_label)
    else:
        assert ep.commongrid.api._parse_x_bin(x_bin, x_label) == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    ["range_var", "lat_lon"], [("depth", False), ("echo_range", False)]
)
def test__groupby_x_along_channels(request, range_var, lat_lon):
    """Testing the underlying function of compute_MVBS and compute_NASC"""
    range_bin = 20
    ping_time_bin = "20S"
    method = "map-reduce"

    flox_kwargs = {"reindex": True}

    # Retrieve the correct dataset
    if range_var == "depth":
        ds_Sv = request.getfixturevalue("ds_Sv_echo_range_regular_w_depth")
    else:
        ds_Sv = request.getfixturevalue("ds_Sv_echo_range_regular")

    # compute range interval
    echo_range_max = ds_Sv[range_var].max()
    range_interval = np.arange(0, echo_range_max + range_bin, range_bin)

    # create bin information needed for ping_time
    d_index = (
        ds_Sv["ping_time"]
        .resample(ping_time=ping_time_bin, skipna=True)
        .asfreq()
        .indexes["ping_time"]
    )
    ping_interval = d_index.union([d_index[-1] + pd.Timedelta(ping_time_bin)])
    
    sv_mean = _groupby_x_along_channels(
        ds_Sv,
        range_interval,
        x_interval=ping_interval,
        x_var="ping_time",
        range_var=range_var,
        method=method,
        **flox_kwargs
    )

    # Check that the range_var is in the dimension
    assert f"{range_var}_bins" in sv_mean.dims


# NASC Tests
@pytest.mark.integration
@pytest.mark.parametrize("compute_mvbs", [True, False])
def test_compute_NASC(request, test_data_samples, compute_mvbs):
    if any(request.node.callspec.id.startswith(id) for id in ["ek80", "azfp"]):
        pytest.skip("Skipping NASC test for ek80 and azfp, no data available")

    (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    ) = test_data_samples
    ed = ep.open_raw(filepath, sonar_model, azfp_xml_path)
    if ed.sonar_model.lower() == "azfp":
        avg_temperature = ed["Environment"]["temperature"].values.mean()
        env_params = {
            "temperature": avg_temperature,
            "salinity": 27.9,
            "pressure": 59,
        }
        range_kwargs["env_params"] = env_params
        if "azfp_cal_type" in range_kwargs:
            range_kwargs.pop("azfp_cal_type")
    ds_Sv = ep.calibrate.compute_Sv(ed, **range_kwargs)

    # Adds location and depth information
    ds_Sv = ds_Sv.pipe(add_location, ed).pipe(
        add_depth, depth_offset=ed["Platform"].water_level.values
    )

    if compute_mvbs:
        range_bin = "2m"
        ping_time_bin = "1s"

        ds_Sv = ds_Sv.pipe(
            ep.commongrid.compute_MVBS,
            range_var="depth",
            range_bin=range_bin,
            ping_time_bin=ping_time_bin,
        )

    dist_bin = "0.5nmi"
    range_bin = "10m"

    ds_NASC = ep.commongrid.compute_NASC(ds_Sv, range_bin=range_bin, dist_bin=dist_bin)
    assert ds_NASC is not None

    dist_nmi = get_distance_from_latlon(ds_Sv)

    # Check dimensions
    dist_bin = _parse_x_bin(dist_bin, "dist_bin")
    range_bin = _parse_x_bin(range_bin)
    da_NASC = ds_NASC["NASC"]
    assert da_NASC.dims == ("channel", "distance", "depth")
    assert np.all(ds_NASC["channel"].values == ds_Sv["channel"].values)
    assert da_NASC["depth"].size == np.ceil(ds_Sv["depth"].max() / range_bin)
    assert da_NASC["distance"].size == np.ceil(dist_nmi.max() / dist_bin)


@pytest.mark.unit
def test_simple_NASC_Echoview_values(mock_Sv_dataset_NASC):
    dist_interval = np.array([-5, 10])
    range_interval = np.array([1, 5])
    raw_NASC = compute_raw_NASC(
        mock_Sv_dataset_NASC,
        range_interval,
        dist_interval,
    )
    for ch_idx, _ in enumerate(raw_NASC.channel):
        NASC_echoview = get_NASC_echoview(mock_Sv_dataset_NASC, ch_idx)
        assert np.allclose(
            raw_NASC.sv.isel(channel=ch_idx)[0, 0], NASC_echoview, atol=1e-10, rtol=1e-10
        )


# MVBS Tests
@pytest.mark.integration
def test_compute_MVBS_index_binning(ds_Sv_echo_range_regular, regular_data_params):
    """Test compute_MVBS_index_binning on mock data"""

    ping_num = 3  # number of pings to average over
    range_sample_num = 7  # number of range_samples to average over
    nchan = regular_data_params["channel_len"]
    npings = regular_data_params["ping_time_len"]
    nrange_samples = regular_data_params["depth_len"]

    # Binned MVBS test
    ds_MVBS = ep.commongrid.compute_MVBS_index_binning(
        ds_Sv_echo_range_regular, range_sample_num=range_sample_num, ping_num=ping_num
    )

    # Shape test
    data_binned_shape = np.ceil(
        (nchan, npings / ping_num, nrange_samples / range_sample_num)
    ).astype(int)
    assert np.all(ds_MVBS.Sv.shape == data_binned_shape)

    # Expected values compute
    # average should be done in linear domain
    da_sv = 10 ** (ds_Sv_echo_range_regular["Sv"] / 10)
    expected = 10 * np.log10(
        da_sv.coarsen(ping_time=ping_num, range_sample=range_sample_num, boundary="pad").mean(
            skipna=True
        )
    )

    # Test all values in MVBS
    assert np.array_equal(ds_MVBS.Sv.data, expected.data)


@pytest.mark.unit
@pytest.mark.parametrize(
    ["range_bin", "ping_time_bin"], [(5, "10S"), ("10m", 10), ("10km", "10S"), ("10", "10S")]
)
def test_compute_MVBS_bin_inputs_fail(ds_Sv_echo_range_regular, range_bin, ping_time_bin):
    expected_error = ValueError
    if isinstance(range_bin, int) or isinstance(ping_time_bin, int):
        expected_error = TypeError
        match = r"must be a string"
    else:
        match = r"Range bin must be in meters"

    with pytest.raises(expected_error, match=match):
        ep.commongrid.compute_MVBS(
            ds_Sv_echo_range_regular, range_bin=range_bin, ping_time_bin=ping_time_bin
        )


@pytest.mark.unit
def test_compute_MVBS_w_latlon(ds_Sv_echo_range_regular_w_latlon, lat_attrs, lon_attrs):
    """Testing for compute_MVBS with latitude and longitude"""
    from echopype.consolidate.api import POSITION_VARIABLES

    ds_MVBS = ep.commongrid.compute_MVBS(
        ds_Sv_echo_range_regular_w_latlon, range_bin="5m", ping_time_bin="10S"
    )
    for var in POSITION_VARIABLES:
        # Check to ensure variable is in dataset
        assert var in ds_MVBS.data_vars
        # Check for correct shape, which is just ping time
        assert ds_MVBS[var].shape == ds_MVBS.ping_time.shape

        # Check if attributes match
        if var == "latitude":
            assert ds_MVBS[var].attrs == lat_attrs
        elif var == "longitude":
            assert ds_MVBS[var].attrs == lon_attrs


@pytest.mark.unit
@pytest.mark.parametrize("range_var", ["my_range", "echo_range", "depth"])
def test_compute_MVBS_invalid_range_var(ds_Sv_echo_range_regular, range_var):
    """Test compute MVBS range_var on mock data"""

    if range_var == "my_range":
        with pytest.raises(ValueError, match="range_var must be one of 'echo_range' or 'depth'."):
            ep.commongrid.compute_MVBS(ds_Sv_echo_range_regular, range_var=range_var)
    elif range_var == "depth":
        with pytest.raises(
            ValueError, match=r"Input Sv dataset must contain all of the following variables"
        ):
            ep.commongrid.compute_MVBS(ds_Sv_echo_range_regular, range_var=range_var)
    else:
        pass


@pytest.mark.integration
def test_compute_MVBS(test_data_samples):
    """
    Test running through from open_raw to compute_MVBS.
    """
    (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    ) = test_data_samples
    ed = ep.open_raw(filepath, sonar_model, azfp_xml_path)
    if ed.sonar_model.lower() == "azfp":
        avg_temperature = ed["Environment"]["temperature"].values.mean()
        env_params = {
            "temperature": avg_temperature,
            "salinity": 27.9,
            "pressure": 59,
        }
        range_kwargs["env_params"] = env_params
        if "azfp_cal_type" in range_kwargs:
            range_kwargs.pop("azfp_cal_type")
    Sv = ep.calibrate.compute_Sv(ed, **range_kwargs)
    ping_time_bin = "20S"
    ds_MVBS = ep.commongrid.compute_MVBS(Sv, ping_time_bin=ping_time_bin)
    assert ds_MVBS is not None

    # Test to see if ping_time was resampled correctly
    expected_ping_time = (
        Sv["ping_time"].resample(ping_time=ping_time_bin, skipna=True).asfreq().indexes["ping_time"]
    )
    assert np.array_equal(ds_MVBS.ping_time.data, expected_ping_time.values)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("er_type"),
    [
        ("regular"),
        ("irregular"),
    ],
)
def test_compute_MVBS_range_output(request, er_type):
    """
    Tests the shape of compute_MVBS output on regular and irregular data.
    The irregularity in the input echo_range would cause some rows or columns
    of the output Sv to contain NaN.
    Here we test for the expected shape after dropping the NaNs
    for specific ping_time bins.
    """
    # set jitter=0 to get predictable number of ping within each echo_range groups
    if er_type == "regular":
        ds_Sv = request.getfixturevalue("ds_Sv_echo_range_regular")
    else:
        ds_Sv = request.getfixturevalue("ds_Sv_echo_range_irregular")

    ds_MVBS = ep.commongrid.compute_MVBS(ds_Sv, range_bin="5m", ping_time_bin="10S")

    if er_type == "regular":
        expected_len = (
            ds_Sv["channel"].size,  # channel
            np.ceil(np.diff(ds_Sv["ping_time"][[0, -1]].astype(int)) / 1e9 / 10),  # ping_time
            np.ceil(ds_Sv["echo_range"].max() / 5),  # depth
        )
        assert ds_MVBS["Sv"].shape == expected_len
    else:
        assert (ds_MVBS["Sv"].isel(ping_time=slice(None, 3)).dropna(dim="echo_range").shape) == (
            2,
            3,
            10,
        )  # full array, no NaN
        assert (ds_MVBS["Sv"].isel(ping_time=slice(3, 12)).dropna(dim="echo_range").shape) == (
            2,
            9,
            7,
        )  # bottom bins contain NaN
        assert (ds_MVBS["Sv"].isel(ping_time=slice(12, None)).dropna(dim="echo_range").shape) == (
            2,
            6,
            3,
        )  # bottom bins contain NaN


@pytest.mark.integration
@pytest.mark.parametrize(
    ("er_type"),
    [
        ("regular"),
        ("irregular"),
    ],
)
def test_compute_MVBS_values(request, er_type):
    """Tests for the values of compute_MVBS on regular and irregular data."""

    def _parse_nans(mvbs, ds_Sv) -> np.ndarray:
        """Go through and figure out nan values in result"""
        echo_range_step = np.unique(np.diff(mvbs.Sv.echo_range.values))[0]
        expected_outs = []
        # Loop over channels
        for chan in mvbs.Sv.channel.values:
            # Get ping times for this channel
            ping_times = mvbs.Sv.ping_time.values
            # Compute the total number of pings
            ping_len = len(ping_times)
            # Variable to store the expected output for this channel
            chan_expected = []
            for idx in range(ping_len):
                # Loop over pings and create slices
                if idx < ping_len - 1:
                    ping_slice = slice(ping_times[idx], ping_times[idx + 1])
                else:
                    ping_slice = slice(ping_times[idx], None)

                # Get the original echo_range data for this channel and ping slice
                da = ds_Sv.echo_range.sel(channel=chan, ping_time=ping_slice)
                # Drop the nan values since this shouldn't be included in actual
                # computation for compute_MVBS, a.k.a. 'nanmean'
                mean_values = da.dropna(dim="ping_time", how="all").values
                # Compute the histogram of the mean values to get distribution
                hist, _ = np.histogram(
                    mean_values[~np.isnan(mean_values)],
                    bins=np.append(
                        mvbs.Sv.echo_range.values,
                        # Add one more bin to account for the last value
                        mvbs.Sv.echo_range.values.max() + echo_range_step,
                    ),
                )
                # Convert any non-zero values to False, and zero values to True
                # to imitate having nan values since there's no value for that bin
                chan_expected.append([False if v > 0 else True for v in hist])
            expected_outs.append(chan_expected)
        return np.array(expected_outs)

    range_bin = "2m"
    ping_time_bin = "1s"

    if er_type == "regular":
        ds_Sv = request.getfixturevalue("mock_Sv_dataset_regular")
        expected_mvbs = request.getfixturevalue("mock_mvbs_array_regular")
    else:
        # Mock irregular dataset contains jitter
        # and NaN values in the bottom echo_range
        ds_Sv = request.getfixturevalue("mock_Sv_dataset_irregular")
        expected_mvbs = request.getfixturevalue("mock_mvbs_array_irregular")

    ds_MVBS = ep.commongrid.compute_MVBS(ds_Sv, range_bin=range_bin, ping_time_bin=ping_time_bin)

    expected_outputs = _parse_nans(ds_MVBS, ds_Sv)

    assert ds_MVBS.Sv.shape == expected_mvbs.shape
    # Floating digits need to check with all close not equal
    # Compare the values of the MVBS array with the expected values
    assert np.allclose(ds_MVBS.Sv.values, expected_mvbs, atol=1e-10, rtol=1e-10, equal_nan=True)

    # Ensures that the computation of MVBS takes doesn't take into account NaN values
    # that are sporadically placed in the echo_range values
    assert np.array_equal(np.isnan(ds_MVBS.Sv.values), expected_outputs)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("er_type"),
    [
        ("regular"),
        ("irregular"),
    ],
)
def test_compute_NASC_values(request, er_type):
    """Tests for the values of compute_NASC on regular and irregular data."""

    range_bin = "2m"
    dist_bin = "0.5nmi"

    if er_type == "regular":
        ds_Sv = request.getfixturevalue("mock_Sv_dataset_regular")
        expected_nasc = request.getfixturevalue("mock_nasc_array_regular")
    else:
        # Mock irregular dataset contains jitter
        # and NaN values in the bottom echo_range
        ds_Sv = request.getfixturevalue("mock_Sv_dataset_irregular")
        expected_nasc = request.getfixturevalue("mock_nasc_array_irregular")

    ds_NASC = ep.commongrid.compute_NASC(ds_Sv, range_bin=range_bin, dist_bin=dist_bin, skipna=True)

    assert ds_NASC.NASC.shape == expected_nasc.shape
    # Floating digits need to check with all close not equal
    # Compare the values of the MVBS array with the expected values
    assert np.allclose(
        ds_NASC.NASC.values, expected_nasc.values, atol=1e-10, rtol=1e-10, equal_nan=True
    )

@pytest.mark.integration
@pytest.mark.parametrize(
    ("operation","skipna", "range_var"),
    [
        ("MVBS", True, "depth"),
        ("MVBS", False, "depth"),
        ("MVBS", True, "echo_range"),
        ("MVBS", False, "echo_range"),
        # NASC `range_var` always defaults to `depth` so we set this as `None``.
        ("NASC", True, None),
        ("NASC", False, None),
    ],
)
def test_compute_MVBS_NASC_skipna_nan_and_non_nan_values(
    request,
    operation,
    skipna,
    range_var,
    caplog,
):
    # Create subset dataset with 2 channels, 2 ping times, and 20 range samples:

    # Get fixture for irregular Sv
    ds_Sv = request.getfixturevalue("mock_Sv_dataset_irregular")
    # Already has 2 channels and 20 range samples, so subset for only ping time
    subset_ds_Sv = ds_Sv.isel(ping_time=slice(0,2))

    # Compute MVBS / Compute NASC
    if operation == "MVBS":
        if range_var == "echo_range":
            # Turn on logger verbosity
            ep.utils.log.verbose(override=False)

        da = ep.commongrid.compute_MVBS(
            subset_ds_Sv,
            range_var=range_var,
            range_bin="2m",
            skipna=skipna
        )["Sv"]

        if range_var == "echo_range":
            # Check for appropriate warning
            aggregation_msg = (
                "Aggregation may be negatively impacted since Flox will not aggregate any "
                "```Sv``` values that have corresponding NaN coordinate values. Consider handling "
                "these values before calling your intended commongrid function."
            )
            expected_warning = f"The ```echo_range``` coordinate array contain NaNs. {aggregation_msg}"
            assert any(expected_warning in record.message for record in caplog.records)

            # Turn off logger verbosity
            ep.utils.log.verbose(override=True)
    else:
        da = ep.commongrid.compute_NASC(subset_ds_Sv, range_bin="2m", skipna=skipna)["NASC"]

    # Create NaN Mask: True if NaN, False if Not
    da_nan_mask = np.isnan(da.data)

    # Check for appropriate NaN/non-NaN values:

    if range_var == "echo_range":
        # Note that ALL `Sv` values that are `NaN` have corresponding `echo_range`
        # values that are also ALL `NaN``. Flox does not bin any values that have `NaN`
        # coordinates, and so none of the values that are aggregated into the 5 bins
        # have any `NaNs` that are aggregated into them.
        expected_values = [
            [[False, False, False, False, False]],
            [[False, False, False, False, False]]
        ]
        assert np.array_equal(da_nan_mask, np.array(expected_values))
    else:
        # Note that the first value along the depth dimension is always NaN due to the minimum
        # depth value in the Sv dataset being 2.5 (2.5m from the added offset), since both MVBS and
        # NASC outputs start at depth=0 and so the first depth bin (0m-2m) doesn't contain anything.
        if skipna:
            expected_values = [
                [[True, False, False, False, False, False]],
                [[True, False, False, False, False, False]]
            ]
            assert np.array_equal(da_nan_mask, np.array(expected_values))
        else:
            expected_values = [
                [[True, True, True, False, False, True]],
                [[True, False, False, True, True, True]]
            ]
            assert np.array_equal(da_nan_mask, np.array(expected_values))
