from typing import Literal

import pytest

import numpy as np
import pandas as pd
import xarray as xr

import echopype as ep
from echopype.consolidate import add_depth
from echopype.commongrid.utils import (
    get_distance_from_latlon,
)
from echopype.testing import (
    _gen_Sv_echo_range_regular,
    _gen_Sv_echo_range_irregular,
    _get_expected_mvbs_val,
)



@pytest.fixture
def random_number_generator():
    """Random number generator for tests"""
    return np.random.default_rng()


@pytest.fixture
def mock_nan_ilocs():
    """NaN i locations for irregular Sv dataset

    It's a list of tuples, each tuple contains
    (channel, ping_time, range_sample)

    Notes
    -----
    This was created with the following code:

    ```
    import numpy as np

    random_positions = []
    for i in range(20):
        random_positions.append((
            np.random.randint(0, 2),
            np.random.randint(0, 5),
            np.random.randint(0, 20))
        )
    ```
    """
    return [
        (1, 1, 10),
        (1, 0, 16),
        (0, 3, 6),
        (0, 2, 11),
        (0, 2, 6),
        (1, 1, 14),
        (0, 1, 17),
        (1, 4, 19),
        (0, 3, 3),
        (0, 0, 19),
        (0, 1, 5),
        (1, 2, 9),
        (1, 4, 18),
        (0, 1, 5),
        (0, 4, 4),
        (0, 1, 6),
        (1, 2, 2),
        (0, 1, 2),
        (0, 4, 8),
        (0, 1, 1),
    ]


@pytest.fixture
def mock_parameters():
    """Small mock parameters"""
    return {
        "channel_len": 2,
        "ping_time_len": 10,
        "depth_len": 20,
        "ping_time_interval": "0.3S",
    }


@pytest.fixture
def mock_Sv_sample(mock_parameters):
    """
    Mock Sv sample

    Dimension: (2, 10, 20)
    """
    channel_len = mock_parameters["channel_len"]
    ping_time_len = mock_parameters["ping_time_len"]
    depth_len = mock_parameters["depth_len"]

    depth_data = np.linspace(0, 1, num=depth_len)
    return np.tile(depth_data, (channel_len, ping_time_len, 1))


def _add_latlon_depth(ds_Sv, latlon=False, depth=False, lat_attrs={}, lon_attrs={}, depth_offset=0):
    """Adds lat lon variables and/or depth to ds_Sv"""
    if latlon:
        # Add lat lon
        n_pings = ds_Sv.ping_time.shape[0]
        latitude = np.linspace(42.48916859, 42.49071833, num=n_pings)
        longitude = np.linspace(-124.88296688, -124.81919229, num=n_pings)

        ds_Sv["latitude"] = (["ping_time"], latitude, lat_attrs)
        ds_Sv["longitude"] = (["ping_time"], longitude, lon_attrs)

        # Need processing level code for compute MVBS to work!
        ds_Sv.attrs["processing_level"] = "Level 2A"

    if depth:
        # Add depth
        ds_Sv = ds_Sv.pipe(add_depth, depth_offset=depth_offset)
    return ds_Sv


@pytest.fixture
def mock_Sv_dataset_regular(mock_parameters, mock_Sv_sample, lat_attrs, lon_attrs, depth_offset):
    ds_Sv = _gen_Sv_echo_range_regular(**mock_parameters, ping_time_jitter_max_ms=0)
    ds_Sv["Sv"].data = mock_Sv_sample

    # Add latlon and depth
    ds_Sv = _add_latlon_depth(
        ds_Sv,
        latlon=True,
        depth=True,
        lat_attrs=lat_attrs,
        lon_attrs=lon_attrs,
        depth_offset=depth_offset,
    )
    return ds_Sv


@pytest.fixture
def mock_Sv_dataset_irregular(
    mock_parameters, mock_Sv_sample, mock_nan_ilocs, lat_attrs, lon_attrs, depth_offset
):
    depth_interval = [0.5, 0.32, 0.2]
    depth_ping_time_len = [2, 3, 5]
    ds_Sv = _gen_Sv_echo_range_irregular(
        **mock_parameters,
        depth_interval=depth_interval,
        depth_ping_time_len=depth_ping_time_len,
        ping_time_jitter_max_ms=30,  # Added jitter to ping_time
    )
    ds_Sv["Sv"].data = mock_Sv_sample

    # Add latlon and depth
    ds_Sv = _add_latlon_depth(
        ds_Sv,
        latlon=True,
        depth=True,
        lat_attrs=lat_attrs,
        lon_attrs=lon_attrs,
        depth_offset=depth_offset,
    )

    # Sprinkle nans around echo_range
    for pos in mock_nan_ilocs:
        ds_Sv["echo_range"][pos] = np.nan
        ds_Sv["Sv"][pos] = np.nan
    return ds_Sv


@pytest.fixture
def mock_mvbs_inputs():
    return dict(range_meter_bin=2, ping_time_bin="1s")


@pytest.fixture
def mock_mvbs_array_regular(mock_Sv_dataset_regular, mock_mvbs_inputs, mock_parameters):
    """
    Mock Sv sample result from compute_MVBS

    Dimension: (2, 3, 5)
    Ping time bin: 1s
    Range bin: 2m
    """
    ds_Sv = mock_Sv_dataset_regular
    ping_time_bin = mock_mvbs_inputs["ping_time_bin"]
    range_bin = mock_mvbs_inputs["range_meter_bin"]
    channel_len = mock_parameters["channel_len"]
    expected_mvbs_val = _get_expected_mvbs_val(ds_Sv, ping_time_bin, range_bin, channel_len)

    return expected_mvbs_val


@pytest.fixture
def mock_nasc_array_regular(mock_Sv_dataset_regular, mock_parameters):
    """
    Mock Sv sample result from compute_MVBS

    Dimension: (2, 3, 5)
    Ping time bin: 1s
    Range bin: 2m
    """
    ds_Sv = mock_Sv_dataset_regular
    dist_bin = 0.5
    range_bin = 2
    channel_len = mock_parameters["channel_len"]
    expected_nasc_val = _get_expected_nasc_val_nanmean(ds_Sv, dist_bin, range_bin, channel_len)

    return expected_nasc_val


@pytest.fixture
def mock_mvbs_array_irregular(mock_Sv_dataset_irregular, mock_mvbs_inputs, mock_parameters):
    """
    Mock Sv sample irregular result from compute_MVBS

    Dimension: (2, 3, 5)
    Ping time bin: 1s
    Range bin: 2m
    """
    ds_Sv = mock_Sv_dataset_irregular
    ping_time_bin = mock_mvbs_inputs["ping_time_bin"]
    range_bin = mock_mvbs_inputs["range_meter_bin"]
    channel_len = mock_parameters["channel_len"]
    expected_mvbs_val = _get_expected_mvbs_val(ds_Sv, ping_time_bin, range_bin, channel_len)

    return expected_mvbs_val


@pytest.fixture
def mock_nasc_array_irregular(mock_Sv_dataset_irregular, mock_parameters):
    """
    Mock Sv sample result from compute_MVBS

    Dimension: (2, 3, 5)
    Ping time bin: 1s
    Range bin: 2m
    """
    ds_Sv = mock_Sv_dataset_irregular
    dist_bin = 0.5
    range_bin = 2
    channel_len = mock_parameters["channel_len"]
    expected_nasc_val = _get_expected_nasc_val_nanmean(ds_Sv, dist_bin, range_bin, channel_len)

    return expected_nasc_val


@pytest.fixture(
    params=[
        (
            ("EK60", "ncei-wcsd", "Summer2017-D20170719-T211347.raw"),
            "EK60",
            None,
            {},
        ),
        (
            ("EK80_NEW", "echopype-test-D20211004-T235930.raw"),
            "EK80",
            None,
            {"waveform_mode": "BB", "encode_mode": "complex"},
        ),
        (
            ("EK80_NEW", "D20211004-T233354.raw"),
            "EK80",
            None,
            {"waveform_mode": "CW", "encode_mode": "power"},
        ),
        (
            ("EK80_NEW", "D20211004-T233115.raw"),
            "EK80",
            None,
            {"waveform_mode": "CW", "encode_mode": "complex"},
        ),
        (("ES70", "D20151202-T020259.raw"), "ES70", None, {}),
        (("AZFP", "17082117.01A"), "AZFP", ("AZFP", "17041823.XML"), {}),
        (
            ("AD2CP", "raw", "090", "rawtest.090.00001.ad2cp"),
            "AD2CP",
            None,
            {},
        ),
    ],
    ids=[
        "ek60_cw_power",
        "ek80_bb_complex",
        "ek80_cw_power",
        "ek80_cw_complex",
        "es70",
        "azfp",
        "ad2cp",
    ],
)
def test_data_samples(request, test_path):
    (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    ) = request.param
    if sonar_model.lower() in ["es70", "ad2cp"]:
        pytest.xfail(
            reason="Not supported at the moment",
        )
    path_model, *paths = filepath
    filepath = test_path[path_model].joinpath(*paths)

    if azfp_xml_path is not None:
        path_model, *paths = azfp_xml_path
        azfp_xml_path = test_path[path_model].joinpath(*paths)
    return (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    )


@pytest.fixture
def regular_data_params():
    return {
        "channel_len": 4,
        "depth_len": 4000,
        "ping_time_len": 100,
        "ping_time_jitter_max_ms": 0,
    }


@pytest.fixture
def ds_Sv_echo_range_regular(regular_data_params, random_number_generator):
    return _gen_Sv_echo_range_regular(
        **regular_data_params,
        random_number_generator=random_number_generator,
    )


@pytest.fixture
def latlon_history_attr():
    return (
        "2023-08-31 12:00:00.000000 +00:00. "
        "Interpolated or propagated from Platform latitude/longitude."  # noqa
    )


@pytest.fixture
def lat_attrs(latlon_history_attr):
    """Latitude attributes"""
    return {
        "long_name": "Platform latitude",
        "standard_name": "latitude",
        "units": "degrees_north",
        "valid_range": "(-90.0, 90.0)",
        "history": latlon_history_attr,
    }


@pytest.fixture
def lon_attrs(latlon_history_attr):
    """Longitude attributes"""
    return {
        "long_name": "Platform longitude",
        "standard_name": "longitude",
        "units": "degrees_east",
        "valid_range": "(-180.0, 180.0)",
        "history": latlon_history_attr,
    }


@pytest.fixture
def depth_offset():
    """Depth offset for calculating depth"""
    return 2.5


@pytest.fixture
def ds_Sv_echo_range_regular_w_latlon(ds_Sv_echo_range_regular, lat_attrs, lon_attrs):
    """Sv dataset with latitude and longitude"""
    ds_Sv_echo_range_regular = _add_latlon_depth(
        ds_Sv_echo_range_regular, latlon=True, lat_attrs=lat_attrs, lon_attrs=lon_attrs
    )
    return ds_Sv_echo_range_regular


@pytest.fixture
def ds_Sv_echo_range_regular_w_depth(ds_Sv_echo_range_regular, depth_offset):
    """Sv dataset with depth"""
    return ds_Sv_echo_range_regular.pipe(add_depth, depth_offset=depth_offset)


@pytest.fixture
def ds_Sv_echo_range_irregular(random_number_generator):
    depth_interval = [0.5, 0.32, 0.13]
    depth_ping_time_len = [100, 300, 200]
    ping_time_len = 600
    ping_time_interval = "0.3S"
    return _gen_Sv_echo_range_irregular(
        depth_interval=depth_interval,
        depth_ping_time_len=depth_ping_time_len,
        ping_time_len=ping_time_len,
        ping_time_interval=ping_time_interval,
        ping_time_jitter_max_ms=0,
        random_number_generator=random_number_generator,
    )


# Helper functions for NASC testing
def _create_dataset(i, sv, dim, rng):
    dims = {
        "range_sample": np.arange(5),
        "distance_nmi": np.arange(5),
    }
    # Add one for other channel
    sv = sv + (rng.random() * 5)
    Sv = ep.utils.compute._lin2log(sv)
    ds_Sv = xr.Dataset(
        {
            "Sv": (list(dims.keys()), Sv),
            "depth": (list(dims.keys()), np.array([dim] * 5).T),
            "ping_time": (
                ["distance_nmi"],
                pd.date_range("2020-01-01", periods=len(dim), freq="1min"),
            ),
        },
        coords=dict(channel=f"ch_{i}", **dims),
    )
    return ds_Sv


def get_NASC_echoview(ds_Sv, ch_idx=0, r0=2, r1=20):
    """
    Computes NASC using echoview's method, 1 channel only,
    as described in https://gist.github.com/leewujung/3b058ab63c3b897b273b33b907b62f6d
    """
    r = ds_Sv.depth.isel(channel=ch_idx, distance_nmi=0).values
    # get r0 and r1 indexes
    # these are used to slice the desired Sv samples
    r0 = np.argmin(abs(r - r0))
    r1 = np.argmin(abs(r - r1))

    sh = np.r_[np.diff(r), np.nan]

    sv = ds_Sv["Sv"].pipe(ep.utils.compute._log2lin).isel(channel=ch_idx).values
    sv_mean_echoview = np.nanmean(sv[r0:r1])
    h_mean_echoview = np.sum(sh[r0:r1]) * sv.shape[1] / sv.shape[1]

    NASC_echoview = sv_mean_echoview * h_mean_echoview * 4 * np.pi * 1852**2
    return NASC_echoview


@pytest.fixture
def mock_Sv_dataset_NASC(mock_parameters, random_number_generator):
    channel_len = mock_parameters["channel_len"]
    dim0 = np.array([0.5, 1.5, 2.5, 3.5, 9])
    sv0 = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, np.nan],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, np.nan],
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ]
    )
    return xr.concat(
        [_create_dataset(i, sv0, dim0, random_number_generator) for i in range(channel_len)],
        dim="channel",
    )


# Helper functions to generate mock Sv and MVBS dataset
def _get_expected_nasc_val_nanmean(
    ds_Sv: xr.Dataset, dist_bin: str, range_bin: float, channel_len: int = 2
) -> np.ndarray:
    """
    Helper functions to generate expected NASC outputs from mock Sv dataset
    by brute-force looping and compute the nanmean

    Parameters
    ----------
    ds_Sv : xr.Dataset
        Mock Sv dataset
    dist_bin : float
        Distance bin
    range_bin : float
        Range bin
    channel_len : int, default 2
        Number of channels
    """
    # Get distance from lat/lon in nautical miles
    dist_nmi = get_distance_from_latlon(ds_Sv)
    ds_Sv = ds_Sv.assign_coords({"distance_nmi": ("ping_time", dist_nmi)}).swap_dims(
        {"ping_time": "distance_nmi"}
    )

    # create bin information along distance_nmi
    # this computes the distance max since there might NaNs in the data
    dist_max = ds_Sv["distance_nmi"].max()
    dist_interval = np.arange(0, dist_max + dist_bin, dist_bin)

    # create bin information for depth
    # this computes the depth max since there might NaNs in the data
    depth_max = ds_Sv["depth"].max()
    range_interval = np.arange(0, depth_max + range_bin, range_bin)

    sv = ds_Sv["Sv"].pipe(ep.utils.compute._log2lin)

    # Compute sv mean
    sv_mean = _brute_nanmean_reduce_3d(
        sv, ds_Sv, "depth", "distance_nmi", channel_len, dist_interval, range_interval
    )

    # Calculate denominator
    h_mean_denom = np.ones(len(dist_interval) - 1) * np.nan
    for x_idx in range(len(dist_interval) - 1):
        x_range = ds_Sv["distance_nmi"].sel(
            distance_nmi=slice(dist_interval[x_idx], dist_interval[x_idx + 1])
        )
        h_mean_denom[x_idx] = float(len(x_range.data))

    # Calculate numerator
    r_diff = ds_Sv["depth"].diff(dim="range_sample", label="lower")
    depth = ds_Sv["depth"].isel(**{"range_sample": slice(0, -1)})
    h_mean_num = np.ones((channel_len, len(dist_interval) - 1, len(range_interval) - 1)) * np.nan
    for ch_idx in range(channel_len):
        for x_idx in range(len(dist_interval) - 1):
            for r_idx in range(len(range_interval) - 1):
                x_range = depth.isel(channel=ch_idx).sel(
                    **{"distance_nmi": slice(dist_interval[x_idx], dist_interval[x_idx + 1])}
                )
                r_idx_active = np.logical_and(
                    x_range.data >= range_interval[r_idx],
                    x_range.data < range_interval[r_idx + 1],
                )
                r_tmp = (
                    r_diff.isel(channel=ch_idx)
                    .sel(**{"distance_nmi": slice(dist_interval[x_idx], dist_interval[x_idx + 1])})
                    .data[r_idx_active]
                )
                if 0 in r_tmp.shape:
                    h_mean_num[ch_idx, x_idx, r_idx] = np.nan
                else:
                    h_mean_num[ch_idx, x_idx, r_idx] = np.sum(r_tmp)

    # Compute raw NASC
    h_mean_num_da = xr.DataArray(h_mean_num, dims=["channel", "distance_nmi", "depth"])
    h_mean_denom_da = xr.DataArray(h_mean_denom, dims=["distance_nmi"])
    h_mean = h_mean_num_da / h_mean_denom_da
    # Combine to compute NASC
    return sv_mean * h_mean * 4 * np.pi * 1852**2


def _brute_nanmean_reduce_3d(
    sv: xr.DataArray,
    ds_Sv: xr.Dataset,
    range_var: Literal["echo_range", "depth"],
    x_var: Literal["ping_time", "distance_nmi"],
    channel_len: int,
    x_interval: list,
    range_interval: list,
) -> np.ndarray:
    """
    Perform brute force reduction on sv data for 3 Dimensions

    Parameters
    ----------
    sv : xr.DataArray
        A DataArray containing ``sv`` data with coordinates
    ds_Sv : xr.Dataset
        A Dataset containing ``Sv`` and other variables,
        depending on computation performed.

        For MVBS computation, this must contain ``Sv`` and ``echo_range`` data
        with coordinates ``channel``, ``ping_time``, and ``range_sample``
        at bare minimum.
        Or this can contain ``Sv`` and ``depth`` data with similar coordinates.

        For NASC computatioon this must contain ``Sv`` and ``depth`` data
        with coordinates ``channel``, ``distance_nmi``, and ``range_sample``.
    range_var: {'echo_range', 'depth'}, default 'echo_range'
        The variable to use for range binning.
        Either ``echo_range`` or ``depth``.

        **For NASC, this must be ``depth``.**
    x_var : {'ping_time', 'distance_nmi'}, default 'ping_time'
        The variable to use for x binning. This will determine
        if computation is for MVBS or NASC.
    channel_len : int
        Number of channels
    x_interval : list
        1D array or interval index representing
        the bins required for ``ping_time`` or ``distance_nmi``.
    range_interval : list
        1D array or interval index representing
        the bins required for ``range_var``
    """
    mean_vals = np.ones((channel_len, len(x_interval) - 1, len(range_interval) - 1)) * np.nan

    for ch_idx in range(channel_len):
        for x_idx in range(len(x_interval) - 1):
            for r_idx in range(len(range_interval) - 1):
                x_range = (
                    ds_Sv[range_var]
                    .isel(channel=ch_idx)
                    .sel(**{x_var: slice(x_interval[x_idx], x_interval[x_idx + 1])})
                )
                r_idx_active = np.logical_and(
                    x_range.data >= range_interval[r_idx],
                    x_range.data < range_interval[r_idx + 1],
                )
                sv_tmp = (
                    sv.isel(channel=ch_idx)
                    .sel(**{x_var: slice(x_interval[x_idx], x_interval[x_idx + 1])})
                    .data[r_idx_active]
                )
                if 0 in sv_tmp.shape:
                    mean_vals[ch_idx, x_idx, r_idx] = np.nan
                else:
                    mean_vals[ch_idx, x_idx, r_idx] = np.nanmean(sv_tmp)
    return mean_vals


# End helper functions
