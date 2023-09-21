"""
Functions for enhancing the spatial and temporal coherence of data.
"""
import re
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from flox.xarray import xarray_reduce

from ..consolidate.api import POSITION_VARIABLES
from ..utils.compute import _log2lin
from ..utils.prov import add_processing_level, echopype_prov_attrs, insert_input_processing_level
from .mvbs import get_MVBS_along_channels
from .nasc import get_distance_from_latlon


def _set_var_attrs(da, long_name, units, round_digits, standard_name=None):
    """
    Attach common attributes to DataArray variable.

    Parameters
    ----------
    da : xr.DataArray
        DataArray that will receive attributes
    long_name : str
        Variable long_name attribute
    units : str
        Variable units attribute
    round_digits : int
        Number of digits after decimal point for rounding off actual_range
    standard_name : str
        CF standard_name, if available (optional)
    """

    da.attrs = {
        "long_name": long_name,
        "units": units,
        "actual_range": [
            round(float(da.min().values), round_digits),
            round(float(da.max().values), round_digits),
        ],
    }
    if standard_name:
        da.attrs["standard_name"] = standard_name


def _set_MVBS_attrs(ds):
    """
    Attach common attributes.

    Parameters
    ----------
    ds : xr.Dataset
        dataset containing MVBS
    """
    ds["ping_time"].attrs = {
        "long_name": "Ping time",
        "standard_name": "time",
        "axis": "T",
    }

    _set_var_attrs(
        ds["Sv"],
        long_name="Mean volume backscattering strength (MVBS, mean Sv re 1 m-1)",
        units="dB",
        round_digits=2,
    )


def _convert_bins_to_interval_index(
    bins: list, closed: Literal["left", "right"] = "left"
) -> pd.IntervalIndex:
    """
    Convert bins to sorted pandas IntervalIndex
    with specified closed end

    Parameters
    ----------
    bins : list
        The bin edges
    closed : {'left', 'right'}, default 'left'
        Which side of bin interval is closed

    Returns
    -------
    pd.IntervalIndex
        The resulting IntervalIndex
    """
    return pd.IntervalIndex.from_breaks(bins, closed=closed).sort_values()


def _parse_x_bin(x_bin: str, x_label="range_bin") -> float:
    """
    Parses x bin string, check unit,
    and returns x bin in the specified unit.

    Currently only available for:
    range_bin: meters (m)
    dist_bin: nautical miles (nmi)

    Parameters
    ----------
    x_bin : str
        X bin string, e.g., "0.5nmi" or "10m"
    x_label : {"range_bin", "dist_bin"}, default "range_bin"
        The label of the x bin.

    Returns
    -------
    float
        The resulting x bin value in x unit,
        based on label.

    Raises
    ------
    ValueError
        If the x bin string doesn't include unit value.
    TypeError
        If the x bin is not a type string.
    KeyError
        If the x label is not one of the available labels.
    """
    x_bin_map = {
        "range_bin": {
            "name": "Range bin",
            "unit": "m",
            "ex": "10m",
            "unit_label": "meters",
            "pattern": r"([\d+]*[.,]{0,1}[\d+]*)(\s+)?(m)",
        },
        "dist_bin": {
            "name": "Distance bin",
            "unit": "nmi",
            "ex": "0.5nmi",
            "unit_label": "nautical miles",
            "pattern": r"([\d+]*[.,]{0,1}[\d+]*)(\s+)?(nmi)",
        },
    }
    x_bin_info = x_bin_map.get(x_label, None)

    if x_bin_info is None:
        raise KeyError(f"x_label must be one of {list(x_bin_map.keys())}")

    # First check for bin types
    if not isinstance(x_bin, str):
        raise TypeError("'x_bin' must be a string")
    # normalize to lower case
    # for x_bin
    x_bin = x_bin.strip().lower()
    # Only matches meters
    match_obj = re.match(x_bin_info["pattern"], x_bin)

    # Do some checks on x_bin inputs
    if match_obj is None:
        # This shouldn't be other units
        raise ValueError(
            f"{x_bin_info['name']} must be in "
            f"{x_bin_info['unit_label']} "
            f"(e.g., '{x_bin_info['ex']}')."
        )

    # Convert back to float
    x_bin = float(match_obj.group(1))
    return x_bin


@add_processing_level("L3*")
def compute_MVBS(
    ds_Sv: xr.Dataset,
    range_var: Literal["echo_range", "depth"] = "echo_range",
    range_bin: str = "20m",
    ping_time_bin: str = "20S",
    method="map-reduce",
    closed: Literal["left", "right"] = "left",
    **flox_kwargs,
):
    """
    Compute Mean Volume Backscattering Strength (MVBS)
    based on intervals of range (``echo_range``) or depth (``depth``)
    and ``ping_time`` specified in physical units.

    Output of this function differs from that of ``compute_MVBS_index_binning``, which computes
    bin-averaged Sv according to intervals of ``echo_range`` and ``ping_time`` specified as
    index number.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        dataset containing Sv and ``echo_range`` [m]
    range_var: {'echo_range', 'depth'}, default 'echo_range'
        The variable to use for range binning.
        Must be one of ``echo_range`` or ``depth``.
        Note that ``depth`` is only available if the input dataset contains
        ``depth`` as a data variable.
    range_bin : str, default '20m'
        bin size along ``echo_range`` or ``depth`` in meters.
    ping_time_bin : str, default '20S'
        bin size along ``ping_time``
    method: str, default 'map-reduce'
        The flox strategy for reduction of dask arrays only.
        See flox `documentation <https://flox.readthedocs.io/en/latest/implementation.html>`_
        for more details.
    closed: {'left', 'right'}, default 'left'
        Which side of bin interval is closed.
    **kwargs
        Additional keyword arguments to be passed
        to flox reduction function.

    Returns
    -------
    A dataset containing bin-averaged Sv
    """

    if not isinstance(ping_time_bin, str):
        raise TypeError("ping_time_bin must be a string")

    range_bin = _parse_x_bin(range_bin, "range_bin")

    # Clean up filenames dimension if it exists
    # not needed here
    if "filenames" in ds_Sv.dims:
        ds_Sv = ds_Sv.drop_dims("filenames")

    # Check if range_var is valid
    if range_var not in ["echo_range", "depth"]:
        raise ValueError("range_var must be one of 'echo_range' or 'depth'.")

    # Check if range_var exists in ds_Sv
    if range_var not in ds_Sv.data_vars:
        raise ValueError(f"range_var '{range_var}' does not exist in the input dataset.")

    # Check for closed values
    if closed not in ["right", "left"]:
        raise ValueError(f"{closed} is not a valid option. Options are 'left' or 'right'.")

    # create bin information for echo_range
    # this computes the echo range max since there might NaNs in the data
    echo_range_max = ds_Sv[range_var].max()
    range_interval = np.arange(0, echo_range_max + range_bin, range_bin)

    # create bin information needed for ping_time
    d_index = (
        ds_Sv["ping_time"]
        .resample(ping_time=ping_time_bin, skipna=True)
        .first()  # Not actually being used, but needed to get the bin groups
        .indexes["ping_time"]
    )
    ping_interval = d_index.union([d_index[-1] + pd.Timedelta(ping_time_bin)]).values

    # Set interval index for groups
    ping_interval = _convert_bins_to_interval_index(ping_interval, closed=closed)
    range_interval = _convert_bins_to_interval_index(range_interval, closed=closed)
    raw_MVBS = get_MVBS_along_channels(
        ds_Sv,
        range_interval,
        ping_interval,
        range_var=range_var,
        method=method,
        **flox_kwargs,
    )

    # create MVBS dataset
    # by transforming the binned dimensions to regular coords
    ds_MVBS = xr.Dataset(
        data_vars={"Sv": (["channel", "ping_time", range_var], raw_MVBS["Sv"].data)},
        coords={
            "ping_time": np.array([v.left for v in raw_MVBS.ping_time_bins.values]),
            "channel": raw_MVBS.channel.values,
            range_var: np.array([v.left for v in raw_MVBS[f"{range_var}_bins"].values]),
        },
    )

    # "has_positions" attribute is inserted in get_MVBS_along_channels
    # when the dataset has position information
    # propagate this to the final MVBS dataset
    if raw_MVBS.attrs.get("has_positions", False):
        for var in POSITION_VARIABLES:
            ds_MVBS[var] = (["ping_time"], raw_MVBS[var].data, ds_Sv[var].attrs)

    # Add water level if uses echo_range and it exists in Sv dataset
    if range_var == "echo_range" and "water_level" in ds_Sv.data_vars:
        ds_MVBS["water_level"] = ds_Sv["water_level"]

    # ping_time_bin parsing and conversions
    # Need to convert between pd.Timedelta and np.timedelta64 offsets/frequency strings
    # https://xarray.pydata.org/en/stable/generated/xarray.Dataset.resample.html
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.resample.html
    # https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html
    # https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.resolution_string.html
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    # https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    timedelta_units = {
        "d": {"nptd64": "D", "unitstr": "day"},
        "h": {"nptd64": "h", "unitstr": "hour"},
        "t": {"nptd64": "m", "unitstr": "minute"},
        "min": {"nptd64": "m", "unitstr": "minute"},
        "s": {"nptd64": "s", "unitstr": "second"},
        "l": {"nptd64": "ms", "unitstr": "millisecond"},
        "ms": {"nptd64": "ms", "unitstr": "millisecond"},
        "u": {"nptd64": "us", "unitstr": "microsecond"},
        "us": {"nptd64": "ms", "unitstr": "millisecond"},
        "n": {"nptd64": "ns", "unitstr": "nanosecond"},
        "ns": {"nptd64": "ms", "unitstr": "millisecond"},
    }
    ping_time_bin_td = pd.Timedelta(ping_time_bin)
    # res = resolution (most granular time unit)
    ping_time_bin_resunit = ping_time_bin_td.resolution_string.lower()
    ping_time_bin_resvalue = int(
        ping_time_bin_td / np.timedelta64(1, timedelta_units[ping_time_bin_resunit]["nptd64"])
    )
    ping_time_bin_resunit_label = timedelta_units[ping_time_bin_resunit]["unitstr"]

    # Attach attributes
    _set_MVBS_attrs(ds_MVBS)
    ds_MVBS[range_var].attrs = {"long_name": "Range distance", "units": "m"}
    ds_MVBS["Sv"] = ds_MVBS["Sv"].assign_attrs(
        {
            "cell_methods": (
                f"ping_time: mean (interval: {ping_time_bin_resvalue} {ping_time_bin_resunit_label} "  # noqa
                "comment: ping_time is the interval start) "
                f"{range_var}: mean (interval: {range_bin} meter "
                f"comment: {range_var} is the interval start)"
            ),
            "binning_mode": "physical units",
            "range_meter_interval": str(range_bin) + "m",
            "ping_time_interval": ping_time_bin,
            "actual_range": [
                round(float(ds_MVBS["Sv"].min().values), 2),
                round(float(ds_MVBS["Sv"].max().values), 2),
            ],
        }
    )

    prov_dict = echopype_prov_attrs(process_type="processing")
    prov_dict["processing_function"] = "commongrid.compute_MVBS"
    ds_MVBS = ds_MVBS.assign_attrs(prov_dict)
    ds_MVBS["frequency_nominal"] = ds_Sv["frequency_nominal"]  # re-attach frequency_nominal

    ds_MVBS = insert_input_processing_level(ds_MVBS, input_ds=ds_Sv)

    return ds_MVBS


@add_processing_level("L3*")
def compute_MVBS_index_binning(ds_Sv, range_sample_num=100, ping_num=100):
    """
    Compute Mean Volume Backscattering Strength (MVBS)
    based on intervals of ``range_sample`` and ping number (``ping_num``) specified in index number.

    Output of this function differs from that of ``compute_MVBS``, which computes
    bin-averaged Sv according to intervals of range (``echo_range``) and ``ping_time`` specified
    in physical units.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        dataset containing ``Sv`` and ``echo_range`` [m]
    range_sample_num : int
        number of samples to average along the ``range_sample`` dimension, default to 100
    ping_num : int
        number of pings to average, default to 100

    Returns
    -------
    A dataset containing bin-averaged Sv
    """
    da_sv = 10 ** (ds_Sv["Sv"] / 10)  # average should be done in linear domain
    da = 10 * np.log10(
        da_sv.coarsen(ping_time=ping_num, range_sample=range_sample_num, boundary="pad").mean(
            skipna=True
        )
    )

    # Attach attributes and coarsened echo_range
    da.name = "Sv"
    ds_MVBS = da.to_dataset()
    ds_MVBS.coords["range_sample"] = (
        "range_sample",
        np.arange(ds_MVBS["range_sample"].size),
        {"long_name": "Along-range sample number, base 0"},
    )  # reset range_sample to start from 0
    ds_MVBS["echo_range"] = (
        ds_Sv["echo_range"]
        .coarsen(  # binned echo_range (use first value in each average bin)
            ping_time=ping_num, range_sample=range_sample_num, boundary="pad"
        )
        .min(skipna=True)
    )
    _set_MVBS_attrs(ds_MVBS)
    ds_MVBS["Sv"] = ds_MVBS["Sv"].assign_attrs(
        {
            "cell_methods": (
                f"ping_time: mean (interval: {ping_num} pings "
                "comment: ping_time is the interval start) "
                f"range_sample: mean (interval: {range_sample_num} samples along range "
                "comment: range_sample is the interval start)"
            ),
            "comment": "MVBS binned on the basis of range_sample and ping number specified as index numbers",  # noqa
            "binning_mode": "sample number",
            "range_sample_interval": f"{range_sample_num} samples along range",
            "ping_interval": f"{ping_num} pings",
            "actual_range": [
                round(float(ds_MVBS["Sv"].min().values), 2),
                round(float(ds_MVBS["Sv"].max().values), 2),
            ],
        }
    )

    prov_dict = echopype_prov_attrs(process_type="processing")
    prov_dict["processing_function"] = "commongrid.compute_MVBS_index_binning"
    ds_MVBS = ds_MVBS.assign_attrs(prov_dict)
    ds_MVBS["frequency_nominal"] = ds_Sv["frequency_nominal"]  # re-attach frequency_nominal

    ds_MVBS = insert_input_processing_level(ds_MVBS, input_ds=ds_Sv)

    return ds_MVBS


def compute_NASC(
    ds_Sv: xr.Dataset,
    range_var: str = "depth",
    range_bin=20,  # TODO: accept "20m" like in compute_MVBS
    dist_bin=0.5,  # TODO: accept "0.5nmi"
) -> xr.Dataset:
    """
    Compute Nautical Areal Scattering Coefficient (NASC) from an Sv dataset.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        A dataset containing Sv data.
        The Sv dataset must contain ``latitude``, ``longitude``, and ``depth`` as data variables.
    cell_dist: int, float
        The horizontal size of each NASC cell, in nautical miles [nmi]
    cell_depth: int, float
        The vertical size of each NASC cell, in meters [m]

    Returns
    -------
    xr.Dataset
        A dataset containing NASC

    Notes
    -----
    The NASC computation implemented here generally corresponds to the Echoview algorithm PRC_NASC
    https://support.echoview.com/WebHelp/Reference/Algorithms/Analysis_Variables/PRC_ABC_and_PRC_NASC.htm#PRC_NASC  # noqa
    The difference is that since in echopype masking of the Sv dataset is done explicitly using
    functions in the ``mask`` subpackage, the computation only involves computing the
    mean Sv and the mean height within each cell.

    In addition, in echopype the binning of pings into individual cells is based on the actual horizontal
    distance computed from the latitude and longitude coordinates of each ping in the Sv dataset.
    Therefore, both regular and irregular horizontal distance in the Sv dataset are allowed.
    This is different from Echoview's assumption of constant ping rate, vessel speed, and sample
    thickness when computing mean Sv
    (see https://support.echoview.com/WebHelp/Reference/Algorithms/Analysis_Variables/Sv_mean.htm#Conversions).  # noqa
    """
    # Check Sv contains lat/lon
    if "latitude" not in ds_Sv or "longitude" not in ds_Sv:
        raise ValueError("Both 'latitude' and 'longitude' must exist in the input Sv dataset.")

    # Get distance from lat/lon in nautical miles
    dist_nmi = get_distance_from_latlon(ds_Sv)
    ds_Sv = ds_Sv.assign_coords({"distance_nmi": ("ping_time", dist_nmi)}).swap_dims(
        {"ping_time": "distance_nmi"}
    )

    # create bin information along range_var
    # this computes the range_var max since there might NaNs in the data
    range_var_max = ds_Sv[range_var].max()
    range_interval = np.arange(0, range_var_max + range_bin, range_bin)

    # create bin information along distance_nmi
    # this computes the distance max since there might NaNs in the data
    dist_max = ds_Sv["distance_nmi"].max()
    dist_interval = np.arange(0, dist_max + dist_bin, dist_bin)

    # TODO: move the below to a function equivalent to get_MVBS_along_channels
    # Mean sv (volume backscattering coefficient, linear scale):
    #   do the equivalent of get_MVBS_along_channels
    #   but reduce along ping_time and echo_range or depth by binning and averaging
    #   for each channel

    # average should be done in linear domain
    sv = ds_Sv["Sv"].pipe(_log2lin)

    sv_mean = xarray_reduce(
        sv,
        ds_Sv["distance_nmi"],
        ds_Sv[range_var],
        func="nanmean",
        expected_groups=(dist_interval, range_interval),
        isbin=[True, True],
        method="map-reduce",
    )

    # Mean height: approach to use flox
    # Numerator (h_mean_num):
    #   - create a dataarray filled with the first difference of sample height
    #     with 2D coordinate (distance, depth)
    #   - flox xarray_reduce along both distance and depth, summing over each 2D bin
    # Denominator (h_mean_denom):
    #   - create a datararray filled with 1, with 1D coordinate (distance)
    #   - flox xarray_reduce along distance, summing over each 1D bin
    # h_mean = N/D
    da_denom = xr.ones_like(ds_Sv["distance_nmi"])
    h_mean_denom = xarray_reduce(
        da_denom,
        ds_Sv["distance_nmi"],
        func="sum",
        expected_groups=(dist_interval),
        isbin=[True],
        method="map-reduce",
    )
    h_mean_num = xarray_reduce(
        ds_Sv["depth"].diff(dim="range_sample", label="lower"),  # use lower end label after diff
        ds_Sv[range_var],
        ds_Sv["depth"][:-1],
        func="sum",
        expected_groups=(range_interval, range_interval),
        isbin=[True, True],
        method="map-reduce",
    )
    h_mean = h_mean_num / h_mean_denom

    # Combine to compute NASC
    raw_NASC = sv_mean * h_mean * 4 * np.pi * 1852**2

    # TODO: add mean lat/lon, as in ds_Pos in get_MVBS_along_channels
    # TODO: add mean ping_time, similar to mean lat/lon

    # create MVBS dataset
    # by transforming the binned dimensions to regular coords
    ds_NASC = xr.Dataset(
        data_vars={"NASC": (["channel", "ping_time", range_var], raw_NASC["Sv"].data)},
        coords={
            "distance_nmi": np.array([v.left for v in raw_NASC.distance_nmi_bins.values]),
            "channel": raw_NASC.channel.values,
            range_var: np.array([v.left for v in raw_NASC[f"{range_var}_bins"].values]),
        },
    )
    ds_NASC["frequency_nominal"] = ds_Sv["frequency_nominal"]  # re-attach frequency_nominal

    # Attach attributes
    _set_var_attrs(
        ds_NASC["NASC"],
        long_name="Nautical Areal Scattering Coefficient (NASC, m2 nmi-2)",
        units="m2 nmi-2",
        round_digits=3,
    )
    _set_var_attrs(ds_NASC["distance"], "Cumulative distance", "m", 3)
    _set_var_attrs(ds_NASC["depth"], "Cell depth", "m", 3, standard_name="depth")

    # Calculate and add ACDD bounding box global attributes
    ds_NASC.attrs["Conventions"] = "CF-1.7,ACDD-1.3"
    ds_NASC.attrs["time_coverage_start"] = np.datetime_as_string(
        ds_Sv["ping_time"].min().values, timezone="UTC"
    )
    ds_NASC.attrs["time_coverage_end"] = np.datetime_as_string(
        ds_Sv["ping_time"].max().values, timezone="UTC"
    )
    ds_NASC.attrs["geospatial_lat_min"] = round(float(ds_Sv["latitude"].min().values), 5)
    ds_NASC.attrs["geospatial_lat_max"] = round(float(ds_Sv["latitude"].max().values), 5)
    ds_NASC.attrs["geospatial_lon_min"] = round(float(ds_Sv["longitude"].min().values), 5)
    ds_NASC.attrs["geospatial_lon_max"] = round(float(ds_Sv["longitude"].max().values), 5)

    return ds_NASC


def regrid():
    return 1
