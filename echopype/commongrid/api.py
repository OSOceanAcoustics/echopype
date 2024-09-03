"""
Functions for enhancing the spatial and temporal coherence of data.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..consolidate.api import POSITION_VARIABLES
from ..utils.prov import add_processing_level, echopype_prov_attrs, insert_input_processing_level
from .utils import (
    _convert_bins_to_interval_index,
    _get_reduced_positions,
    _parse_x_bin,
    _set_MVBS_attrs,
    _set_var_attrs,
    _setup_and_validate,
    compute_raw_MVBS,
    compute_raw_NASC,
    get_distance_from_latlon,
)

logger = logging.getLogger(__name__)


@add_processing_level("L3*")
def compute_MVBS(
    ds_Sv: xr.Dataset,
    range_var: Literal["echo_range", "depth"] = "echo_range",
    range_bin: str = "20m",
    ping_time_bin: str = "20s",
    method="map-reduce",
    skipna=True,
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
    ping_time_bin : str, default '20s'
        bin size along ``ping_time``
    method: str, default 'map-reduce'
        The flox strategy for reduction of dask arrays only.
        See flox `documentation <https://flox.readthedocs.io/en/latest/implementation.html>`_
        for more details.
    skipna: bool, default True
        If true, the mean operation skips NaN values.
        Else, the mean operation includes NaN values.
    closed: {'left', 'right'}, default 'left'
        Which side of bin interval is closed.
    **flox_kwargs
        Additional keyword arguments to be passed
        to flox reduction function.

    Returns
    -------
    A dataset containing bin-averaged Sv
    """

    # Setup and validate
    # * Sv dataset must contain specified range_var
    # * Parse range_bin
    # * Check closed value
    ds_Sv, range_bin = _setup_and_validate(ds_Sv, range_var, range_bin, closed)

    if not isinstance(ping_time_bin, str):
        raise TypeError("ping_time_bin must be a string")

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
    raw_MVBS = compute_raw_MVBS(
        ds_Sv,
        range_interval,
        ping_interval,
        range_var=range_var,
        method=method,
        skipna=skipna,
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

    # If dataset has position information
    # propagate this to the final MVBS dataset
    ds_MVBS = _get_reduced_positions(ds_Sv, ds_MVBS, "MVBS", ping_interval)

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


@add_processing_level("L4")
def compute_NASC(
    ds_Sv: xr.Dataset,
    range_bin: str = "10m",
    dist_bin: str = "0.5nmi",
    method: str = "map-reduce",
    skipna=True,
    closed: Literal["left", "right"] = "left",
    **flox_kwargs,
) -> xr.Dataset:
    """
    Compute Nautical Areal Scattering Coefficient (NASC) from an Sv dataset.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        A dataset containing Sv data.
        The Sv dataset must contain ``latitude``, ``longitude``, and ``depth`` as data variables.
    range_bin : str, default '10m'
        bin size along ``depth`` in meters (m).
    dist_bin : str, default '0.5nmi'
        bin size along ``distance`` in nautical miles (nmi).
    method: str, default 'map-reduce'
        The flox strategy for reduction of dask arrays only.
        See flox `documentation <https://flox.readthedocs.io/en/latest/implementation.html>`_
        for more details.
    skipna: bool, default True
        If true, the mean operation skips NaN values.
        Else, the mean operation includes NaN values.
    closed: {'left', 'right'}, default 'left'
        Which side of bin interval is closed.
    **flox_kwargs
        Additional keyword arguments to be passed
        to flox reduction function.

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
    mean Sv and the mean height within each cell, where some Sv "pixels" may have been
    masked as NaN.

    In addition, in echopype the binning of pings into individual cells is based on the actual horizontal
    distance computed from the latitude and longitude coordinates of each ping in the Sv dataset.
    Therefore, both regular and irregular horizontal distance in the Sv dataset are allowed.
    This is different from Echoview's assumption of constant ping rate, vessel speed, and sample
    thickness when computing mean Sv
    (see https://support.echoview.com/WebHelp/Reference/Algorithms/Analysis_Variables/Sv_mean.htm#Conversions).  # noqa
    """
    # Set range_var to be 'depth'
    range_var = "depth"

    # Setup and validate
    # * Sv dataset must contain latitude, longitude, and depth
    # * Parse range_bin
    # * Check closed value
    ds_Sv, range_bin = _setup_and_validate(
        ds_Sv, range_var, range_bin, closed, required_data_vars=POSITION_VARIABLES
    )

    # Check if dist_bin is a string
    if not isinstance(dist_bin, str):
        raise TypeError("dist_bin must be a string")

    # Parse the dist_bin string and convert to float
    dist_bin = _parse_x_bin(dist_bin, "dist_bin")

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

    # Set interval index for groups
    dist_interval = _convert_bins_to_interval_index(dist_interval, closed=closed)
    range_interval = _convert_bins_to_interval_index(range_interval, closed=closed)

    raw_NASC = compute_raw_NASC(
        ds_Sv,
        range_interval,
        dist_interval,
        method=method,
        skipna=skipna,
        **flox_kwargs,
    )

    # create MVBS dataset
    # by transforming the binned dimensions to regular coords
    ds_NASC = xr.Dataset(
        data_vars={"NASC": (["channel", "distance", range_var], raw_NASC["sv"].data)},
        coords={
            "distance": np.array([v.left for v in raw_NASC["distance_nmi_bins"].values]),
            "channel": raw_NASC["channel"].values,
            range_var: np.array([v.left for v in raw_NASC[f"{range_var}_bins"].values]),
        },
    )

    # If dataset has position information
    # propagate this to the final NASC dataset
    ds_NASC = _get_reduced_positions(ds_Sv, ds_NASC, "NASC", dist_interval)

    # Set ping time binning information
    ds_NASC["ping_time"] = (["distance"], raw_NASC["ping_time"].data, ds_Sv["ping_time"].attrs)

    ds_NASC["frequency_nominal"] = ds_Sv["frequency_nominal"]  # re-attach frequency_nominal

    # Attach attributes
    _set_var_attrs(
        ds_NASC["NASC"],
        long_name="Nautical Areal Scattering Coefficient (NASC, m2 nmi-2)",
        units="m2 nmi-2",
        round_digits=3,
    )
    _set_var_attrs(ds_NASC["distance"], "Cumulative distance", "nmi", 3)
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
