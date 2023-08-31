"""
Functions for enhancing the spatial and temporal coherence of data.
"""
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..consolidate.api import POSITION_VARIABLES
from ..utils.prov import add_processing_level, echopype_prov_attrs, insert_input_processing_level
from .mvbs import get_MVBS_along_channels


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


@add_processing_level("L3*")
def compute_MVBS(
    ds_Sv: xr.Dataset,
    range_var: Literal["echo_range", "depth"] = "echo_range",
    range_meter_bin: int = 20,
    ping_time_bin: str = "20S",
    method="map-reduce",
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
    range_var: str
        The variable to use for range binning.
        Must be one of ``echo_range`` or ``depth``.
        Note that ``depth`` is only available if the input dataset contains
        ``depth`` as a data variable.
    range_meter_bin : Union[int, float]
        bin size along ``echo_range`` or ``depth`` in meters,
        default to ``20``
    ping_time_bin : str
        bin size along ``ping_time``, default to ``20S``
    method: str
        The flox strategy for reduction of dask arrays only.
        See flox `documentation <https://flox.readthedocs.io/en/latest/implementation.html>`_
        for more details.
    **kwargs
        Additional keyword arguments to be passed
        to flox reduction function.

    Returns
    -------
    A dataset containing bin-averaged Sv
    """

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

    # create bin information for echo_range
    # this computes the echo range max since there might be missing values
    echo_range_max = ds_Sv[range_var].max()
    range_interval = np.arange(0, echo_range_max + range_meter_bin, range_meter_bin)

    # create bin information needed for ping_time
    d_index = (
        ds_Sv["ping_time"]
        .resample(ping_time=ping_time_bin, skipna=True)
        .asfreq()
        .indexes["ping_time"]
    )
    ping_interval = d_index.union([d_index[-1] + pd.Timedelta(ping_time_bin)])

    # calculate the MVBS along each channel
    if method == "map-reduce":
        # set to the faster compute if not specified
        flox_kwargs.setdefault("reindex", True)
    raw_MVBS = get_MVBS_along_channels(
        ds_Sv, range_interval, ping_interval, range_var=range_var, method=method, **flox_kwargs
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

    # Add the position variables
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
                f"{range_var}: mean (interval: {range_meter_bin} meter "
                f"comment: {range_var} is the interval start)"
            ),
            "binning_mode": "physical units",
            "range_meter_interval": str(range_meter_bin) + "m",
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


# def compute_NASC(
#     ds_Sv: xr.Dataset,
#     cell_dist: Union[int, float],  # TODO: allow xr.DataArray
#     cell_depth: Union[int, float],  # TODO: allow xr.DataArray
# ) -> xr.Dataset:
#     """
#     Compute Nautical Areal Scattering Coefficient (NASC) from an Sv dataset.

#     Parameters
#     ----------
#     ds_Sv : xr.Dataset
#         A dataset containing Sv data.
#         The Sv dataset must contain ``latitude``, ``longitude``, and ``depth`` as data variables.
#     cell_dist: int, float
#         The horizontal size of each NASC cell, in nautical miles [nmi]
#     cell_depth: int, float
#         The vertical size of each NASC cell, in meters [m]

#     Returns
#     -------
#     xr.Dataset
#         A dataset containing NASC

#     Notes
#     -----
#     The NASC computation implemented here corresponds to the Echoview algorithm PRC_NASC
#     https://support.echoview.com/WebHelp/Reference/Algorithms/Analysis_Variables/PRC_ABC_and_PRC_NASC.htm#PRC_NASC  # noqa
#     The difference is that since in echopype masking of the Sv dataset is done explicitly using
#     functions in the ``mask`` subpackage so the computation only involves computing the
#     mean Sv and the mean height within each cell.

#     In addition, here the binning of pings into individual cells is based on the actual horizontal
#     distance computed from the latitude and longitude coordinates of each ping in the Sv dataset.
#     Therefore, both regular and irregular horizontal distance in the Sv dataset are allowed.
#     This is different from Echoview's assumption of constant ping rate, vessel speed, and sample
#     thickness when computing mean Sv.
#     """
#     # Check Sv contains lat/lon
#     if "latitude" not in ds_Sv or "longitude" not in ds_Sv:
#         raise ValueError("Both 'latitude' and 'longitude' must exist in the input Sv dataset.")

#     # Check if depth vectors are identical within each channel
#     if not ds_Sv["depth"].groupby("channel").map(check_identical_depth).all():
#         raise ValueError(
#             "Only Sv data with identical depth vectors across all pings "
#             "are allowed in the current compute_NASC implementation."
#         )

#     # Get distance from lat/lon in nautical miles
#     dist_nmi = get_distance_from_latlon(ds_Sv)

#     # Find binning indices along distance
#     bin_num_dist, dist_bin_idx = get_dist_bin_info(dist_nmi, cell_dist)  # dist_bin_idx is 1-based

#     # Find binning indices along depth: channel-dependent
#     bin_num_depth, depth_bin_idx = get_depth_bin_info(ds_Sv, cell_depth)  # depth_bin_idx is 1-based  # noqa

#     # Compute mean sv (volume backscattering coefficient, linear scale)
#     # This is essentially to compute MVBS over a the cell defined here,
#     # which are typically larger than those used for MVBS.
#     # The implementation below is brute force looping, but can be optimized
#     # by experimenting with different delayed schemes.
#     # The optimized routines can then be used here and
#     # in commongrid.compute_MVBS and clean.estimate_noise
#     sv_mean = []
#     for ch_seq in np.arange(ds_Sv["channel"].size):
#         # TODO: .compute each channel sequentially?
#         #       dask.delay within each channel?
#         ds_Sv_ch = ds_Sv["Sv"].isel(channel=ch_seq).data  # preserve the underlying type

#         sv_mean_dist_depth = []
#         for dist_idx in np.arange(bin_num_dist) + 1:  # along ping_time
#             sv_mean_depth = []
#             for depth_idx in np.arange(bin_num_depth) + 1:  # along depth
#                 # Sv dim: ping_time x depth
#                 Sv_cut = ds_Sv_ch[dist_idx == dist_bin_idx, :][
#                     :, depth_idx == depth_bin_idx[ch_seq]
#                 ]
#                 sv_mean_depth.append(np.nanmean(10 ** (Sv_cut / 10)))
#             sv_mean_dist_depth.append(sv_mean_depth)

#         sv_mean.append(sv_mean_dist_depth)

#     # Compute mean height
#     # For data with uniform depth step size, mean height = vertical size of cell
#     height_mean = cell_depth
#     # TODO: generalize to variable depth step size

#     ds_NASC = xr.DataArray(
#         np.array(sv_mean) * height_mean,
#         dims=["channel", "distance", "depth"],
#         coords={
#             "channel": ds_Sv["channel"].values,
#             "distance": np.arange(bin_num_dist) * cell_dist,
#             "depth": np.arange(bin_num_depth) * cell_depth,
#         },
#         name="NASC",
#     ).to_dataset()

#     ds_NASC["frequency_nominal"] = ds_Sv["frequency_nominal"]  # re-attach frequency_nominal

#     # Attach attributes
#     _set_var_attrs(
#         ds_NASC["NASC"],
#         long_name="Nautical Areal Scattering Coefficient (NASC, m2 nmi-2)",
#         units="m2 nmi-2",
#         round_digits=3,
#     )
#     _set_var_attrs(ds_NASC["distance"], "Cumulative distance", "m", 3)
#     _set_var_attrs(ds_NASC["depth"], "Cell depth", "m", 3, standard_name="depth")

#     # Calculate and add ACDD bounding box global attributes
#     ds_NASC.attrs["Conventions"] = "CF-1.7,ACDD-1.3"
#     ds_NASC.attrs["time_coverage_start"] = np.datetime_as_string(
#         ds_Sv["ping_time"].min().values, timezone="UTC"
#     )
#     ds_NASC.attrs["time_coverage_end"] = np.datetime_as_string(
#         ds_Sv["ping_time"].max().values, timezone="UTC"
#     )
#     ds_NASC.attrs["geospatial_lat_min"] = round(float(ds_Sv["latitude"].min().values), 5)
#     ds_NASC.attrs["geospatial_lat_max"] = round(float(ds_Sv["latitude"].max().values), 5)
#     ds_NASC.attrs["geospatial_lon_min"] = round(float(ds_Sv["longitude"].min().values), 5)
#     ds_NASC.attrs["geospatial_lon_max"] = round(float(ds_Sv["longitude"].max().values), 5)

#     return ds_NASC


def regrid():
    return 1
