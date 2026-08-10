"""
Functions for enhancing the spatial and temporal coherence of data.
"""

import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..consolidate.api import POSITION_VARIABLES
from ..qc.api import coerce_increasing_time, exist_reversed_time
from ..utils.prov import add_processing_level, echopype_prov_attrs, insert_input_processing_level
from .utils import (
    _convert_bins_to_interval_index,
    _get_reduced_positions,
    _lin2log,
    _log2lin,
    _parse_x_bin,
    _set_MVBS_attrs,
    _set_var_attrs,
    _setup_and_validate,
    _weighted_mean_kernel,
    compute_raw_MVBS,
    compute_raw_NASC,
    get_distance_from_latlon,
    ping_time_bin_parsing_and_conversion,
)

logger = logging.getLogger(__name__)


@add_processing_level("L3*")
def compute_MVBS(
    ds_Sv: xr.Dataset,
    range_var: Literal["echo_range", "depth"] = "echo_range",
    range_bin: str = "20m",
    ping_time_bin: str = "20s",
    method: str = "map-reduce",
    reindex: bool = False,
    skipna: bool = True,
    fill_value: float = np.nan,
    closed: Literal["left", "right"] = "left",
    range_var_max: str = None,
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
    reindex: bool, default False
        If False, reindex after the blockwise stage. If True, reindex at the blockwise stage.
        Generally, `reindex=False` results in less memory at the cost of computation speed.
        Can only be used when method='map-reduce'.
        See flox `documentation <https://flox.readthedocs.io/en/latest/implementation.html>`_
        for more details.
    skipna: bool, default True
        If true, the mean operation skips NaN values.
        Else, the mean operation includes NaN values.
    fill_value: float, default np.nan
        Fill value when no group data exists to aggregate.
    closed: {'left', 'right'}, default 'left'
        Which side of bin interval is closed.
    range_var_max: str, default None
        Range variable maximum. Can be true range variable maximum or the maximum depth the
        user wishes to regrid to. If known, users can pass in range variable maximum to
        ensure that `compute_MVBS` can lazily run without any computation.
    **flox_kwargs
        Additional keyword arguments to be passed
        to flox reduction function.

    Returns
    -------
    A dataset containing bin-averaged Sv
    """
    if method != "map-reduce" and reindex is not None:
        raise ValueError(f"Passing in reindex={reindex} is only allowed when method='map_reduce'.")

    # Setup and validate
    # * Sv dataset must contain specified range_var
    # * Parse range_bin
    # * Check closed value
    ds_Sv, range_bin = _setup_and_validate(ds_Sv, range_var, range_bin, closed)

    if not isinstance(ping_time_bin, str):
        raise TypeError("ping_time_bin must be a string")

    # Create bin information for the range variable
    if range_var_max is None:
        # This computes the range variable max since there might be NaNs in the data
        range_var_max = ds_Sv[range_var].max(skipna=True)
    else:
        # Parse string and small increase to ensure that we get the bin
        # corresponding to range_var_max
        range_var_max = _parse_x_bin(range_var_max) + 1e-8
    range_interval = np.arange(0, range_var_max + range_bin, range_bin)

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
        reindex=reindex,
        skipna=skipna,
        fill_value=fill_value,
        **flox_kwargs,
    )

    # Generalize the first dimension name to support multiple like channel and frequency_nominal
    dim_0 = list(raw_MVBS.sizes.keys())[0]

    # create MVBS dataset
    # by transforming the binned dimensions to regular coords
    ds_MVBS = xr.Dataset(
        data_vars={"Sv": ([dim_0, "ping_time", range_var], raw_MVBS["Sv"].data)},
        coords={
            "ping_time": np.array([v.left for v in raw_MVBS.ping_time_bins.values]),
            dim_0: getattr(raw_MVBS, dim_0).values,
            range_var: np.array([v.left for v in raw_MVBS[f"{range_var}_bins"].values]),
        },
    )

    # If dataset has position information
    # propagate this to the final MVBS dataset
    ds_MVBS = _get_reduced_positions(ds_Sv, ds_MVBS, "MVBS", ping_interval)

    # Add water level if uses echo_range and it exists in Sv dataset
    if range_var == "echo_range" and "water_level" in ds_Sv.data_vars:
        ds_MVBS["water_level"] = ds_Sv["water_level"]

    # Attach attributes
    _set_MVBS_attrs(ds_MVBS)
    ds_MVBS[range_var].attrs = {"long_name": "Range distance", "units": "m"}
    ping_time_bin_resvalue, ping_time_bin_resunit_label = ping_time_bin_parsing_and_conversion(
        ping_time_bin
    )
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

    # Preserve the channel order returned by compute_raw_MVBS and align
    # frequency_nominal to that order.
    freq = ds_Sv["frequency_nominal"]

    if "channel" in ds_MVBS.dims:
        ds_MVBS["frequency_nominal"] = freq.sel(channel=ds_MVBS["channel"])
    else:
        ds_MVBS["frequency_nominal"] = freq

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

    ds_MVBS["frequency_nominal"] = ds_Sv["frequency_nominal"]

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
    TODO: Add range_var_max and reindex parameters to match `compute_MVBS`.

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
    https://support.echoview.com/WebHelp/Reference/Algorithms/Analysis_Variables/PRC_ABC_and_PRC_NASC.htm#PRC_NASC
    The difference is that since in echopype masking of the Sv dataset is done explicitly using
    functions in the ``mask`` subpackage, the computation only involves computing the
    mean Sv and the mean height within each cell, where some Sv "pixels" may have been
    masked as NaN.

    In addition, in echopype the binning of pings into individual cells is based on the actual horizontal
    distance computed from the latitude and longitude coordinates of each ping in the Sv dataset.
    Therefore, both regular and irregular horizontal distance in the Sv dataset are allowed.
    This is different from Echoview's assumption of constant ping rate, vessel speed, and sample
    thickness when computing mean Sv
    (see https://support.echoview.com/WebHelp/Reference/Algorithms/Analysis_Variables/Sv_mean.htm#Conversions).
    """  # noqa: E501
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
    freq = ds_Sv["frequency_nominal"]

    if "channel" in ds_NASC.dims:
        ds_NASC["frequency_nominal"] = freq.sel(channel=ds_NASC["channel"])
    else:
        ds_NASC["frequency_nominal"] = freq

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


def resample_to_geometry(
    ds_Sv,
    target_variable: str = "Sv",
    is_log: bool = True,
    target_channel: str | None = None,
    target_grid: xr.DataArray | None = None,
):
    """
    Regrids a variable across all channels in an Sv or Sp dataset to
    a common range geometry specified by either a target channel or a custom target grid.
    Ping time is assumed identical for all input channels.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        Input Dataset containing Sv data

    target_variable : str, default "Sv"
        Name of the variable to resample. The variable must exist in
        ``ds_Sv``.

    is_log : bool, default True
        Whether ``target_variable`` contains logarithmic values. If True,
        values are converted to the linear domain before weighted averaging
        and converted back afterward.

    target_channel : str, optional
        Channel used as reference grid. Must be provided if target_grid is None.

    target_grid : xr.DataArray, optional
        Custom range grid. Must be provided if ``target_channel`` is None.
        The grid may have dimension ``("range_sample",)`` for a common
        grid across all pings, or ``("ping_time", "range_sample")`` for
        a ping-dependent grid. Its ``range_sample`` length may differ
        from that of the input dataset.

    Returns
    -------
    xr.Dataset
        A new Dataset where all channels share the same `ping_time`,
        `range_sample`, and `echo_range` as the target.
        `ping_time` is assumed identical across all input channels
        and preserved throughout the resampling process.
        The resulting target geometry, whether specified using
        `target_channel`` or `target_grid`, can be retrieved with
        `target_grid = regridded_ds["echo_range"]`.
    """

    if target_variable not in ds_Sv:
        raise ValueError(
            f"'{target_variable}' is not a variable in the input dataset. "
            f"Available variables are: {list(ds_Sv.data_vars)}"
        )

    if target_variable != "Sv":
        warnings.warn(
            f"Resampling '{target_variable}' with overlap-weighted averaging. "
            "This function is primarily intended and validated for Sv. "
            "Ensure that `is_log` correctly describes the variable's domain. "
            "Angle variables are resampled geometrically, and this is not "
            "physically equivalent to recomputing angles from complex data.",
            UserWarning,
            stacklevel=2,
        )

    if (target_channel is None) == (target_grid is None):
        raise ValueError("Provide exactly one of target_channel or target_grid.")

    if exist_reversed_time(ds_Sv, "ping_time"):
        warnings.warn(
            "Reversed ping_time values detected. The ping_time variable "
            "has been modified to increase monotonically before resampling. "
            "See echopype.qc.coerce_increasing_time() for detail.",
            UserWarning,
        )
        coerce_increasing_time(ds_Sv)

    channels = ds_Sv.channel.values

    expected_dims = {"channel", "ping_time", "range_sample"}
    actual_dims = set(ds_Sv[target_variable].dims)
    if actual_dims != expected_dims:
        raise ValueError(
            f"Target variable '{target_variable}' must have exactly the dimensions "
            f"('channel', 'ping_time', 'range_sample'). Found: {ds_Sv[target_variable].dims}"
        )

    if target_channel and target_channel not in channels:
        raise ValueError(f"{target_channel} is not part of the channel names in : {channels}")

    valid_target_grid_dims = {
        ("range_sample",),
        ("ping_time", "range_sample"),
        ("range_sample", "ping_time"),
    }

    if target_grid is not None and target_grid.dims not in valid_target_grid_dims:
        raise ValueError(
            "target_grid must have dimensions ('range_sample',) or "
            "('ping_time', 'range_sample'). "
            f"Found: {target_grid.dims}"
        )

    da_var = ds_Sv[target_variable]

    if target_channel:
        ds_target = ds_Sv.sel(channel=target_channel).copy()
        target_range_da = ds_target["echo_range"]
    else:  # Target grid is given
        target_range_da = target_grid

    # Use a distinct core dimension so the target grid may have a
    # different length from the input range_sample dimension.
    target_range_da = target_range_da.rename({"range_sample": "target_range_sample"})

    target_range_size = target_range_da.sizes["target_range_sample"]

    # List to hold the aligned DataArrays
    aligned_arrays = []

    for channel in channels:

        ds_source = da_var.sel(channel=channel)

        if is_log:
            source_linear = _log2lin(ds_source)
        else:
            source_linear = ds_source
        source_range_da = ds_Sv["echo_range"].sel(channel=channel)

        # Apply weighted mean resampling as ufunc
        result_linear = xr.apply_ufunc(
            _weighted_mean_kernel,
            target_range_da,
            source_range_da,
            source_linear,
            input_core_dims=[
                ["target_range_sample"],
                ["range_sample"],
                ["range_sample"],
            ],
            output_core_dims=[["target_range_sample"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={
                "output_sizes": {
                    "target_range_sample": target_range_size,
                }
            },
        )

        result_linear = result_linear.rename({"target_range_sample": "range_sample"}).assign_coords(
            range_sample=np.arange(target_range_size)
        )

        # Convert back to log domain
        if is_log:
            result_linear = result_linear.where(result_linear > 0)
            resample_variable = _lin2log(result_linear)
        else:
            resample_variable = result_linear

        resample_variable.name = target_variable
        resample_variable = resample_variable.assign_coords(channel=channel)

        aligned_arrays.append(resample_variable)

    ds_combined = xr.concat(aligned_arrays, dim="channel")

    target_range_output = target_range_da.rename(
        {"target_range_sample": "range_sample"}
    ).assign_coords(range_sample=np.arange(target_range_size))

    echo_range_aligned = target_range_output.broadcast_like(ds_combined)
    echo_range_aligned = echo_range_aligned.transpose(*ds_combined.dims)

    new_ds = xr.Dataset(
        data_vars={
            target_variable: ds_combined,
            "echo_range": echo_range_aligned,
            "frequency_nominal": ds_Sv["frequency_nominal"],
        },
        coords={
            "channel": ds_Sv["channel"],
        },
    )
    if "water_level" in ds_Sv:
        new_ds["water_level"] = ds_Sv["water_level"]

    # Attach attributes
    new_ds[target_variable].attrs = ds_Sv[target_variable].attrs

    if target_channel:
        new_ds[target_variable].attrs["resampling_mode"] = "target_channel"
        new_ds[target_variable].attrs["target_channel"] = target_channel
    else:
        new_ds[target_variable].attrs["resampling_mode"] = "target_grid"

    prov_dict = echopype_prov_attrs(process_type="processing")
    prov_dict["processing_function"] = "commongrid.resample_to_geometry"
    new_ds = new_ds.assign_attrs(prov_dict)
    new_ds = insert_input_processing_level(new_ds, ds_Sv)
    new_ds["echo_range"].attrs = ds_Sv["echo_range"].attrs

    return new_ds
