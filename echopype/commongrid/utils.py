import logging
import re
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from flox.xarray import xarray_reduce
from geopy import distance

from ..consolidate.api import POSITION_VARIABLES
from ..utils.compute import _lin2log, _log2lin

logger = logging.getLogger(__name__)


def compute_raw_MVBS(
    ds_Sv: xr.Dataset,
    range_interval: Union[pd.IntervalIndex, np.ndarray],
    ping_interval: Union[pd.IntervalIndex, np.ndarray],
    range_var: Literal["echo_range", "depth"] = "echo_range",
    method="map-reduce",
    skipna=True,
    **flox_kwargs,
):
    """
    Compute the raw unformatted MVBS of ``ds_Sv``.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        A Dataset containing ``Sv`` and ``echo_range`` data
        with coordinates ``channel``, ``ping_time``, and ``range_sample``
        at bare minimum.
        Or this can contain ``Sv`` and ``depth`` data with similar coordinates.
    range_interval: pd.IntervalIndex or np.ndarray
        1D array or interval index representing
        the bins required for ``range_var``
    ping_interval : pd.IntervalIndex or np.ndarray
        1D array or interval index representing
        the bins required for ``ping_time``.
    range_var: {'echo_range', 'depth'}, default 'echo_range'
        The variable to use for range binning.
        Either ``echo_range`` or ``depth``.
    method: str
        The flox strategy for reduction of dask arrays only.
        See flox `documentation <https://flox.readthedocs.io/en/latest/implementation.html>`_
        for more details.
    skipna: bool, default True
        If true, the mean operation skips NaN values.
        Else, the mean operation includes NaN values.
    **flox_kwargs
        Additional keyword arguments to be passed
        to flox reduction function.

    Returns
    -------
    xr.Dataset
        The MVBS dataset of the input ``ds_Sv`` for all channels.
    """
    # Set initial variables
    ds = xr.Dataset()
    x_var = "ping_time"

    sv_mean = _groupby_x_along_channels(
        ds_Sv,
        range_interval,
        x_interval=ping_interval,
        x_var=x_var,
        range_var=range_var,
        method=method,
        func="nanmean" if skipna else "mean",
        skipna=skipna,
        **flox_kwargs,
    )

    # This is MVBS computation
    # apply inverse mapping to get back to the original domain and store values
    da_MVBS = sv_mean.pipe(_lin2log)
    # return xr.merge([ds_Pos, da_MVBS])
    return xr.merge([ds, da_MVBS])


def compute_raw_NASC(
    ds_Sv: xr.Dataset,
    range_interval: Union[pd.IntervalIndex, np.ndarray],
    dist_interval: Union[pd.IntervalIndex, np.ndarray],
    method="map-reduce",
    skipna=True,
    **flox_kwargs,
):
    """
    Compute the raw unformatted NASC of ``ds_Sv``.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        A Dataset containing ``Sv`` and ``depth`` data
        with coordinates ``channel``, ``distance_nmi``, and ``range_sample``.
    range_interval: pd.IntervalIndex or np.ndarray
        1D array or interval index representing
        the bins required for ``range_var``.
    dist_interval : pd.IntervalIndex or np.ndarray
        1D array or interval index representing
        the bins required for ``distance_nmi``.
    method: str
        The flox strategy for reduction of dask arrays only.
        See flox `documentation <https://flox.readthedocs.io/en/latest/implementation.html>`_
        for more details.
    skipna: bool, default True
        If true, the mean operation skips NaN values.
        Else, the mean operation includes NaN values.
    **flox_kwargs
        Additional keyword arguments to be passed
        to flox reduction function.

    Returns
    -------
    xr.Dataset
        The MVBS or NASC dataset of the input ``ds_Sv`` for all channels
    """
    # Set initial variables
    ds = xr.Dataset()
    x_var = "distance_nmi"
    range_var = "depth"

    # Determine range_dim for NASC computation
    range_dim = "range_sample"
    if range_dim not in ds_Sv.dims:
        range_dim = "depth"

    sv_mean = _groupby_x_along_channels(
        ds_Sv,
        range_interval,
        x_interval=dist_interval,
        x_var=x_var,
        range_var=range_var,
        method=method,
        func="nanmean" if skipna else "mean",
        skipna=skipna,
        **flox_kwargs,
    )

    # Get mean ping_time along distance_nmi
    # this is only done for NASC computation,
    # since for MVBS the ping_time is used for binning already.
    ds_ping_time = xarray_reduce(
        ds_Sv["ping_time"],
        ds_Sv[x_var],
        func="nanmean",
        skipna=True,
        expected_groups=(dist_interval),
        isbin=True,
        method=method,
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
    da_denom = xr.ones_like(ds_Sv[x_var])
    h_mean_denom = xarray_reduce(
        da_denom,
        ds_Sv[x_var],
        func="nansum",
        skipna=True,
        expected_groups=(dist_interval),
        isbin=[True],
        method=method,
    )

    h_mean_num = xarray_reduce(
        ds_Sv[range_var].diff(dim=range_dim, label="lower"),  # use lower end label after diff
        ds_Sv["channel"],
        ds_Sv[x_var],
        ds_Sv[range_var].isel(**{range_dim: slice(0, -1)}),
        func="nansum",
        skipna=True,
        expected_groups=(None, dist_interval, range_interval),
        isbin=[False, True, True],
        method=method,
    )
    h_mean = h_mean_num / h_mean_denom

    # Combine to compute NASC and name it
    raw_NASC = sv_mean * h_mean * 4 * np.pi * 1852**2
    raw_NASC.name = "sv"

    return xr.merge([ds, ds_ping_time, raw_NASC])


def get_distance_from_latlon(ds_Sv):
    # Get distance from lat/lon in nautical miles
    df_pos = ds_Sv["latitude"].to_dataframe().join(ds_Sv["longitude"].to_dataframe())
    df_pos["latitude_prev"] = df_pos["latitude"].shift(-1)
    df_pos["longitude_prev"] = df_pos["longitude"].shift(-1)
    df_latlon_nonan = df_pos.dropna().copy()

    if len(df_latlon_nonan) == 0:  # lat/lon entries are all NaN
        raise ValueError("All lat/lon entries are NaN!")

    df_latlon_nonan["dist"] = df_latlon_nonan.apply(
        lambda x: distance.distance(
            (x["latitude"], x["longitude"]),
            (x["latitude_prev"], x["longitude_prev"]),
        ).nm,
        axis=1,
    )
    df_pos = df_pos.join(df_latlon_nonan["dist"], how="left")
    df_pos["dist"] = df_pos["dist"].cumsum()
    df_pos["dist"] = df_pos["dist"].ffill().bfill()

    return df_pos["dist"].values


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


def _setup_and_validate(
    ds_Sv: xr.Dataset,
    range_var: Literal["echo_range", "depth"] = "echo_range",
    range_bin: Optional[str] = None,
    closed: Literal["left", "right"] = "left",
    required_data_vars: Optional[list] = None,
) -> Tuple[xr.Dataset, float]:
    """
    Setup and validate shared arguments for
    ``compute_X`` functions.

    For now this is only used by ``compute_MVBS`` and ``compute_NASC``.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        Sv dataset
    range_var : {'echo_range', 'depth'}, default 'echo_range'
        The variable to use for range binning.
        Must be one of ``echo_range`` or ``depth``.
        Note that ``depth`` is only available if the input dataset contains
        ``depth`` as a data variable.
    range_bin : str, default None
        bin size along ``echo_range`` or ``depth`` in meters.
    closed: {'left', 'right'}, default 'left'
        Which side of bin interval is closed.
    required_data_vars : list, optional
        List of required data variables in ds_Sv.
        If None, defaults to empty list.

    Returns
    -------
    ds_Sv : xr.Dataset
        Modified Sv dataset
    range_bin : float
        The range bin value in meters
    """

    # Check if range_var is valid
    if range_var not in ["echo_range", "depth"]:
        raise ValueError("range_var must be one of 'echo_range' or 'depth'.")

    # Set to default empty list if None
    if required_data_vars is None:
        required_data_vars = []

    # Check if required data variables exists in ds_Sv
    # Use set to ensure no duplicates
    required_data_vars = set(required_data_vars + [range_var])
    if not all([var in ds_Sv.variables for var in required_data_vars]):
        raise ValueError(
            "Input Sv dataset must contain all of " f"the following variables: {required_data_vars}"
        )

    # Check if range_bin is a string
    if not isinstance(range_bin, str):
        raise TypeError("range_bin must be a string")

    # Parse the range_bin string and convert to float
    range_bin = _parse_x_bin(range_bin, "range_bin")

    # Check for closed values
    if closed not in ["right", "left"]:
        raise ValueError(f"{closed} is not a valid option. Options are 'left' or 'right'.")

    # Clean up filenames dimension if it exists
    # not needed here
    if "filenames" in ds_Sv.dims:
        ds_Sv = ds_Sv.drop_dims("filenames")

    return ds_Sv, range_bin


def _get_reduced_positions(
    ds_Sv: xr.Dataset,
    ds_X: xr.Dataset,
    X: Literal["MVBS", "NASC"],
    x_interval: Union[pd.IntervalIndex, np.ndarray],
) -> xr.Dataset:
    """Helper function to get reduced positions

    Parameters
    ----------
    ds_Sv : xr.Dataset
        The input Sv dataset
    ds_X : xr.Dataset
        The input X dataset, either ``ds_MVBS`` or ``ds_NASC``
    X : {'MVBS', 'NASC'}
        The type of X dataset
    x_interval : pd.IntervalIndex or np.ndarray
        1D array or interval index representing
        the bins required for X dataset.

        MVBS: ``ping_time``
        NASC: ``distance_nmi``

    Returns
    -------
    xr.Dataset
        The X dataset with reduced position variables
        such as latitude and longitude
    """
    # Get positions if exists
    # otherwise return the input ds_X
    if all(v in ds_Sv for v in POSITION_VARIABLES):
        x_var = x_dim = "ping_time"
        if X == "NASC":
            x_var = "distance_nmi"
            x_dim = "distance"

        ds_Pos = xarray_reduce(
            ds_Sv[POSITION_VARIABLES],
            ds_Sv[x_var],
            func="nanmean",
            expected_groups=(x_interval),
            isbin=True,
            method="map-reduce",
        )

        for var in POSITION_VARIABLES:
            ds_X[var] = ([x_dim], ds_Pos[var].data, ds_Sv[var].attrs)
    return ds_X


def _groupby_x_along_channels(
    ds_Sv: xr.Dataset,
    range_interval: Union[pd.IntervalIndex, np.ndarray],
    x_interval: Union[pd.IntervalIndex, np.ndarray],
    x_var: Literal["ping_time", "distance_nmi"] = "ping_time",
    range_var: Literal["echo_range", "depth"] = "echo_range",
    method: str = "map-reduce",
    func: str = "nanmean",
    skipna: bool = True,
    **flox_kwargs,
) -> xr.Dataset:
    """
    Perform groupby of ``ds_Sv`` along each channel for the given
    intervals to get the 'sv' mean value.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        A Dataset containing ``Sv`` and other variables,
        depending on computation performed.

        For MVBS computation, this must contain ``Sv`` and ``echo_range`` data
        with coordinates ``channel``, ``ping_time``, and ``range_sample``
        at bare minimum.
        Or this can contain ``Sv`` and ``depth`` data with similar coordinates.

        For NASC computatioon this must contain ``Sv`` and ``depth`` data
        with coordinates ``channel``, ``distance_nmi``, and ``range_sample``.
    range_interval: pd.IntervalIndex or np.ndarray
        1D array or interval index representing
        the bins required for ``range_var``
    x_interval : pd.IntervalIndex or np.ndarray
        1D array or interval index representing
        the bins required for ``ping_time`` or ``distance_nmi``.
    x_var : {'ping_time', 'distance_nmi'}, default 'ping_time'
        The variable to use for x binning. This will determine
        if computation is for MVBS or NASC.
    range_var: {'echo_range', 'depth'}, default 'echo_range'
        The variable to use for range binning.
        Either ``echo_range`` or ``depth``.

        **For NASC, this must be ``depth``.**
    method: str
        The flox strategy for reduction of dask arrays only.
        See flox `documentation <https://flox.readthedocs.io/en/latest/implementation.html>`_
        for more details.
    func: str, default 'nanmean'
        The aggregation function used for reducing the data array.
        By default, 'nanmean' is used. Other options can be found in the flox `documentation
        <https://flox.readthedocs.io/en/latest/generated/flox.xarray.xarray_reduce.html>`_.
    skipna: bool, default True
        If true, aggregation function skips NaN values.
        Else, aggregation function includes NaN values.
        Note that if ``func`` is set to 'mean' and ``skipna`` is set to True, then aggregation
        will have the same behavior as if func is set to 'nanmean'.
    **flox_kwargs
        Additional keyword arguments to be passed
        to flox reduction function.

    Returns
    -------
    xr.Dataset
        The MVBS or NASC dataset of the input ``ds_Sv`` for all channels
    """
    # Check if x_var is valid, currently only support
    # ping_time and distance_nmi, which indicates
    # either a MVBS or NASC computation
    if x_var not in ["ping_time", "distance_nmi"]:
        raise ValueError("x_var must be 'ping_time' or 'distance_nmi'")

    # Set correct range_var just in case
    if x_var == "distance_nmi" and range_var != "depth":
        logger.warning("x_var is 'distance_nmi', setting range_var to 'depth'")
        range_var = "depth"

    # average should be done in linear domain
    sv = ds_Sv["Sv"].pipe(_log2lin)

    # Check for any NaNs in the coordinate arrays:
    named_arrays = {
        x_var: ds_Sv[x_var].data,
        range_var: ds_Sv[range_var].data,
    }
    aggregation_msg = (
        "Aggregation may be negatively impacted since Flox will not aggregate any "
        "```Sv``` values that have corresponding NaN coordinate values. Consider handling "
        "these values before calling your intended commongrid function."
    )
    for array_name, array in named_arrays.items():
        if np.isnan(array).any():
            logging.warning(
                f"The ```{array_name}``` coordinate array contain NaNs. {aggregation_msg}"
            )

    # reduce along ping_time or distance_nmi
    # and echo_range or depth
    # by binning and averaging
    sv_mean = xarray_reduce(
        sv,
        ds_Sv["channel"],
        ds_Sv[x_var],
        ds_Sv[range_var],
        expected_groups=(None, x_interval, range_interval),
        isbin=[False, True, True],
        method=method,
        func=func,
        skipna=skipna,
        **flox_kwargs,
    )
    return sv_mean
