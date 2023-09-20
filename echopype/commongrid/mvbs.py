"""
Contains core functions needed to compute the MVBS of an input dataset.
"""

from typing import Literal, Union

import numpy as np
import pandas as pd
import xarray as xr
from flox.xarray import xarray_reduce

from ..consolidate.api import POSITION_VARIABLES
from ..utils.compute import _lin2log, _log2lin


def get_MVBS_along_channels(
    ds_Sv: xr.Dataset,
    range_interval: Union[pd.IntervalIndex, np.ndarray],
    ping_interval: Union[pd.IntervalIndex, np.ndarray],
    range_var: Literal["echo_range", "depth"] = "echo_range",
    method: str = "map-reduce",
    **kwargs
) -> xr.Dataset:
    """
    Computes the MVBS of ``ds_Sv`` along each channel for the given
    intervals.

    Parameters
    ----------
    ds_Sv: xr.Dataset
        A Dataset containing ``Sv`` and ``echo_range`` data with coordinates
        ``channel``, ``ping_time``, and ``range_sample``
    range_interval: pd.IntervalIndex or np.ndarray
        1D array or interval index representing
        the bins required for ``range_var``
    ping_interval: pd.IntervalIndex or np.ndarray
        1D array or interval index representing
        the bins required for ``ping_time``
    range_var: str
        The variable to use for range binning.
        Either ``echo_range`` or ``depth``.
    method: str
        The flox strategy for reduction of dask arrays only.
        See flox `documentation <https://flox.readthedocs.io/en/latest/implementation.html>`_
        for more details.
    **kwargs
        Additional keyword arguments to be passed
        to flox reduction function

    Returns
    -------
    xr.Dataset
        The MVBS dataset of the input ``ds_Sv`` for all channels
    """

    # average should be done in linear domain
    sv = ds_Sv["Sv"].pipe(_log2lin)

    # Get positions if exists
    # otherwise just use an empty dataset
    ds_Pos = xr.Dataset(attrs={"has_positions": False})
    if all(v in ds_Sv for v in POSITION_VARIABLES):
        ds_Pos = xarray_reduce(
            ds_Sv[POSITION_VARIABLES],
            ds_Sv["ping_time"],
            func="nanmean",
            expected_groups=(ping_interval),
            isbin=True,
            method=method,
        )
        ds_Pos.attrs["has_positions"] = True

    # reduce along ping_time and echo_range or depth
    # by binning and averaging
    mvbs = xarray_reduce(
        sv,
        sv["channel"],
        ds_Sv["ping_time"],
        ds_Sv[range_var],
        func="nanmean",
        expected_groups=(None, ping_interval, range_interval),
        isbin=[False, True, True],
        method=method,
        **kwargs
    )

    # apply inverse mapping to get back to the original domain and store values
    da_MVBS = mvbs.pipe(_lin2log)
    return xr.merge([ds_Pos, da_MVBS])
