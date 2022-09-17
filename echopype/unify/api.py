from typing import Union

import xarray as xr


def compute_MVBS(
    ds: xr.Dataset, range_bin_size: Union[int, float], ping_bin_size: str
) -> xr.Dataset:
    """
    Compute the mean volume backscattering strength (MVBS)
    based on intervals of range (``echo_range`` or ``depth``)
    and ``ping_time`` specified in physical units.

    Output of this function differs from that of ``compute_MVBS_index_binning``, which computes
    bin-averaged Sv according to intervals of ``echo_range`` and ``ping_time`` specified as
    index number.

    Parameters
    ----------
    ds : xr.Dataset
        An Sv dataset containing Sv and ``echo_range`` or ``depth`` [m]
    range_bin_size : Union[int, float]
        Bin size along ``echo_range`` or `depth`, default to ``20``
    ping_bin_size : str
        Bin size along ``ping_time`` in seconds, default to ``20S``

    Returns
    -------
    A dataset containing bin-averaged Sv.
    """
    pass


def compute_MVBS_index_binning(ds: xr.Dataset, range_sample_num=100, ping_num=100) -> xr.Dataset:
    """
    Compute Mean Volume Backscattering Strength (MVBS)
    based on intervals of ``range_sample`` and ping number (``ping_num``) specified in index number.

    Output of this function differs from that of ``compute_MVBS``, which computes
    bin-averaged Sv according to intervals of range (``echo_range``) and ``ping_time`` specified
    in physical units.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        An Sv dataset containing Sv
    range_sample_num : int
        Number of samples to average along the ``range_sample`` dimension, default to 100
    ping_num : int
        Number of pings to average, default to 100

    Returns
    -------
    A dataset containing bin-averaged Sv
    """
    pass


def regrid_Sv(df: xr.Dataset, range_bin_size: Union[int, float]) -> xr.Dataset:
    """
    Regrid the Sv data along the ``echo_range`` or ``depth`` dimension.

    Parameters
    ----------
    ds : xr.Dataset
        An dataset containing Sv and ``echo_range`` or ``depth`` [m]
    range_size : {int, float}
        Size of range bin to be regridded

    Returns
    -------
    A dataset containing regridded Sv.
    """
    pass
