"""
Algorithms for masking Impulse noise.

    Copyright (c) 2020 Echopy

    This code is licensed under the MIT License.
    See https://opensource.org/license/mit/ for details.
    Original code sourced from:
    https://github.com/open-ocean-sounding/echopy/blob/master/echopy/processing/mask_impulse.py

"""
__authors__ = [
    "Alejandro Ariza",  # wrote ryan(), ryan_iterable(), and wang()
    "Raluca Simedroni",  # adapted the impulse noise masking algorithms to echopype
    "Ruxandra Valcu",  # adapted the ryan algorithm to use xarray data structures
]

import numpy as np
import xarray as xr

from ..utils.mask_transformation_xr import downsample, upsample

RYAN_DEFAULT_PARAMS = {"thr": 10, "m": 5, "n": 1}
RYAN_ITERABLE_DEFAULT_PARAMS = {"thr": 10, "m": 5, "n": (1, 2)}
WANG_DEFAULT_PARAMS = {
    "thr": (-70, -40),
    "erode": [(3, 3)],
    "dilate": [(5, 5), (7, 7)],
    "median": [(7, 7)],
}


def _ryan(
    Sv_ds: xr.Dataset,
    desired_channel: str,
    parameters: dict = RYAN_DEFAULT_PARAMS,
) -> xr.DataArray:
    """
    Mask impulse noise following the two-sided comparison method described in:
    Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in
    open-ocean echo integration data’, ICES Journal of Marine Science, 72: 2482–2493.

    Parameters
    ----------
        Sv_ds (xarray.Dataset): xr.DataArray with Sv data for multiple channels (dB).
        desired_channel (str): Name of the desired frequency channel.
        parameters (dict): Dictionary of parameters. Must contain the following:
            m (int/float): Vertical binning length (n samples or range).
            n (int): Number of pings either side for comparisons.
            thr (int/float): User-defined threshold value (dB).

    Returns
    -------
        xarray.DataArray: xr.DataArray with IN mask.

    Notes
    -----
    In the original 'ryan' function (echopy), two masks are returned:
        - 'mask', where True values represent likely impulse noise, and
        - 'mask_', where True values represent valid samples for side comparison.

    When adapting for echopype, we must ensure the mask aligns with our data orientation.
    Hence, we transpose 'mask' and 'mask_' to match the shape of the data in 'Sv_ds'.

    Then, we create a combined mask using a bitwise AND operation between 'mask' and '~mask_'.

    """
    parameter_names = ("m", "n", "thr")
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError("Missing parameters - should be m, n, thr, are" + str(parameters.keys()))
    m = parameters["m"]
    n = parameters["n"]
    thr = parameters["thr"]

    # Select the desired frequency channel directly using 'sel'
    selected_channel_ds = Sv_ds.sel(channel=desired_channel)

    Sv = selected_channel_ds.Sv
    Sv_ = downsample(Sv, coordinates={"range_sample": m}, is_log=True)
    Sv_ = upsample(Sv_, Sv)

    # get valid sample mask
    mask = Sv_.isnull()

    # get IN mask
    forward = Sv_ - Sv_.shift(shifts={"ping_time": n}, fill_value=np.nan)
    backward = Sv_ - Sv_.shift(shifts={"ping_time": -n}, fill_value=np.nan)
    forward = forward.fillna(np.inf)
    backward = backward.fillna(np.inf)
    mask_in = (forward > thr) & (backward > thr)
    # add to the mask areas that have had data shifted out of range
    mask_in[0:n, :] = True
    mask_in[-n:, :] = True

    mask = mask | mask_in
    mask = mask.drop("channel")
    return mask


def _ryan_iterable(
    Sv_ds: xr.Dataset,
    desired_channel: str,
    parameters: dict = RYAN_ITERABLE_DEFAULT_PARAMS,
) -> xr.DataArray:
    """
    Modified from "ryan" so that the parameter "n" can be provided multiple
    times. It enables the algorithm to iterate and perform comparisons at
    different n distances. Resulting masks at each iteration are combined in
    a single mask. By setting multiple n distances the algorithm can detect
    spikes adjacent each other.

    Parameters
    ----------
        Sv_ds (xarray.Dataset): xr.DataArray with Sv data for multiple channels (dB).
        desired_channel (str): Name of the desired frequency channel.
        parameters (dict): Dictionary of parameters. Must contain the following:
            m (int/float): Vertical binning length (n samples or range).
            n (int): Number of pings either side for comparisons.
            thr (int/float): User-defined threshold value (dB).

    Returns
    -------
        xarray.DataArray: xr.DataArray with IN mask.


    Notes
    -----
    In the original 'ryan' function (echopy), two masks are returned:
        - 'mask', where True values represent likely impulse noise, and
        - 'mask_', where True values represent valid samples for side comparison.

    When adapting for echopype, we must ensure the mask aligns with our data orientation.
    Hence, we transpose 'mask' and 'mask_' to match the shape of the data in 'Sv_ds'.

    Then, we create a combined mask using a bitwise AND operation between 'mask' and '~mask_'.

    """
    parameter_names = ("m", "n", "thr")
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError("Missing parameters - should be m, n, thr, are" + str(parameters.keys()))
    m = parameters["m"]
    n = parameters["n"]
    thr = parameters["thr"]

    mask_list = []
    for n_i in n:
        parameter_dict = {"m": m, "n": n_i, "thr": thr}
        mask = _ryan(Sv_ds, desired_channel, parameter_dict)
        mask_list.append(mask)
    mask_xr = mask_list[0]
    if len(mask_list) > 1:
        for mask in mask_list[1:]:
            mask_xr |= mask
    return mask_xr
