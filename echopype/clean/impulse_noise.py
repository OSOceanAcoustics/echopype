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
]

import warnings

import numpy as np
import xarray as xr
from scipy.ndimage import median_filter
from skimage.morphology import dilation, erosion

from ..utils import mask_transformation
from ..utils.mask_transformation_xr import downsample, upsample

RYAN_DEFAULT_PARAMS = {"thr": 10, "m": 5, "n": 1}
RYAN_ITERABLE_DEFAULT_PARAMS = {"thr": 10, "m": 5, "n": (1, 2)}
WANG_DEFAULT_PARAMS = {
    "thr": (-70, -40),
    "erode": [(3, 3)],
    "dilate": [(5, 5), (7, 7)],
    "median": [(7, 7)],
}


def _wang(
    Sv_ds: xr.Dataset,
    desired_channel: str,
    parameters: dict = WANG_DEFAULT_PARAMS,
) -> xr.DataArray:
    """
    Clean impulse noise from Sv data following the method described by:

        Wang et al. (2015) ’A noise removal algorithm for acoustic data with
        strong interference based on post-processing techniques’, CCAMLR
        SG-ASAM: 15/02.

    This algorithm runs different cycles of erosion, dilation, and median
    filtering to clean impulse noise from Sv.
    Returns a boolean mask indicating the location of impulse noise in Sv data.

    Parameters
    ----------
        Sv_ds (xarray.Dataset): xr.DataArray with Sv data for multiple channels (dB).
        desired_channel (str): Name of the desired frequency channel.
        parameters {}: parameter dict, should contain:
            thr    : 2-element tuple with bottom/top Sv thresholds (dB).
            erode  : List of 2-element tuples indicating the window's size
                        for each erosion cycle.
            dilate : List of 2-element tuples indicating the window's size
                        for each dilation cycle.
            median : List of 2-element tuples indicating the window's size
                        for each median filter cycle.

    Returns
    -------
        xarray.DataArray: xr.DataArray with mask indicating the presence of impulse noise.

    Warning
    -------
    Input Sv data shouldn't contain NaN values.
    These values are not processed correctly by the impulse noise removal algorithm and
    will be marked as noise in the output mask.
    Please ensure that Sv data is cleaned or appropriately preprocessed
    before using this function.

    This method identifies the locations of noise in the Sv data but
    does not follow the exact same process as the wang function from echopy,
    which replaces the identified noise values with -999. The visual representation in echograms
    produced from the output of this method may therefore differ from those generated using
    the wang from echopy function. Users should take into
    account that regions marked as True in the returned mask have been identified as noise.


    """
    parameter_names = ("thr", "erode", "dilate", "median")
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be thr, erode, dilate, median, are"
            + str(parameters.keys())
        )
    thr = parameters["thr"]
    erode = parameters["erode"]
    dilate = parameters["dilate"]
    median = parameters["median"]

    # Select the desired frequency channel directly using 'sel'
    selected_channel_ds = Sv_ds.sel(channel=desired_channel)

    # Extract Sv values for the desired frequency channel
    Sv = selected_channel_ds["Sv"].values

    # Check if there are any NaN values in the Sv data
    if np.isnan(Sv).any():
        warnings.warn(
            "Input Sv data contains NaN values."
            "These values are not processed correctly by the impulse noise removal algorithm"
            "and will be marked as noise in the output mask."
            "Please ensure that Sv data is cleaned or appropriately "
            "preprocessed before using this function."
        )

    # Transpose the Sv data so that the vertical dimension is the first dimension (axis 0)
    Sv = np.transpose(Sv)

    """
    Call the wang function to get the cleaned Sv data and the mask indicating edges,
    where swarms analysis couldn't be performed
    The variable mask_ is a boolean mask where
    True represents edges where cleaning wasn't applied,
    and False represents areas where cleaning was applied
    """
    # Sv_cleaned, mask_ = _wang(Sv, thr, erode, dilate, median)
    # set weak noise and strong interference as vacant samples (-999)
    Sv_thresholded = Sv.copy()
    Sv_thresholded[(Sv < thr[0]) | (Sv > thr[1])] = -999

    # remaining weak interferences will take neighbouring vacant values
    # by running erosion cycles
    Sv_eroded = Sv.copy()
    for e in erode:
        Sv_eroded = erosion(Sv_thresholded, np.ones(e))

    # the last step might have turned interferences inside biology into vacant
    # samples, this is solved by running dilation cycles
    Sv_dilated = Sv_eroded.copy()
    for d in dilate:
        Sv_dilated = dilation(Sv_dilated, np.ones(d))

    # dilation has modified the Sv value of biological features, so these are
    # now corrected to corresponding Sv values before the erosion/dilation
    Sv_corrected = Sv_dilated.copy()
    mask_bio = (Sv_dilated >= thr[0]) & (Sv_dilated < thr[1])
    Sv_corrected[mask_bio] = Sv_thresholded[mask_bio]

    # compute median convolution in Sv corrected array
    Sv_median = Sv_corrected.copy()
    for m in median:
        Sv_median = mask_transformation.log(
            median_filter(mask_transformation.lin(Sv_median), footprint=np.ones(m))
        )

    # any vacant sample inside biological features will be corrected with
    # the median of corresponding neighbouring samples
    Sv_cleaned = Sv_corrected.copy()
    mask_bio = (Sv >= thr[0]) & (Sv < thr[1])
    mask_vacant = Sv_corrected == -999
    Sv_cleaned[mask_vacant & mask_bio] = Sv_median[mask_vacant & mask_bio]

    # get mask indicating edges, where swarms analysis couldn't be performed
    mask_ = np.ones_like(Sv_cleaned, dtype=bool)
    idx = int((max([e[0], d[0]]) - 1) / 2)
    jdx = int((max([e[1], d[1]]) - 1) / 2)
    mask_[idx:-idx, jdx:-jdx] = False

    # return Sv_corrected2, mask_

    """
    Create a boolean mask comparing the original and cleaned Sv data
    Creates a boolean mask where True denotes locations where the original Sv values
    are different from the cleaned Sv values.
    """

    noise_mask = Sv != Sv_cleaned

    # Combined mask
    # The bitwise negation ~ operator is applied to mask_.
    # So, ~mask_ is True where cleaning was applied and
    # False where cleaning wasn't applied (the edges).
    combined_mask = np.logical_and(~mask_, noise_mask)

    # Transpose the mask back to its original shape
    # Combined_mask is a mask that marks valid (non-edge) locations where
    # noise has been identified and cleaned.
    combined_mask = np.transpose(noise_mask)

    # Create a new xarray for the mask with the correct dimensions and coordinates
    mask_xr = xr.DataArray(
        combined_mask,
        dims=("ping_time", "range_sample"),
        coords={
            "ping_time": selected_channel_ds.ping_time.values,
            "range_sample": selected_channel_ds.range_sample.values,
        },
    )

    warnings.warn(
        "The output mask from this function identifies regions of noise in the Sv data, "
        "but does not modify them in the same way as the `wang` function from echopy."
        "Visualizations using this mask may therefore differ from"
        "those generated using the `wang` function from echopy. "
        "Be aware that regions marked as True in the mask are identified as noise."
    )

    return mask_xr


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
