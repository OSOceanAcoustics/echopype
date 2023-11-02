"""
Algorithms for masking seabed.
  These methods are based on:

    Copyright (c) 2020 Echopy

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    __authors__ = ['Alejandro Ariza'   # wrote maxSv(), deltaSv(), blackwell(),
                                       # blackwell_mod(), aliased2seabed(),
                                       # seabed2aliased(), ariza(), experimental()
  __authors__ = ['Mihai Boldeanu'
                  # adapted the mask seabed algorithms from the Echopy library
                  and implemented them for use with the Echopype library.
                ]
"""

import warnings

import numpy as np
import scipy.ndimage as nd_img
import xarray as xr
from scipy.signal import convolve2d
from skimage.measure import label
from skimage.morphology import dilation, erosion, remove_small_objects, square

from ..utils.mask_transformation import lin, log

MAX_SV_DEFAULT_PARAMS = {"r0": 10, "r1": 1000, "roff": 0, "thr": (-40, -60)}
DELTA_SV_DEFAULT_PARAMS = {"r0": 10, "r1": 1000, "roff": 0, "thr": 20}
BLACKWELL_DEFAULT_PARAMS = {
    "theta": None,
    "phi": None,
    "r0": 10,
    "r1": 1000,
    "tSv": -75,
    "ttheta": 702,
    "tphi": 282,
    "wtheta": 28,
    "wphi": 52,
}
BLACKWELL_MOD_DEFAULT_PARAMS = {
    "theta": None,
    "phi": None,
    "r0": 10,
    "r1": 1000,
    "tSv": -75,
    "ttheta": 702,
    "tphi": 282,
    "wtheta": 28,
    "wphi": 52,
    "rlog": None,
    "tpi": None,
    "freq": None,
    "rank": 50,
}
EXPERIMENTAL_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1000,
    "roff": 0,
    "thr": (-30, -70),
    "ns": 150,
    "n_dil": 3,
}
ARIZA_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1000,
    "roff": 0,
    "thr": -40,
    "ec": 1,
    "ek": (1, 3),
    "dc": 10,
    "dk": (3, 7),
}


def _maxSv(Sv_ds: xr.DataArray, desired_channel: str, parameters: dict = MAX_SV_DEFAULT_PARAMS):
    """
    Initially detects the seabed as the ping sample with the strongest Sv value,
    as long as it exceeds a dB threshold. Then it searches up along the ping
    until Sv falls below a secondary (lower) dB threshold, where the final
    seabed is set.

    Args:
        Sv_ds (xr.DataArray): xr.DataArray with Sv data for multiple channels (dB)
        desired_channel(str): Name of the desired frequency channel
        parameters: parameter dict, should contain:
            r0 (int): minimum range below which the search will be performed (m).
            r1 (int): maximum range above which the search will be performed (m).
            roff (int): seabed range offset (m).
            thr (tuple): 2 integers with 1st and 2nd Sv threshold (dB).

    Returns:
        xr.DataArray: A DataArray containing the mask for the Sv data.
            Regions satisfying the thresholding criteria are True, others are False
    """
    parameter_names = ["r0", "r1", "roff", "thr"]
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be: "
            + str(parameter_names)
            + ", are: "
            + str(parameters.keys())
        )
    r0 = parameters["r0"]
    r1 = parameters["r1"]
    roff = parameters["roff"]
    thr = parameters["thr"]

    channel_Sv = Sv_ds.sel(channel=desired_channel)
    Sv = channel_Sv["Sv"].values.T
    r = Sv_ds["echo_range"].values[0, 0]

    # get offset and range indexes
    roff = np.nanargmin(abs(r - roff))
    r0 = np.nanargmin(abs(r - r0))
    r1 = np.nanargmin(abs(r - r1))

    # get indexes for maximum Sv along every ping,
    idx = np.int64(np.zeros(Sv.shape[1]))
    idx[~np.isnan(Sv).all(axis=0)] = np.nanargmax(Sv[r0:r1, ~np.isnan(Sv).all(axis=0)], axis=0) + r0

    # indexes with maximum Sv < main threshold are discarded (=0)
    maxSv = Sv[idx, range(len(idx))]
    maxSv[np.isnan(maxSv)] = -999
    idx[maxSv < thr[0]] = 0

    # mask seabed, proceed only with acepted seabed indexes (!=0)
    idx = idx
    mask = np.zeros(Sv.shape, dtype=bool)
    for j, i in enumerate(idx):
        if i != 0:
            # decrease indexes until Sv mean falls below the 2nd threshold
            if np.isnan(Sv[i - 5 : i, j]).all():
                Svmean = thr[1] + 1
            else:
                Svmean = log(np.nanmean(lin(Sv[i - 5 : i, j])))

            while (Svmean > thr[1]) & (i >= 5):
                i -= 1

            # subtract range offset & mask all the way down
            i -= roff
            if i < 0:
                i = 0
            mask[i:, j] = True

    mask = np.logical_not(mask.T)
    return_mask = xr.DataArray(
        mask,
        dims=("ping_time", "range_sample"),
        coords={"ping_time": channel_Sv.ping_time, "range_sample": channel_Sv.range_sample},
    )
    return return_mask


def _deltaSv(Sv_ds: xr.DataArray, desired_channel: str, parameters: dict = MAX_SV_DEFAULT_PARAMS):
    """
    Examines the difference in Sv over a 2-samples moving window along
    every ping, and returns the range of the first value that exceeded
    a user-defined dB threshold (likely, the seabed).

    Args:
        Sv_ds (xr.DataArray): xr.DataArray with Sv data for multiple channels (dB)
        desired_channel(str): Name of the desired frequency channel
        parameters: parameter dict, should contain:
            r0 (int): minimum range below which the search will be performed (m).
            r1 (int): maximum range above which the search will be performed (m).
            roff (int): seabed range offset (m).
            thr (int): threshold value (dB).

    Returns:
        xr.DataArray: A DataArray containing the mask for the Sv data.
            Regions satisfying the thresholding criteria are True, others are False
    """
    parameter_names = ["r0", "r1", "roff", "thr"]
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be: "
            + str(parameter_names)
            + ", are: "
            + str(parameters.keys())
        )
    r0 = parameters["r0"]
    r1 = parameters["r1"]
    roff = parameters["roff"]
    thr = parameters["thr"]

    channel_Sv = Sv_ds.sel(channel=desired_channel)
    Sv = channel_Sv["Sv"].values.T
    r = Sv_ds["echo_range"].values[0, 0]

    # get offset as number of samples
    roff = np.nanargmin(abs(r - roff))

    # compute Sv difference along every ping
    Svdiff = np.diff(Sv, axis=0)
    dummy = np.zeros((1, Svdiff.shape[1])) * np.nan
    Svdiff = np.r_[dummy, Svdiff]

    # get range indexes
    r0 = np.nanargmin(abs(r - r0))
    r1 = np.nanargmin(abs(r - r1))

    # get indexes for the first value above threshold, along every ping
    idx = np.nanargmax((Svdiff[r0:r1, :] > thr), axis=0) + r0

    # mask seabed, proceed only with acepted seabed indexes (!=0)
    idx = idx
    mask = np.zeros(Sv.shape, dtype=bool)
    for j, i in enumerate(idx):
        if i != 0:
            # subtract range offset & mask all the way down
            i -= roff
            if i < 0:
                i = 0
            mask[i:, j] = True

    mask = np.logical_not(mask.T)
    return_mask = xr.DataArray(
        mask,
        dims=("ping_time", "range_sample"),
        coords={"ping_time": channel_Sv.ping_time, "range_sample": channel_Sv.range_sample},
    )
    return return_mask


def _blackwell(Sv_ds: xr.DataArray, desired_channel: str, parameters: dict = MAX_SV_DEFAULT_PARAMS):
    """
    Detects and mask seabed using the split-beam angle and Sv, based in
    "Blackwell et al (2019), Aliased seabed detection in fisheries acoustic
    data". Complete article here: https://arxiv.org/abs/1904.10736

    Args:
        Sv_ds (xr.DataArray): xr.DataArray with Sv data for multiple channels (dB)
        desired_channel(str): Name of the desired frequency channel
        parameters: parameter dict, should contain:
            r0 (int): minimum range below which the search will be performed (m)
            r1 (int): maximum range above which the search will be performed (m)
            tSv (float): Sv threshold above which seabed is pre-selected (dB)
            ttheta (int): Theta threshold above which seabed is pre-selected (dB)
            tphi (int): Phi threshold above which seabed is pre-selected (dB)
            wtheta (int): window's size for mean square operation in Theta field
            wphi (int): window's size for mean square operation in Phi field

    Returns:
        xr.DataArray: A DataArray containing the mask for the Sv data.
            Regions satisfying the thresholding criteria are True, others are False
    """
    parameter_names = ["r0", "r1", "tSv", "ttheta", "tphi", "wtheta", "wphi"]
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be: "
            + str(parameter_names)
            + ", are: "
            + str(parameters.keys())
        )
    r0 = parameters["r0"]
    r1 = parameters["r1"]
    tSv = parameters["tSv"]
    ttheta = parameters["ttheta"]
    tphi = parameters["tphi"]
    wtheta = parameters["wtheta"]
    wphi = parameters["wphi"]

    channel_Sv = Sv_ds.sel(channel=desired_channel)
    Sv = channel_Sv["Sv"].values.T
    r = Sv_ds["echo_range"].values[0, 0]
    theta = channel_Sv["angle_alongship"].values.T
    phi = channel_Sv["angle_athwartship"].values.T
    # apply reverse correction on theta & phi to match Blackwell's constants
    theta = theta * 22 * 128 / 180
    phi = phi * 22 * 128 / 180

    # delimit the analysis within user-defined range limits
    r0 = np.nanargmin(abs(r - r0))
    r1 = np.nanargmin(abs(r - r1)) + 1
    Svchunk = Sv[r0:r1, :]
    thetachunk = theta[r0:r1, :]
    phichunk = phi[r0:r1, :]

    # get blur kernels with theta & phi width dimensions
    ktheta = np.ones((wtheta, wtheta)) / wtheta**2
    kphi = np.ones((wphi, wphi)) / wphi**2

    # perform mean square convolution and mask if above theta & phi thresholds
    thetamaskchunk = convolve2d(thetachunk, ktheta, "same", boundary="symm") ** 2 > ttheta
    phimaskchunk = convolve2d(phichunk, kphi, "same", boundary="symm") ** 2 > tphi
    anglemaskchunk = thetamaskchunk | phimaskchunk

    # if aliased seabed, mask Sv above the Sv median of angle-masked regions
    if anglemaskchunk.any():
        Svmedian_anglemasked = log(np.nanmedian(lin(Svchunk[anglemaskchunk])))
        if np.isnan(Svmedian_anglemasked):
            Svmedian_anglemasked = np.inf
        if Svmedian_anglemasked < tSv:
            Svmedian_anglemasked = tSv
        Svmaskchunk = Svchunk > Svmedian_anglemasked

        # label connected items in Sv mask
        items = nd_img.label(Svmaskchunk, nd_img.generate_binary_structure(2, 2))[0]

        # get items intercepted by angle mask (likely, the seabed)
        intercepted = list(set(items[anglemaskchunk]))
        if 0 in intercepted:
            intercepted.remove(intercepted == 0)

        # combine angle-intercepted items in a single mask
        maskchunk = np.zeros(Svchunk.shape, dtype=bool)
        for i in intercepted:
            maskchunk = maskchunk | (items == i)

        # add data above r0 and below r1 (removed in first step)
        above = np.zeros((r0, maskchunk.shape[1]), dtype=bool)
        below = np.zeros((len(r) - r1, maskchunk.shape[1]), dtype=bool)
        mask = np.r_[above, maskchunk, below]

    # give empty mask if aliased-seabed was not detected in Theta & Phi
    else:
        warnings.warn(
            "No aliased seabed detected in Theta & Phi. "
            "A default mask with all True values is returned."
        )
        mask = np.zeros_like(Sv, dtype=bool)

    mask = np.logical_not(mask.T)
    return_mask = xr.DataArray(
        mask,
        dims=("ping_time", "range_sample"),
        coords={"ping_time": channel_Sv.ping_time, "range_sample": channel_Sv.range_sample},
    )
    return return_mask


def _blackwell_mod(
    Sv_ds: xr.DataArray, desired_channel: str, parameters: dict = MAX_SV_DEFAULT_PARAMS
):
    """
    Detects and mask seabed using the split-beam angle and Sv, based in
    "Blackwell et al (2019), Aliased seabed detection in fisheries acoustic
    data". Complete article here: https://arxiv.org/abs/1904.10736

    This is a modified version from the original algorithm. It includes extra
    arguments to evaluate whether aliased seabed items can occur, given the
    true seabed detection range, and the possibility of tuning the percentile's
    rank.

    Args:
        Sv_ds (xr.DataArray): xr.DataArray with Sv data for multiple channels (dB)
        desired_channel(str): Name of the desired frequency channel
        parameters: parameter dict, should contain:
            r0 (int): minimum range below which the search will be performed (m)
            r1 (int): maximum range above which the search will be performed (m)
            tSv (float): Sv threshold above which seabed is pre-selected (dB)
            ttheta (int): Theta threshold above which seabed is pre-selected (dB)
            tphi (int): Phi threshold above which seabed is pre-selected (dB)
            wtheta (int): window's size for mean square operation in Theta field
            wphi (int): window's size for mean square operation in Phi field
            rlog (float): Maximum logging range of the echosounder (m)
            tpi (float): Transmit pulse interval, or ping rate (s)
            freq (int): frequecy (kHz)
            rank (int): Rank for percentile operation: [0, 100]

    Returns:
        xr.DataArray: A DataArray containing the mask for the Sv data.
            Regions satisfying the thresholding criteria are True, others are False
    """
    parameter_names = [
        "r0",
        "r1",
        "tSv",
        "ttheta",
        "tphi",
        "wtheta",
        "wphi",
        "rlog",
        "tpi",
        "freq",
        "rank",
    ]
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be: "
            + str(parameter_names)
            + ", are: "
            + str(parameters.keys())
        )
    r0 = parameters["r0"]
    r1 = parameters["r1"]
    tSv = parameters["tSv"]
    ttheta = parameters["ttheta"]
    tphi = parameters["tphi"]
    wtheta = parameters["wtheta"]
    wphi = parameters["wphi"]
    rlog = parameters["rlog"]
    tpi = parameters["tpi"]
    freq = parameters["freq"]
    rank = parameters["rank"]

    channel_Sv = Sv_ds.sel(channel=desired_channel)
    Sv = channel_Sv["Sv"].values.T
    r = Sv_ds["echo_range"].values[0, 0]
    theta = channel_Sv["angle_alongship"].values.T
    phi = channel_Sv["angle_athwartship"].values.T
    # apply reverse correction on theta & phi to match Blackwell's constants
    theta = theta * 22 * 128 / 180
    phi = phi * 22 * 128 / 180

    # raise errors if wrong arguments
    if r0 > r1:
        raise Exception("Minimum range has to be shorter than maximum range")

    # give empty mask if searching range is outside the echosounder range
    if (r0 > r[-1]) or (r1 < r[0]):
        warnings.warn(
            "Search range is outside the echosounder range."
            "A default mask with all True values is returned."
        )
        mask = np.zeros_like(Sv, dtype=bool)

    # delimit the analysis within user-defined range limits
    i0 = np.nanargmin(abs(r - r0))
    i1 = np.nanargmin(abs(r - r1)) + 1
    Svchunk = Sv[i0:i1, :]
    thetachunk = theta[i0:i1, :]
    phichunk = phi[i0:i1, :]

    # get blur kernels with theta & phi width dimensions
    ktheta = np.ones((wtheta, wtheta)) / wtheta**2
    kphi = np.ones((wphi, wphi)) / wphi**2

    # perform mean square convolution and mask if above theta & phi thresholds
    thetamaskchunk = convolve2d(thetachunk, ktheta, "same", boundary="symm") ** 2 > ttheta
    phimaskchunk = convolve2d(phichunk, kphi, "same", boundary="symm") ** 2 > tphi
    anglemaskchunk = thetamaskchunk | phimaskchunk

    # remove aliased seabed items when estimated True seabed can not be
    # detected below the logging range
    if (rlog is not None) and (tpi is not None) and (freq is not None):
        items = label(anglemaskchunk)
        item_labels = np.unique(label(anglemaskchunk))[1:]
        for il in item_labels:
            item = items == il
            ritem = np.nanmean(r[i0:i1][np.where(item)[0]])
            rseabed = _aliased2seabed(ritem, rlog, tpi, freq)
            if rseabed == []:
                anglemaskchunk[item] = False

    anglemaskchunk = anglemaskchunk & (Svchunk > tSv)

    # if aliased seabed, mask Sv above the Sv median of angle-masked regions
    if anglemaskchunk.any():
        Svmedian_anglemasked = log(np.nanpercentile(lin(Svchunk[anglemaskchunk]), rank))
        if np.isnan(Svmedian_anglemasked):
            Svmedian_anglemasked = np.inf
        if Svmedian_anglemasked < tSv:
            Svmedian_anglemasked = tSv
        Svmaskchunk = Svchunk > Svmedian_anglemasked

        # label connected items in Sv mask
        items = nd_img.label(Svmaskchunk, nd_img.generate_binary_structure(2, 2))[0]

        # get items intercepted by angle mask (likely, the seabed)
        intercepted = list(set(items[anglemaskchunk]))
        if 0 in intercepted:
            intercepted.remove(intercepted == 0)

        # combine angle-intercepted items in a single mask
        maskchunk = np.zeros(Svchunk.shape, dtype=bool)
        for i in intercepted:
            maskchunk = maskchunk | (items == i)

        # add data above r0 and below r1 (removed in first step)
        above = np.zeros((i0, maskchunk.shape[1]), dtype=bool)
        below = np.zeros((len(r) - i1, maskchunk.shape[1]), dtype=bool)
        mask = np.r_[above, maskchunk, below]

    # give empty mask if aliased-seabed was not detected in Theta & Phi
    else:
        warnings.warn(
            "Aliased seabed not detected in Theta & Phi."
            "A default mask with all True values is returned."
        )
        mask = np.zeros_like(Sv, dtype=bool)

    mask = np.logical_not(mask.T)
    return_mask = xr.DataArray(
        mask,
        dims=("ping_time", "range_sample"),
        coords={"ping_time": channel_Sv.ping_time, "range_sample": channel_Sv.range_sample},
    )
    return return_mask


def _aliased2seabed(
    aliased, rlog, tpi, f, c=1500, rmax={18: 7000, 38: 2800, 70: 1100, 120: 850, 200: 550}
):
    """
    Estimate true seabed, given the aliased seabed range. It might provide
    a list of ranges, corresponding to seabed reflections from several pings
    before, or provide an empty list if true seabed occurs within the logging
    range or beyond the maximum detection range.

    Args:
      aliased (float): Range of aliased seabed (m).
      rlog (float): Maximum logging range (m).
      tpi (float): Transmit pulse interval (s).
      f (int): Frequency (kHz).
      c (int): Sound speed in seawater (m s-1). Defaults to 1500.
      rmax (dict): Maximum seabed detection range per frequency. Defaults
                   to {18:7000, 38:2800, 70:1100, 120:850, 200:550}.

    Returns:
        float: list with estimated seabed ranges, reflected from preceding
        pings (ping -1, ping -2, ping -3, etc.).

    """
    ping = 0
    seabed = 0
    seabeds = []
    while seabed <= rmax[f]:
        ping = ping + 1
        seabed = (c * tpi * ping) / 2 + aliased
        if (seabed > rlog) & (seabed < rmax[f]):
            seabeds.append(seabed)

    return seabeds


def _seabed2aliased(
    seabed, rlog, tpi, f, c=1500, rmax={18: 7000, 38: 2800, 70: 1100, 120: 850, 200: 550}
):
    """
    Estimate aliased seabed range, given the true seabed range. The answer will
    be 'None' if true seabed occurs within the logging range or if it's beyond
    the detection limit of the echosounder.

    Args:
        seabed (float): True seabed range (m).
        rlog (float): Maximum logging range (m).
        tpi (float): Transmit pulse interval (s).
        f (int): frequency (kHz).
        c (float): Sound speed in seawater (m s-1). Defaults to 1500.
        rmax (dict): Maximum seabed detection range per frequency. Defaults
                     to {18:7000, 38:2800, 70:1100, 120:850, 200:550}.

    Returns:
        float: Estimated range of aliased seabed (m

    """
    if (not seabed < rlog) and (not seabed > rmax[f]):
        aliased = ((2 * seabed) % (c * tpi)) / 2
    else:
        aliased = None

    return aliased


def _experimental(
    Sv_ds: xr.DataArray, desired_channel: str, parameters: dict = MAX_SV_DEFAULT_PARAMS
):
    """
    Mask Sv above a threshold to get a potential seabed mask. Then, the mask is
    dilated to fill seabed breaches, and small objects are removed to prevent
    masking high Sv features that are not seabed (e.g. fish schools or spikes).
    Once this is done, the mask is built up until Sv falls below a 2nd
    threshold, Finally, the mask is extended all the way down.

    Args:
        Sv_ds (xr.DataArray): xr.DataArray with Sv data for multiple channels (dB)
        desired_channel(str): Name of the desired frequency channel
        parameters: parameter dict, should contain:
            r0 (int): minimum range below which the search will be performed (m).
            r1 (int): maximum range above which the search will be performed (m).
            roff (int): seabed range offset (m).
            thr (tuple): 2 integers with 1st and 2nd Sv threshold (dB).
            ns (int): maximum number of samples for an object to be removed.
            n_dil (int): number of dilations performed to the seabed mask.

    Returns:
        xr.DataArray: A DataArray containing the mask for the Sv data.
            Regions satisfying the thresholding criteria are True, others are False
    """
    parameter_names = ["r0", "r1", "roff", "thr", "ns", "n_dil"]
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be: "
            + str(parameter_names)
            + ", are: "
            + str(parameters.keys())
        )
    r0 = parameters["r0"]
    r1 = parameters["r1"]
    roff = parameters["roff"]
    thr = parameters["thr"]
    ns = parameters["ns"]
    n_dil = parameters["n_dil"]

    channel_Sv = Sv_ds.sel(channel=desired_channel)
    Sv = channel_Sv["Sv"].values.T
    r = Sv_ds["echo_range"].values[0, 0]

    # get indexes for range offset and range limits
    roff = np.nanargmin(abs(r - roff))
    r0 = np.nanargmin(abs(r - r0))
    r1 = np.nanargmin(abs(r - r1)) + 1

    # mask Sv above the first Sv threshold
    mask = Sv[r0:r1, :] > thr[0]
    maskabove = np.zeros((r0, mask.shape[1]), dtype=bool)
    maskbelow = np.zeros((len(r) - r1, mask.shape[1]), dtype=bool)
    mask = np.r_[maskabove, mask, maskbelow]

    # remove small to prevent other high Sv features to be masked as seabed
    # (e.g fish schools, impulse noise not properly masked. etc)
    mask = remove_small_objects(mask, ns)

    # dilate mask to fill seabed breaches
    # (e.g. attenuated pings or gaps from previous masking)
    mask = dilation(np.uint8(mask), square(n_dil))
    mask = np.array(mask, dtype="bool")

    # proceed with the following only if seabed was detected
    idx = np.argmax(mask, axis=0)
    for j, i in enumerate(idx):
        if i != 0:
            # rise up seabed until Sv falls below the 2nd threshold
            while (log(np.nanmean(lin(Sv[i - 5 : i, j]))) > thr[1]) & (i >= 5):
                i -= 1

            # subtract range offset & mask all the way down
            i -= roff
            if i < 0:
                i = 0
            mask[i:, j] = True

    #    # dilate again to ensure not leaving seabed behind
    #    kernel = np.ones((3,3))
    #    mask = cv2.dilate(np.uint8(mask), kernel, iterations = 2)
    #    mask = np.array(mask, dtype = 'bool')

    mask = np.logical_not(mask.T)
    return_mask = xr.DataArray(
        mask,
        dims=("ping_time", "range_sample"),
        coords={"ping_time": channel_Sv.ping_time, "range_sample": channel_Sv.range_sample},
    )
    return return_mask


def _ariza(Sv_ds: xr.DataArray, desired_channel: str, parameters: dict = MAX_SV_DEFAULT_PARAMS):
    """
    Mask Sv above a threshold to get potential seabed features. These features
    are eroded first to get rid of fake seabeds (spikes, schools, etc.) and
    dilated afterwards to fill in seabed breaches. Seabed detection is coarser
    than other methods (it removes water nearby the seabed) but the seabed line
    never drops when a breach occurs. Suitable for pelagic assessments and
    reconmended for non-supervised processing.

    Args:
        Sv_ds (xr.DataArray): xr.DataArray with Sv data for multiple channels (dB)
        desired_channel(str): Name of the desired frequency channel
        parameters: parameter dict, should contain:
            r0 (int): minimum range below which the search will be performed (m).
            r1 (int): maximum range above which the search will be performed (m).
            roff (int): seabed range offset (m).
            thr (int): Sv threshold above which seabed might occur (dB).
            ec (int): number of erosion cycles.
            ek (int): 2-elements tuple with vertical and horizontal dimensions
                      of the erosion kernel.
            dc (int): number of dilation cycles.
            dk (int): 2-elements tuple with vertical and horizontal dimensions
                      of the dilation kernel.

    Returns:
        xr.DataArray: A DataArray containing the mask for the Sv data.
            Regions satisfying the thresholding criteria are True, others are False
    """
    parameter_names = ["r0", "r1", "roff", "thr", "ec", "ek", "dc", "dk"]
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be: "
            + str(parameter_names)
            + ", are: "
            + str(parameters.keys())
        )
    r0 = parameters["r0"]
    r1 = parameters["r1"]
    roff = parameters["roff"]
    thr = parameters["thr"]
    ec = parameters["ec"]
    ek = parameters["ek"]
    dc = parameters["dc"]
    dk = parameters["dk"]

    channel_Sv = Sv_ds.sel(channel=desired_channel)
    Sv = channel_Sv["Sv"].values.T
    r = Sv_ds["echo_range"].values[0, 0]

    # raise errors if wrong arguments
    if r0 > r1:
        raise Exception("Minimum range has to be shorter than maximum range")

    # give empty mask if searching range is outside the echosounder range
    if (r0 > r[-1]) or (r1 < r[0]):
        warnings.warn(
            "Search range is outside the echosounder range. "
            "A default mask with all True values is returned."
        )
        mask = np.zeros_like(Sv, dtype=bool)

    # get indexes for range offset and range limits
    r0 = np.nanargmin(abs(r - r0))
    r1 = np.nanargmin(abs(r - r1))
    roff = np.nanargmin(abs(r - roff))

    # set to -999 shallow and deep waters (prevents seabed detection)
    Sv_ = Sv.copy()
    Sv_[0:r0, :] = -999
    Sv_[r1:, :] = -999

    # give empty mask if there is nothing above threshold
    if not (Sv_ > thr).any():
        warnings.warn(
            "Nothing found above the threshold. " "A default mask with all True values is returned."
        )
        mask = np.zeros_like(Sv_, dtype=bool)

    # search for seabed otherwise
    else:
        # potential seabed will be everything above the threshold, the rest
        # will be set as -999
        seabed = Sv_.copy()
        seabed[Sv_ < thr] = -999

        # run erosion cycles to remove fake seabeds (e.g: spikes, small shoals)
        for i in range(ec):
            seabed = erosion(seabed, np.ones(ek))

        # run dilation cycles to fill seabed breaches
        for i in range(dc):
            seabed = dilation(seabed, np.ones(dk))

        # mask as seabed everything greater than -999
        mask = seabed > -999

        # if seabed occur in a ping...
        idx = np.argmax(mask, axis=0)
        for j, i in enumerate(idx):
            if i != 0:
                # ...apply range offset & mask all the way down
                i -= roff
                if i < 0:
                    i = 0
                mask[i:, j] = True

    mask = np.logical_not(mask.T)
    return_mask = xr.DataArray(
        mask,
        dims=("ping_time", "range_sample"),
        coords={"ping_time": channel_Sv.ping_time, "range_sample": channel_Sv.range_sample},
    )
    return return_mask
