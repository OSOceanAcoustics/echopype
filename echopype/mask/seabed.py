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

import dask.array as da
import numpy as np
import xarray as xr
from dask_image.ndfilters import convolve
from dask_image.ndmeasure import label
from dask_image.ndmorph import binary_dilation, binary_erosion

from ..utils.mask_transformation_xr import dask_nanpercentile, line_to_square

MAX_SV_DEFAULT_PARAMS = {"r0": 10, "r1": 1000, "roff": 0, "thr": (-40, -60)}
DELTA_SV_DEFAULT_PARAMS = {"r0": 10, "r1": 1000, "roff": 0, "thr": 20}
BLACKWELL_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1000,
    "tSv": -75,
    "ttheta": 702,
    "tphi": 282,
    "wtheta": 28,
    "wphi": 52,
}
BLACKWELL_MOD_DEFAULT_PARAMS = {
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
    "ek": (3, 3),
    "dc": 3,
    "dk": (3, 3),
}

ARIZA_SPIKE_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1000,
    "roff": 0,
    "thr": (-40, -40),
    "ec": 1,
    "ek": (3, 3),
    "dc": 3,
    "dk": (3, 3),
    "maximum_spike": 200,
}

ARIZA_EXPERIMENTAL_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1000,
    "roff": 0,
    "thr": (-40, -70),
    "ec": 1,
    "ek": (3, 3),
    "dc": 3,
    "dk": (3, 3),
}


def _get_seabed_range(mask: xr.DataArray):
    """
    Given a seabed mask, returns the range_sample depth of the seabed

    Args:
        mask (xr.DataArray): seabed mask

    Returns:
        xr.DataArray: a ping_time-sized array containing the range_sample seabed depth,
        or max range_sample if no seabed is detected

    """
    seabed_depth = mask.argmax(dim="range_sample").compute()
    seabed_depth[seabed_depth == 0] = mask.range_sample.max().item()
    return seabed_depth


def _morpho(mask: xr.DataArray, operation: str, c: int, k: int):
    """
    Given a preexisting 1/0 mask, run erosion or dilation cycles on it to remove noise

    Args:
        mask (xr.DataArray): xr.DataArray with 1 and 0 data
        operation(str): dilation, erosion
        c (int): number of cycles.
        k (int): 2-elements tuple with vertical and horizontal dimensions
                      of the kernel.

    Returns:
        xr.DataArray: A DataArray containing the denoised mask.
            Regions satisfying the criteria are 1, others are 0
    """
    function_dict = {"dilation": binary_dilation, "erosion": binary_erosion}

    if c > 0:
        dask_mask = da.asarray(mask, allow_unknown_chunksizes=False)
        dask_mask.compute_chunk_sizes()
        dask_mask = function_dict[operation](
            dask_mask,
            structure=da.ones(shape=k, dtype=bool),
            iterations=c,
        ).compute()
        dask_mask = da.asarray(dask_mask, allow_unknown_chunksizes=False)
        dask_mask.compute()
        mask.values = dask_mask.compute()
    return mask


def _erode_dilate(mask: xr.DataArray, ec: int, ek: int, dc: int, dk: int):
    """
    Given a preexisting 1/0 mask, run erosion and dilation cycles on it to remove noise

    Args:
        mask (xr.DataArray): xr.DataArray with 1 and 0 data
        ec (int): number of erosion cycles.
        ek (int): 2-elements tuple with vertical and horizontal dimensions
                      of the erosion kernel.
        dc (int): number of dilation cycles.
        dk (int): 2-elements tuple with vertical and horizontal dimensions
                      of the dilation kernel.

    Returns:
        xr.DataArray: A DataArray containing the denoised mask.
            Regions satisfying the criteria are 1, others are 0
    """
    mask = _morpho(mask, "erosion", ec, ek)
    mask = _morpho(mask, "dilation", dc, dk)
    return mask


def _create_range_mask(Sv_ds: xr.DataArray, desired_channel: str, thr: int, r0: int, r1: int):
    """
    Return a raw threshold/range mask for a certain dataset and desired channel

    Args:
        Sv_ds (xr.DataArray): xr.DataArray with Sv data for multiple channels (dB)
        desired_channel(str): Name of the desired frequency channel
        r0 (int): minimum range below which the search will be performed (m).
        r1 (int): maximum range above which the search will be performed (m).
        thr (int): Sv threshold above which seabed might occur (dB).

    Returns:
        dict: a dict containing the mask and whether or not further processing is necessary
            mask (xr.DataArray): a basic range/threshold mask.
                                Regions satisfying the criteria are 1, others are 0
            ok (bool): should the mask be further processed  or is there no data to be found?

    """
    channel_Sv = Sv_ds.sel(channel=desired_channel)
    Sv = channel_Sv["Sv"]
    r = channel_Sv["echo_range"][0]

    # return empty mask if searching range is outside the echosounder range
    if (r0 > r[-1]) or (r1 < r[0]):
        # Raise a warning to inform the user
        warnings.warn(
            "The searching range is outside the echosounder range. "
            "A default mask with all True values is returned, "
            "which won't mask any data points in the dataset."
        )
        mask = xr.DataArray(
            np.ones_like(Sv, dtype=bool),
            dims=("ping_time", "range_sample"),
            coords={"ping_time": Sv.ping_time, "range_sample": Sv.range_sample},
        )
        return {"mask": mask, "ok": False, "Sv": Sv, "range": r}

    # get upper and lower range indexes
    up = abs(r - r0).argmin(dim="range_sample").item()
    lw = abs(r - r1).argmin(dim="range_sample").item()

    # get threshold mask with shallow and deep waters masked
    mask = xr.where(Sv > thr, 1, 0).drop("channel")
    mask.fillna(0)
    range_filter = (mask["range_sample"] >= up) & (mask["range_sample"] <= lw)
    mask = mask.where(range_filter, other=0)

    # give empty mask if there is nothing above threshold
    if mask.sum() == 0:
        warnings.warn(
            "Nothing found above the threshold. " "A default mask with all True values is returned."
        )
        mask = xr.DataArray(
            np.ones_like(Sv, dtype=bool),
            dims=("ping_time", "range_sample"),
            coords={"ping_time": Sv.ping_time, "range_sample": Sv.range_sample},
        )
        return {"mask": mask, "ok": False, "Sv": Sv, "range": r}
    return {"mask": mask, "ok": True, "Sv": Sv, "range": r}


def _mask_down(mask: xr.DataArray):
    """
    Given a seabed mask, masks all signal under the detected seabed

    Args:
          mask (xr.DataArray): seabed mask

    Returns:
           xr.DataArray(mask with area under seabed masked)
    """
    seabed_depth = _get_seabed_range(mask)
    mask = (mask["range_sample"] <= seabed_depth).transpose()
    return mask


# move to utils and rewrite transient noise/fielding to use this once merged
def _erase_floating_zeros(mask: xr.DataArray):
    """
    Given a boolean mask, turns back to True any "floating" False values,
    e.g. not attached to the max range

    Args:
        mask: xr.DataArray - mask to remove floating values from

    Returns:
        xr.DataArray - mask with floating False values removed

    """
    flipped_mask = mask.isel(range_sample=slice(None, None, -1))
    flipped_mask["range_sample"] = mask["range_sample"]
    ft = len(flipped_mask.range_sample) - flipped_mask.argmax(dim="range_sample")

    first_true_indices = xr.DataArray(
        line_to_square(ft, mask, dim="range_sample").transpose(),
        dims=("ping_time", "range_sample"),
        coords={"ping_time": mask.ping_time, "range_sample": mask.range_sample},
    )

    indices = xr.DataArray(
        line_to_square(mask["range_sample"], mask, dim="ping_time"),
        dims=("ping_time", "range_sample"),
        coords={"ping_time": mask.ping_time, "range_sample": mask.range_sample},
    )
    spike_mask = mask.where(indices > first_true_indices, True)

    mask = spike_mask
    return mask


def _experimental_correction(mask: xr.DataArray, Sv: xr.DataArray, thr: int):
    """
    Given an existing seabed mask, the single-channel dataset it was created on
    and a secondary, lower threshold, it builds the mask up until the Sv falls below the threshold

    Args:
          mask (xr.DataArray): seabed mask
          Sv (xr.DataArray): single-channel Sv data the mask was build on
          thr (int): secondary threshold

    Returns:
          xr.DataArray: mask with secondary threshold correction applied

    """
    secondary_mask = xr.where(Sv < thr, 1, 0).drop("channel")
    secondary_mask.fillna(1)
    fill_mask = secondary_mask & mask
    spike_mask = _erase_floating_zeros(fill_mask)
    return spike_mask


def _cut_spikes(mask: xr.DataArray, maximum_spike: int):
    """
    In the Ariza seabed detecting method, large shoals can be falsely detected as
    seabed. Their appearance on the seabed mask is large vertical "spikes".
    We want to remove any such spikes from the dataset using maximum_spike
    as a control parameter.

    If this option is used, we also recommend applying the _experimental_correction,
    even if with the same threshold as the initial threshold, to fill up any
    imprecisions in the interpolated seabed

    Args:
        mask (xr.DataArray):
        maximum_spike(int): maximum height, in range samples, acceptable before
                            we start removing that data

    Returns:
        xr.DataArray: the corrected mask
    """
    int_mask = (~mask.copy()).astype(int)
    seabed = _get_seabed_range(int_mask)
    shifted_seabed = seabed.shift(ping_time=-1, fill_value=seabed[-1])
    spike = seabed - shifted_seabed
    spike_sign = xr.where(abs(spike) > maximum_spike, xr.where(spike > 0, 1, -1), 0)
    spike_cs = spike_sign.cumsum(dim="ping_time")
    # for i in spike:
    #     print(i.item())

    # mask spikes
    nan_mask = xr.where(spike_cs > 0, np.nan, xr.where(spike_sign == -1, np.nan, mask))

    # fill in with interpolated values from non-spikes
    mask_interpolated = nan_mask.interpolate_na(dim="ping_time", method="nearest").astype(bool)

    return mask_interpolated


def _ariza(Sv_ds: xr.DataArray, desired_channel: str, parameters: dict = ARIZA_DEFAULT_PARAMS):
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
                Can be a tuple, case in which a secondary experimental
                thresholding correction is applied
            ec (int): number of erosion cycles.
            ek (int): 2-elements tuple with vertical and horizontal dimensions
                      of the erosion kernel.
            dc (int): number of dilation cycles.
            dk (int): 2-elements tuple with vertical and horizontal dimensions
                      of the dilation kernel.
            maximum_spike(int): optional, if not None, used to determine the maximum
                    allowed height of the "spikes" potentially created by
                    dense shoals before masking them out. If used, applying
                    a secondary threshold correction is recommended


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
    thr = parameters["thr"]
    ec = parameters["ec"]
    ek = parameters["ek"]
    dc = parameters["dc"]
    dk = parameters["dk"]
    secondary_thr = None
    maximum_spike = None
    if "maximum_spike" in parameters.keys():
        maximum_spike = parameters["maximum_spike"]

    if isinstance(thr, int) is False:
        secondary_thr = thr[1]
        thr = thr[0]

    # create raw range and threshold mask, if no seabed is detected return empty
    raw = _create_range_mask(Sv_ds, desired_channel=desired_channel, thr=thr, r0=r0, r1=r1)
    mask = raw["mask"]
    if raw["ok"] is False:
        return mask

    # run erosion and dilation denoising cycles
    mask = _erode_dilate(mask, ec, ek, dc, dk)

    # mask areas under the detected seabed
    mask = _mask_down(mask)

    # apply spike correction
    if maximum_spike is not None:
        mask = _cut_spikes(mask, maximum_spike)

    # apply experimental correction, if specified
    if secondary_thr is not None:
        mask = _experimental_correction(mask, raw["Sv"], secondary_thr)

    return mask


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

    rlog = None
    tpi = None
    freq = None
    rank = 50

    if "rlog" in parameters.keys():
        rlog = parameters["rlog"]
    if "tpi" in parameters.keys():
        tpi = parameters["tpi"]
    if "freq" in parameters.keys():
        freq = parameters["freq"]
    if "rank" in parameters.keys():
        rank = parameters["rank"]

    channel_Sv = Sv_ds.sel(channel=desired_channel).drop("channel")
    Sv = channel_Sv["Sv"]
    r = channel_Sv["echo_range"][0]
    theta = channel_Sv["angle_alongship"].copy() * 22 * 128 / 180
    phi = channel_Sv["angle_athwartship"].copy() * 22 * 128 / 180

    dask_theta = da.asarray(theta, allow_unknown_chunksizes=False)
    dask_theta.compute_chunk_sizes()
    theta.values = (
        convolve(
            dask_theta,
            weights=da.ones(shape=(wtheta, wtheta), dtype=float) / wtheta**2,
            mode="nearest",
        )
        ** 2
    )

    dask_phi = da.asarray(phi, allow_unknown_chunksizes=False)
    dask_phi.compute_chunk_sizes()
    phi.values = (
        convolve(
            dask_phi,
            weights=da.ones(shape=(wphi, wphi), dtype=float) / wphi**2,
            mode="nearest",
        )
        ** 2
    )

    angle_mask = ~((theta > ttheta) | (phi > tphi)).compute()

    if angle_mask.all():
        warnings.warn(
            "No aliased seabed detected in Theta & Phi. "
            "A default mask with all True values is returned."
        )
        return angle_mask
    # negate for further processing
    angle_mask = ~angle_mask

    # remove aliased seabed items when estimated True seabed can not be
    # detected below the logging range
    if (rlog is not None) and (tpi is not None) and (freq is not None):
        items = label(angle_mask)
        item_labels = np.unique(items)[1:]
        for il in item_labels:
            item = items == il
            ritem = np.nanmean(r[np.where(item)[0]])
            rseabed = _aliased2seabed(ritem, rlog, tpi, freq)
            if rseabed == []:
                angle_mask[item] = False

        angle_mask = angle_mask & (Sv > tSv)

    # calculate rank percentile Sv of angle-masked regions, and mask Sv above
    Sv_masked = Sv.where(angle_mask)
    # anglemasked_threshold = Sv_masked.median(skipna=True).item()
    anglemasked_threshold = dask_nanpercentile(Sv_masked.values, rank)

    if np.isnan(anglemasked_threshold):
        anglemasked_threshold = np.inf
    if anglemasked_threshold < tSv:
        anglemasked_threshold = tSv
    Sv_threshold_mask = Sv > anglemasked_threshold

    # create structure element that defines connections
    structure = da.ones(shape=(3, 3), dtype=bool)
    items = label(Sv_threshold_mask, structure)[0]

    items_data = xr.DataArray(
        items,
        dims=angle_mask.dims,
        coords=angle_mask.coords,
    )

    mask_items = items_data.where(angle_mask, 0)

    # get items intercepted by angle mask
    keep_items = np.unique(mask_items.values)
    keep_items = keep_items[keep_items > 0]
    angle_items = xr.where(items_data.isin(keep_items), items_data, 0)
    angle_items_mask = ~(angle_items > 0)

    mask = angle_items_mask

    # apply range filter
    # get upper and lower range indexes
    up = abs(r - r0).argmin(dim="range_sample").item()
    lw = abs(r - r1).argmin(dim="range_sample").item()

    # get threshold mask with shallow and deep waters masked
    range_filter = (mask["range_sample"] >= up) & (mask["range_sample"] <= lw)
    mask = mask.where(range_filter, other=True)

    mask.data = mask.data.compute()
    return mask
