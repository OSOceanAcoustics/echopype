import pathlib
from typing import Union

import numpy as np
import xarray as xr
import scipy.ndimage as nd

"""
Algorithms for masking shoals.
    
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


authors = ['Alejandro Ariza'   # wrote weill(), echoview()
            'Ruxandra Valcu'    # adapted the code for echopype]
credits = ['Rob Blackwell'     # supervised the code and provided ideas
               'Sophie Fielding'   # supervised the code and provided ideas
               'Nhan Vu'           # contributed to weill()]
"""


def get_shoal_mask(
    source_Sv: Union[xr.Dataset, str, pathlib.Path], mask_type: str = "weill", **kwargs
):
    """
    Wrapper function for (future) multiple shoal masking algorithms (currently, only MOVIES-B (Weill) is implemented)

    Args:
        source_Sv: xr.Dataset or str or pathlib.Path
                    If a Dataset this value contains the Sv data to create a mask for,
                    else it specifies the path to a zarr or netcdf file containing
                    a Dataset. This input must correspond to a Dataset that has the
                    coordinate ``channel`` and variables ``frequency_nominal`` and ``Sv``.
        mask_type: string specifying the algorithm to use - currently, 'weill' is the only one implemented

    Returns
    -------
    mask: xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.
    mask_: xr.DataArray
        A DataArray containing the mask for areas in which shoals were searched.
        Edge regions are filled with 'False', whereas the portion in which shoals could be detected is 'True'


    Raises
    ------
    ValueError
        If 'weill' is not given
    """
    assert mask_type in ["weill"]
    if mask_type == "weill":
        # Define a list of the keyword arguments your function can handle
        valid_args = {"thr", "maxvgap", "maxhgap", "minvlen", "minhlen"}
        # Filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask, mask_ = weill(source_Sv, **filtered_kwargs)
    else:
        raise ValueError("The provided mask type must be Weill")
    return_mask = xr.DataArray(
        mask,
        dims=("ping_time", "range_sample"),
        coords={"ping_time": source_Sv.ping_time, "range_sample": source_Sv.range_sample},
    )
    return_mask_ = xr.DataArray(
        mask_,
        dims=("ping_time", "range_sample"),
        coords={"ping_time": source_Sv.ping_time, "range_sample": source_Sv.range_sample},
    )
    return return_mask, return_mask_


def weill(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    thr=-70,
    maxvgap=5,
    maxhgap=0,
    minvlen=0,
    minhlen=0,
):
    """
    Detects and masks shoals following the algorithm described in:

        "Weill et al. (1993): MOVIES-B — an acoustic detection description
        software . Application to shoal species' classification".

    Contiguous Sv samples above a given threshold will be considered as the
    same shoal, as long as it meets the contiguity criteria described by Weill
    et al. (1993):

        * Vertical contiguity: along-ping gaps are allowed to some extent.
        Typically, no more than the number of samples equivalent to half of the
        pulse length.

        * Horizontal contiguity: above-threshold features from contiguous pings
        will be regarded as the same shoal if there is at least one sample in
        each ping at the same range depth.

    Although the default settings strictly complies with Weill's contiguity
    criteria, other contiguity arguments has been enabled in this function to
    increase operability. For instance, the possibility to allow gaps in the
    horizontal, or to set minimum vertical and horizontal lengths for a feature
    to be regarded as a shoal. These settings are set to zero (not applicable)
    by default.

    Args:
        source_Sv: xr.Dataset or str or pathlib.Path
                    If a Dataset this value contains the Sv data to create a mask for,
                    else it specifies the path to a zarr or netcdf file containing
                    a Dataset. This input must correspond to a Dataset that has the
                    coordinate ``channel`` and variables ``frequency_nominal`` and ``Sv``.
        thr (int): Sv threshold (dB).
        maxvgap (int): maximum vertical gap allowed (n samples).
        maxhgap (int): maximum horizontal gap allowed (n pings).
        minvlen (int): minimum vertical length for a shoal to be eligible
                       (n samples).
        minhlen (int): minimum horizontal length for a shoal to be eligible
                       (n pings).
        start (int): ping index to start processing. If greater than zero, it
                     means that Sv carries data from a preceding file and
                     the algorithm needs to know where to start processing.

    Returns
    -------
    mask: xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.
    mask_: xr.DataArray
        A DataArray containing the mask for areas in which shoals were searched.
        Edge regions are filled with 'False', whereas the portion in which shoals could be detected is 'True'
    """
    Sv = source_Sv["Sv"].values[0]

    # mask Sv above threshold
    mask = np.ma.masked_greater(Sv, thr).mask

    # for each ping in the mask...
    for jdx, ping in enumerate(list(np.transpose(mask))):
        # find gaps between masked features, and give them a label number
        pinglabelled = nd.label(np.invert(ping))[0]

        # proceed only if the ping presents gaps
        if (not (pinglabelled == 0).all()) & (not (pinglabelled == 1).all()):
            # get list of gap labels and iterate through gaps
            labels = np.arange(1, np.max(pinglabelled) + 1)
            for label in labels:
                # if vertical gaps are equal/shorter than maxvgap...
                gap = pinglabelled == label
                if np.sum(gap) <= maxvgap:
                    # get gap indexes and fill in with True values (masked)
                    idx = np.where(gap)[0]
                    if (not 0 in idx) & (not len(mask) - 1 in idx):  # (exclude edges)
                        mask[idx, jdx] = True

    # for each depth in the mask...
    for idx, depth in enumerate(list(mask)):
        # find gaps between masked features, and give them a label number
        depthlabelled = nd.label(np.invert(depth))[0]

        # proceed only if the ping presents gaps
        if (not (depthlabelled == 0).all()) & (not (depthlabelled == 1).all()):
            # get list of gap labels and iterate through gaps
            labels = np.arange(1, np.max(depthlabelled) + 1)
            for label in labels:
                # if horizontal gaps are equal/shorter than maxhgap...
                gap = depthlabelled == label
                if np.sum(gap) <= maxhgap:
                    # get gap indexes and fill in with True values (masked)
                    jdx = np.where(gap)[0]
                    if (not 0 in jdx) & (not len(mask) - 1 in jdx):  # (exclude edges)
                        mask[idx, jdx] = True

    # label connected features in the mask
    masklabelled = nd.label(mask)[0]

    # get list of features labelled and iterate through them
    labels = np.arange(1, np.max(masklabelled) + 1)
    for label in labels:
        # target feature & calculate its maximum vertical/horizontal length
        feature = masklabelled == label
        idx, jdx = np.where(feature)
        featurehlen = max(idx + 1) - min(idx)
        featurevlen = max(jdx + 1) - min(jdx)

        # remove feature from mask if its maximum vertical length < minvlen
        if featurevlen < minvlen:
            mask[idx, jdx] = False

        # remove feature from mask if its maximum horizontal length < minhlen
        if featurehlen < minhlen:
            mask[idx, jdx] = False

    # get mask_ indicating the valid samples for mask
    mask_ = np.zeros_like(mask, dtype=bool)
    mask_[minvlen : len(mask_) - minvlen, minhlen : len(mask_[0]) - minhlen] = True

    # return masks, from the start ping onwards
    return mask, mask_
