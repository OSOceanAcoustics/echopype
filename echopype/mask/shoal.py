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


authors = ['Alejandro Ariza'   # wrote will(), echoview()
            'Ruxandra Valcu'    # rewrote will() to involve xarray+dask functionality]
credits = ['Rob Blackwell'     # supervised the code and provided ideas
               'Sophie Fielding'   # supervised the code and provided ideas
               'Nhan Vu'           # contributed to will()]
"""
import pathlib
from typing import Union

import numpy as np
import xarray as xr
from dask_image.ndmorph import binary_closing, binary_opening

WEILL_DEFAULT_PARAMETERS = {
    "thr": -70,
    "maxvgap": 5,
    "maxhgap": 5,
    "minvlen": 0,
    "minhlen": 0,
    "dask_chunking": {"ping_time": 1000, "range_sample": 1000},
}


def _weill(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    desired_channel: str,
    parameters: dict = WEILL_DEFAULT_PARAMETERS,
):
    """
    Detects and masks shoals following the algorithm described in:

        "Will et al. (1993): MOVIES-B — an acoustic detection description
        software . Application to shoal species' classification".

    Contiguous Sv samples above a given threshold will be considered as the
    same shoal, as long as it meets the contiguity criteria described by Will
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
        desired_channel (str): channel to generate the mask on
        parameters (dict): containing the required parameters
            thr (int): Sv threshold (dB).
            maxvgap (int): maximum vertical gap allowed (n samples).
            maxhgap (int): maximum horizontal gap allowed (n pings).
            minvlen (int): minimum vertical length for a shoal to be eligible
                           (n samples).
            minhlen (int): minimum horizontal length for a shoal to be eligible
                           (n pings).
            dask_chunking (dict): dask chunking to use

    Returns
    -------
    mask: xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.
    mask_: xr.DataArray
        A DataArray containing the mask for areas in which shoals were searched.
        Edge regions are filled with 'False', whereas the portion in which shoals
        could be detected is 'True'
    """
    parameter_names = ["thr", "maxvgap", "maxhgap", "minvlen", "minhlen", "dask_chunking"]
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be: "
            + str(parameter_names)
            + ", are: "
            + str(parameters.keys())
        )
    thr = parameters["thr"]
    maxvgap = parameters["maxvgap"]
    maxhgap = parameters["maxhgap"]
    minvlen = parameters["minvlen"]
    minhlen = parameters["minhlen"]
    dask_chunking = parameters["dask_chunking"]

    channel_Sv = source_Sv.sel(channel=desired_channel)
    Sv = channel_Sv["Sv"].chunk(dask_chunking)

    mask = xr.where(Sv > thr, True, False).chunk(dask_chunking)

    # close shoal gaps smaller than the specified box
    if maxvgap > 0 & maxhgap > 0:
        closing_array = np.ones(maxhgap, maxvgap)
        mask = binary_closing(
            mask,
            structure=closing_array,
            iterations=1,
        )

    # drop shoals smaller than the specified box
    if minvlen > 0 & minhlen > 0:
        opening_array = np.ones(minhlen, minhlen)
        mask = binary_opening(
            mask,
            structure=opening_array,
            iterations=1,
        )

    mask = mask.drop("channel")
    return mask
