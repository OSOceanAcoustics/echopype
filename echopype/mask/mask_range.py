from typing import Optional, Union

import numpy as np
import xarray as xr

"""

    Filters for masking data based on depth range.
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

            __authors__ = ['Alejandro Ariza'   # wrote above(), below(), inside(), outside()]

    __authors__ = ['Raluca Simedroni'
                    # adapted the range masking algorithms from the Echopy library and
                    implemented them for use with the Echopype library.
                    ]
"""


def get_range_mask(
    Sv_ds: xr.Dataset,
    channel: str,
    r0: Union[int, float],
    r1: Optional[Union[int, float]] = np.nan,
    method: str = "above",
):
    """
    Creates a mask for a given data set based on the range and method provided.
    This function can mask data that's `above`, `below`, `inside`
    or `outside` a given range. The desired frequency channel
    from the data set is selected, then the mask is created based on the method chosen.

    Parameters
    ----------
    Sv_ds: xr.Dataset
        The dataset that contains the Sv and the range data to create a mask.
    channel: str
        The name of the desired frequency channel.
    r0: Union[int, float]
        The lower limit of the range for masking.
    r1: Optional[Union[int, float]]
        The upper limit of the range for masking. Defaults to NaN which signifies no upper limit.
    method: str
        The method to create the mask. Can be 'above', 'below',
        'inside', or 'outside'. Defaults to 'above'.

    Returns
    -------
    xr.DataArray
        The mask in the form of a DataArray. True values indicate valid data.
    """

    # Select the desired frequency channel directly using 'sel'
    selected_channel_ds = Sv_ds.sel(channel=channel)

    # Extract Sv and iax for the desired frequency channel
    Sv = selected_channel_ds["Sv"].values

    # But first, transpose the Sv data so that the vertical dimension is axis 0
    Sv = np.transpose(Sv)

    r = selected_channel_ds.range_sample.values

    def _above(Sv, r, r0):
        """
        Mask data above a given range.

            Parameters
            ----------
                Sv (float): 2D array with data to be masked.
                r (float): 1D array with range data.
                r0 (int):  range above which data will be masked.

            Returns
            -------
                bool: 2D array mask (above range = True).
        """

        idx = np.where(np.ma.masked_less(r, r0).mask)[0]
        mask = np.zeros((Sv.shape), dtype=bool)
        mask[idx, :] = True
        return mask

    def _below(Sv, r, r0):
        """
        Mask data below a given range.

            Parameters
            ----------
                Sv (float): 2D array with data to be masked.
                r (float): 1D array with range data.
                r0 (int):  range below which data will be masked.

            Returns
            -------
                bool: 2D array mask (below range = True).
        """

        idx = np.where(np.ma.masked_greater(r, r0).mask)[0]
        mask = np.zeros((Sv.shape), dtype=bool)
        mask[idx, :] = True
        return mask

    def _inside(Sv, r, r0, r1):
        """
        Mask data inside a given range.

            Parameters
            ----------
                Sv (float): 2D array with data to be masked.
                r (float): 1D array with range data.
                r0 (int): Upper range limit.
                r1 (int): Lower range limit.

            Returns
            -------
                bool: 2D array mask (inside range = True).
        """
        masku = np.ma.masked_greater_equal(r, r0).mask
        maskl = np.ma.masked_less_equal(r, r1).mask
        idx = np.where(masku & maskl)[0]
        mask = np.zeros((Sv.shape), dtype=bool)
        mask[idx, :] = True
        return mask

    def _outside(Sv, r, r0, r1):
        """
        Mask data outside a given range.

            Parameters
            ----------
                Sv (float): 2D array with data to be masked.
                r (float): 1D array with range data.
                r0 (int): Upper range limit.
                r1 (int): Lower range limit.

            Returns
            -------
                bool: 2D array mask (out of range = True).
        """
        masku = np.ma.masked_less(r, r0).mask
        maskl = np.ma.masked_greater_equal(r, r1).mask
        idx = np.where(masku | maskl)[0]
        mask = np.zeros((Sv.shape), dtype=bool)
        mask[idx, :] = True

        return mask

    # Call the existing chosen method
    if method == "above":
        mask = _above(Sv, r, r0)
    elif method == "below":
        mask = _below(Sv, r, r0)
    elif method == "inside":
        mask = _inside(Sv, r, r0, r1)
    elif method == "outside":
        mask = _outside(Sv, r, r0, r1)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Transpose the mask back to its original shape
    mask = np.transpose(mask)

    # Create a new xarray for the mask with the correct dimensions and coordinates
    mask_xr = xr.DataArray(
        mask,
        dims=("ping_time", "range_sample"),
        coords={
            "ping_time": selected_channel_ds.ping_time.values,
            "range_sample": selected_channel_ds.range_sample.values,
        },
    )

    return mask_xr
