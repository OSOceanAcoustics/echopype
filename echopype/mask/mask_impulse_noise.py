"""
    Adaptation and Implementation of Impulse Noise Masking Algorithms: From Echopy to Echopype

        Algorithms for masking Impulse noise.
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

        __authors__ = ['Alejandro Ariza'   # wrote ryan(), ryan_iterable(), and wang()

    __authors__ = [ 'Raluca Simedroni'
                    # adapted the impulse noise masking algorithms
                    from the Echopy library and implemented them for use with the Echopype library.
                   ]
"""

from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import xarray as xr
from scipy.ndimage.filters import median_filter as medianf
from skimage.morphology import dilation, erosion

from ..utils import mask_transformation



def get_impulse_noise_mask(
    source_Sv: xr.Dataset,
    desired_channel: str,
    thr: Union[Tuple[float, float], int, float],
    m: Optional[Union[int, float]] = None,
    n: Optional[Union[int, Tuple[int, int]]] = None,
    erode: Optional[List[Tuple[int, int]]] = None,
    dilate: Optional[List[Tuple[int, int]]] = None,
    median: Optional[List[Tuple[int, int]]] = None,
    method: str = "ryan",
) -> xr.DataArray:
    """
    Algorithms for masking Impulse noise.

    Parameters
    ----------
    source_Sv: xr.Dataset
        Dataset  containing the Sv data to create a mask
    desired_channel: str
        Name of the desired frequency channel.
    thr: float or tuple
        User-defined threshold value (dB) (ryan and ryan iterable) o
        r a 2-element tuple specifying the range of threshold values (wang).
    m: int or float, optional
        Vertical binning length (in number of samples or range) (ryan and ryan iterable).
        Defaults to None.
    n: int or tuple, optional
        Number of pings either side for comparisons (ryan),
        or a 2-element tuple specifying the range (ryan iterable).
        Defaults to None.
    erode: List of 2-element tuples, optional
        List indicating the window's size for each erosion cycle (wang). Defaults to None.
    dilate: List of 2-element tuples, optional
        List indicating the window's size for each dilation cycle (wang). Defaults to None.
    median: List of 2-element tuples, optional
        List indicating the window's size for each median filter cycle (wang). Defaults to None.
    method: str, optional
        The method (ryan, ryan iterable or wang) used to mask impulse noise. Defaults to 'ryan'.

    Returns
    -------
    xr.DataArray
        A DataArray consisting of a mask for the Sv data, wherein True values signify
        samples that are free of noise.
    """

    def ryan(
        Sv: np.ndarray, iax: np.ndarray, m: Union[int, float], n: int, thr: Union[int, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mask impulse noise following the two-sided comparison method described
        in:
            Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in
            open-ocean echo integration data’, ICES Journal of Marine Science,
            72: 2482–2493.

        Parameters
        ----------
            Sv  (float)    : 2D array with Sv data to be masked (dB).
            iax (int/float): 1D array with i axis data (n samples or range).
            m   (int/float): vertical binning length (n samples or range).
            n   (int)      : number of pings either side for comparisons.
            thr (int/float): user-defined threshold value (dB).

        Returns
        -------
            bool: 2D array with IN mask
            bool: 2D array with mask indicating valid IN mask samples.
        """

        # resample down vertically
        iax_ = np.arange(iax[0], iax[-1], m)
        Sv_ = mask_transformation.oned(Sv, iax, iax_, 0, log_var=True)[0]

        # resample back to full resolution
        jax = np.arange(len(Sv[0]))
        Sv_, mask_ = mask_transformation.full(Sv_, iax_, jax, iax, jax)

        # side comparison (±n)
        dummy = np.zeros((iax.shape[0], n)) * np.nan
        comparison_forward = Sv_ - np.c_[Sv_[:, n:], dummy]
        comparison_backward = Sv_ - np.c_[dummy, Sv_[:, 0:-n]]

        # get IN mask
        comparison_forward[np.isnan(comparison_forward)] = np.inf
        maskf = comparison_forward > thr
        comparison_backward[np.isnan(comparison_backward)] = np.inf
        maskb = comparison_backward > thr
        mask = maskf & maskb

        # get second mask indicating valid samples in IN mask
        mask_[:, 0:n] = False
        mask_[:, -n:] = False

        return mask, mask_

    def ryan_iterable(
        Sv: Union[float, np.ndarray],
        iax: Union[int, float, np.ndarray],
        m: Union[int, float],
        n: Tuple[int, ...],
        thr: Union[int, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Modified from "ryan" so that the parameter "n" can be provided multiple
        times. It enables the algorithm to iterate and perform comparisons at
        different n distances. Resulting masks at each iteration are combined in
        a single mask. By setting multiple n distances the algorithm can detect
        spikes adjacent each other.

        Parameters
        ----------
            Sv  (float)     : 2D array with Sv data to be masked (dB).
            iax (int, float): 1D array with i axis data (n samples or range).
            m   (int, float): vertical binning length (n samples or range).
            n   (int)       : number of pings either side for comparisons.
            thr (int,float) : user-defined threshold value (dB).

        Returns
        -------
            bool: 2D array with IN mask
            bool: 2D array with mask indicating valid IN mask samples.
        """

        # resample down vertically
        iax_ = np.arange(iax[0], iax[-1], m)
        Sv_ = mask_transformation.oned(Sv, iax, iax_, 0, log_var=True)[0]

        # resample back to full resolution
        jax = np.arange(len(Sv[0]))
        Sv_, mask_ = mask_transformation.full(Sv_, iax_, jax, iax, jax)

        # perform side comparisons and combine masks in one unique mask
        mask = np.zeros_like(Sv, dtype=bool)
        for i in n:
            dummy = np.zeros((iax.shape[0], i))
            dummy[:] = np.nan
            forward = Sv_ - np.c_[Sv_[:, i:], dummy]
            backward = Sv_ - np.c_[dummy, Sv_[:, 0:-i]]
            maskf = np.ma.masked_greater(forward, thr).mask
            maskb = np.ma.masked_greater(backward, thr).mask
            mask = mask | (maskf & maskb)

        # get second mask indicating valid samples in IN mask
        mask_[:, 0 : max(n)] = True
        mask_[:, -max(n) :] = True

        return mask, mask_

    def wang(
        Sv: np.ndarray,
        thr: Tuple[float, float],
        erode: List[Tuple[int, int]],
        dilate: List[Tuple[int, int]],
        median: List[Tuple[int, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clean impulse noise from Sv data following the method decribed by:

            Wang et al. (2015) ’A noise removal algorithm for acoustic data with
            strong interference based on post-processing techniques’, CCAMLR
            SG-ASAM: 15/02.

        This algorithm runs different cycles of erosion, dilation, and median
        filtering to clean impulse noise from Sv. Note that this function
        returns a clean/corrected Sv array, instead of a boolean array indicating
        the occurrence of impulse noise.

        Parameters
        ----------
            Sv     (float)    : 2D numpy array with Sv data (dB).
            thr    (int/float): 2-element tupple with bottom/top Sv thresholds (dB)
            erode  (int)      : list of 2-element tupples indicating the window's
                                size for each erosion cycle.
            dilate (int)      : list of 2-element tupples indicating the window's
                                size for each dilation cycle.
            median (int)      : list of 2-element tupples indicating the window's
                                size for each median filter cycle.

        Returns
        -------
            float             : 2D array with clean Sv data.
            bool              : 2D array with mask indicating valid clean Sv data.
        """

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
        Sv_corrected1 = Sv_dilated.copy()
        mask_bio = (Sv_dilated >= thr[0]) & (Sv_dilated < thr[1])
        Sv_corrected1[mask_bio] = Sv_thresholded[mask_bio]

        # compute median convolution in Sv corrected array
        Sv_median = Sv_corrected1.copy()
        for m in median:
            Sv_median = mask_transformation.log(
                medianf(mask_transformation.lin(Sv_median), footprint=np.ones(m))
            )

        # any vacant sample inside biological features will be corrected with
        # the median of corresponding neighbouring samples
        Sv_corrected2 = Sv_corrected1.copy()
        mask_bio = (Sv >= thr[0]) & (Sv < thr[1])
        mask_vacant = Sv_corrected1 == -999
        Sv_corrected2[mask_vacant & mask_bio] = Sv_median[mask_vacant & mask_bio]

        # get mask indicating edges, where swarms analysis couldn't be performed
        mask_ = np.ones_like(Sv_corrected2, dtype=bool)
        idx = int((max([e[0], d[0]]) - 1) / 2)
        jdx = int((max([e[1], d[1]]) - 1) / 2)
        mask_[idx:-idx, jdx:-jdx] = False

        return Sv_corrected2, mask_

    def find_impulse_mask_wang(
        Sv_ds: xr.Dataset,
        desired_channel: str,
        thr: Tuple[float, float],
        erode: List[Tuple[int, int]],
        dilate: List[Tuple[int, int]],
        median: List[Tuple[int, int]],
    ) -> xr.DataArray:
        """
        Return a boolean mask indicating the location of impulse noise in Sv data.

        Parameters
        ----------
            Sv_ds (xarray.Dataset): xr.DataArray with Sv data for multiple channels (dB).
            desired_channel (str): Name of the desired frequency channel.
            thr    : 2-element tuple with bottom/top Sv thresholds (dB).
            erode  : List of 2-element tuples indicating the window's size for each erosion cycle.
            dilate : List of 2-element tuples indicating the window's size for each dilation cycle.
            median : List of 2-element tuples indicating the window's
                    size for each median filter cycle.

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
        Sv_cleaned, mask_ = wang(Sv, thr, erode, dilate, median)

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

    def find_impulse_mask_ryan(
        Sv_ds: xr.Dataset,
        desired_channel: str,
        m: Union[int, float],
        n: int,
        thr: Union[int, float],
    ) -> xr.DataArray:
        """
        Mask impulse noise following the two-sided comparison method described in:
        Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in
        open-ocean echo integration data’, ICES Journal of Marine Science, 72: 2482–2493.

        Parameters
        ----------
            Sv_ds (xarray.Dataset): xr.DataArray with Sv data for multiple channels (dB).
            desired_channel (str): Name of the desired frequency channel.
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

        # Select the desired frequency channel directly using 'sel'
        selected_channel_ds = Sv_ds.sel(channel=desired_channel)

        # Extract Sv and iax for the desired frequency channel
        Sv = selected_channel_ds["Sv"].values

        # But first, transpose the Sv data so that the vertical dimension is axis 0
        Sv = np.transpose(Sv)

        iax = selected_channel_ds.range_sample.values

        # Call the existing ryan function
        mask, mask_ = ryan(Sv, iax, m, n, thr)

        # Transpose the mask back to its original shape
        mask = np.transpose(mask)
        mask_ = np.transpose(mask_)
        combined_mask = mask & (~mask_)

        # Create a new xarray for the mask with the correct dimensions and coordinates
        mask_xr = xr.DataArray(
            combined_mask,
            dims=("ping_time", "range_sample"),
            coords={
                "ping_time": selected_channel_ds.ping_time.values,
                "range_sample": selected_channel_ds.range_sample.values,
            },
        )

        return mask_xr

    def find_impulse_mask_ryan_iterable(
        Sv_ds: xr.Dataset,
        desired_channel: str,
        m: Union[int, float],
        n: Tuple[int, ...],
        thr: Union[int, float],
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

        # Select the desired frequency channel directly using 'sel'
        selected_channel_ds = Sv_ds.sel(channel=desired_channel)

        # Extract Sv and iax for the desired frequency channel
        Sv = selected_channel_ds["Sv"].values

        # But first, transpose the Sv data so that the vertical dimension is axis 0
        Sv = np.transpose(Sv)

        iax = selected_channel_ds.range_sample.values

        # Call the existing ryan function
        mask, mask_ = ryan_iterable(Sv, iax, m, n, thr)

        # Transpose the mask back to its original shape
        mask = np.transpose(mask)
        mask_ = np.transpose(mask_)
        combined_mask = mask & (~mask_)

        # Create a new xarray for the mask with the correct dimensions and coordinates
        mask_xr = xr.DataArray(
            combined_mask,
            dims=("ping_time", "range_sample"),
            coords={
                "ping_time": selected_channel_ds.ping_time.values,
                "range_sample": selected_channel_ds.range_sample.values,
            },
        )

        return mask_xr

    # Our goal is to have a mask where True represents samples that are NOT impulse noise.
    # So, we negate the obtained mask.

    if method == "ryan":
        impulse_mask_ryan = find_impulse_mask_ryan(source_Sv, desired_channel, m, n, thr)
        noise_free_mask = ~impulse_mask_ryan
    elif method == "ryan_iterable":
        impulse_mask_ryan_iterable = find_impulse_mask_ryan_iterable(
            source_Sv, desired_channel, m, n, thr
        )
        noise_free_mask = ~impulse_mask_ryan_iterable
    elif method == "wang":
        impulse_mask_wang = find_impulse_mask_wang(
            source_Sv, desired_channel, thr, erode, dilate, median
        )
        noise_free_mask = ~impulse_mask_wang
    else:
        raise ValueError(f"Unsupported method: {method}")

    return noise_free_mask
