"""
Functions for reducing variabilities in backscatter data.
"""
import pathlib
from typing import List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from ..utils.prov import add_processing_level, echopype_prov_attrs, insert_input_processing_level
from . import transient_noise
from .impulse_noise import (
    _find_impulse_mask_ryan,
    _find_impulse_mask_ryan_iterable,
    _find_impulse_mask_wang,
)
from . import signal_attenuation
from .noise_est import NoiseEst


def estimate_noise(ds_Sv, ping_num, range_sample_num, noise_max=None):
    """
    Estimate background noise by computing mean calibrated power of a collection of pings.

    See ``remove_noise`` for reference.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        dataset containing ``Sv`` and ``echo_range`` [m]
    ping_num : int
        number of pings to obtain noise estimates
    range_sample_num : int
        number of samples along the ``range_sample`` dimension to obtain noise estimates
    noise_max : float
        the upper limit for background noise expected under the operating conditions

    Returns
    -------
    A DataArray containing noise estimated from the input ``ds_Sv``
    """
    noise_obj = NoiseEst(ds_Sv=ds_Sv.copy(), ping_num=ping_num, range_sample_num=range_sample_num)
    noise_obj.estimate_noise(noise_max=noise_max)
    return noise_obj.Sv_noise


@add_processing_level("L*B")
def remove_noise(ds_Sv, ping_num, range_sample_num, noise_max=None, SNR_threshold=3):
    """
    Remove noise by using estimates of background noise
    from mean calibrated power of a collection of pings.

    Reference: De Robertis & Higginbottom. 2007.
    A post-processing technique to estimate the signal-to-noise ratio
    and remove echosounder background noise.
    ICES Journal of Marine Sciences 64(6): 1282–1291.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        dataset containing ``Sv`` and ``echo_range`` [m]
    ping_num : int
        number of pings to obtain noise estimates
    range_sample_num : int
        number of samples along the ``range_sample`` dimension to obtain noise estimates
    noise_max : float
        the upper limit for background noise expected under the operating conditions
    SNR_threshold : float
        acceptable signal-to-noise ratio, default to 3 dB

    Returns
    -------
    The input dataset with additional variables, including
    the corrected Sv (``Sv_corrected``) and the noise estimates (``Sv_noise``)
    """
    noise_obj = NoiseEst(ds_Sv=ds_Sv.copy(), ping_num=ping_num, range_sample_num=range_sample_num)
    noise_obj.remove_noise(noise_max=noise_max, SNR_threshold=SNR_threshold)
    ds_Sv = noise_obj.ds_Sv

    prov_dict = echopype_prov_attrs(process_type="processing")
    prov_dict["processing_function"] = "clean.remove_noise"
    ds_Sv = ds_Sv.assign_attrs(prov_dict)

    # The output ds_Sv is built as a copy of the input ds_Sv, so the step below is
    # not needed, strictly speaking. But doing makes the decorator function more generic
    ds_Sv = insert_input_processing_level(ds_Sv, input_ds=ds_Sv)

    return ds_Sv


def get_transient_noise_mask(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    desired_channel: str,
    mask_type: str = "ryan",
    **kwargs,
) -> xr.DataArray:
    """
    Create a mask based on the identified signal attenuations of Sv values at 38KHz.
    This method is based on:
    Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in
        open-ocean echo integration data’, ICES Journal of Marine Science,
        72: 2482–2493.

    Parameters
    ----------
    source_Sv: xr.Dataset or str or pathlib.Path
        If a Dataset this value contains the Sv data to create a mask for,
        else it specifies the path to a zarr or netcdf file containing
        a Dataset. This input must correspond to a Dataset that has the
        coordinate ``channel`` and variables ``frequency_nominal`` and ``Sv``.
    mask_type: str with either "ryan" or "fielding" based on
        the preferred method for signal attenuation mask generation
    Returns
    -------
    xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.

    Raises
    ------
    ValueError
        If neither ``ryan`` or ``fielding`` are given

    """
    assert mask_type in ["ryan", "fielding"], "mask_type must be either 'ryan' or 'fielding'"
    selected_channel_Sv = source_Sv.sel(channel=desired_channel)
    Sv = selected_channel_Sv["Sv"].values
    r = source_Sv["echo_range"].values[0, 0]
    if mask_type == "ryan":
        # Define a list of the keyword arguments your function can handle
        valid_args = {"m", "n", "thr", "excludeabove", "operation"}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = transient_noise._ryan(Sv, r, m=5, **filtered_kwargs)
    elif mask_type == "fielding":
        # Define a list of the keyword arguments your function can handle
        valid_args = {"r0", "r1", "roff", "n", "thr"}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = transient_noise._fielding(Sv, r, **filtered_kwargs)
    else:
        raise ValueError("The provided mask_type must be ryan or fielding!")

    mask = np.logical_not(mask)
    return_mask = xr.DataArray(
        mask,
        dims=("ping_time", "range_sample"),
        coords={"ping_time": source_Sv.ping_time, "range_sample": source_Sv.range_sample},
    )
    return return_mask


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

    # Our goal is to have a mask where True represents samples that are NOT impulse noise.
    # So, we negate the obtained mask.

    if method == "ryan":
        impulse_mask_ryan = _find_impulse_mask_ryan(source_Sv, desired_channel, m, n, thr)
        noise_free_mask = ~impulse_mask_ryan
    elif method == "ryan_iterable":
        impulse_mask_ryan_iterable = _find_impulse_mask_ryan_iterable(
            source_Sv, desired_channel, m, n, thr
        )
        noise_free_mask = ~impulse_mask_ryan_iterable
    elif method == "wang":
        impulse_mask_wang = _find_impulse_mask_wang(
            source_Sv, desired_channel, thr, erode, dilate, median
        )
        noise_free_mask = ~impulse_mask_wang
    else:
        raise ValueError(f"Unsupported method: {method}")

    return noise_free_mask
  
  
def get_attenuation_mask(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    desired_channel: str,
    mask_type: str = "ryan",
    **kwargs
) -> xr.DataArray:
    """
    Create a mask based on the identified signal attenuations of Sv values at 38KHz.
    This method is based on:
    Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in
        open-ocean echo integration data’, ICES Journal of Marine Science,
        72: 2482–2493.
    and,
    Ariza et al. (2022) 'Acoustic seascape partitioning through functional data analysis',
    Journal of Biogeography, 00, 1– 15. https://doi.org/10.1111/jbi.14534


    Parameters
    ----------
    source_Sv: xr.Dataset or str or pathlib.Path
        If a Dataset this value contains the Sv data to create a mask for,
        else it specifies the path to a zarr or netcdf file containing
        a Dataset. This input must correspond to a Dataset that has the
        coordinate ``channel`` and variables ``frequency_nominal`` and ``Sv``.
    desired_channel: the Dataset channel to be used for identifying the signal attenuation.
    mask_type: str with either "ryan" or "ariza" based on the
                preferred method for signal attenuation mask generation
    Returns
    -------
    xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.

    Raises
    ------
    ValueError
        If neither ``ryan`` or ``azira`` are given

    Notes
    -----


    Examples
    --------

    """
    assert mask_type in ["ryan", "ariza"], "mask_type must be either 'ryan' or 'ariza'"
    selected_channel_Sv = source_Sv.sel(channel=desired_channel)
    Sv = selected_channel_Sv["Sv"].values
    r = source_Sv["echo_range"].values[0, 0]
    if mask_type == "ryan":
        # Define a list of the keyword arguments your function can handle
        valid_args = {"r0", "r1", "n", "thr", "start"}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = signal_attenuation._ryan(Sv, r, **filtered_kwargs)
    elif mask_type == "ariza":
        # Define a list of the keyword arguments your function can handle
        valid_args = {"offset", "thr", "m", "n"}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = signal_attenuation._ariza(Sv, r, **filtered_kwargs)
    else:
        raise ValueError("The provided mask_type must be ryan or ariza!")

    return_mask = xr.DataArray(
        mask,
        dims=("ping_time", "range_sample"),
        coords={"ping_time": source_Sv.ping_time, "range_sample": source_Sv.range_sample},
    )
    return return_mask
