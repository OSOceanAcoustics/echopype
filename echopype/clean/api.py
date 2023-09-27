"""
Functions for reducing variabilities in backscatter data.
"""
import pathlib
from typing import Union

import xarray as xr

from ..utils.misc import frequency_nominal_to_channel
from ..utils.prov import add_processing_level, echopype_prov_attrs, insert_input_processing_level
from . import impulse_noise, signal_attenuation, transient_noise

# from .impulse_noise import _ryan, _ryan_iterable, _wang
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
    parameters: dict,
    desired_channel: str = None,
    desired_frequency: int = None,
    method: str = "ryan",
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
    desired_channel: str
        Name of the desired frequency channel.
    desired_frequency: int
        Desired frequency, in case the channel is not directly specified
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
    mask_map = {
        "ryan": transient_noise._ryan,
        "fielding": transient_noise._fielding,
    }
    if method not in mask_map.keys():
        raise ValueError(f"Unsupported method: {method}")
    if desired_channel is None:
        if desired_frequency is None:
            raise ValueError("Must specify either desired channel or desired frequency")
        else:
            desired_channel = frequency_nominal_to_channel(source_Sv, desired_frequency)

    mask = mask_map[method](source_Sv, desired_channel, parameters)
    return mask


def get_impulse_noise_mask(
    source_Sv: xr.Dataset,
    parameters: {},
    desired_channel: str = None,
    desired_frequency: int = None,
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
    desired_frequency: int
        Desired frequency, in case the channel is not directly specified
    parameters: {}
        Parameter dictionary containing function-specific arguments.
        Can contain the following:
            thr: Union[Tuple[float, float], int, float]
                User-defined threshold value (dB) (ryan and ryan iterable)
                or a 2-element tuple with the range of threshold values (wang).
            m:  Optional[Union[int, float]] = None,
                Vertical binning length (in number of samples or range)
                (ryan and ryan iterable).
                Defaults to None.
            n: Optional[Union[int, Tuple[int, int]]] = None,
                Number of pings either side for comparisons (ryan),
                or a 2-element tuple specifying the range (ryan iterable).
                Defaults to None.
            erode: Optional[List[Tuple[int, int]]] = None,
                Window size for each erosion cycle (wang).
                Defaults to None.
            dilate: Optional[List[Tuple[int, int]]] = None,
                Window size for each dilation cycle (wang).
                Defaults to None.
            median: Optional[List[Tuple[int, int]]] = None,
                Window size for each median filter cycle (wang).
                Defaults to None.
    method: str, optional
        The method (ryan, ryan_iterable or wang) used to mask impulse noise.
        Defaults to 'ryan'.

    Returns
    -------
    xr.DataArray
        A DataArray consisting of a mask for the Sv data, wherein True values signify
        samples that are free of noise.
    """
    # Our goal is to have a mask True on samples that are NOT impulse noise.
    # So, we negate the obtained mask.
    mask_map = {
        "ryan": impulse_noise._ryan,
        "ryan_iterable": impulse_noise._ryan_iterable,
        "wang": impulse_noise._wang,
    }
    if method not in mask_map.keys():
        raise ValueError(f"Unsupported method: {method}")
    if desired_channel is None:
        if desired_frequency is None:
            raise ValueError("Must specify either desired channel or desired frequency")
        else:
            desired_channel = frequency_nominal_to_channel(source_Sv, desired_frequency)
    impulse_mask = mask_map[method](source_Sv, desired_channel, parameters)
    noise_free_mask = ~impulse_mask

    return noise_free_mask


def get_attenuation_mask(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    parameters: dict,
    desired_channel: str = None,
    desired_frequency: int = None,
    method: str = "ryan",
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
    parameters: dict
        Dictionary of parameters to pass to the relevant subfunctions.
    desired_channel: str
        Name of the desired frequency channel.
    desired_frequency: int
        Desired frequency, in case the channel is not directly specified
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
        If neither ``ryan`` or ``ariza`` are given

    Notes
    -----


    Examples
    --------

    """
    mask_map = {
        "ryan": signal_attenuation._ryan,
        "ariza": signal_attenuation._ariza,
    }
    if method not in mask_map.keys():
        raise ValueError(f"Unsupported method: {method}")
    if desired_channel is None:
        if desired_frequency is None:
            raise ValueError("Must specify either desired channel or desired frequency")
        else:
            desired_channel = frequency_nominal_to_channel(source_Sv, desired_frequency)

    mask = mask_map[method](source_Sv, desired_channel, parameters)
    return mask
