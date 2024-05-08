"""
Functions for reducing variabilities in backscatter data.
"""

from functools import partial
from typing import Union

import numpy as np
import xarray as xr

from ..utils.prov import add_processing_level, echopype_prov_attrs, insert_input_processing_level
from .background_noise_est import BackgroundNoiseEst
from .utils import (
    downsample_upsample_along_depth,
    echopy_attenuated_signal_mask,
    echopy_impulse_noise_mask,
)


def mask_impulse_noise(
    ds_Sv: xr.Dataset,
    depth_bin: str = "5m",
    num_side_pings: int = 1,
    impulse_threshold: Union[int, float] = 10.0,
):
    """
    Locate and create a mask for impulse noise using a ping-wise two-sided comparison.

    Parameters
    ----------
    ds_Sv : xarray.Dataset
        Calibrated Sv data with depth data variable.
    depth_bin : str, default '5m'
        Donwsampling bin size along ``depth`` in meters.
    num_side_pings : int, default `1`
        Number of side pings to look at for the two-side comparison.
    impulse_threshold : Union[int, float], default `10.0`
        User-defined impulse threshold value in dB for the two-side comparison.

    Returns
    -------
    xr.Dataset
        Xarray boolean array with impulse noise mask.

    References
    ----------
    Ryan et al. (2015) Reducing bias due to noise and attenuation in
    open-ocean echo integration data, ICES Journal of Marine Science, 72: 2482–2493.

    Code was derived from echopy's numpy single-channel implementation of impulse noise masking
    and translated into xarray code:
    https://github.com/open-ocean-sounding/echopy/blob/master/echopy/processing/mask_impulse.py # noqa
    """
    if "depth" not in ds_Sv.data_vars:
        raise ValueError(
            "Masking attenuated signal requires depth data variable in `ds_Sv`. "
            "Consider adding depth with `ds_Sv = ep.consolidate.add_depth(ds_Sv)`."
        )

    # Copy `ds_Sv`
    ds_Sv_copy = ds_Sv.copy()

    # Downsample and Upsample Sv along depth
    _, upsampled_Sv = downsample_upsample_along_depth(ds_Sv_copy, depth_bin)

    # Create partial of `echopy_impulse_noise_mask`
    partial_echopy_impulse_noise_mask = partial(
        echopy_impulse_noise_mask,
        num_side_pings=num_side_pings,
        impulse_threshold=impulse_threshold,
    )

    # Create noise mask
    impulse_noise_mask = xr.apply_ufunc(
        partial_echopy_impulse_noise_mask,
        upsampled_Sv,
        input_core_dims=[["range_sample", "ping_time"]],
        output_core_dims=[["range_sample", "ping_time"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.float64],
    )

    return impulse_noise_mask


def mask_attenuated_signal(ds_Sv: xr.Dataset, r0: float, r1: float, n: int, threshold: float):
    """
    Locate attenuated signals and create an attenuated mask and a mask where attenuation
    masking was unfeasible.

    Parameters
    ----------
    ds_Sv : xarray.Dataset
        Calibrated Sv data with depth data variable.
    r0 : float
        Upper limit of SL (m).
    r1 : float
        Lower limit of SL (m).
    n : int
        Number of preceding & subsequent pings defining the block.
    threshold : float
        User-defined threshold value (dB).

    Returns
    -------
    tuple
        Xarray boolean array with attenuated signal mask.
        Xarray boolean array with mask indicating where attenuated signal detection was unfeasible.

    References
    ----------
    Ryan et al. (2015) Reducing bias due to noise and attenuation in
    open-ocean echo integration data, ICES Journal of Marine Science, 72: 2482–2493.

    Code was derived from echopy's numpy single-channel implementation of attenuation signal masking
    and translated into xarray code:
    https://github.com/open-ocean-sounding/echopy/blob/master/echopy/processing/mask_attenuated.py # noqa
    """
    if "depth" not in ds_Sv.data_vars:
        raise ValueError(
            "Masking attenuated signal requires depth data variable in `ds_Sv`. "
            "Consider adding depth with `ds_Sv = ep.consolidate.add_depth(ds_Sv)`."
        )

    # Check range values
    if r0 > r1:
        raise ValueError("Minimum range has to be shorter than maximum range")

    # Copy `ds_Sv`
    ds_Sv_copy = ds_Sv.copy()

    # Return empty masks if searching range is outside the echosounder range
    if (r0 > ds_Sv_copy["depth"].max()) or (r1 < ds_Sv_copy["depth"].min()):
        attenuated_mask = xr.zeros_like(ds_Sv_copy["Sv"], dtype=bool)
        unfeasible_mask = xr.full_like(ds_Sv_copy["Sv"], True, dtype=bool)
        return attenuated_mask, unfeasible_mask

    # Create partial of echopy attenuation mask computation
    partial_echopy_attenuation_mask = partial(
        echopy_attenuated_signal_mask, r0=r0, r1=r1, n=n, threshold=threshold
    )

    # Compute attenuated signal and unfeasible (incapable of computing attenuated signal) masks
    attenuated_mask, unfeasible_mask = xr.apply_ufunc(
        partial_echopy_attenuation_mask,
        ds_Sv_copy["Sv"],
        ds_Sv_copy["depth"],
        input_core_dims=[["ping_time", "range_sample"], ["ping_time", "range_sample"]],
        output_core_dims=[["ping_time", "range_sample"], ["ping_time", "range_sample"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[bool, bool],
    )

    return attenuated_mask, unfeasible_mask


def estimate_background_noise(ds_Sv, ping_num, range_sample_num, noise_max=None):
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
    noise_obj = BackgroundNoiseEst(
        ds_Sv=ds_Sv.copy(), ping_num=ping_num, range_sample_num=range_sample_num
    )
    noise_obj.estimate_noise(noise_max=noise_max)
    return noise_obj.Sv_noise


@add_processing_level("L*B")
def remove_background_noise(ds_Sv, ping_num, range_sample_num, noise_max=None, SNR_threshold=3):
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
    noise_obj = BackgroundNoiseEst(
        ds_Sv=ds_Sv.copy(), ping_num=ping_num, range_sample_num=range_sample_num
    )
    noise_obj.remove_noise(noise_max=noise_max, SNR_threshold=SNR_threshold)
    ds_Sv = noise_obj.ds_Sv

    prov_dict = echopype_prov_attrs(process_type="processing")
    prov_dict["processing_function"] = "clean.remove_noise"
    ds_Sv = ds_Sv.assign_attrs(prov_dict)

    # The output ds_Sv is built as a copy of the input ds_Sv, so the step below is
    # not needed, strictly speaking. But doing makes the decorator function more generic
    ds_Sv = insert_input_processing_level(ds_Sv, input_ds=ds_Sv)

    return ds_Sv
