"""
Functions for reducing variabilities in backscatter data.
"""

from functools import partial
from typing import Union

import numpy as np
import xarray as xr

from ..commongrid.utils import _setup_and_validate
from ..utils.compute import _lin2log, _log2lin
from ..utils.log import _init_logger
from ..utils.prov import add_processing_level, echopype_prov_attrs, insert_input_processing_level
from .utils import (
    add_remove_background_noise_attrs,
    calc_transient_noise_pooled_Sv,
    downsample_upsample_along_depth,
    echopy_attenuated_signal_mask,
    echopy_impulse_noise_mask,
)

logger = _init_logger(__name__)


def mask_transient_noise(
    ds_Sv: xr.Dataset,
    func: str = "nanmean",
    depth_bin: str = "20m",
    num_side_pings: int = 50,
    exclude_above: Union[int, float] = 250.0,
    transient_noise_threshold: Union[int, float] = 12.0,
) -> xr.DataArray:
    """
    Locate and create a mask for transient noise using a pooling comparison.

    Parameters
    ----------
    ds_Sv : xarray.Dataset
        Calibrated Sv data with depth data variable.
    func: str, default `nanmean`
        Pooling function used in the pooled Sv aggregation.
    depth_bin : str, default `20`m
        Pooling radius along ``depth`` in meters.
    num_side_pings : int, default `50`
        Number of side pings to look at for the pooling.
    exclude_above : Union[int, float], default `250`m
        Exclude all depth above (closer to the surface than) this value.
    transient_noise_threshold : Union[int, float], default `10.0`dB
        Transient noise threshold value (dB) for the pooling comparison.

    Returns
    -------
    xr.Dataset
        Xarray boolean array transient noise mask.

    References
    ----------
    This function's implementation is based on the following text reference:

        Ryan et al. (2015) Reducing bias due to noise and attenuation in
        open-ocean echo integration data, ICES Journal of Marine Science, 72: 2482–2493.

    Additionally, this code was derived from echopy's numpy single-channel implementation of
    transient noise masking and translated into xarray code:
    https://github.com/open-ocean-sounding/echopy/blob/master/echopy/processing/mask_transient.py # noqa
    """
    if "depth" not in ds_Sv.data_vars:
        raise ValueError(
            "Masking attenuated signal requires depth data variable in `ds_Sv`. "
            "Consider adding depth with `ds_Sv = ep.consolidate.add_depth(ds_Sv)`."
        )

    # Copy `ds_Sv`
    ds_Sv = ds_Sv.copy()

    # Check for appropriate function passed in
    if func != "nanmean" and func != "nanmedian":
        raise ValueError("Input `func` is `nanmode`. `func` must be `nanmean` or `nanmedian`.")

    # Warn when `func=nanmedian` since the sorting overhead makes it incredibly slow compared to
    # `nanmean`.
    logger.warning(
        "Consider using `func=nanmean`. `func=nanmedian` is an incredibly slow operation due to "
        "the overhead sorting."
    )

    # Setup depth bin
    ds_Sv, depth_bin = _setup_and_validate(ds_Sv, "depth", depth_bin)

    # Compute pooled Sv
    pooled_Sv = calc_transient_noise_pooled_Sv(
        ds_Sv, func, depth_bin, num_side_pings, exclude_above
    )

    # Compute transient noise mask
    transient_noise_mask = ds_Sv["Sv"] - pooled_Sv > transient_noise_threshold

    return transient_noise_mask


def mask_impulse_noise(
    ds_Sv: xr.Dataset,
    depth_bin: str = "5m",
    num_side_pings: int = 2,
    impulse_noise_threshold: Union[int, float] = 10.0,
) -> xr.DataArray:
    """
    Locate and create a mask for impulse noise using a ping-wise two-sided comparison.

    Parameters
    ----------
    ds_Sv : xarray.Dataset
        Calibrated Sv data with depth data variable.
    depth_bin : str, default `5`m
        Donwsampling bin size along ``depth`` in meters.
    num_side_pings : int, default `2`
        Number of side pings to look at for the two-side comparison.
    impulse_noise_threshold : Union[int, float], default `10.0`dB
        Impulse noise threshold value (dB) for the two-side comparison.

    Returns
    -------
    xr.Dataset
        Xarray boolean array impulse noise mask.

    References
    ----------
    This function's implementation is based on the following text reference:

        Ryan et al. (2015) Reducing bias due to noise and attenuation in
        open-ocean echo integration data, ICES Journal of Marine Science, 72: 2482–2493.

    Additionally, code was derived from echopy's numpy single-channel implementation of
    impulse noise masking and translated into xarray code:
    https://github.com/open-ocean-sounding/echopy/blob/master/echopy/processing/mask_impulse.py # noqa
    """
    if "depth" not in ds_Sv.data_vars:
        raise ValueError(
            "Masking attenuated signal requires depth data variable in `ds_Sv`. "
            "Consider adding depth with `ds_Sv = ep.consolidate.add_depth(ds_Sv)`."
        )

    # Copy `ds_Sv`
    ds_Sv = ds_Sv.copy()

    # Setup and validate Sv and depth bin
    ds_Sv, depth_bin = _setup_and_validate(ds_Sv, "depth", depth_bin)

    # Downsample and Upsample Sv along depth
    _, upsampled_Sv = downsample_upsample_along_depth(ds_Sv, depth_bin)

    # Create partial of `echopy_impulse_noise_mask`
    partial_echopy_impulse_noise_mask = partial(
        echopy_impulse_noise_mask,
        num_side_pings=num_side_pings,
        impulse_noise_threshold=impulse_noise_threshold,
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


def mask_attenuated_signal(
    ds_Sv: xr.Dataset,
    upper_limit_sl: Union[int, float] = 400.0,
    lower_limit_sl: Union[int, float] = 500.0,
    num_side_pings: int = 15,
    attenuation_signal_threshold: Union[int, float] = 8.0,
) -> xr.DataArray:
    """
    Locate attenuated signals and create an attenuated signal mask.

    Parameters
    ----------
    ds_Sv : xarray.Dataset
        Calibrated Sv data with depth data variable.
    upper_limit_sl : Union[int, float], default `400`m
        Upper limit of deep scattering layer line (m).
    lower_limit_sl : Union[int, float], default `500`m
        Lower limit of deep scattering layer line (m).
    num_side_pings : int, default `15`
        Number of preceding & subsequent pings defining the block.
    attenuation_signal_threshold : Union[int, float], default `8.0`dB
        Attenuation signal threshold value (dB) for the ping-block comparison.

    Returns
    -------
    xr.Dataset
        Xarray boolean array attenuated signal mask.

    References
    ----------
    This function's implementation is based on the following text reference:

        Ryan et al. (2015) Reducing bias due to noise and attenuation in
        open-ocean echo integration data, ICES Journal of Marine Science, 72: 2482–2493.

    Additionally, code was derived from echopy's numpy single-channel implementation of
    attenuation signal masking and translated into xarray code:
    https://github.com/open-ocean-sounding/echopy/blob/master/echopy/processing/mask_attenuated.py # noqa
    """
    if "depth" not in ds_Sv.data_vars:
        raise ValueError(
            "Masking attenuated signal requires depth data variable in `ds_Sv`. "
            "Consider adding depth with `ds_Sv = ep.consolidate.add_depth(ds_Sv)`."
        )

    # Check range values
    if upper_limit_sl > lower_limit_sl:
        raise ValueError("Minimum range has to be shorter than maximum range")

    # Copy `ds_Sv`
    ds_Sv = ds_Sv.copy()

    # Return empty masks if searching range is outside the echosounder range
    if (upper_limit_sl > ds_Sv["depth"].max()) or (lower_limit_sl < ds_Sv["depth"].min()):
        attenuated_mask = xr.zeros_like(ds_Sv["Sv"], dtype=bool)
        return attenuated_mask

    # Create partial of echopy attenuation mask computation
    partial_echopy_attenuation_mask = partial(
        echopy_attenuated_signal_mask,
        upper_limit_sl=upper_limit_sl,
        lower_limit_sl=lower_limit_sl,
        num_side_pings=num_side_pings,
        attenuation_signal_threshold=attenuation_signal_threshold,
    )

    # Compute attenuated signal mask
    attenuated_mask = xr.apply_ufunc(
        partial_echopy_attenuation_mask,
        ds_Sv["Sv"],
        ds_Sv["depth"],
        input_core_dims=[["ping_time", "range_sample"], ["ping_time", "range_sample"]],
        output_core_dims=[["ping_time", "range_sample"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[bool],
    )

    return attenuated_mask


def estimate_background_noise(
    ds_Sv: xr.Dataset, ping_num: int, range_sample_num: int, noise_max: float = None
) -> xr.DataArray:
    """
    Estimate background noise by computing mean calibrated power of a collection of pings.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        Dataset containing ``Sv`` and ``echo_range`` [m].
    ping_num : int
        Number of pings to obtain noise estimates
    range_sample_num : int
        Number of samples along the ``range_sample`` dimension to obtain noise estimates.
    noise_max : float, default `None`
        The upper limit for background noise expected under the operating conditions.

    Returns
    -------
    A DataArray containing noise estimated from the input ``ds_Sv``

    Notes
    -----
    This function's implementation is based on the following text reference:

        De Robertis & Higginbottom. 2007.
        A post-processing technique to estimate the signal-to-noise ratio
        and remove echosounder background noise.
        ICES Journal of Marine Sciences 64(6): 1282–1291.
    """
    # Compute transmission loss
    spreading_loss = 20 * np.log10(ds_Sv["echo_range"].where(ds_Sv["echo_range"] >= 1, other=1))
    absorption_loss = 2 * ds_Sv["sound_absorption"] * ds_Sv["echo_range"]

    # Compute power binned averages
    power_cal = _log2lin(ds_Sv["Sv"] - spreading_loss - absorption_loss)
    power_cal_binned_avg = 10 * np.log10(
        power_cal.coarsen(
            ping_time=ping_num,
            range_sample=range_sample_num,
            boundary="pad",
        ).mean()
    )

    # Compute noise
    noise = power_cal_binned_avg.min(dim="range_sample", skipna=True)

    # Align noise `ping_time` to the first index of each coarsened `ping_time` bin
    noise = noise.assign_coords(ping_time=ping_num * np.arange(len(noise["ping_time"])))

    # Limit max noise level
    noise = noise.where(noise < noise_max, noise_max) if noise_max is not None else noise

    # Upsample noise to original ping time dimension
    Sv_noise = (
        noise.reindex({"ping_time": power_cal["ping_time"]}, method="ffill")
        + spreading_loss
        + absorption_loss
    )

    return Sv_noise


@add_processing_level("L*B")
def remove_background_noise(
    ds_Sv: xr.Dataset,
    ping_num: int,
    range_sample_num: int,
    noise_max: float = None,
    SNR_threshold: float = 3.0,
) -> xr.Dataset:
    """
    Remove noise by using estimates of background noise
    from mean calibrated power of a collection of pings.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        dataset containing ``Sv`` and ``echo_range`` [m]
    ping_num : int
        Number of pings to obtain noise estimates.
    range_sample_num : int
        Number of samples along the ``range_sample`` dimension to obtain noise estimates.
    noise_max : float, default `None`
        The upper limit for background noise expected under the operating conditions.
    SNR_threshold : float, default `3.0`
        Acceptable signal-to-noise ratio, default to 3 dB.

    Returns
    -------
    The input dataset with additional variables, including
    the corrected Sv (``Sv_corrected``) and the noise estimates (``Sv_noise``)

    Notes
    -----
    This function's implementation is based on the following text reference:

        De Robertis & Higginbottom. 2007.
        A post-processing technique to estimate the signal-to-noise ratio
        and remove echosounder background noise.
        ICES Journal of Marine Sciences 64(6): 1282–1291.
    """
    # Compute Sv_noise
    Sv_noise = estimate_background_noise(ds_Sv, ping_num, range_sample_num, noise_max=noise_max)

    # Correct Sv for noise
    linear_corrected_Sv = _log2lin(ds_Sv["Sv"]) - _log2lin(Sv_noise)
    corrected_Sv = _lin2log(linear_corrected_Sv.where(linear_corrected_Sv > 0, other=np.nan))
    corrected_Sv = corrected_Sv.where(corrected_Sv - Sv_noise > SNR_threshold, other=np.nan)

    # Assemble output dataset
    ds_Sv["Sv_noise"] = Sv_noise
    ds_Sv["Sv_noise"] = add_remove_background_noise_attrs(
        ds_Sv["Sv_noise"], "noise", ping_num, range_sample_num, SNR_threshold, noise_max
    )
    ds_Sv["Sv_corrected"] = corrected_Sv
    ds_Sv["Sv_corrected"] = add_remove_background_noise_attrs(
        ds_Sv["Sv_corrected"], "corrected", ping_num, range_sample_num, SNR_threshold, noise_max
    )
    prov_dict = echopype_prov_attrs(process_type="processing")
    prov_dict["processing_function"] = "clean.remove_background_noise"
    ds_Sv = ds_Sv.assign_attrs(prov_dict)

    # The output `ds_Sv` is built as a copy of the input `ds_Sv`, so the step below is
    # not needed, strictly speaking. But doing makes the decorator function more generic
    ds_Sv = insert_input_processing_level(ds_Sv, input_ds=ds_Sv)

    return ds_Sv
