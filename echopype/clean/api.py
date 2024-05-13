"""
Functions for reducing variabilities in backscatter data.
"""

from functools import partial
from typing import Union

import flox.xarray
import numpy as np
import xarray as xr

from ..commongrid.utils import _setup_and_validate
from ..utils.compute import _lin2log, _log2lin
from ..utils.prov import add_processing_level, echopype_prov_attrs, insert_input_processing_level
from .background_noise_est import BackgroundNoiseEst
from .utils import (
    downsample_upsample_along_depth,
    echopy_attenuated_signal_mask,
    echopy_impulse_noise_mask,
    setup_transient_noise_bins,
)


def mask_transient_noise(
    ds_Sv: xr.Dataset,
    func: str = "nanmean",
    depth_bin: str = "20m",
    num_side_pings: int = 50,
    exclude_above: Union[int, float] = 250.0,
    transient_noise_threshold: Union[int, float] = 12.0,
):
    """
    Locate and create a mask for transient noise using a pooling comparison.

    Parameters
    ----------
    ds_Sv : xarray.Dataset
        Calibrated Sv data with depth data variable.
    func: str, default `nanmean`
        Pooling function used in flox reduction.
        `nanmedian` can also be used, but then the masking operation cannot be Dask delayed.
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
        Xarray boolean array impulse noise mask.

    References
    ----------
    Ryan et al. (2015) Reducing bias due to noise and attenuation in
    open-ocean echo integration data, ICES Journal of Marine Science, 72: 2482–2493.
    """
    if "depth" not in ds_Sv.data_vars:
        raise ValueError(
            "Masking attenuated signal requires depth data variable in `ds_Sv`. "
            "Consider adding depth with `ds_Sv = ep.consolidate.add_depth(ds_Sv)`."
        )

    # Copy `ds_Sv`
    ds_Sv_copy = ds_Sv.copy()

    # Setup and validate Sv and depth bin
    ds_Sv_copy, depth_bin = _setup_and_validate(ds_Sv_copy, "depth", depth_bin)

    # Setup binning variables
    (
        ds_Sv_copy,
        range_sample_values_kept,
        depth_intervals,
        ping_time_indices,
        ping_values_kept,
        ping_intervals,
    ) = setup_transient_noise_bins(ds_Sv_copy, depth_bin, num_side_pings, exclude_above)

    # The `nanmedian` aggregation function is only implemented with `blockwise` reduction for dask
    # arrays; however, `blockwise` cannot be used since we have blocks that exist in separate chunks
    # due to the overlapping ping and depth intervals. For now, we will always compute Sv arrays
    # that are chunked and when `func==median`.
    # TODO: Allow for Dask Delay when `func==nanmedian` is implemented for flox `map-reduce`.
    if func == "nanmedian" and ds_Sv.chunks != {}:
        ds_Sv_copy = ds_Sv_copy.compute()

    # Pool Sv based on defined ping and depth intervals
    pooled_Sv = flox.xarray.xarray_reduce(
        ds_Sv_copy["Sv"].pipe(_log2lin),
        ds_Sv_copy["channel"],
        ping_time_indices,
        ds_Sv_copy["depth"],
        expected_groups=(None, ping_intervals, depth_intervals),
        isbin=(False, True, True),
        method="map-reduce",
        func=func,
        engine="numpy",  # numpy is currently faster than the flox internal implementations
        skipna=True,
    ).pipe(_lin2log)

    # Rename dimensions, assign coords, and broadcast to original Sv dimensions
    pooled_Sv = (
        pooled_Sv.rename({"ping_time_indices_bins": "ping_time", "depth_bins": "range_sample"})
        .assign_coords(ping_time=ping_values_kept, range_sample=range_sample_values_kept)
        .broadcast_like(ds_Sv_copy["Sv"])
    )

    # Compute transient noise mask
    transient_noise_mask = ds_Sv_copy["Sv"] - pooled_Sv > transient_noise_threshold

    return transient_noise_mask


def mask_impulse_noise(
    ds_Sv: xr.Dataset,
    depth_bin: str = "5m",
    num_side_pings: int = 1,
    impulse_noise_threshold: Union[int, float] = 10.0,
):
    """
    Locate and create a mask for impulse noise using a ping-wise two-sided comparison.

    Parameters
    ----------
    ds_Sv : xarray.Dataset
        Calibrated Sv data with depth data variable.
    depth_bin : str, default `5`m
        Donwsampling bin size along ``depth`` in meters.
    num_side_pings : int, default `1`
        Number of side pings to look at for the two-side comparison.
    impulse_noise_threshold : Union[int, float], default `10.0`dB
        Impulse noise threshold value (dB) for the two-side comparison.

    Returns
    -------
    xr.Dataset
        Xarray boolean array impulse noise mask.

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

    # Setup and validate Sv and depth bin
    ds_Sv_copy, depth_bin = _setup_and_validate(ds_Sv_copy, "depth", depth_bin)

    # Downsample and Upsample Sv along depth
    _, upsampled_Sv = downsample_upsample_along_depth(ds_Sv_copy, depth_bin)

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
    num_pings: int = 30,
    attenuation_signal_threshold: Union[int, float] = 8.0,
):
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
    num_pings : int, default `30`
        Number of preceding & subsequent pings defining the block.
    attenuation_signal_threshold : Union[int, float], default `8.0`dB
        Attenuation signal threshold value (dB) for the ping-block comparison.

    Returns
    -------
    xr.Dataset
        Xarray boolean array attenuated signal mask.

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
    if upper_limit_sl > lower_limit_sl:
        raise ValueError("Minimum range has to be shorter than maximum range")

    # Copy `ds_Sv`
    ds_Sv_copy = ds_Sv.copy()

    # Return empty masks if searching range is outside the echosounder range
    if (upper_limit_sl > ds_Sv_copy["depth"].max()) or (lower_limit_sl < ds_Sv_copy["depth"].min()):
        attenuated_mask = xr.zeros_like(ds_Sv_copy["Sv"], dtype=bool)
        return attenuated_mask

    # Create partial of echopy attenuation mask computation
    partial_echopy_attenuation_mask = partial(
        echopy_attenuated_signal_mask,
        upper_limit_sl=upper_limit_sl,
        lower_limit_sl=lower_limit_sl,
        num_pings=num_pings,
        attenuation_signal_threshold=attenuation_signal_threshold,
    )

    # Compute attenuated signal mask
    attenuated_mask = xr.apply_ufunc(
        partial_echopy_attenuation_mask,
        ds_Sv_copy["Sv"],
        ds_Sv_copy["depth"],
        input_core_dims=[["ping_time", "range_sample"], ["ping_time", "range_sample"]],
        output_core_dims=[["ping_time", "range_sample"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[bool],
    )

    return attenuated_mask


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
    prov_dict["processing_function"] = "clean.remove_background_noise"
    ds_Sv = ds_Sv.assign_attrs(prov_dict)

    # The output ds_Sv is built as a copy of the input ds_Sv, so the step below is
    # not needed, strictly speaking. But doing makes the decorator function more generic
    ds_Sv = insert_input_processing_level(ds_Sv, input_ds=ds_Sv)

    return ds_Sv
