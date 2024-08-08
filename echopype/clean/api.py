"""
Functions for reducing variabilities in backscatter data.
"""

from functools import partial

import numpy as np
import xarray as xr

from ..commongrid.utils import _parse_x_bin
from ..utils.compute import _lin2log, _log2lin
from ..utils.log import _init_logger
from ..utils.prov import add_processing_level, echopype_prov_attrs, insert_input_processing_level
from .utils import (
    add_remove_background_noise_attrs,
    downsample_upsample_along_depth,
    echopy_attenuated_signal_mask,
    echopy_impulse_noise_mask,
    extract_dB,
    index_binning_downsample_upsample_along_depth,
    index_binning_pool_Sv,
    pool_Sv,
)

logger = _init_logger(__name__)


def mask_transient_noise(
    ds_Sv: xr.Dataset,
    func: str = "nanmean",
    depth_bin: str = "10m",
    num_side_pings: int = 25,
    exclude_above: str = "250.0m",
    transient_noise_threshold: str = "12.0dB",
    range_var: str = "depth",
    use_index_binning: bool = False,
    chunk_dict: dict = {},
) -> xr.DataArray:
    """
    Locate and create a mask for transient noise using a pooling comparison.

    Parameters
    ----------
    ds_Sv : xarray.Dataset
        Calibrated Sv data with depth data variable.
    func: str, default "nanmedian"
        Pooling function used in the pooled Sv aggregation.
    depth_bin : str, default "10m"
        Pooling bin size vertically along the vertical range variable (`range_var`).
    num_side_pings : int, default 25
        Number of side pings to look at for the pooling.
    exclude_above : str, default "250m"
        Exclude all depth above (closer to the surface than) this value.
    transient_noise_threshold : str, default "10.0dB"
        Transient noise threshold value (dB) for the pooling comparison.
    range_var : str, default "depth"
        Vertical Range Variable. Can be either `depth` or `echo_range`.
    use_index_binning : bool, default False
        Speeds up aggregations by assuming depth is uniform and binning based
        on `range_sample` indices instead of `depth` values.
    chunk_dict : dict, default {}
        Dictionary containing chunk sizes for use in the Dask-Image
        pooling function. Only used when `use_index_binning=True`.

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

    Examples
    --------

    Default Transient Noise Mask Usage: Without using index binning along the vertical axis:

    >>> mask_transient_noise(
    >>>     ds_Sv
    >>>     func="nanmean",
    >>>     depth_bin = "10m",
    >>>     num_side_pings = 25,
    >>>     exclude_above = "250.0m",
    >>>     transient_noise_threshold = "12.0dB",
    >>>     range_var = "depth",
    >>>     use_index_binning = False
    >>> )

    Vertical Index Binning Transient Noise Mask Usage with Chunking across Vertical
    and Horizontal Axis:

    >>> mask_transient_noise(
    >>>     ds_Sv
    >>>     func="nanmean",
    >>>     depth_bin = "10m",
    >>>     num_side_pings = 25,
    >>>     exclude_above = "250.0m",
    >>>     transient_noise_threshold = "12.0dB",
    >>>     range_var = "depth",
    >>>     use_index_binning = True,
    >>>     chunk_dict = {"ping_time": 1000, "range_sample": 50}
    >>> )

    To compute the pooled Sv, prior to the binning operation, the number of range sample
    indices needed to encapsulate "10m" will be calculated, and this number will be used to
    do a completely index based nanmean binning operation. This operation is exactly the
    same as applying a mean filter over an image, but in this case, it is applied to an
    Echogram.
    """
    # Check range variable
    if range_var not in ["echo_range", "depth"]:
        raise ValueError("`range_var` must be either `echo_range` or `depth`.")
    if range_var not in ds_Sv.data_vars and not use_index_binning:
        raise ValueError(
            f"Masking transient noise requires `{range_var}` data variable in `ds_Sv`."
        )

    # Copy `ds_Sv`
    ds_Sv = ds_Sv.copy()

    # Check for appropriate function passed in
    if func != "nanmean" and func != "nanmedian":
        raise ValueError(f"Input `func` is `{func}`. `func` must be `nanmean` or `nanmedian`.")
    # Assign string to numpy function
    if func == "nanmean":
        func = np.nanmean
    elif func == "nanmedian":
        # Warn when `func=nanmedian` since the sorting overhead makes it incredibly slow compared to
        # other non-sorting aggregations like `nanmean`.
        logger.warning(
            "`func=nanmedian` is an incredibly slow operation due to the overhead sorting. "
            "We plan to add the Fielding Transient Noise Filter in the future"
            "described here: https://github.com/OSOceanAcoustics/echopype/issues/1352"
        )
        func = np.nanmedian

    # Extract dB float value
    transient_noise_threshold = extract_dB(transient_noise_threshold)

    # Setup and validate depth bin and `exclude_above` values
    depth_bin = _parse_x_bin(depth_bin, "range_bin")
    exclude_above = _parse_x_bin(exclude_above, "range_bin")

    if not use_index_binning:
        # Compute pooled Sv with assumption that depth is not uniform across
        # `ping_time` and `channel`
        pooled_Sv = pool_Sv(ds_Sv, func, depth_bin, num_side_pings, exclude_above, range_var)
    else:
        # Compute pooled Sv using Dask-Image's Generic Filter with assumption that depth is uniform
        # across `ping_time` per `channel` dimension.
        pooled_Sv = index_binning_pool_Sv(
            ds_Sv, func, depth_bin, num_side_pings, exclude_above, range_var, chunk_dict
        )

    # Compute transient noise mask
    transient_noise_mask = ds_Sv["Sv"] - pooled_Sv > transient_noise_threshold

    return transient_noise_mask


def mask_impulse_noise(
    ds_Sv: xr.Dataset,
    depth_bin: str = "5m",
    num_side_pings: int = 2,
    impulse_noise_threshold: str = "10.0dB",
    range_var: str = "depth",
    use_index_binning: bool = False,
) -> xr.DataArray:
    """
    Locate and create a mask for impulse noise using a ping-wise two-sided comparison.

    Parameters
    ----------
    ds_Sv : xarray.Dataset
        Calibrated Sv data with depth data variable.
    depth_bin : str, default "5m"
        Donwsampling bin size along vertical range variable (`range_var`) in meters.
    num_side_pings : int, default 2
        Number of side pings to look at for the two-side comparison.
    impulse_noise_threshold : str, default "10.0dB"
        Impulse noise threshold value (dB) for the two-side comparison.
    range_var : str, default "depth"
        Vertical Axis Range Variable. Can be either "depth" or "echo_range".
    use_index_binning : bool, default False
        Speeds up aggregations by assuming depth is uniform and binning based
        on `range_sample` indices instead of `depth` values.

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
    # Check range variable
    if range_var not in ["echo_range", "depth"]:
        raise ValueError("`range_var` must be either `echo_range` or `depth`.")
    if range_var not in ds_Sv.data_vars and not use_index_binning:
        raise ValueError(f"Masking impulse noise requires `{range_var}` data variable in `ds_Sv`.")

    # Copy `ds_Sv`
    ds_Sv = ds_Sv.copy()

    # Extract dB float value
    impulse_noise_threshold = extract_dB(impulse_noise_threshold)

    # Setup and validate depth bin
    depth_bin = _parse_x_bin(depth_bin, "range_bin")

    if not use_index_binning:
        # Compute Upsampled Sv with assumption that depth is not uniform across
        # `ping_time` and `channel`
        _, upsampled_Sv = downsample_upsample_along_depth(ds_Sv, depth_bin, range_var)
    else:
        # Compute Upsampled Sv using Coarsen with assumption that depth is uniform
        # across `ping_time` per `channel` dimension.
        upsampled_Sv = index_binning_downsample_upsample_along_depth(ds_Sv, depth_bin, range_var)

    # Create partial of `echopy_impulse_noise_mask`
    partial_echopy_impulse_noise_mask = partial(
        echopy_impulse_noise_mask,
        num_side_pings=num_side_pings,
        impulse_noise_threshold=impulse_noise_threshold,
    )

    # Must rechunk `range_sample` to full chunk because of the following `ValueError`
    # from `apply_ufunc`:
    # 'dimension range_sample on 0th function argument to apply_ufunc with dask='parallelized'
    # consists of multiple chunks, but is also a core dimension. To fix, either rechunk into
    # a single array chunk along this dimension, i.e., ``.chunk(dict(range_sample=-1))``, or
    # pass ``allow_rechunk=True`` in ``dask_gufunc_kwargs`` but beware that this may
    # significantly increase memory usage'.
    if hasattr(upsampled_Sv, "chunks") and upsampled_Sv.chunks is not None:
        upsampled_Sv = upsampled_Sv.chunk(dict(range_sample=-1, ping_time=-1))

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
    upper_limit_sl: str = "400.0m",
    lower_limit_sl: str = "500.0m",
    num_side_pings: int = 15,
    attenuation_signal_threshold: str = "8.0dB",
    range_var: str = "depth",
) -> xr.DataArray:
    """
    Locate attenuated signals and create an attenuated signal mask.

    Parameters
    ----------
    ds_Sv : xarray.Dataset
        Calibrated Sv data with depth data variable.
    upper_limit_sl : str, default "400m"
        Upper limit of deep scattering layer line (m).
    lower_limit_sl : str, default "500m"
        Lower limit of deep scattering layer line (m).
    num_side_pings : int, default 15
        Number of preceding & subsequent pings defining the block.
    attenuation_signal_threshold : str, default "8.0dB"
        Attenuation signal threshold value (dB) for the ping-block comparison.
    range_var : str, default "depth"
        Vertical Axis Range Variable. Can be either `depth` or `echo_range`.

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
    # Check range variable
    if range_var not in ["echo_range", "depth"]:
        raise ValueError("`range_var` must be either `echo_range` or `depth`.")
    if range_var not in ds_Sv.data_vars:
        raise ValueError(
            f"Masking attenuated signal requires `{range_var}` data variable in `ds_Sv`."
        )

    # Check range values
    if upper_limit_sl > lower_limit_sl:
        raise ValueError("Minimum range has to be shorter than maximum range")

    # Copy `ds_Sv`
    ds_Sv = ds_Sv.copy()

    # Extract dB float value
    attenuation_signal_threshold = extract_dB(attenuation_signal_threshold)

    # Setup and validate upper and lower limit SL range values
    lower_limit_sl = _parse_x_bin(lower_limit_sl, "range_bin")
    upper_limit_sl = _parse_x_bin(upper_limit_sl, "range_bin")

    # Return empty masks if searching range is outside the echosounder range
    if (upper_limit_sl > ds_Sv[range_var].max()) or (lower_limit_sl < ds_Sv[range_var].min()):
        attenuated_mask = xr.zeros_like(ds_Sv[range_var], dtype=bool)
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
        ds_Sv[range_var],
        input_core_dims=[["ping_time", "range_sample"], ["ping_time", "range_sample"]],
        output_core_dims=[["ping_time", "range_sample"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[bool],
    )

    return attenuated_mask


def estimate_background_noise(
    ds_Sv: xr.Dataset, ping_num: int, range_sample_num: int, background_noise_max: str = None
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
    background_noise_max : str, default None
        The upper limit (dB) for background noise expected under the operating conditions.

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
    if background_noise_max is not None:
        # Extract dB float value
        background_noise_max = extract_dB(background_noise_max)

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
    power_cal = power_cal.assign_coords(ping_time=np.arange(len(power_cal["ping_time"])))

    # Limit max noise level
    noise = (
        noise.where(noise < background_noise_max, background_noise_max)
        if background_noise_max is not None
        else noise
    )

    # Upsample noise to original ping time dimension
    Sv_noise = (
        noise.reindex({"ping_time": power_cal["ping_time"]}, method="ffill").assign_coords(
            ping_time=ds_Sv["ping_time"]
        )
        + spreading_loss
        + absorption_loss
    )

    return Sv_noise


@add_processing_level("L*B")
def remove_background_noise(
    ds_Sv: xr.Dataset,
    ping_num: int,
    range_sample_num: int,
    background_noise_max: str = None,
    SNR_threshold: float = "3.0dB",
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
    background_noise_max : str, default None
        The upper limit for background noise expected under the operating conditions.
    SNR_threshold : str, default "3.0dB"
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
    if SNR_threshold is not None:
        # Extract dB float value
        SNR_threshold = extract_dB(SNR_threshold)

    # Compute Sv_noise
    Sv_noise = estimate_background_noise(
        ds_Sv, ping_num, range_sample_num, background_noise_max=background_noise_max
    )

    # Correct Sv for noise
    linear_corrected_Sv = _log2lin(ds_Sv["Sv"]) - _log2lin(Sv_noise)
    corrected_Sv = _lin2log(linear_corrected_Sv.where(linear_corrected_Sv > 0, other=np.nan))
    corrected_Sv = corrected_Sv.where(corrected_Sv - Sv_noise > SNR_threshold, other=np.nan)

    # Assemble output dataset
    ds_Sv["Sv_noise"] = Sv_noise
    ds_Sv["Sv_noise"] = add_remove_background_noise_attrs(
        ds_Sv["Sv_noise"], "noise", ping_num, range_sample_num, SNR_threshold, background_noise_max
    )
    ds_Sv["Sv_corrected"] = corrected_Sv
    ds_Sv["Sv_corrected"] = add_remove_background_noise_attrs(
        ds_Sv["Sv_corrected"],
        "corrected",
        ping_num,
        range_sample_num,
        SNR_threshold,
        background_noise_max,
    )
    prov_dict = echopype_prov_attrs(process_type="processing")
    prov_dict["processing_function"] = "clean.remove_background_noise"
    ds_Sv = ds_Sv.assign_attrs(prov_dict)

    # The output `ds_Sv` is built as a copy of the input `ds_Sv`, so the step below is
    # not needed, strictly speaking. But doing makes the decorator function more generic
    ds_Sv = insert_input_processing_level(ds_Sv, input_ds=ds_Sv)

    return ds_Sv
