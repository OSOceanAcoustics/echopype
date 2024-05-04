"""
Functions for reducing variabilities in backscatter data.
"""

from functools import partial

import numpy as np
import xarray as xr

from ..utils.compute import _lin2log, _log2lin
from ..utils.prov import add_processing_level, echopype_prov_attrs, insert_input_processing_level
from .noise_est import NoiseEst


def _echopy_attenuated_signal_mask(Sv, depth, r0, r1, n, threshold):
    """Single-channel attenuated signal mask computation from echopy."""
    # Initialize masks
    attenuated_mask = np.zeros(Sv.shape, dtype=bool)
    unfeasible_mask = np.zeros(Sv.shape, dtype=bool)

    for ping_time_idx in range(Sv.shape[0]):

        # Find indices for upper and lower SL limits
        up = np.argmin(abs(depth[ping_time_idx, :] - r0))
        lw = np.argmin(abs(depth[ping_time_idx, :] - r1))

        # Mask where attenuation masking is unfeasible (e.g. edge issues, all-NANs)
        if (
            (ping_time_idx - n < 0)
            | (ping_time_idx + n > Sv.shape[0] - 1)
            | np.all(np.isnan(Sv[ping_time_idx, up:lw]))
        ):
            unfeasible_mask[ping_time_idx, :] = True

        # Compare ping and block medians, and mask ping if too difference greater than
        # threshold.
        else:
            pingmedian = _lin2log(np.nanmedian(_log2lin(Sv[ping_time_idx, up:lw])))
            blockmedian = _lin2log(
                np.nanmedian(_log2lin(Sv[(ping_time_idx - n) : (ping_time_idx + n), up:lw]))
            )
            if (pingmedian - blockmedian) < threshold:
                attenuated_mask[ping_time_idx, :] = True

    return attenuated_mask, unfeasible_mask


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
        2D boolean array with attenuated signal mask.
        2D boolean array with mask indicating where attenuated signal detection was unfeasible.

    References
    ----------
    Ryan et al. (2015) Reducing bias due to noise and attenuation in
    open-ocean echo integration data, ICES Journal of Marine Science, 72: 2482–2493.

    Code was derived from echopy's numpy single-channel implementation and translated into
    xarray code:
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
    _partial_echopy_attenuation_mask = partial(
        _echopy_attenuated_signal_mask, r0=r0, r1=r1, n=n, threshold=threshold
    )

    # Compute attenuated signal and unfeasible (incapable of computing attenuated signal) masks
    attenuated_mask, unfeasible_mask = xr.apply_ufunc(
        _partial_echopy_attenuation_mask,
        ds_Sv_copy["Sv"],
        ds_Sv_copy["depth"],
        input_core_dims=[["ping_time", "range_sample"], ["ping_time", "range_sample"]],
        output_core_dims=[["ping_time", "range_sample"], ["ping_time", "range_sample"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[bool, bool],
    )

    return attenuated_mask, unfeasible_mask


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
