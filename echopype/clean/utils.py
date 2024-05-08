from functools import partial

import flox.xarray
import numpy as np
import xarray as xr

from ..commongrid.utils import _convert_bins_to_interval_index, _setup_and_validate
from ..utils.compute import _lin2log, _log2lin


def _upsample_using_mapping(downsampled_Sv, original_Sv, raw_resolution_Sv_index_to_bin_index):
    """Use Sv index to bin index mapping to upsample Sv."""
    # Initialize upsampled Sv
    upsampled_Sv = np.zeros_like(original_Sv)

    # Iterate through index mapping dictionary
    for depth_index, bin_index in raw_resolution_Sv_index_to_bin_index.items():
        upsampled_Sv[depth_index] = downsampled_Sv[bin_index]

    return upsampled_Sv


def downsample_upsample_along_depth(ds_Sv, depth_bin: str = "5min"):
    """
    Downsample and upsample Sv to mimic what was done in echopy impulse
    noise masking.
    """
    # Validate Sv and compute range interval
    ds_Sv, depth_bin = _setup_and_validate(ds_Sv, "depth", "5min")
    echo_range_max = ds_Sv["depth"].max()
    range_interval = np.arange(0, echo_range_max + depth_bin, depth_bin)
    range_interval = _convert_bins_to_interval_index(range_interval)

    # Downsample Sv along range sample
    downsampled_Sv = flox.xarray.xarray_reduce(
        ds_Sv["Sv"].pipe(_log2lin),
        ds_Sv["channel"],
        ds_Sv["ping_time"],
        ds_Sv["depth"],
        expected_groups=(None, None, range_interval),
        isbin=[False, False, True],
        method="map-reduce",
        func="nanmean",
        skipna=True,
    ).pipe(_lin2log)

    # Create mapping from original depth/Sv variable to binned depth and
    # upsample. This upsampling operation assumes that depth values are
    # uniform across channel and ping time.
    raw_resolution_Sv_index_to_bin_index = {}
    for depth_index, depth in enumerate(ds_Sv["depth"].isel(channel=0, ping_time=0).values):
        for bin_index, bin in enumerate(downsampled_Sv["depth_bins"].values):
            if bin.left <= depth < bin.right:
                raw_resolution_Sv_index_to_bin_index[depth_index] = bin_index
                break
    _upsample_using_mapping_partial = partial(
        _upsample_using_mapping,
        raw_resolution_Sv_index_to_bin_index=raw_resolution_Sv_index_to_bin_index,
    )
    upsampled_Sv = xr.apply_ufunc(
        _upsample_using_mapping_partial,
        downsampled_Sv.compute(),
        ds_Sv["Sv"].compute(),
        input_core_dims=[["depth_bins"], ["range_sample"]],
        output_core_dims=[["range_sample"]],
        exclude_dims=set(["depth_bins", "range_sample"]),
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.float64],
    )

    return downsampled_Sv, upsampled_Sv


def echopy_impulse_noise_mask(Sv, num_side_pings, impulse_threshold):
    """Single-channel impulse noise mask computation from echopy."""
    # Construct the two ping side-by-side comparison arrays
    dummy = np.zeros((Sv.shape[0], num_side_pings)) * np.nan
    comparison_forward = Sv - np.c_[Sv[:, num_side_pings:], dummy]
    comparison_backward = Sv - np.c_[dummy, Sv[:, 0:-num_side_pings]]
    comparison_forward[np.isnan(comparison_forward)] = np.inf
    comparison_backward[np.isnan(comparison_backward)] = np.inf

    # Create mask by checking if comparison arrays are above `impulse_threshold`
    maskf = comparison_forward > impulse_threshold
    maskb = comparison_backward > impulse_threshold
    mask = maskf & maskb

    return mask


def echopy_attenuated_signal_mask(Sv, depth, r0, r1, n, threshold):
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
