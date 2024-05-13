from functools import partial

import flox.xarray
import numpy as np
import pandas as pd
import xarray as xr

from ..commongrid.utils import _convert_bins_to_interval_index
from ..utils.compute import _lin2log, _log2lin


def setup_transient_noise_bins(ds_Sv, depth_bin, num_side_pings, exclude_above):
    """
    Setup range bin intervals and ping time bin intervals, and also return ping time
    and range sample values that are used.
    """
    # Create depth bin intervals
    depth_values_min = ds_Sv["depth"].min()
    depth_values_max = ds_Sv["depth"].max()
    depth_subset = ds_Sv["depth"].isel(channel=0, ping_time=0)
    valid_depth_mask = (
        (depth_subset - depth_bin >= depth_values_min)
        & (depth_subset + depth_bin <= depth_values_max)
        & (depth_subset - depth_bin >= exclude_above)
    )
    valid_depth_subset = depth_subset.where(valid_depth_mask).dropna(dim="range_sample").compute()
    depth_intervals = pd.IntervalIndex.from_tuples(
        tuple(zip((valid_depth_subset.data - depth_bin), (valid_depth_subset.data + depth_bin)))
    )
    range_sample_values_kept = valid_depth_subset.indexes["range_sample"].to_numpy()

    # Create ping time indices array
    ping_time_indices = xr.DataArray(
        np.arange(len(ds_Sv["ping_time"]), dtype=int),
        dims=["ping_time"],
        coords=[ds_Sv["ping_time"]],
        name="ping_time_indices",
    )

    # Create ping bin intervals
    ping_indices_min = 0
    ping_indices_max = len(ping_time_indices)
    valid_ping_time_mask = (ping_time_indices - num_side_pings >= ping_indices_min) & (
        ping_time_indices + num_side_pings <= ping_indices_max
    )
    ping_indices_kept = (
        ping_time_indices.where(valid_ping_time_mask).dropna(dim="ping_time").compute().to_numpy()
    )
    ping_intervals = pd.IntervalIndex.from_tuples(
        tuple(zip((ping_indices_kept - num_side_pings), (ping_indices_kept + num_side_pings)))
    )
    ping_values_kept = ds_Sv["ping_time"].isel(ping_time=ping_indices_kept.astype(int))

    return (
        ds_Sv,
        range_sample_values_kept,
        depth_intervals,
        ping_time_indices,
        ping_values_kept,
        ping_intervals,
    )


def _upsample_using_mapping(downsampled_Sv, original_Sv, raw_resolution_Sv_index_to_bin_index):
    """Use Sv index to bin index mapping to upsample Sv."""
    # Initialize upsampled Sv
    upsampled_Sv = np.zeros_like(original_Sv)

    # Iterate through index mapping dictionary
    for depth_index, bin_index in raw_resolution_Sv_index_to_bin_index.items():
        upsampled_Sv[depth_index] = downsampled_Sv[bin_index]

    return upsampled_Sv


def downsample_upsample_along_depth(ds_Sv, depth_bin):
    """
    Downsample and upsample Sv to mimic what was done in echopy impulse
    noise masking.
    """
    # Validate and compute range interval
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
    # TODO: Find a better way to do this? Perhaps something vectorized.
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


def echopy_impulse_noise_mask(Sv, num_side_pings, impulse_noise_threshold):
    """Single-channel impulse noise mask computation from echopy."""
    # Construct the two ping side-by-side comparison arrays
    dummy = np.zeros((Sv.shape[0], num_side_pings)) * np.nan
    comparison_forward = Sv - np.c_[Sv[:, num_side_pings:], dummy]
    comparison_backward = Sv - np.c_[dummy, Sv[:, 0:-num_side_pings]]
    comparison_forward[np.isnan(comparison_forward)] = np.inf
    comparison_backward[np.isnan(comparison_backward)] = np.inf

    # Create mask by checking if comparison arrays are above `impulse_noise_threshold`
    maskf = comparison_forward > impulse_noise_threshold
    maskb = comparison_backward > impulse_noise_threshold
    mask = maskf & maskb

    return mask


def echopy_attenuated_signal_mask(
    Sv, depth, upper_limit_sl, lower_limit_sl, num_pings, attenuation_signal_threshold
):
    """Single-channel attenuated signal mask computation from echopy."""
    # Initialize mask
    attenuated_mask = np.zeros(Sv.shape, dtype=bool)

    for ping_time_idx in range(Sv.shape[0]):

        # Find indices for upper and lower SL limits
        up = np.argmin(abs(depth[ping_time_idx, :] - upper_limit_sl))
        lw = np.argmin(abs(depth[ping_time_idx, :] - lower_limit_sl))

        # Mask when attenuation masking is feasible
        if not (
            (ping_time_idx - num_pings < 0)
            | (ping_time_idx + num_pings > Sv.shape[0] - 1)
            | np.all(np.isnan(Sv[ping_time_idx, up:lw]))
        ):
            # Compare ping and block medians, and mask ping if difference greater than
            # threshold.
            pingmedian = _lin2log(np.nanmedian(_log2lin(Sv[ping_time_idx, up:lw])))
            blockmedian = _lin2log(
                np.nanmedian(
                    _log2lin(Sv[(ping_time_idx - num_pings) : (ping_time_idx + num_pings), up:lw])
                )
            )
            if (pingmedian - blockmedian) < attenuation_signal_threshold:
                attenuated_mask[ping_time_idx, :] = True

    return attenuated_mask
