import re
from typing import Callable

import dask_image.ndfilters
import flox.xarray
import numpy as np
import xarray as xr

from ..commongrid.utils import _convert_bins_to_interval_index
from ..utils.compute import _lin2log, _log2lin


def extract_dB(dB_str: str) -> float:
    """Extract float value from decibal string in the form of 'NUMdB'."""
    # Search for numeric part using regular expressions and convert to float
    if not isinstance(dB_str, str):
        raise TypeError(
            "Decibal input must be a string formatted as `NUMdB` or `NUMdb."
            f"Cannot be of type `{type(dB_str)}`."
        )
    pattern = r"^[-+]?\d+\.?\d*(?:dB|db)$"  # Ensures exact match with format
    match = re.search(pattern, dB_str, flags=re.IGNORECASE)  # Case-insensitive search
    if match:
        return float(match.group(0)[:-2])  # Extract number and remove "dB"
    else:
        raise ValueError("Decibal string must be formatted as 'NUMdB' or `NUMdb")


def pool_Sv(
    ds_Sv: xr.Dataset,
    func: Callable,
    depth_bin: float,
    num_side_pings: int,
    exclude_above: float,
    range_var: str,
) -> xr.DataArray:
    """
    Compute pooled Sv array for transient noise masking.
    """
    # Create ping time indices array
    ping_time_indices = xr.DataArray(
        np.arange(len(ds_Sv["ping_time"]), dtype=int),
        dims=["ping_time"],
        coords=[ds_Sv["ping_time"]],
        name="ping_time_indices",
    )

    # Create NaN pooled Sv array
    pooled_Sv = xr.full_like(ds_Sv["Sv"], np.nan)

    # Set min max values
    depth_values_min = ds_Sv[range_var].min()
    depth_values_max = ds_Sv[range_var].max()
    ping_time_index_min = 0
    ping_time_index_max = len(ds_Sv["ping_time"])

    # Iterate through the channel dimension
    for channel_index in range(len(ds_Sv["channel"])):

        # Set channel arrays
        chan_Sv = ds_Sv["Sv"].isel(channel=channel_index)
        chan_depth = ds_Sv[range_var].isel(channel=channel_index)

        # Iterate through the range sample dimension
        for range_sample_index in range(len(ds_Sv["range_sample"])):

            # Iterate through the ping time dimension
            for ping_time_index in range(len(ds_Sv["ping_time"])):

                # Grab current depth
                current_depth = ds_Sv[range_var].isel(
                    channel=channel_index,
                    range_sample=range_sample_index,
                    ping_time=ping_time_index,
                )

                # Check if current value is within a valid window
                if (
                    (current_depth - depth_bin >= depth_values_min)
                    & (current_depth + depth_bin <= depth_values_max)
                    & (current_depth - depth_bin >= exclude_above)
                    & (ping_time_index - num_side_pings >= ping_time_index_min)
                    & (ping_time_index + num_side_pings <= ping_time_index_max)
                ):

                    # Compute aggregate window Sv value
                    window_mask = (
                        (current_depth - depth_bin <= chan_depth)
                        & (chan_depth <= current_depth + depth_bin)
                        & (ping_time_index - num_side_pings <= ping_time_indices)
                        & (ping_time_indices <= ping_time_index + num_side_pings)
                    )
                    window_Sv = chan_Sv.where(window_mask, other=np.nan).pipe(_log2lin)
                    aggregate_window_Sv_value = window_Sv.pipe(func)
                    aggregate_window_Sv_value = _lin2log(aggregate_window_Sv_value)

                    # Put aggregate value in pooled Sv array
                    pooled_Sv[
                        dict(
                            channel=channel_index,
                            range_sample=range_sample_index,
                            ping_time=ping_time_index,
                        )
                    ] = aggregate_window_Sv_value

    return pooled_Sv


def index_binning_pool_Sv(
    ds_Sv: xr.Dataset,
    func: Callable,
    depth_bin: int,
    num_side_pings: int,
    exclude_above: float,
    range_var: str,
    chunk_dict: dict,
) -> xr.DataArray:
    """
    Compute pooled Sv array for transient noise masking using index binning.

    This function makes the assumption that within each channel, the difference
    between depth values is uniform across all pings. Thus, computing the number of
    range sample indices needed to cover the depth bin is a channel-specific task.
    """
    # Drop `filenames` dimension if exists and transpose Dataset
    ds_Sv = ds_Sv.drop_dims("filenames", errors="ignore").transpose(
        "channel", "ping_time", "range_sample"
    )

    # Compute number of range sample indices that are needed to encapsulate the `depth_bin`
    # value per channel.
    all_chan_num_range_sample_indices = np.ceil(
        depth_bin / np.nanmean(np.diff(ds_Sv[range_var], axis=2), axis=(1, 2))
    ).astype(int)

    # Create list for pooled Sv DataArrays
    pooled_Sv_list = []

    # Iterate through channels
    for channel_index in range(len(ds_Sv["channel"])):
        # Create calibrated Sv DataArray copies and remove values too close to the surface
        min_range_sample = (ds_Sv[range_var] <= exclude_above).argmin().values
        chan_Sv = ds_Sv["Sv"].isel(
            channel=channel_index,
            range_sample=slice(min_range_sample, None),
        )
        chan_pooled_Sv = ds_Sv["Sv"].isel(
            channel=channel_index,
            range_sample=slice(min_range_sample, None),
        )

        # Grab channel-specific number of range sample indices for vertical binning
        chan_num_range_sample_indices = all_chan_num_range_sample_indices[channel_index]

        # Create pooling size list
        pooling_size = [(2 * num_side_pings) + 1, (2 * chan_num_range_sample_indices) + 1]

        # Rechunk Sv since `generic_filter` expects a Dask Array
        chan_Sv = chan_Sv.chunk(chunk_dict)

        # Compute `chan_pooled_Sv` values using dask-image's generic filter
        chan_pooled_Sv.values = _lin2log(
            dask_image.ndfilters.generic_filter(
                chan_Sv.pipe(_log2lin).data,
                function=func,
                size=pooling_size,
                mode="reflect",
            ).compute()
        )

        # Expand `chan_pooled_Sv` to original Sv dimensions, effectively NaN'ing values close
        # to the surface.
        chan_pooled_Sv = chan_pooled_Sv.reindex_like(ds_Sv["Sv"].isel(channel=channel_index))

        # Place in pooled Sv list
        pooled_Sv_list.append(chan_pooled_Sv)

    # Concatenate arrays along channel dimension
    pooled_Sv = xr.concat(pooled_Sv_list, dim="channel")

    return pooled_Sv


def downsample_upsample_along_depth(
    ds_Sv: xr.Dataset, depth_bin: float, range_var: str
) -> xr.DataArray:
    """
    Downsample and upsample Sv to mimic what was done in echopy impulse
    noise masking.
    """
    # Validate and compute range interval
    depth_min = ds_Sv[range_var].min()
    depth_max = ds_Sv[range_var].max()
    range_interval = np.arange(depth_min, depth_max + depth_bin, depth_bin)
    range_interval = _convert_bins_to_interval_index(range_interval)

    # Downsample Sv along range sample
    downsampled_Sv = flox.xarray.xarray_reduce(
        ds_Sv["Sv"].pipe(_log2lin),
        ds_Sv["channel"],
        ds_Sv["ping_time"],
        ds_Sv[range_var],
        expected_groups=(None, None, range_interval),
        isbin=[False, False, True],
        method="map-reduce",
        func="nanmean",
        skipna=True,
    ).pipe(_lin2log)

    # Assign a depth bin index to each Sv depth value
    depth_bin_assignment = xr.DataArray(
        np.digitize(
            ds_Sv[range_var], [interval.left for interval in downsampled_Sv["depth_bins"].data]
        ),
        dims=["channel", "ping_time", "range_sample"],
    )

    # Initialize upsampled Sv
    upsampled_Sv = ds_Sv["Sv"].copy()

    # Iterate through all channels
    for channel_index in range(len(depth_bin_assignment["channel"])):
        # Iterate through all ping times
        for ping_time_index in range(len(depth_bin_assignment["ping_time"])):
            # Get unique range sample values along a single ping from the digitized depth array:
            # NOTE: The unique index corresponds to the first unique value's position which in
            # turn corresponds to the first range sample value contained in each depth bin.
            _, unique_range_sample_indices = np.unique(
                depth_bin_assignment.isel(channel=channel_index, ping_time=ping_time_index).data,
                return_index=True,
            )

            # Select a single ping downsampled Sv vector
            subset_downsampled_Sv = downsampled_Sv.isel(
                channel=channel_index, ping_time=ping_time_index
            )

            # Substitute depth bin coordinate in the downsampled Sv to be the range sample value
            # corresponding to the first element (lowest depth value) of each depth bin, and rename
            # `depth_bin` coordinate to `range_sample`.
            subset_downsampled_Sv = subset_downsampled_Sv.assign_coords(
                {"depth_bins": unique_range_sample_indices}
            ).rename({"depth_bins": "range_sample"})

            # Upsample via `reindex` `ffill`
            upsampled_Sv[dict(channel=channel_index, ping_time=ping_time_index)] = (
                subset_downsampled_Sv.reindex(
                    {"range_sample": ds_Sv["range_sample"]}, method="ffill"
                )
            )

    return downsampled_Sv, upsampled_Sv


def index_binning_downsample_upsample_along_depth(
    ds_Sv: xr.Dataset, depth_bin: float, range_var: str
) -> xr.DataArray:
    """
    Downsample and upsample Sv using index binning to mimic what was done in echopy
    impulse noise masking.

    This function makes the assumption that within each channel, the difference
    between depth values is uniform across all pings. Thus, computing the number of
    range sample indices needed to cover the depth bin is a channel-specific task.
    """
    # Drop `filenames` dimension if exists and transpose Dataset
    ds_Sv = ds_Sv.drop_dims("filenames", errors="ignore").transpose(
        "channel", "ping_time", "range_sample"
    )

    # Compute number of range sample indices that are needed to encapsulate the `depth_bin`
    # value per channel.
    all_chan_num_range_sample_indices = np.ceil(
        depth_bin / np.nanmean(np.diff(ds_Sv[range_var], axis=2), axis=(1, 2))
    ).astype(int)

    # Create list for upsampled Sv DataArrays
    upsampled_Sv_list = []

    # Iterate through channels
    for channel_index in range(len(ds_Sv["channel"])):
        # Grab channel-specific number of range sample indices for vertical binning
        chan_num_range_sample_indices = all_chan_num_range_sample_indices[channel_index]

        # Compute channel-specific coarsened Sv
        chan_coarsened_Sv = (
            ds_Sv["Sv"]
            .isel(channel=channel_index)
            .pipe(_log2lin)
            .coarsen(
                range_sample=chan_num_range_sample_indices,
                boundary="pad",
            )
            .mean(skipna=True)
            .pipe(_lin2log)
        )

        # Align coarsened Sv `range_sample` to the first index of each coarsened `range_sample` bin
        chan_coarsened_Sv = chan_coarsened_Sv.assign_coords(
            range_sample=chan_num_range_sample_indices
            * np.arange(len(chan_coarsened_Sv["range_sample"]))
        )

        # Upsample Sv using reindex
        chan_upsampled_Sv = chan_coarsened_Sv.reindex(
            {"range_sample": ds_Sv["range_sample"]}, method="ffill"
        )

        # Place channel-specific upsampled Sv into list
        upsampled_Sv_list.append(chan_upsampled_Sv)

    # Concatenate arrays along channel dimension
    upsampled_Sv = xr.concat(upsampled_Sv_list, dim="channel")

    return upsampled_Sv


def echopy_impulse_noise_mask(
    Sv: np.ndarray, num_side_pings: int, impulse_noise_threshold: float
) -> np.ndarray:
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
    Sv: np.ndarray,
    range_var: np.ndarray,
    upper_limit_sl: float,
    lower_limit_sl: float,
    num_side_pings: int,
    attenuation_signal_threshold: float,
) -> np.ndarray:
    """Single-channel attenuated signal mask computation from echopy."""
    # Initialize mask
    attenuated_mask = np.zeros(Sv.shape, dtype=bool)

    for ping_time_idx in range(Sv.shape[0]):

        # Find indices for upper and lower SL limits
        up = np.argmin(abs(range_var[ping_time_idx, :] - upper_limit_sl))
        lw = np.argmin(abs(range_var[ping_time_idx, :] - lower_limit_sl))

        # Mask when attenuation masking is feasible
        if not (
            (ping_time_idx - num_side_pings < 0)
            | (ping_time_idx + num_side_pings > Sv.shape[0] - 1)
            | np.all(np.isnan(Sv[ping_time_idx, up:lw]))
        ):
            # Compare ping and block medians, and mask ping if difference greater than
            # threshold.
            pingmedian = _lin2log(np.nanmedian(_log2lin(Sv[ping_time_idx, up:lw])))
            blockmedian = _lin2log(
                np.nanmedian(
                    _log2lin(
                        Sv[
                            (ping_time_idx - num_side_pings) : (ping_time_idx + num_side_pings),
                            up:lw,
                        ]
                    )
                )
            )
            if (pingmedian - blockmedian) < attenuation_signal_threshold:
                attenuated_mask[ping_time_idx, :] = True

    return attenuated_mask


def add_remove_background_noise_attrs(
    da: xr.DataArray,
    sv_type: str,
    ping_num: int,
    range_sample_num: int,
    SNR_threshold: float,
    noise_max: float,
) -> xr.DataArray:
    """Add attributes to a `remove_background_noise` data array."""
    da.attrs = {
        "long_name": f"Volume backscattering strength, {sv_type} (Sv re 1 m-1)",
        "units": "dB",
        "actual_range": [
            round(float(da.min().values), 2),
            round(float(da.max().values), 2),
        ],
        "noise_ping_num": ping_num,
        "noise_range_sample_num": range_sample_num,
        "SNR_threshold": SNR_threshold,
        "noise_max": noise_max,
    }
    return da
