import flox.xarray
import numpy as np
import xarray as xr

from ..commongrid.utils import _convert_bins_to_interval_index
from ..utils.compute import _lin2log, _log2lin


def calc_transient_noise_pooled_Sv(
    ds_Sv: xr.Dataset, func: str, depth_bin: float, num_side_pings: int, exclude_above: float
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
    depth_values_min = ds_Sv["depth"].min()
    depth_values_max = ds_Sv["depth"].max()
    ping_time_index_min = 0
    ping_time_index_max = len(ds_Sv["ping_time"])

    # Iterate through the channel dimension
    for channel_index in range(len(ds_Sv["channel"])):

        # Set channel arrays
        chan_Sv = ds_Sv["Sv"].isel(channel=channel_index)
        chan_depth = ds_Sv["depth"].isel(channel=channel_index)

        # Iterate through the range sample dimension
        for range_sample_index in range(len(ds_Sv["range_sample"])):

            # Iterate through the ping time dimension
            for ping_time_index in range(len(ds_Sv["ping_time"])):

                # Grab current depth
                current_depth = ds_Sv["depth"].isel(
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
                    aggregate_window_Sv_value = window_Sv.pipe(
                        np.nanmean if func == "nanmean" else np.nanmedian
                    )
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


def downsample_upsample_along_depth(ds_Sv: xr.Dataset, depth_bin: float) -> xr.DataArray:
    """
    Downsample and upsample Sv to mimic what was done in echopy impulse
    noise masking.
    """
    # Validate and compute range interval
    depth_min = ds_Sv["depth"].min()
    depth_max = ds_Sv["depth"].max()
    range_interval = np.arange(depth_min, depth_max + depth_bin, depth_bin)
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

    # Assign a depth bin index to each Sv depth value
    depth_bin_assignment = xr.DataArray(
        np.digitize(
            ds_Sv["depth"], [interval.left for interval in downsampled_Sv["depth_bins"].data]
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
    depth: np.ndarray,
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
        up = np.argmin(abs(depth[ping_time_idx, :] - upper_limit_sl))
        lw = np.argmin(abs(depth[ping_time_idx, :] - lower_limit_sl))

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
