# detect_basic.py
from __future__ import annotations

import numpy as np
import xarray as xr


def _check_bottom_inputs(ds: xr.Dataset, var_name: str, channel: int | str):
    """Validate dataset and select the reference channel for bottom detection."""
    if var_name not in ds:
        raise KeyError(f"{var_name!r} not found in dataset")
    if "depth" not in ds:
        raise KeyError("'depth' variable not found in dataset")
    if "channel" not in ds.coords:
        raise ValueError("Dataset must have 'channel' coordinate")

    Sv_all = ds[var_name]
    depth_all = ds["depth"]
    Sv_sel = Sv_all.sel(channel=channel)
    depth_sel = depth_all.sel(channel=channel)

    # Ensure uniform depth grid across pings
    depth_ref = depth_sel.isel(ping_time=0)
    is_uniform = (abs(depth_sel - depth_ref).max(dim="range_sample") < 1e-16).all()
    if not bool(is_uniform):
        raise ValueError("Depth grid varies across ping_time for the selected channel.")

    return Sv_sel, depth_sel, Sv_all, depth_all


def _validate_threshold(threshold):
    """Ensure threshold is a valid (tmin, tmax) tuple."""
    if isinstance(threshold, (int, float)):
        tmin, tmax = float(threshold), float(threshold) + 10.0
    else:
        tmin, tmax = map(float, threshold)
        if tmax <= tmin:
            raise ValueError("threshold upper bound must be > lower bound")
    return tmin, tmax


def _detect_bottom_on_selected_channel(Sv_sel, depth_sel, *, surface_skip, tmin, tmax, offset_m):
    """Detect bottom depth for each ping using threshold conditions."""
    depth_ref = depth_sel.isel(ping_time=0)
    Sv_deep = Sv_sel.isel(range_sample=slice(surface_skip, None))
    cond = (Sv_deep > tmin) & (Sv_deep < tmax)
    bottom_idx = cond.argmax("range_sample") + surface_skip

    # Map index → depth and apply offset
    bottom_depth = xr.apply_ufunc(
        lambda idx: depth_ref.values[int(idx)] if np.isfinite(idx) else np.nan,
        bottom_idx.astype(float),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ) - float(offset_m)

    return bottom_depth  # dims: ping_time


def _map_bottom_depth_to_rs_all_channels(
    bottom_depth: xr.DataArray, depth_all: xr.DataArray
) -> xr.DataArray:
    """
    Map bottom depth (ping_time) to nearest range_sample index for each channel.
    Assumes per-channel depth grid is time-invariant (uses ping_time=0).
    Returns: (channel, ping_time) float array of range_sample indices.
    """

    def nearest_rs(depth_ref_vec: np.ndarray, bd: float) -> float:
        if np.isnan(bd):
            return np.nan
        valid = np.where(~np.isnan(depth_ref_vec))[0]
        if valid.size == 0:
            return np.nan
        return float(valid[np.argmin(np.abs(depth_ref_vec[valid] - bd))])

    channels = [str(c) for c in depth_all["channel"].values]
    rs_idx_list = []

    for ch in channels:
        depth_ref_ch = depth_all.sel(channel=ch).isel(ping_time=0).values  # 1D by range_sample
        rs_idx_ch = (
            xr.apply_ufunc(
                lambda bd: nearest_rs(depth_ref_ch, bd),
                bottom_depth,  # (ping_time,)
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            .rename("bottom_rs")
            .expand_dims(channel=[ch])
            .assign_coords(channel=[ch])
        )
        rs_idx_list.append(rs_idx_ch)

    chan_dim = xr.DataArray(channels, dims="channel", name="channel")
    rs_idx_all = xr.concat(rs_idx_list, dim=chan_dim)  # (channel, ping_time)
    return rs_idx_all
