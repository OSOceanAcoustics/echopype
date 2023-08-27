"""
An overhaul is required for the below compute_NASC implementation, to:
- increase the computational efficiency
- debug the current code of any discrepancy against Echoview implementation
- potentially provide an alternative based on ping-by-ping Sv

This script contains functions used by commongrid.compute_NASC,
but a subset of these overlap with operations needed
for commongrid.compute_MVBS and clean.estimate_noise.
The compute_MVBS and remove_noise code needs to be refactored,
and the compute_NASC needs to be optimized.
The plan is to create a common util set of functions for use in
these functions in an upcoming release.
"""

import numpy as np
import xarray as xr
from geopy import distance


def check_identical_depth(ds_ch):
    """
    Check if all pings have the same depth vector.
    """
    # Depth vector are identical for all pings, if:
    #   - the number of non-NaN range_sample is the same for all pings, AND
    #   - all pings have the same max range
    num_nan = np.isnan(ds_ch.values).sum(axis=1)
    nan_check = True if np.all(num_nan == 0) or np.unique(num_nan).size == 1 else False

    if not nan_check:
        return xr.DataArray(False, coords={"channel": ds_ch["channel"]})
    else:
        # max range of each ping should be identical
        max_range_ping = ds_ch.values[np.arange(ds_ch.shape[0]), ds_ch.shape[1] - num_nan - 1]
        if np.unique(max_range_ping).size == 1:
            return xr.DataArray(True, coords={"channel": ds_ch["channel"]})
        else:
            return xr.DataArray(False, coords={"channel": ds_ch["channel"]})


def get_depth_bin_info(ds_Sv, cell_depth):
    """
    Find binning indices along depth
    """
    depth_ping1 = ds_Sv["depth"].isel(ping_time=0)
    num_nan = np.isnan(depth_ping1.values).sum(axis=1)
    # ping 1 max range of each channel
    max_range_ch = depth_ping1.values[
        np.arange(depth_ping1.shape[0]), depth_ping1.shape[1] - num_nan - 1
    ]
    bin_num_depth = np.ceil(max_range_ch.max() / cell_depth)  # use max range of all channel
    depth_bin_idx = [
        np.digitize(dp1, np.arange(bin_num_depth + 1) * cell_depth, right=False)
        for dp1 in depth_ping1
    ]
    return bin_num_depth, depth_bin_idx


def get_distance_from_latlon(ds_Sv):
    # Get distance from lat/lon in nautical miles
    df_pos = ds_Sv["latitude"].to_dataframe().join(ds_Sv["longitude"].to_dataframe())
    df_pos["latitude_prev"] = df_pos["latitude"].shift(-1)
    df_pos["longitude_prev"] = df_pos["longitude"].shift(-1)
    df_latlon_nonan = df_pos.dropna().copy()
    df_latlon_nonan["dist"] = df_latlon_nonan.apply(
        lambda x: distance.distance(
            (x["latitude"], x["longitude"]),
            (x["latitude_prev"], x["longitude_prev"]),
        ).nm,
        axis=1,
    )
    df_pos = df_pos.join(df_latlon_nonan["dist"], how="left")
    df_pos["dist"] = df_pos["dist"].cumsum()
    df_pos["dist"] = df_pos["dist"].fillna(method="ffill").fillna(method="bfill")

    return df_pos["dist"].values


def get_dist_bin_info(dist_nmi, cell_dist):
    bin_num_dist = np.ceil(dist_nmi.max() / cell_dist)
    if np.mod(dist_nmi.max(), cell_dist) == 0:
        # increment bin num if last element coincides with bin edge
        bin_num_dist = bin_num_dist + 1
    dist_bin_idx = np.digitize(dist_nmi, np.arange(bin_num_dist + 1) * cell_dist, right=False)
    return bin_num_dist, dist_bin_idx
