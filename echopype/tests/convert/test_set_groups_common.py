from typing import Dict, List, Any

import xarray as xr
import numpy as np


def test_backscatter_concat_jitter_ping_time(mock_ping_data_dict_power_angle_jitter):
    """
    Test parser and set groups for EK60 data with ping time jitter
    for ensuring that ping times are properly merged together
    with xr.concat across channels.
    """
    ping_data_dict: Dict[str, Any] = mock_ping_data_dict_power_angle_jitter
    ping_times: Dict[int, List[np.ndarray]] = ping_data_dict["timestamp"]

    # Go through each power data similarly to "backscatter_r"
    # in `set_beam` method for both `set_groups_ek60` and `set_groups_ek80`
    ds_backscatter = []
    for ch, arr_list in ping_data_dict["power"].items():
        data = np.array(arr_list)
        var_dict = {}
        var_dict["backscatter_r"] = (
            ["ping_time", "range_sample"],
            data,
        )

        ds_tmp = xr.Dataset(
            var_dict,
            coords={
                "ping_time": (["ping_time"], ping_times[ch]),
                "range_sample": (["range_sample"], np.arange(data.shape[1])),
            },
        )
        ds_tmp = ds_tmp.expand_dims({"channel": [ch]})
        ds_backscatter.append(ds_tmp)

    # Perform the concat across channels
    merged_ds = xr.concat(ds_backscatter, dim="channel")

    # Check that the ping times are properly merged together
    # and that values didn't change
    xr_ping_times = merged_ds["ping_time"].to_numpy()

    # Create manual concatenated ping times with just numpy.
    # This merges all the ping time arrays together, get the unique values,
    # and then sort them.
    manual_ping_times = np.sort(
        np.unique(np.concatenate([np.array(pts) for pts in ping_times.values()]))
    )

    # Check for ping time shape and value equality
    assert manual_ping_times.shape == xr_ping_times.shape
    assert np.array_equal(manual_ping_times, xr_ping_times)

    # Iterate over each channel and check for
    # original values equivalency
    for ch in merged_ds["channel"].to_numpy():
        # This filters the NaNs from the merged dataset
        # for the given channel, so this should result
        # in the original data
        da = merged_ds["backscatter_r"].sel(channel=ch).dropna(dim="ping_time")

        # Get the original data for the given channel
        orig_data = np.array(ping_data_dict["power"][ch])

        # Check equivalent values
        assert np.array_equal(orig_data, da.to_numpy())

        # Check equivalent ping times
        assert np.array_equal(da["ping_time"].to_numpy(), np.array(ping_times[ch]))
