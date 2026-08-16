from typing import Dict, List, Any
from types import SimpleNamespace

import xarray as xr
import numpy as np
import pytest

from echopype.convert.set_groups_base import SetGroupsBase

pytestmark = pytest.mark.unit


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
    merged_ds = xr.concat(ds_backscatter, dim="channel", join="outer")

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


# Regression tests for _nan_timestamp_handler on partial/truncated raw files.
# See PR #1624.


def _call_nan_timestamp_handler(sonar_model, parser_obj, time_val):
    """Invoke the unbound method to avoid instantiating the abstract SetGroupsBase."""
    fake_self = SimpleNamespace(sonar_model=sonar_model, parser_obj=parser_obj)
    return SetGroupsBase._nan_timestamp_handler(fake_self, time_val)


@pytest.mark.parametrize("sonar_model", ["EK60", "ES70", "EK80", "ES80", "EA640"])
def test_nan_timestamp_handler_ek_empty_ping_time_dict(sonar_model):
    """No channels at all -> [nan] instead of crashing on np.array([]).min()."""
    parser_obj = SimpleNamespace(ping_time={})

    result = _call_nan_timestamp_handler(sonar_model, parser_obj, [np.nan])

    assert len(result) == 1
    assert np.isnan(result[0])


@pytest.mark.parametrize("sonar_model", ["EK60", "EK80"])
def test_nan_timestamp_handler_ek_all_channels_empty(sonar_model):
    """All channels present but each has an empty ping_time array -> [nan]."""
    parser_obj = SimpleNamespace(
        ping_time={
            "ch1": [np.array([], dtype="datetime64[ns]")],
            "ch2": [np.array([], dtype="datetime64[ns]")],
        }
    )

    result = _call_nan_timestamp_handler(sonar_model, parser_obj, [np.nan])

    assert len(result) == 1
    assert np.isnan(result[0])


def test_nan_timestamp_handler_ek_returns_earliest_when_populated():
    """Sanity check: non-empty ping times still return the earliest one."""
    t_ch1 = np.array(["2024-01-01T00:00:05"], dtype="datetime64[ns]")
    t_ch2 = np.array(["2024-01-01T00:00:02"], dtype="datetime64[ns]")
    parser_obj = SimpleNamespace(ping_time={"ch1": [t_ch1], "ch2": [t_ch2]})

    result = _call_nan_timestamp_handler("EK60", parser_obj, [np.nan])

    assert len(result) == 1
    assert result[0] == np.datetime64("2024-01-01T00:00:02", "ns")


def test_nan_timestamp_handler_passthrough_when_time_val_valid():
    """Non-NaN input should be returned unchanged without touching parser_obj."""
    time_val = [np.datetime64("2024-01-01T00:00:00", "ns")]
    parser_obj = SimpleNamespace(ping_time={})

    result = _call_nan_timestamp_handler("EK60", parser_obj, time_val)

    assert result is time_val
