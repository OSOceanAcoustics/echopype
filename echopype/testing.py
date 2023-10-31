from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .utils.compute import _lin2log, _log2lin

HERE = Path(__file__).parent.absolute()
TEST_DATA_FOLDER = HERE / "test_data"

# Data length for each data type
_DATA_LEN = {
    "power": 1,
    "angle": 2,
    "complex": 4,  # assume 4 transducer sectors, can also be 3
}


def _gen_ping_time(ping_time_len, ping_time_interval, ping_time_jitter_max_ms=0):
    ping_time = pd.date_range("2018-07-01", periods=ping_time_len, freq=ping_time_interval)
    if ping_time_jitter_max_ms != 0:  # if to add jitter
        jitter = (
            np.random.randint(ping_time_jitter_max_ms, size=ping_time_len) / 1000
        )  # convert to seconds
        ping_time = pd.to_datetime(ping_time.astype(int) / 1e9 + jitter, unit="s")
    return ping_time


# Helper functions to generate mock Sv and MVBS dataset
def _get_expected_mvbs_val(
    ds_Sv: xr.Dataset, ping_time_bin: str, range_bin: float, channel_len: int = 2
) -> np.ndarray:
    """
    Helper functions to generate expected MVBS outputs from mock Sv dataset
    by brute-force looping and compute the mean

    Parameters
    ----------
    ds_Sv : xr.Dataset
        Mock Sv dataset
    ping_time_bin : str
        Ping time bin
    range_bin : float
        Range bin
    channel_len : int, default 2
        Number of channels
    """
    # create bin information needed for ping_time
    d_index = (
        ds_Sv["ping_time"]
        .resample(ping_time=ping_time_bin, skipna=True)
        .first()  # Not actually being used, but needed to get the bin groups
        .indexes["ping_time"]
    )
    ping_interval = d_index.union([d_index[-1] + pd.Timedelta(ping_time_bin)]).values

    # create bin information for echo_range
    # this computes the echo range max since there might NaNs in the data
    echo_range_max = ds_Sv["echo_range"].max()
    range_interval = np.arange(0, echo_range_max + 2, range_bin)

    sv = ds_Sv["Sv"].pipe(_log2lin)

    expected_mvbs_val = np.ones((2, len(ping_interval) - 1, len(range_interval) - 1)) * np.nan

    for ch_idx in range(channel_len):
        for p_idx in range(len(ping_interval) - 1):
            for r_idx in range(len(range_interval) - 1):
                echo_range = (
                    ds_Sv["echo_range"]
                    .isel(channel=ch_idx)
                    .sel(ping_time=slice(ping_interval[p_idx], ping_interval[p_idx + 1]))
                )
                r_idx_active = np.logical_and(
                    echo_range.data >= range_interval[r_idx],
                    echo_range.data < range_interval[r_idx + 1],
                )
                sv_tmp = (
                    sv.isel(channel=ch_idx)
                    .sel(ping_time=slice(ping_interval[p_idx], ping_interval[p_idx + 1]))
                    .data[r_idx_active]
                )
                if 0 in sv_tmp.shape:
                    expected_mvbs_val[ch_idx, p_idx, r_idx] = np.nan
                else:
                    expected_mvbs_val[ch_idx, p_idx, r_idx] = np.mean(sv_tmp)
    return _lin2log(expected_mvbs_val)


def _gen_Sv_echo_range_regular(
    channel_len=2,
    depth_len=100,
    depth_interval=0.5,
    ping_time_len=600,
    ping_time_interval="0.3S",
    ping_time_jitter_max_ms=0,
    random_number_generator=None,
):
    """
    Generate a Sv dataset with uniform echo_range across all ping_time.

    ping_time_jitter_max_ms controlled jitter in milliseconds in ping_time.

    Parameters
    ------------
    channel_len
        number of channels
    depth_len
        number of total depth bins
    depth_interval
        depth intervals, may have multiple values
    ping_time_len
        total number of ping_time
    ping_time_interval
        interval between pings
    ping_time_jitter_max_ms
        jitter of ping_time in milliseconds
    """

    if random_number_generator is None:
        random_number_generator = np.random.default_rng()

    # regular echo_range
    echo_range = np.array([[np.arange(depth_len)] * ping_time_len] * channel_len) * depth_interval

    # generate dataset
    ds_Sv = xr.Dataset(
        data_vars={
            "Sv": (
                ["channel", "ping_time", "range_sample"],
                random_number_generator.random(size=(channel_len, ping_time_len, depth_len)),
            ),
            "echo_range": (["channel", "ping_time", "range_sample"], echo_range),
            "frequency_nominal": (["channel"], np.arange(channel_len)),
        },
        coords={
            "channel": [f"ch_{ch}" for ch in range(channel_len)],
            "ping_time": _gen_ping_time(ping_time_len, ping_time_interval, ping_time_jitter_max_ms),
            "range_sample": np.arange(depth_len),
        },
    )

    return ds_Sv


def _gen_Sv_echo_range_irregular(
    channel_len=2,
    depth_len=100,
    depth_interval=[0.5, 0.32, 0.13],
    depth_ping_time_len=[100, 300, 200],
    ping_time_len=600,
    ping_time_interval="0.3S",
    ping_time_jitter_max_ms=0,
    random_number_generator=None,
):
    """
    Generate a Sv dataset with uniform echo_range across all ping_time.

    ping_time_jitter_max_ms controlled jitter in milliseconds in ping_time.

    Parameters
    ------------
    channel_len
        number of channels
    depth_len
        number of total depth bins
    depth_interval
        depth intervals, may have multiple values
    depth_ping_time_len
        the number of pings to use each of the depth_interval
        for example, with depth_interval=[0.5, 0.32, 0.13]
        and depth_ping_time_len=[100, 300, 200],
        the first 100 pings have echo_range with depth intervals of 0.5 m,
        the next 300 pings have echo_range with depth intervals of 0.32 m,
        and the last 200 pings have echo_range with depth intervals of 0.13 m.
    ping_time_len
        total number of ping_time
    ping_time_interval
        interval between pings
    ping_time_jitter_max_ms
        jitter of ping_time in milliseconds
    """
    if random_number_generator is None:
        random_number_generator = np.random.default_rng()

    # check input
    if len(depth_interval) != len(depth_ping_time_len):
        raise ValueError("The number of depth_interval and depth_ping_time_len must be equal!")

    if ping_time_len != np.array(depth_ping_time_len).sum():
        raise ValueError("The number of total pings does not match!")

    # irregular echo_range
    echo_range_list = []
    for d, dp in zip(depth_interval, depth_ping_time_len):
        echo_range_list.append(np.array([[np.arange(depth_len)] * dp] * channel_len) * d)
    echo_range = np.hstack(echo_range_list)

    # generate dataset
    ds_Sv = xr.Dataset(
        data_vars={
            "Sv": (
                ["channel", "ping_time", "range_sample"],
                random_number_generator.random(size=(channel_len, ping_time_len, depth_len)),
            ),
            "echo_range": (["channel", "ping_time", "range_sample"], echo_range),
            "frequency_nominal": (["channel"], np.arange(channel_len)),
        },
        coords={
            "channel": [f"ch_{ch}" for ch in range(channel_len)],
            "ping_time": _gen_ping_time(ping_time_len, ping_time_interval, ping_time_jitter_max_ms),
            "range_sample": np.arange(depth_len),
        },
    )

    return ds_Sv


# End helper functions for mock Sv and MVBS


# Helper functions to generate ping data dict
def _gen_channel_data(
    range_sample_len=[100, 500, 10000],
    range_sample_ping_time_len=[20, 30, 10],
    data_type="power",
    has_angle=True,
) -> np.ndarray:
    """
    Create data for one channel with variable length
    along the range_sample dimension.

    To generate channel data with uniform length across ping_time,
    set range_sample_len and range_sample_ping_time_len both to single-element lists.

    To generate channel data with variable length across ping_time,
    set range_sample_len and range_sample_ping_time_len as lists similar to the default.

    Parameters
    ----------
    range_sample_len
        length along the range_sample dimension for each block of pings
    range_sample_ping_time_len
        number of pings in each block
    data_type
        whether the generated channel data is mimicking the
        power, angle, or complex data generated from EK60 and EK80.

    Returns
    -------
    A numpy array containing mock data for one channel.
    """
    ch_data = []
    for rs_len, pt_len in zip(range_sample_len, range_sample_ping_time_len):
        # Generate data for each ping
        for pt in np.arange(pt_len):  # looping since this needs to be a list of np arrays
            if (data_type != "angle") or (has_angle is True):
                if data_type == "complex":
                    rand_samples = (
                        (
                            np.random.randn(rs_len, _DATA_LEN[data_type])
                            + 1j * np.random.randn(rs_len, _DATA_LEN[data_type])
                        )
                        .reshape(-1, 1)
                        .squeeze()
                    )
                else:
                    rand_samples = np.random.randint(
                        0, 10000, size=(rs_len, _DATA_LEN[data_type])
                    ).squeeze()
                ch_data.append(rand_samples)
            else:
                ch_data.append(None)

    return ch_data


def _gen_timestamp_data(ch_name, ch_range_sample_ping_time_len, ping_time_jitter_max_ms=0):
    timestamp_data = defaultdict(list)
    for ch_seq, ch in enumerate(ch_name):
        mock_time = _gen_ping_time(
            ping_time_len=sum(ch_range_sample_ping_time_len[ch_seq]),
            ping_time_interval="1S",
            ping_time_jitter_max_ms=ping_time_jitter_max_ms,
        )
        timestamp_data[ch] = [np.datetime64(t) for t in mock_time.tolist()]
    return timestamp_data


def _gen_echo_data(
    ch_name, ch_range_sample_len, ch_range_sample_ping_time_len, data_type, has_angle
):
    echo_data = defaultdict(list)
    for ch_seq, ch in enumerate(ch_name):
        echo_data[ch] = _gen_channel_data(
            range_sample_len=ch_range_sample_len[ch_seq],
            range_sample_ping_time_len=ch_range_sample_ping_time_len[ch_seq],
            data_type=data_type,
            has_angle=has_angle[ch_seq],
        )
    return echo_data


def _gen_ping_data_dict_power_angle(
    ch_name=[1, 2, 3],
    ch_range_sample_len=[[100], [100], [100]],
    ch_range_sample_ping_time_len=[[20], [20], [20]],
    has_angle=[True, True, True],
    ping_time_jitter_max_ms=0,
):
    """
    Mock parser.ping_data_dict for EK60/EK80 power-angle data.

    To generate regular data:
        # all pings in each channel have length=100 along the range_sample dimension
        ch_range_sample_len=[[100], [100], [100]]

        # all channels have 20 pings
        ch_range_sample_ping_time_len=[[20], [20], [20]]

    To generate irregular data:
        # the length along range_sample changes across ping_time in different ways for each channel
        ch_range_sample_len=[[10, 20, 100], [130], [20, 100, 10]]

        # the number of pings in each block (each block has different length along range_sample)
        # is different for each channel
        ch_range_sample_ping_time_len=[[20, 100, 20], [120, 10, 5], [50, 20, 20]]

    To generate data with a subset channels containing no angle data:
        # set has_angle of the channel without angle data to False
        has_angle=[True, False, True]

    If ping_time_jitter_max_ms!=0 in gen_timestamp_data(),
    each ping_time will be different by some small jitter across all channels,
    i.e., the ping_time across will NOT be aligned.
    """

    if (len(ch_name) != len(ch_range_sample_len)) or (
        len(ch_name) != len(ch_range_sample_ping_time_len)
    ):
        raise ValueError("Channel length mismatches!")

    ping_data_dict = defaultdict(lambda: defaultdict(list))

    # Echo data (power, angle, complex) generation
    ping_data_dict["power"] = _gen_echo_data(
        ch_name,
        ch_range_sample_len,
        ch_range_sample_ping_time_len,
        data_type="power",
        has_angle=has_angle,
    )
    ping_data_dict["angle"] = _gen_echo_data(
        ch_name,
        ch_range_sample_len,
        ch_range_sample_ping_time_len,
        data_type="angle",
        has_angle=has_angle,
    )

    # Ping time generation
    ping_data_dict["timestamp"] = _gen_timestamp_data(
        ch_name, ch_range_sample_ping_time_len, ping_time_jitter_max_ms
    )

    return ping_data_dict


def _gen_ping_data_dict_complex(
    ch_name=["WBT_1", "WBT_2", "WBT_4"],
    ch_range_sample_len=[[100], [100], [100]],
    ch_range_sample_ping_time_len=[[20], [20], [20]],
    has_angle=[False, False, False],
    ping_time_jitter_max_ms=0,
):
    """
    Mock parser.ping_data_dict for EK80 complex data.

    There is no angle data with complex samples.

    To generate regular data:
        # all pings in each channel have length=100 along the range_sample dimension
        ch_range_sample_len=[[100], [100], [100]]

        # all channels have 20 pings
        ch_range_sample_ping_time_len=[[20], [20], [20]]

    To generate irregular data:
        # the length along range_sample changes across ping_time in different ways for each channel
        ch_range_sample_len=[[10, 20, 100], [130], [20, 100, 10]]

        # the number of pings in each block (each block has different length along range_sample)
        # is different for each channel
        ch_range_sample_ping_time_len=[[20, 100, 20], [120, 10, 5], [50, 20, 20]]

    If ping_time_jitter_max_ms!=0 in gen_timestamp_data(),
    each ping_time will be different by some small jitter across all channels,
    i.e., the ping_time across will NOT be aligned.
    """

    if (len(ch_name) != len(ch_range_sample_len)) or (
        len(ch_name) != len(ch_range_sample_ping_time_len)
    ):
        raise ValueError("Channel length mismatches!")

    ping_data_dict = defaultdict(lambda: defaultdict(list))

    # Echo data (power, angle, complex) generation
    ping_data_dict["complex"] = _gen_echo_data(
        ch_name,
        ch_range_sample_len,
        ch_range_sample_ping_time_len,
        data_type="complex",
        has_angle=has_angle,
    )

    # Ping time generation
    ping_data_dict["timestamp"] = _gen_timestamp_data(
        ch_name, ch_range_sample_ping_time_len, ping_time_jitter_max_ms
    )

    return ping_data_dict


# End helper functions for ping data dict
