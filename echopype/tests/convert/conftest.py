import pytest

from collections import defaultdict

import numpy as np
import pandas as pd


DATA_LEN = {
    "power": 1,
    "angle": 2,
    "complex": 4,
}


@pytest.fixture
def mock_channel(
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
                ch_data.append(np.random.randint(0, 10000, size=(rs_len, DATA_LEN[data_type])).squeeze())
            else:
                ch_data.append(None)

    return ch_data


@pytest.fixture
def mock_channel_timestamp(ping_time_len, ping_time_interval="1S", ping_time_jitter_max_ms=0):
    # TODO: this is the same function as in tests/commongrid/conftest.py::_gen_ping_time
    #       consider moving this to consolidate
    ping_time = pd.date_range("2018-07-01", periods=ping_time_len, freq=ping_time_interval)
    if ping_time_jitter_max_ms != 0:  # if to add jitter
        jitter = (
            np.random.randint(ping_time_jitter_max_ms, size=ping_time_len) / 1000
        )  # convert to seconds
        ping_time = pd.to_datetime(ping_time.astype(int) / 1e9 + jitter, unit="s")
    return ping_time


@pytest.fixture
def gen_timestamp_data(ch_num, ch_range_sample_ping_time_len, ping_time_jitter_max_ms=0):
    timestamp_data = defaultdict(list)
    for ch_seq, ch in enumerate(ch_num):
        mock_time = mock_channel_timestamp(
            ping_time_len=sum(ch_range_sample_ping_time_len[ch_seq]),
            ping_time_interval="1S",
            ping_time_jitter_max_ms=ping_time_jitter_max_ms,
        )
        timestamp_data[ch] = [np.datetime64(t) for t in mock_time.tolist()]
    return timestamp_data


@pytest.fixture
def gen_echo_data(ch_num, ch_range_sample_len, ch_range_sample_ping_time_len, data_type, has_angle):
    echo_data = defaultdict(list)
    for ch_seq, ch in enumerate(ch_num):
        echo_data[ch] = mock_channel(
            range_sample_len=ch_range_sample_len[ch_seq],
            range_sample_ping_time_len=ch_range_sample_ping_time_len[ch_seq],
            data_type=data_type,
            has_angle=has_angle[ch_seq],
        )
    return echo_data


@pytest.fixture
def mock_ping_data_dict(
    ch_num=[1, 2, 3],
    ch_range_sample_len=[[100], [100], [100]],
    ch_range_sample_ping_time_len=[[20], [20], [20]],
    has_angle=[True, True, True],
):
    """
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

    if (len(ch_num) != len(ch_range_sample_len)) or (len(ch_num) != len(ch_range_sample_ping_time_len)):
        raise ValueError("Channel length mismatches!")

    ping_data_dict = defaultdict(list)

    # Echo data (power, angle, complex) generation
    ping_data_dict["power"] = gen_echo_data(
        ch_num, ch_range_sample_len, ch_range_sample_ping_time_len, data_type="power", has_angle=has_angle
    )
    ping_data_dict["angle"] = gen_echo_data(
        ch_num, ch_range_sample_len, ch_range_sample_ping_time_len, data_type="angle", has_angle=has_angle
    )
    
    # Ping time generation
    ping_data_dict["timestamp"] = gen_timestamp_data(ch_num, ch_range_sample_ping_time_len)

    return ping_data_dict
