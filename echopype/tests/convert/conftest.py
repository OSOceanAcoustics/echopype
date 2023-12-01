import pytest

from echopype.testing import _gen_ping_data_dict_power_angle, _gen_ping_data_dict_complex


@pytest.fixture
def irregular_ch_range_sample_len():
    return [[10, 20, 100], [130], [20, 100, 10]]


@pytest.fixture
def irregular_ch_range_sample_ping_time_len():
    return [[20, 100, 20], [120, 10, 5], [50, 20, 20]]


@pytest.fixture
def ping_time_jitter_max_ms():
    return 30


@pytest.fixture
def mock_ping_data_dict_power_angle_simple():
    """Simple Mock ping data dict power angle"""
    return _gen_ping_data_dict_power_angle()


@pytest.fixture
def mock_ping_data_dict_power_angle_jitter(
    ping_time_jitter_max_ms,
):
    """Mock ping data dict power angle with ping time jitter"""
    return _gen_ping_data_dict_power_angle(ping_time_jitter_max_ms=ping_time_jitter_max_ms)


@pytest.fixture(params=["regular", "irregular"])
def mock_ping_data_dict_power_angle(
    request, irregular_ch_range_sample_len, irregular_ch_range_sample_ping_time_len
):
    if request.param == "regular":
        return request.param, _gen_ping_data_dict_power_angle()
    elif request.param == "irregular":
        return request.param, _gen_ping_data_dict_power_angle(
            ch_range_sample_len=irregular_ch_range_sample_len,
            ch_range_sample_ping_time_len=irregular_ch_range_sample_ping_time_len,
            has_angle=[True, False, True],
        )


@pytest.fixture(params=["regular", "irregular"])
def mock_ping_data_dict_complex(
    request, irregular_ch_range_sample_len, irregular_ch_range_sample_ping_time_len
):
    if request.param == "regular":
        return request.param, _gen_ping_data_dict_complex()
    elif request.param == "irregular":
        return request.param, _gen_ping_data_dict_complex(
            ch_range_sample_len=irregular_ch_range_sample_len,
            ch_range_sample_ping_time_len=irregular_ch_range_sample_ping_time_len,
        )
