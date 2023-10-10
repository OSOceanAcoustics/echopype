import pytest

import numpy as np

from echopype.consolidate import add_depth
from echopype.testing import (
    _gen_Sv_echo_range_regular,
    _gen_Sv_echo_range_irregular,
    _get_expected_mvbs_val,
)


@pytest.fixture
def random_number_generator():
    """Random number generator for tests"""
    return np.random.default_rng()


@pytest.fixture
def mock_nan_ilocs():
    """NaN i locations for irregular Sv dataset

    It's a list of tuples, each tuple contains
    (channel, ping_time, range_sample)

    Notes
    -----
    This was created with the following code:

    ```
    import numpy as np

    random_positions = []
    for i in range(20):
        random_positions.append((
            np.random.randint(0, 2),
            np.random.randint(0, 5),
            np.random.randint(0, 20))
        )
    ```
    """
    return [
        (1, 1, 10),
        (1, 0, 16),
        (0, 3, 6),
        (0, 2, 11),
        (0, 2, 6),
        (1, 1, 14),
        (0, 1, 17),
        (1, 4, 19),
        (0, 3, 3),
        (0, 0, 19),
        (0, 1, 5),
        (1, 2, 9),
        (1, 4, 18),
        (0, 1, 5),
        (0, 4, 4),
        (0, 1, 6),
        (1, 2, 2),
        (0, 1, 2),
        (0, 4, 8),
        (0, 1, 1),
    ]


@pytest.fixture
def mock_parameters():
    """Small mock parameters"""
    return {
        "channel_len": 2,
        "ping_time_len": 10,
        "depth_len": 20,
        "ping_time_interval": "0.3S",
    }


@pytest.fixture
def mock_Sv_sample(mock_parameters):
    """
    Mock Sv sample

    Dimension: (2, 10, 20)
    """
    channel_len = mock_parameters["channel_len"]
    ping_time_len = mock_parameters["ping_time_len"]
    depth_len = mock_parameters["depth_len"]

    depth_data = np.linspace(0, 1, num=depth_len)
    return np.tile(depth_data, (channel_len, ping_time_len, 1))


@pytest.fixture
def mock_Sv_dataset_regular(mock_parameters, mock_Sv_sample):
    ds_Sv = _gen_Sv_echo_range_regular(**mock_parameters, ping_time_jitter_max_ms=0)
    ds_Sv["Sv"].data = mock_Sv_sample
    return ds_Sv


@pytest.fixture
def mock_Sv_dataset_irregular(mock_parameters, mock_Sv_sample, mock_nan_ilocs):
    depth_interval = [0.5, 0.32, 0.2]
    depth_ping_time_len = [2, 3, 5]
    ds_Sv = _gen_Sv_echo_range_irregular(
        **mock_parameters,
        depth_interval=depth_interval,
        depth_ping_time_len=depth_ping_time_len,
        ping_time_jitter_max_ms=30,  # Added jitter to ping_time
    )
    ds_Sv["Sv"].data = mock_Sv_sample
    # Sprinkle nans around echo_range
    for pos in mock_nan_ilocs:
        ds_Sv["echo_range"][pos] = np.nan
    return ds_Sv


@pytest.fixture
def mock_mvbs_inputs():
    return dict(range_meter_bin=2, ping_time_bin="1s")


@pytest.fixture
def mock_mvbs_array_regular(mock_Sv_dataset_regular, mock_mvbs_inputs, mock_parameters):
    """
    Mock Sv sample result from compute_MVBS

    Dimension: (2, 3, 5)
    Ping time bin: 1s
    Range bin: 2m
    """
    ds_Sv = mock_Sv_dataset_regular
    ping_time_bin = mock_mvbs_inputs["ping_time_bin"]
    range_bin = mock_mvbs_inputs["range_meter_bin"]
    channel_len = mock_parameters["channel_len"]
    expected_mvbs_val = _get_expected_mvbs_val(ds_Sv, ping_time_bin, range_bin, channel_len)

    return expected_mvbs_val


@pytest.fixture
def mock_mvbs_array_irregular(mock_Sv_dataset_irregular, mock_mvbs_inputs, mock_parameters):
    """
    Mock Sv sample irregular result from compute_MVBS

    Dimension: (2, 3, 5)
    Ping time bin: 1s
    Range bin: 2m
    """
    ds_Sv = mock_Sv_dataset_irregular
    ping_time_bin = mock_mvbs_inputs["ping_time_bin"]
    range_bin = mock_mvbs_inputs["range_meter_bin"]
    channel_len = mock_parameters["channel_len"]
    expected_mvbs_val = _get_expected_mvbs_val(ds_Sv, ping_time_bin, range_bin, channel_len)

    return expected_mvbs_val


@pytest.fixture(
    params=[
        (
            ("EK60", "ncei-wcsd", "Summer2017-D20170719-T211347.raw"),
            "EK60",
            None,
            {},
        ),
        (
            ("EK80_NEW", "echopype-test-D20211004-T235930.raw"),
            "EK80",
            None,
            {"waveform_mode": "BB", "encode_mode": "complex"},
        ),
        (
            ("EK80_NEW", "D20211004-T233354.raw"),
            "EK80",
            None,
            {"waveform_mode": "CW", "encode_mode": "power"},
        ),
        (
            ("EK80_NEW", "D20211004-T233115.raw"),
            "EK80",
            None,
            {"waveform_mode": "CW", "encode_mode": "complex"},
        ),
        (("ES70", "D20151202-T020259.raw"), "ES70", None, {}),
        (("AZFP", "17082117.01A"), "AZFP", ("AZFP", "17041823.XML"), {}),
        (
            ("AD2CP", "raw", "090", "rawtest.090.00001.ad2cp"),
            "AD2CP",
            None,
            {},
        ),
    ],
    ids=[
        "ek60_cw_power",
        "ek80_bb_complex",
        "ek80_cw_power",
        "ek80_cw_complex",
        "es70",
        "azfp",
        "ad2cp",
    ],
)
def test_data_samples(request, test_path):
    (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    ) = request.param
    if sonar_model.lower() in ["es70", "ad2cp"]:
        pytest.xfail(
            reason="Not supported at the moment",
        )
    path_model, *paths = filepath
    filepath = test_path[path_model].joinpath(*paths)

    if azfp_xml_path is not None:
        path_model, *paths = azfp_xml_path
        azfp_xml_path = test_path[path_model].joinpath(*paths)
    return (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    )


@pytest.fixture
def regular_data_params():
    return {
        "channel_len": 4,
        "depth_len": 4000,
        "ping_time_len": 100,
        "ping_time_jitter_max_ms": 0,
    }


@pytest.fixture
def ds_Sv_echo_range_regular(regular_data_params, random_number_generator):
    return _gen_Sv_echo_range_regular(
        **regular_data_params,
        random_number_generator=random_number_generator,
    )


@pytest.fixture
def latlon_history_attr():
    return (
        "2023-08-31 12:00:00.000000 +00:00. "
        "Interpolated or propagated from Platform latitude/longitude."  # noqa
    )


@pytest.fixture
def lat_attrs(latlon_history_attr):
    """Latitude attributes"""
    return {
        "long_name": "Platform latitude",
        "standard_name": "latitude",
        "units": "degrees_north",
        "valid_range": "(-90.0, 90.0)",
        "history": latlon_history_attr,
    }


@pytest.fixture
def lon_attrs(latlon_history_attr):
    """Longitude attributes"""
    return {
        "long_name": "Platform longitude",
        "standard_name": "longitude",
        "units": "degrees_east",
        "valid_range": "(-180.0, 180.0)",
        "history": latlon_history_attr,
    }


@pytest.fixture
def depth_offset():
    """Depth offset for calculating depth"""
    return 2.5


@pytest.fixture
def ds_Sv_echo_range_regular_w_latlon(ds_Sv_echo_range_regular, lat_attrs, lon_attrs):
    """Sv dataset with latitude and longitude"""
    n_pings = ds_Sv_echo_range_regular.ping_time.shape[0]
    latitude = np.linspace(42, 43, num=n_pings)
    longitude = np.linspace(-124, -125, num=n_pings)

    ds_Sv_echo_range_regular["latitude"] = (["ping_time"], latitude, lat_attrs)
    ds_Sv_echo_range_regular["longitude"] = (["ping_time"], longitude, lon_attrs)

    # Need processing level code for compute MVBS to work!
    ds_Sv_echo_range_regular.attrs["processing_level"] = "Level 2A"
    return ds_Sv_echo_range_regular


@pytest.fixture
def ds_Sv_echo_range_regular_w_depth(ds_Sv_echo_range_regular, depth_offset):
    """Sv dataset with depth"""
    return ds_Sv_echo_range_regular.pipe(add_depth, depth_offset=depth_offset)


@pytest.fixture
def ds_Sv_echo_range_irregular(random_number_generator):
    depth_interval = [0.5, 0.32, 0.13]
    depth_ping_time_len = [100, 300, 200]
    ping_time_len = 600
    ping_time_interval = "0.3S"
    return _gen_Sv_echo_range_irregular(
        depth_interval=depth_interval,
        depth_ping_time_len=depth_ping_time_len,
        ping_time_len=ping_time_len,
        ping_time_interval=ping_time_interval,
        ping_time_jitter_max_ms=0,
        random_number_generator=random_number_generator,
    )
