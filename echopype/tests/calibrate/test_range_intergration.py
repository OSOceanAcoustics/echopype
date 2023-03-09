import pytest
import echopype as ep


@pytest.fixture
def azfp_path(test_path):
    return test_path['AZFP']


@pytest.fixture
def ek60_path(test_path):
    return test_path['EK60']


@pytest.fixture
def ek80_cal_path(test_path):
    return test_path['EK80_CAL']


@pytest.mark.parametrize(
    ("sonar_model", "raw_file", "xml_file", "env_params", "cal_params", "waveform_mode", "encode_mode"),
    [
        # AZFP
        ("AZFP", "17082117.01A", "17041823.XML", {"salinity": 30, "pressure": 10}, {}, None, None)
    ],
    ids=[
        "azfp",
    ]
)
def test_range_dimensions(azfp_path, sonar_model, raw_file, xml_file, env_params, cal_params, waveform_mode, encode_mode):
    ed = ep.open_raw(
        raw_file=azfp_path / raw_file, sonar_model=sonar_model, xml_path=azfp_path / xml_file
    )
    ds_Sv = ep.calibrate.compute_Sv(echodata=ed, env_params=env_params)
    assert ds_Sv["echo_range"].dims == ("channel", "ping_time", "range_sample")
