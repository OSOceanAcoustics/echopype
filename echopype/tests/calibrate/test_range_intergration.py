import pytest
import echopype as ep


@pytest.mark.parametrize(
    (
        "test_path_key", "sonar_model", "raw_file", "xml_file",
        "env_params", "cal_params", "waveform_mode", "encode_mode"
    ),
    [
        # AZFP
        ("AZFP", "AZFP", "17082117.01A", "17041823.XML", {"salinity": 30, "pressure": 10}, {}, None, None),
        # EK60
        ("EK60", "EK60", "DY1801_EK60-D20180211-T164025.raw", None, None, None, None, None),
    ],
    ids=[
        "azfp",
        "ek60",
    ]
)
def test_range_dimensions(
    test_path, test_path_key, sonar_model, raw_file, xml_file,
    env_params, cal_params, waveform_mode, encode_mode,
):
    if xml_file is not None:
        ed = ep.open_raw(
            raw_file=test_path[test_path_key] / raw_file, sonar_model=sonar_model,
            xml_path=test_path[test_path_key] / xml_file,
        )
    else:
        ed = ep.open_raw(raw_file=test_path[test_path_key] / raw_file, sonar_model=sonar_model)        
    ds_Sv = ep.calibrate.compute_Sv(echodata=ed, env_params=env_params)
    assert ds_Sv["echo_range"].dims == ("channel", "ping_time", "range_sample")
