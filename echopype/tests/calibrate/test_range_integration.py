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
        # EK80 BB complex
        ("EK80_CAL", "EK80", "2018115-D20181213-T094600.raw", None, None, None, "BB", "complex"),
        # EK80 CW complex
        ("EK80_CAL", "EK80", "2018115-D20181213-T094600.raw", None, None, None, "CW", "complex"),
        # EK80 CW power
        ("EK80", "EK80", "Summer2018--D20180905-T033113.raw", None, None, None, "CW", "power"),
        # TODO: EK80 reduced sampling rate
        ("EK80", "EK80", "Summer2018--D20180905-T033113.raw", None, None, None, "BB", "complex"),
    ],
    ids=[
        "azfp",
        "ek60",
        "ek80_bb_complex",
        "ek80_cw_complex",
        "ek80_cw_power",
        "ek80_bb_complex"
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
    ds_Sv = ep.calibrate.compute_Sv(
        echodata=ed, env_params=env_params, cal_params=cal_params,
        waveform_mode=waveform_mode, encode_mode=encode_mode
    )
    assert ds_Sv["echo_range"].dims == ("channel", "ping_time", "range_sample")
