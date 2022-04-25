from pathlib import Path
from datetime import datetime

from echopype.calibrate.ecs_parser import ECSParser

data_dir = Path("./echopype/test_data/ecs")


CORRECT_PARSED_PARAMS = {
    "fileset": {
        "SoundSpeed": 1496.0,
        "TvgRangeCorrection": "BySamples",
        "TvgRangeCorrectionOffset": 2.0,
    },
    "sourcecal": {
        "T1": {
            "AbsorptionCoefficient": 0.002822,
            "EK60SaCorrection": -0.7,
            "Ek60TransducerGain": 22.95,
            "MajorAxis3dbBeamAngle": 10.82,
            "MajorAxisAngleOffset": 0.25,
            "MajorAxisAngleSensitivity": 13.89,
            "MinorAxis3dbBeamAngle": 10.9,
            "MinorAxisAngleOffset": -0.18,
            "MinorAxisAngleSensitivity": 13.89,
            "SoundSpeed": 1480.6,
            "TwoWayBeamAngle": -17.37,
        },
        "T2": {
            "AbsorptionCoefficient": 0.009855,
            "EK60SaCorrection": -0.52,
            "Ek60TransducerGain": 26.07,
            "MajorAxis3dbBeamAngle": 6.85,
            "MajorAxisAngleOffset": 0.0,
            "MajorAxisAngleSensitivity": 21.970001,
            "MinorAxis3dbBeamAngle": 6.81,
            "MinorAxisAngleOffset": -0.08,
            "MinorAxisAngleSensitivity": 21.970001,
            "SoundSpeed": 1480.6,
            "TwoWayBeamAngle": -21.01,
        },
        "T3": {
            "AbsorptionCoefficient": 0.032594,
            "EK60SaCorrection": -0.3,
            "Ek60TransducerGain": 26.55,
            "MajorAxis3dbBeamAngle": 6.52,
            "MajorAxisAngleOffset": 0.37,
            "MajorAxisAngleSensitivity": 23.12,
            "MinorAxis3dbBeamAngle": 6.58,
            "MinorAxisAngleOffset": -0.05,
            "MinorAxisAngleSensitivity": 23.12,
            "SoundSpeed": 1480.6,
            "TwoWayBeamAngle": -20.47,
        },
    },
    "localcal": {"MyCal": {"TwoWayBeamAngle": -17.37}},
}


def test_convert_ecs():
    # Test converting an EV calibration file (ECS)
    ecs_path = data_dir / "Summer2017_JuneCal_3freq_mod.ecs"

    ecs = ECSParser(ecs_path)
    ecs.parse()

    # Spot test parsed outcome
    assert ecs.data_type == "SimradEK60Raw"
    assert ecs.version == "1.00"
    assert ecs.file_creation_time == datetime(
        year=2015, month=6, day=19, hour=23, minute=26, second=4
    )
    assert ecs.parsed_params == CORRECT_PARSED_PARAMS

    cal_params = ecs.get_cal_params()

    # Test SourceCal overwrite FileSet settings
    assert cal_params["T1"]["SoundSpeed"] == 1480.60

    # Test overwrite by LocalCal
    assert cal_params["T2"]["TwoWayBeamAngle"] == -17.37
