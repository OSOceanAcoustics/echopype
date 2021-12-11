from pathlib import Path
from echopype.calibrate.ecs_parser import ECSParser

data_dir = Path("./echopype/test_data/ecs")


def test_convert_ecs():
    # Test converting an EV calibration file (ECS)
    ecs_path = data_dir / "Summer2017_JuneCal_3freq_mod.ecs"

    ecs = ECSParser(ecs_path)
    ecs.parse()

    # Spot test parsed outcome
    assert ecs.parsed_params["fileset"]["SoundSpeed"] == "1496"
    assert ecs.parsed_params["sourcecal"]["T1"]["MinorAxisAngleOffset"] == "-0.18"
    assert ecs.parsed_params["sourcecal"]["T2"]["MajorAxis3dbBeamAngle"] == "6.85"
    assert ecs.parsed_params["sourcecal"]["T3"]["Ek60TransducerGain"] == "26.55"
    assert ecs.parsed_params["localcal"]["MyCal"]["TwoWayBeamAngle"] == "-17.37"

    cal_params = ecs.get_cal_params()

    # Test SourceCal overwrite FileSet settings
    assert cal_params["T1"]["SoundSpeed"] == "1480.60"

    # Test overwrite by LocalCal
    assert cal_params["T2"]["TwoWayBeamAngle"] == "-17.37"
