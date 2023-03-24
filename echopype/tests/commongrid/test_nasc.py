
from echopype import open_raw
from echopype.calibrate import compute_Sv
from echopype.commongrid import compute_NASC
from echopype.consolidate import add_location, add_depth



def test_compute_NASC():
    raw_path = "/Users/wu-junglee/Downloads/Summer2017-D20170620-T011027.raw"

    ed = open_raw(raw_path, sonar_model="EK60")
    ds_Sv = add_depth(add_location(compute_Sv(ed), ed, nmea_sentence="GGA"))

    assert 1 == 1
