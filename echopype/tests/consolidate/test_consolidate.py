import echopype as ep


def test_add_location(test_path):
    ed = ep.open_raw(
        test_path["EK60"] / "Winter2017-D20170115-T150122.raw",
        sonar_model="EK60"
    )
    ds = ep.calibrate.compute_Sv(ed)

    def _check_var(ds_test):
        assert "latitude" in ds_test
        assert "longitude" in ds_test
        assert "time1" not in ds_test

    ds_all = ep.consolidate.add_location(ds=ds, echodata=ed)
    _check_var(ds_all)

    ds_sel = ep.consolidate.add_location(ds=ds, echodata=ed, nmea_sentence="GGA")
    _check_var(ds_sel)
