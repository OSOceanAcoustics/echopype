import pytest

import numpy as np

from echopype import open_raw
from echopype.calibrate import compute_Sv
from echopype.commongrid.api import compute_NASC, _parse_x_bin
from echopype.commongrid.nasc import (
    get_distance_from_latlon,
    get_depth_bin_info,
    get_dist_bin_info,
    get_distance_from_latlon,
)
from echopype.consolidate import add_location, add_depth


@pytest.fixture
def ek60_path(test_path):
    return test_path['EK60']


def test_compute_NASC(ek60_path):
    raw_path = ek60_path / "ncei-wcsd/Summer2017-D20170620-T011027.raw"

    ed = open_raw(raw_path, sonar_model="EK60")
    ds_Sv = add_depth(add_location(compute_Sv(ed), ed, nmea_sentence="GGA"))
    dist_bin = "0.1nmi"
    range_bin = "20m"
    ds_NASC = compute_NASC(ds_Sv, range_bin, dist_bin)

    dist_nmi = get_distance_from_latlon(ds_Sv)

    # Check dimensions
    dist_bin = _parse_x_bin(dist_bin, "dist_bin")
    range_bin = _parse_x_bin(range_bin)
    da_NASC = ds_NASC["NASC"]
    assert da_NASC.dims == ("channel", "distance", "depth")
    assert np.all(ds_NASC["channel"].values == ds_Sv["channel"].values)
    assert da_NASC["depth"].size == np.ceil(ds_Sv["depth"].max() / range_bin)
    assert da_NASC["distance"].size == np.ceil(dist_nmi.max() / dist_bin)
