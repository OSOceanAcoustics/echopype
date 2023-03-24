import numpy as np

from echopype import open_raw
from echopype.calibrate import compute_Sv
from echopype.commongrid import compute_NASC
from echopype.commongrid.nasc import (
    get_distance_from_latlon,
    get_depth_bin_info,
    get_dist_bin_info,
    get_distance_from_latlon,
)
from echopype.consolidate import add_location, add_depth



def test_compute_NASC():
    raw_path = "/Users/wujung/Downloads/Summer2017-D20170620-T011027.raw"

    ed = open_raw(raw_path, sonar_model="EK60")
    ds_Sv = add_depth(add_location(compute_Sv(ed), ed, nmea_sentence="GGA"))
    cell_dist = 0.1
    cell_depth = 20
    ds_NASC = compute_NASC(ds_Sv, cell_dist, cell_depth)

    dist_nmi = get_distance_from_latlon(ds_Sv)

    # Check dimensions
    assert ds_NASC.dims == ("channel", "distance", "depth")
    assert np.all(ds_NASC["channel"].values == ds_Sv["channel"].values)
    assert ds_NASC["depth"].size == np.ceil(ds_Sv["depth"].max() / cell_depth)
    assert ds_NASC["distance"].size == np.ceil(dist_nmi.max() / cell_dist)
