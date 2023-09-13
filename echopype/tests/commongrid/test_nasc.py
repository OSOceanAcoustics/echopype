import pytest

import numpy as np

from echopype import open_raw
from echopype.calibrate import compute_Sv
# from echopype.commongrid import compute_NASC
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


# def test_compute_NASC(ek60_path):
#     raw_path = ek60_path / "ncei-wcsd/Summer2017-D20170620-T011027.raw"

#     ed = open_raw(raw_path, sonar_model="EK60")
#     ds_Sv = add_depth(add_location(compute_Sv(ed), ed, nmea_sentence="GGA"))
#     cell_dist = 0.1
#     cell_depth = 20
#     ds_NASC = compute_NASC(ds_Sv, cell_dist, cell_depth)

#     dist_nmi = get_distance_from_latlon(ds_Sv)

#     # Check dimensions
#     da_NASC = ds_NASC["NASC"]
#     assert da_NASC.dims == ("channel", "distance", "depth")
#     assert np.all(ds_NASC["channel"].values == ds_Sv["channel"].values)
#     assert da_NASC["depth"].size == np.ceil(ds_Sv["depth"].max() / cell_depth)
#     assert da_NASC["distance"].size == np.ceil(dist_nmi.max() / cell_dist)
