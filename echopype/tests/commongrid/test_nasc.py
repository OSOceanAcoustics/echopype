import pytest

import numpy as np

from echopype import open_raw
from echopype.calibrate import compute_Sv
from echopype.commongrid.api import compute_NASC, _parse_x_bin, get_x_along_channels
from echopype.commongrid.nasc import (
    get_distance_from_latlon,
    get_distance_from_latlon,
)
from echopype.consolidate import add_location, add_depth
from echopype.tests.commongrid.conftest import get_NASC_echoview


@pytest.fixture
def ek60_path(test_path):
    return test_path["EK60"]


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


@pytest.mark.unit
def test_NASC_Echoview_values(mock_Sv_dataset_NASC):
    dist_interval = np.array([-5, 10])
    range_interval = np.array([1, 5])
    raw_NASC = get_x_along_channels(
        mock_Sv_dataset_NASC,
        range_interval,
        dist_interval,
        x_var="distance_nmi",
        range_var="range_sample",
    )
    for ch_idx, _ in enumerate(raw_NASC.channel):
        NASC_echoview = get_NASC_echoview(mock_Sv_dataset_NASC, ch_idx)
        assert np.allclose(
            raw_NASC.sv.isel(channel=ch_idx)[0, 0], NASC_echoview, atol=1e-10, rtol=1e-10
        )
