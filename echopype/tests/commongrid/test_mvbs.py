import numpy as np
import pandas as pd
import pytest
from echopype.commongrid.mvbs import get_MVBS_along_channels
from echopype.consolidate.api import POSITION_VARIABLES
from flox.xarray import xarray_reduce

@pytest.mark.unit
@pytest.mark.parametrize(["range_var", "lat_lon"], [("depth", False), ("echo_range", True), ("echo_range", False)])
def test_get_MVBS_along_channels(request, range_var, lat_lon):
    """Testing the underlying function of compute_MVBS"""
    range_bin = 20
    ping_time_bin = "20S"
    method = "map-reduce"
    
    flox_kwargs = {
        "reindex": True
    }
    
    # Retrieve the correct dataset
    if range_var == "depth":
        ds_Sv = request.getfixturevalue("ds_Sv_er_regular_w_depth")
    elif range_var == "echo_range" and lat_lon is True:
        ds_Sv = request.getfixturevalue("ds_Sv_er_regular_w_latlon")
    else:
        ds_Sv = request.getfixturevalue("ds_Sv_er_regular")
    
    # compute range interval
    echo_range_max = ds_Sv[range_var].max()
    range_interval = np.arange(0, echo_range_max + range_bin, range_bin)
    
    # create bin information needed for ping_time
    d_index = (
        ds_Sv["ping_time"]
        .resample(ping_time=ping_time_bin, skipna=True)
        .asfreq()
        .indexes["ping_time"]
    )
    ping_interval = d_index.union([d_index[-1] + pd.Timedelta(ping_time_bin)])
    
    raw_MVBS = get_MVBS_along_channels(
        ds_Sv, range_interval, ping_interval,
        range_var=range_var, method=method, **flox_kwargs
    )
    
    # Check that the range_var is in the dimension
    assert f"{range_var}_bins" in raw_MVBS.dims
    
    # When it's echo_range and lat_lon, the dataset should have positions
    if range_var == "echo_range" and lat_lon is True:
        assert raw_MVBS.attrs["has_positions"] is True
        assert all(v in raw_MVBS for v in POSITION_VARIABLES)

        # Compute xarray reduce manually for this
        expected_Pos = xarray_reduce(
            ds_Sv[POSITION_VARIABLES],
            ds_Sv["ping_time"],
            func="nanmean",
            expected_groups=(ping_interval),
            isbin=True,
            method=method,
        )
        
        for v in POSITION_VARIABLES:
            assert np.array_equal(raw_MVBS[v].data, expected_Pos[v].data)
    else:
        assert raw_MVBS.attrs["has_positions"] is False
