from echopype.calibrate.env_params import harmonize_env_param_time

import xarray as xr
import numpy as np


def test_harmonize_env_param_time():
    # Scalar
    p = 10.05
    assert harmonize_env_param_time(p=p) == 10.05

    # time1 length=1, should return length=1 numpy array
    p = xr.DataArray(
        data=[1],
        coords={
            "time1": np.array(["2017-06-20T01:00:00"], dtype="datetime64[ns]")
        },
        dims=["time1"]
    )
    assert harmonize_env_param_time(p=p) == 1

    # time1 length>1, interpolate to tareget ping_time
    p = xr.DataArray(
        data=np.array([0, 1]),
        coords={
            "time1": np.arange("2017-06-20T01:00:00", "2017-06-20T01:00:31", np.timedelta64(30, "s"), dtype="datetime64[ns]")
        },
        dims=["time1"]
    )
    # ping_time target is identical to time1
    ping_time_target = p["time1"].rename({"time1": "ping_time"})
    p_new = harmonize_env_param_time(p=p, ping_time=ping_time_target)
    assert (p_new["ping_time"] == ping_time_target).all()
    assert (p_new.data == p.data).all()
    # ping_time target requires actual interpolation
    ping_time_target = xr.DataArray(
        data=[1],
        coords={
            "ping_time": np.array(["2017-06-20T01:00:15"], dtype="datetime64[ns]")
        },
        dims=["ping_time"]
    )
    p_new = harmonize_env_param_time(p=p, ping_time=ping_time_target["ping_time"])
    assert p_new["ping_time"] == ping_time_target["ping_time"]
    assert p_new.data == 0.5


# TODO: unit test for get_env_params_AZFP/EK60/EK80
# - make sure the combination is correctly passed in
# - make sure the sound speed and absorption are correctly calculated
