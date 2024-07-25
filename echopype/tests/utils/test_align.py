import xarray as xr
import numpy as np
import pytest

import echopype as ep
from echopype.utils.align import align_to_ping_time


@pytest.mark.unit
def test_align_to_ping_time():
    """
    Test aligning external pitch data to Echosounder ping time
    """
    # Open RAW and extract ping time
    ping_time_da = ep.open_raw(
        raw_file="echopype/test_data/azfp/rutgers_glider_external_nc/18011107.01A",
        xml_path="echopype/test_data/azfp/rutgers_glider_external_nc/18011107.XML",
        sonar_model="azfp"
    )["Sonar/Beam_group1"]["ping_time"]

    # Open external glider dataset
    glider_ds = xr.open_dataset(
        "echopype/test_data/azfp/rutgers_glider_external_nc/ru32-20180109T0531-profile-sci-delayed-subset.nc",
        engine="netcdf4"
    )

    # Drop NaNs from pitch and align data array
    aligned_da = align_to_ping_time(glider_ds["m_pitch"].dropna("time"), "time", ping_time_da)

    # Extract earliest non-NaN time of the aligned dataset
    earliest_non_nan_time_in_aligned_da = aligned_da.dropna("ping_time")["ping_time"].min()

    # Grab all values past the earliest non-NaN time in the aligned DataArray
    subset_aligned_da = aligned_da.where(aligned_da["ping_time"] >= earliest_non_nan_time_in_aligned_da, drop=True)

    # Check that all values past the earliest non-NaN time in the aligned DataArray are non-NaN
    assert np.all(~np.isnan(subset_aligned_da))

    # Test if aligned_da matches interpolation
    assert np.allclose(
        aligned_da,
        glider_ds["m_pitch"].dropna("time").interp(
            {"time": ping_time_da},
            method="nearest",
            kwargs={"fill_value": "extrapolate"},
        ),
        equal_nan=True,
    )
