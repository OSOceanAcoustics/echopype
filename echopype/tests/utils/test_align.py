import xarray as xr
import numpy as np
import pytest

import echopype as ep
from echopype.utils.align import align_to_ping_time


@pytest.mark.unit
@pytest.mark.parametrize(
    "method, expected_aligned_da",
    [
        (
            "nearest",
            xr.DataArray(
                [0.0, 2.0, 3.0],
                coords={
                    "ping_time": np.array(
                        ["2017-06-20T01:15:00", "2017-06-20T01:30:00", "2017-06-20T01:45:00"],
                        dtype="datetime64[ns]"
                    )
                },
                dims=["ping_time"]
            )
        ),
        (
            "linear",
            xr.DataArray(
                [0.5, 2.0, 3.5],
                coords={
                    "ping_time": np.array(
                        ["2017-06-20T01:15:00", "2017-06-20T01:30:00", "2017-06-20T01:45:00"],
                        dtype="datetime64[ns]"
                    )
                },
                dims=["ping_time"]
            )
        ),
    ]
)
def test_align_to_ping_time_values(method, expected_aligned_da):
    """
    Test 'nearest' and 'linear' `align_to_ping_time` output values on mock external and
    ping time DataArrays.
    """
    # Create external and sounder DataArrays
    external_da = xr.DataArray(
        [0, 1, 2, 3],
        coords={
            "time1": np.array(
                ["2017-06-20T01:10:00", "2017-06-20T01:20:00", "2017-06-20T01:30:00", "2017-06-20T01:40:00"],
                dtype="datetime64[ns]"
            )
        },
        dims=["time1"]
    )
    sounder_da = xr.DataArray(
        [10, 20, 30],
        coords={
            "ping_time": np.array(
                ["2017-06-20T01:15:00", "2017-06-20T01:30:00", "2017-06-20T01:45:00"],
                dtype="datetime64[ns]"
            )
        },
        dims=["ping_time"]
    )

    # Align external DataArray
    aligned_da = align_to_ping_time(external_da, "time1", sounder_da["ping_time"], method=method)

    # Check that expected_aligned_da equals aligned_da
    expected_aligned_da.equals(aligned_da)



@pytest.mark.unit
@pytest.mark.parametrize(
    "external_da, expected_aligned_da",
    [
        (
            xr.DataArray(
                [2.0],
                coords={
                    "time1": np.array(
                        ["2017-06-20T01:10:00"], # Before earliest ping time
                        dtype="datetime64[ns]"
                    )
                },
                dims=["time1"]
            ),
            xr.DataArray(
                [2.0, 2.0, 2.0],
                coords={
                    "ping_time": np.array(
                        ["2017-06-20T01:15:00", "2017-06-20T01:30:00", "2017-06-20T01:45:00"],
                        dtype="datetime64[ns]"
                    )
                },
                dims=["ping_time"]
            )
        ),
        (
            xr.DataArray(
                [2.0],
                coords={
                    "time1": np.array(
                        ["2017-06-20T01:20:00"], # Between min and max ping times
                        dtype="datetime64[ns]"
                    )
                },
                dims=["time1"]
            ),
            xr.DataArray(
                [2.0, 2.0, 2.0],
                coords={
                    "ping_time": np.array(
                        ["2017-06-20T01:15:00", "2017-06-20T01:30:00", "2017-06-20T01:45:00"],
                        dtype="datetime64[ns]"
                    )
                },
                dims=["ping_time"]
            )
        ),
        (
            xr.DataArray(
                [2.0],
                coords={
                    "time1": np.array(
                        ["2017-06-20T01:50:00"], # After latest ping time
                        dtype="datetime64[ns]"
                    )
                },
                dims=["time1"]
            ),
            xr.DataArray(
                [2.0, 2.0, 2.0],
                coords={
                    "ping_time": np.array(
                        ["2017-06-20T01:15:00", "2017-06-20T01:30:00", "2017-06-20T01:45:00"],
                        dtype="datetime64[ns]"
                    )
                },
                dims=["ping_time"]
            )
        ),
        (
            xr.DataArray(
                [],
                coords={
                    "time1": np.array([])
                },
                dims=["time1"]
            ),
            xr.DataArray(
                [np.nan, np.nan, np.nan],
                coords={
                    "ping_time": np.array(
                        ["2017-06-20T01:15:00", "2017-06-20T01:30:00", "2017-06-20T01:45:00"],
                        dtype="datetime64[ns]"
                    )
                },
                dims=["ping_time"]
            )
        ),
    ]
)
def test_align_to_ping_time_values_length_1_and_0(external_da, expected_aligned_da):
    """
    Test `align_to_ping_time` output values on mock external and
    ping time DataArrays that have length 1 or length 0 on the time
    dimension.
    """
    # Create sounder DataArray
    sounder_da = xr.DataArray(
        [10, 20, 30],
        coords={
            "ping_time": np.array(
                ["2017-06-20T01:15:00", "2017-06-20T01:30:00", "2017-06-20T01:45:00"],
                dtype="datetime64[ns]"
            )
        },
        dims=["ping_time"]
    )

    # Align external DataArray
    aligned_da = align_to_ping_time(external_da, "time1", sounder_da["ping_time"])

    # Check that expected_aligned_da equals aligned_da
    expected_aligned_da.equals(aligned_da)


@pytest.mark.unit
def test_align_to_ping_time_values_when_matching_time_values():
    """
    Test 'nearest' and 'linear' `align_to_ping_time` output values on mock external and
    ping time DataArrays where the time values of the mock external and the ping time DataArrays
    are equal.
    """
    # Create external, sounder, and expected aligned output DataArrays
    external_da = xr.DataArray(
        [0, 1, 2],
        coords={
            "time1": np.array(
                ["2017-06-20T01:15:00", "2017-06-20T01:30:00", "2017-06-20T01:45:00"],
                dtype="datetime64[ns]"
            )
        },
        dims=["time1"]
    )
    sounder_da = xr.DataArray(
        [10, 20, 30],
        coords={
            "ping_time": np.array(
                ["2017-06-20T01:15:00", "2017-06-20T01:30:00", "2017-06-20T01:45:00"],
                dtype="datetime64[ns]"
            )
        },
        dims=["ping_time"]
    )
    expected_aligned_da = xr.DataArray(
        [0, 1, 2],
        coords={
            "ping_time": np.array(
                ["2017-06-20T01:15:00", "2017-06-20T01:30:00", "2017-06-20T01:45:00"],
                dtype="datetime64[ns]"
            )
        },
        dims=["ping_time"]
    )

    # Align external DataArray
    aligned_da = align_to_ping_time(external_da, "time1", sounder_da["ping_time"])

    # Check that sounder ping time equals aligned_da
    expected_aligned_da.equals(aligned_da)


@pytest.mark.integration
def test_align_to_ping_time_glider_azfp():
    """
    Test aligning external Glider pitch data to Echosounder ping time.
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

    # Check that all values in the aligned DataArray are non-NaN
    assert np.all(~np.isnan(aligned_da))

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

    # To check that the correct extrapolation calculation was made, we take all values with times
    # before the first non-NaN pitch value and check that there is a single unique value and that
    # this single unique value matches the first non-NaN pitch value. Extrapolation was done on the
    # 'left-side' of this array because the pitch data starts before the Echosounder started pinging.
    first_non_NaN_pitch = glider_ds["m_pitch"].dropna("time").isel(time=0)
    assert np.isclose(
        first_non_NaN_pitch.values, 
        np.unique(aligned_da.where(first_non_NaN_pitch["time"].values > aligned_da["ping_time"], drop=True))
    )
