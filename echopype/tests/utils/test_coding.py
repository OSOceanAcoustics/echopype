import pytest
import numpy as np
import xarray as xr
import math
import dask
import warnings

from echopype.utils.coding import (
    _get_dask_auto_chunk,
    set_netcdf_encodings,
    set_time_encodings,
    set_zarr_encodings,
    _encode_time_dataarray,
    COMPRESSION_SETTINGS,
    DEFAULT_TIME_ENCODING,
)

@pytest.mark.unit
@pytest.mark.parametrize(
    "chunk",
    ["auto", "5MB", "10MB", "30MB", "70MB", "100MB", "default"],
)
def test__get_dask_auto_chunk(chunk):
    random_data = 15 + 8 * np.random.randn(10, 1000, 1000)

    da = xr.DataArray(
        data=random_data,
        dims=["x", "y", "z"]
    )
    
    if chunk == "auto":
        dask_data = da.chunk('auto').data
    elif chunk == "default":
        dask_data = da.chunk(_get_dask_auto_chunk(da)).data
    else:
        dask_data = da.chunk(_get_dask_auto_chunk(da, chunk)).data
    
    chunk_byte_size = math.prod(dask_data.chunksize + (dask_data.itemsize,))
    
    if chunk in ["auto", "100MB", "default"]:
        assert chunk_byte_size == dask_data.nbytes, "Default chunk is not equal to data array size!"
    else:
        assert chunk_byte_size <= dask.utils.parse_bytes(chunk), "Calculated chunk exceeded max chunk!"  # noqa: E501
        
@pytest.mark.unit
def test_set_netcdf_encodings():
    # create a test dataset
    ds = xr.Dataset(
        {
            "var1": xr.DataArray(np.random.rand(10), dims="dim1"),
            "var2": xr.DataArray(np.random.rand(10), dims="dim1", attrs={"attr1": "value1"}),
            "var3": xr.DataArray(["a", "b", "c"], dims="dim2"),
        },
        attrs={"global_attr": "global_value"},
    )

    # test with default compression settings
    encoding = set_netcdf_encodings(ds, {})
    assert isinstance(encoding, dict)
    assert len(encoding) == 3
    assert "var1" in encoding
    assert "var2" in encoding
    assert "var3" in encoding
    assert encoding["var1"]["zlib"] is True
    assert encoding["var1"]["complevel"] == 4
    assert encoding["var2"]["zlib"] is True
    assert encoding["var2"]["complevel"] == 4
    assert encoding["var3"]["zlib"] is False

    # test with custom compression settings
    compression_settings = {"zlib": True, "complevel": 5}
    encoding = set_netcdf_encodings(ds, compression_settings)
    assert isinstance(encoding, dict)
    assert len(encoding) == 3
    assert "var1" in encoding
    assert "var2" in encoding
    assert "var3" in encoding
    assert encoding["var1"]["zlib"] is True
    assert encoding["var1"]["complevel"] == 5
    assert encoding["var2"]["zlib"] is True
    assert encoding["var2"]["complevel"] == 5
    assert encoding["var3"]["zlib"] is False

@pytest.mark.unit
def test_encode_time_dataarray_on_nanosecond_resolution_encoding():
    """Test to ensure that the expected warning / lack of warnings comes up."""
    # Create an array with a multiple datetime64 elements
    datetime_array = np.array(
        [
            '2023-11-22T16:22:41.088137000', 
            '2023-11-22T16:22:46.150034000',
            '2023-11-22T16:22:51.140442000', 
            '2023-11-22T16:22:56.143124000'
        ],
        dtype='datetime64[ns]'
    )

    # This should pass without error since int64 should be sufficient to encompass nanosecond scale granularity  # noqa: E501
    # between time differences in 2023 and 1970
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decoded_datetime_array = _encode_time_dataarray(
            datetime_array,
        )

    # Check if datetime_array and decoded_datetime_array are equal
    assert np.array_equal(datetime_array, decoded_datetime_array), "Arrays are not equal"

@pytest.mark.unit
def test_encode_time_dataarray_on_encoded_time_data():
    """Test to ensure that the array equality and expected error hold."""
    # Create an array with a multiple datetime64 elements
    datetime_array = np.array(
        [
            '2023-11-22T16:22:41.088137000', 
            '2023-11-22T16:22:46.150034000',
            '2023-11-22T16:22:51.140442000', 
            '2023-11-22T16:22:56.143124000'
        ],
        dtype='datetime64[ns]'
    )
    
    # Encode datetime
    encoded_datetime_array, _, _ = xr.coding.times.encode_cf_datetime(
            datetime_array, **{
                "units": DEFAULT_TIME_ENCODING["units"],
                "calendar": DEFAULT_TIME_ENCODING["calendar"],
            }
        )

    # Check that no warning is raised
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decoded_datetime_array = _encode_time_dataarray(
            encoded_datetime_array
        )

    # Check if datetime_array and decoded_datetime_array are equal
    assert np.array_equal(datetime_array, decoded_datetime_array), "Arrays are not equal"

    # Check to see if returns empty array
    assert np.array_equal(np.empty(0), _encode_time_dataarray(np.empty(0)))
    
    # Check to see if value error is raised when we pass in an encoded float datetime array
    with pytest.raises(ValueError, match="Encoded time data array must be of type ```np.int64```."):
        _encode_time_dataarray(encoded_datetime_array.astype(np.float64))


# Regression tests for zero-length / all-NaN handling on partial/truncated raw files.
# See PR #1624.

@pytest.mark.unit
def test_set_time_encodings_skips_all_nan_time_var():
    """All-NaN time variables must be skipped so xarray's encoder isn't called on them."""
    ds = xr.Dataset(
        {
            "ping_time": xr.DataArray(np.array([np.nan, np.nan]), dims="ping_time"),
            "backscatter_r": xr.DataArray(np.zeros((2, 3)), dims=("ping_time", "range_sample")),
        }
    )

    new_ds = set_time_encodings(ds)

    # Values are preserved and no encoding was forced onto the all-NaN time var.
    assert np.all(np.isnan(new_ds["ping_time"].values))
    assert new_ds["ping_time"].encoding == {}


@pytest.mark.unit
def test_set_time_encodings_all_nan_mixed_with_valid_time():
    """A valid time var should still be encoded when another time var is all-NaN."""
    valid_times = np.array(
        ["2024-01-01T00:00:00", "2024-01-01T00:00:01"], dtype="datetime64[ns]"
    )
    ds = xr.Dataset(
        {
            "ping_time": xr.DataArray(np.array([np.nan, np.nan]), dims="ping_time"),
            "time1": xr.DataArray(valid_times, dims="time1"),
        }
    )

    new_ds = set_time_encodings(ds)

    assert np.all(np.isnan(new_ds["ping_time"].values))
    assert new_ds["ping_time"].encoding == {}
    assert np.issubdtype(new_ds["time1"].dtype, np.datetime64)
    assert new_ds["time1"].encoding == DEFAULT_TIME_ENCODING


@pytest.mark.unit
def test_set_zarr_encodings_zero_length_dim_sets_chunks_to_none():
    """A variable with a zero-length dim must not trigger division by zero in chunk calc."""
    ds = xr.Dataset(
        {
            "backscatter_r": xr.DataArray(
                np.zeros((0, 3), dtype=np.float32), dims=("ping_time", "range_sample")
            ),
        }
    )

    encoding = set_zarr_encodings(ds, COMPRESSION_SETTINGS["zarr"])

    assert encoding["backscatter_r"]["chunks"] is None


@pytest.mark.unit
def test_set_zarr_encodings_scalar_variable_sets_chunks_to_none():
    """A scalar (zero-dim) variable must be given chunks=None rather than tripping the chunker."""
    ds = xr.Dataset({"scalar_var": xr.DataArray(np.float32(1.5))})

    encoding = set_zarr_encodings(ds, COMPRESSION_SETTINGS["zarr"])

    assert encoding["scalar_var"]["chunks"] is None


@pytest.mark.unit
def test_set_zarr_encodings_normal_variable_still_chunked():
    """Sanity check: non-empty, non-scalar variables still receive a chunk list."""
    ds = xr.Dataset(
        {
            "backscatter_r": xr.DataArray(
                np.zeros((10, 20), dtype=np.float32), dims=("ping_time", "range_sample")
            ),
        }
    )

    encoding = set_zarr_encodings(ds, COMPRESSION_SETTINGS["zarr"])

    chunks = encoding["backscatter_r"]["chunks"]
    assert chunks is not None
    assert len(chunks) == 2
