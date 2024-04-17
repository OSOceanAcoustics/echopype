import pytest
import numpy as np
import xarray as xr
import math
import dask
import warnings

from echopype.utils.coding import _get_auto_chunk, set_netcdf_encodings, _encode_time_dataarray, DEFAULT_TIME_ENCODING

@pytest.mark.parametrize(
    "chunk",
    ["auto", "5MB", "10MB", "30MB", "70MB", "100MB", "default"],
)
def test__get_auto_chunk(chunk):
    random_data = 15 + 8 * np.random.randn(10, 1000, 1000)

    da = xr.DataArray(
        data=random_data,
        dims=["x", "y", "z"]
    )
    
    if chunk == "auto":
        dask_data = da.chunk('auto').data
    elif chunk == "default":
        dask_data = da.chunk(_get_auto_chunk(da)).data
    else:
        dask_data = da.chunk(_get_auto_chunk(da, chunk)).data
    
    chunk_byte_size = math.prod(dask_data.chunksize + (dask_data.itemsize,))
    
    if chunk in ["auto", "100MB", "default"]:
        assert chunk_byte_size == dask_data.nbytes, "Default chunk is not equal to data array size!"
    else:
        assert chunk_byte_size <= dask.utils.parse_bytes(chunk), "Calculated chunk exceeded max chunk!"
        
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

    # This should pass without error since int64 should be sufficient to encompass nanosecond scale granularity
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
