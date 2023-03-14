import pytest
import numpy as np
import xarray as xr
import math
import dask

from echopype.utils.coding import _get_auto_chunk

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
