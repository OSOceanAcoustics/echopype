from re import search
from typing import Tuple

import numpy as np
import xarray as xr
import zarr
from dask.array.core import auto_chunks
from xarray import coding

DEFAULT_TIME_ENCODING = {
    "units": "seconds since 1900-01-01T00:00:00+00:00",
    "calendar": "gregorian",
    "_FillValue": np.nan,
    "dtype": np.dtype("float64"),
}

COMPRESSION_SETTINGS = {
    "netcdf4": {"zlib": True, "complevel": 4},
    # zarr compressors were chosen based on xarray results
    "zarr": {
        "float": {"compressor": zarr.Blosc(cname="zstd", clevel=3, shuffle=2)},
        "int": {"compressor": zarr.Blosc(cname="lz4", clevel=5, shuffle=1, blocksize=0)},
        "string": {"compressor": zarr.Blosc(cname="lz4", clevel=5, shuffle=1, blocksize=0)},
        "time": {"compressor": zarr.Blosc(cname="lz4", clevel=5, shuffle=1, blocksize=0)},
    },
}


DEFAULT_ENCODINGS = {
    "ping_time": DEFAULT_TIME_ENCODING,
    "ping_time_transmit": DEFAULT_TIME_ENCODING,
    "time1": DEFAULT_TIME_ENCODING,
    "time2": DEFAULT_TIME_ENCODING,
    "time3": DEFAULT_TIME_ENCODING,
}


def _encode_dataarray(da, dtype):
    """Encodes and decode datetime64 array similar to writing to file"""
    if da.size == 0:
        return da
    read_encoding = {
        "units": "seconds since 1900-01-01T00:00:00+00:00",
        "calendar": "gregorian",
    }

    if dtype in [np.float64, np.int64]:
        encoded_data = da
    else:
        # fmt: off
        encoded_data, _, _ = coding.times.encode_cf_datetime(
            da, **read_encoding
        )
        # fmt: on
    return coding.times.decode_cf_datetime(encoded_data, **read_encoding)


def _get_auto_chunk(
    variable: xr.DataArray, chunk_size: "int | str | float" = "100MB"
) -> Tuple[int]:
    """
    Calculate default chunks for a data array based on desired chunk size

    Parameters
    ----------
    variable : xr.DataArray
        The data array variable to be calculated
    chunk_size : int or str or float
        The desired max chunk size for the array.
        Default is 100MB

    Returns
    -------
    tuple
        The chunks
    """
    auto_tuple = tuple(["auto" for i in variable.shape])
    chunks = auto_chunks(auto_tuple, variable.shape, chunk_size, variable.dtype)
    return tuple([c[0] if isinstance(c, tuple) else c for c in chunks])


def set_time_encodings(ds: xr.Dataset) -> xr.Dataset:
    """
    Set the default encoding for variables.
    """
    new_ds = ds.copy(deep=True)
    for var, encoding in DEFAULT_ENCODINGS.items():
        if var in new_ds:
            da = new_ds[var].copy()
            # Process all variable names matching the patterns *_time* or time<digits>
            # Examples: ping_time, ping_time_2, time1, time2
            if bool(search(r"_time|^time[\d]+$", var)):
                new_ds[var] = xr.apply_ufunc(
                    _encode_dataarray,
                    da,
                    keep_attrs=True,
                    kwargs={"dtype": da.dtype},
                )

            new_ds[var].encoding = encoding

    return new_ds


def get_zarr_compression(var: xr.Variable, compression_settings: dict) -> dict:
    """Returns the proper zarr compressor for a given variable type"""

    if np.issubdtype(var.dtype, np.floating):
        return compression_settings["float"]
    elif np.issubdtype(var.dtype, np.integer):
        return compression_settings["int"]
    elif np.issubdtype(var.dtype, np.str_):
        return compression_settings["string"]
    elif np.issubdtype(var.dtype, np.datetime64):
        return compression_settings["time"]
    else:
        raise NotImplementedError(f"Zarr Encoding for dtype = {var.dtype} has not been set!")


def set_zarr_encodings(ds: xr.Dataset, compression_settings: dict) -> dict:
    """
    Obtains all variable encodings based on zarr default values
    """

    # create zarr specific encoding
    encoding = dict()
    for name, val in ds.variables.items():
        val_encoding = val.encoding
        val_encoding.update(get_zarr_compression(val, compression_settings))

        # If data array is not a dask array yet,
        # create a custom chunking encoding
        # currently defaults to 100MB
        if not ds.chunks and len(val.shape) > 0:
            chunks = _get_auto_chunk(val)
            val_encoding.update({"chunks": chunks})
        encoding[name] = val_encoding

    return encoding


def set_netcdf_encodings(ds: xr.Dataset, compression_settings: dict) -> dict:
    """
    Obtains all variable encodings based on netcdf default values
    """

    # TODO: below is the encoding we were using for netcdf, we need to make
    #  sure that the encoding is appropriate for all data variables
    encoding = (
        {var: compression_settings for var in ds.data_vars}
        if compression_settings is not None
        else {}
    )

    return encoding


def set_storage_encodings(ds: xr.Dataset, compression_settings: dict, engine: str) -> dict:
    """
    Obtains the appropriate zarr or netcdf specific encodings for
    each variable in ``ds``.
    """

    if compression_settings is not None:
        if engine == "zarr":
            encoding = set_zarr_encodings(ds, compression_settings)

        elif engine == "netcdf4":
            encoding = set_netcdf_encodings(ds, compression_settings)

        else:
            raise RuntimeError(f"Obtaining encodings for the engine {engine} is not allowed.")

    else:
        encoding = dict()

    return encoding
