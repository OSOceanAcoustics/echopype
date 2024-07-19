from re import search
from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr
import zarr
from dask.array.core import auto_chunks
from dask.utils import parse_bytes
from xarray import coding

DEFAULT_TIME_ENCODING = {
    "units": "nanoseconds since 1970-01-01T00:00:00Z",
    "calendar": "gregorian",
    "dtype": np.dtype("int64"),
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
    "time4": DEFAULT_TIME_ENCODING,
    "time5": DEFAULT_TIME_ENCODING,
}


EXPECTED_VAR_DTYPE = {
    "channel": np.str_,
    "cal_channel_id": np.str_,
    "beam": np.str_,
    "channel_mode": np.float32,
    "beam_stabilisation": np.byte,
    "non_quantitative_processing": np.int16,
}  # channel name  # beam name

PREFERRED_CHUNKS = "preferred_chunks"


def sanitize_dtypes(ds: xr.Dataset) -> xr.Dataset:
    """
    Validates and fixes data type for expected variables
    """

    if isinstance(ds, xr.Dataset):
        for name, var in ds.variables.items():
            if name in EXPECTED_VAR_DTYPE:
                expected_dtype = EXPECTED_VAR_DTYPE[name]
            elif np.issubdtype(var.dtype, np.object_):
                # Defaulting to strings dtype for object data types
                expected_dtype = np.str_
            else:
                # For everything else, this should be the same
                expected_dtype = var.dtype

            if not np.issubdtype(var.dtype, expected_dtype):
                ds[name] = var.astype(expected_dtype)
    return ds


def _encode_time_dataarray(da):
    """Encodes and decode datetime64 array similar to writing to file"""
    if da.size == 0:
        return da
    if da.dtype == np.int64:
        encoded_data = da
    elif da.dtype == np.float64:
        raise ValueError("Encoded time data array must be of type ```np.int64```.")
    else:
        # fmt: off
        encoded_data, _, _ = coding.times.encode_cf_datetime(
            da, **{
                "units": DEFAULT_TIME_ENCODING["units"],
                "calendar": DEFAULT_TIME_ENCODING["calendar"],
            }
        )
        # fmt: on
    return coding.times.decode_cf_datetime(
        encoded_data,
        **{
            "units": DEFAULT_TIME_ENCODING["units"],
            "calendar": DEFAULT_TIME_ENCODING["calendar"],
        },
    )


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
                    _encode_time_dataarray,
                    da,
                    keep_attrs=True,
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


def set_zarr_encodings(
    ds: xr.Dataset, compression_settings: dict, chunk_size: str = "100MB", ctol: str = "10MB"
) -> dict:
    """
    Obtains all variable encodings based on zarr default values

    Parameters
    ----------
    ds : xr.Dataset
        The dataset object to generate encoding for
    compression_settings : dict
        The compression settings dictionary
    chunk_size : dict
        The desired chunk size
    ctol : dict
        The chunk size tolerance before rechunking

    Returns
    -------
    dict
        The encoding dictionary
    """

    # create zarr specific encoding
    encoding = dict()
    for name, val in ds.variables.items():
        encoding[name] = {**val.encoding}
        encoding[name].update(get_zarr_compression(val, compression_settings))

        # Always optimize chunk if not specified already
        # user can specify desired chunk in encoding
        existing_chunks = encoding[name].get("chunks", None)
        optimal_chunk_size = parse_bytes(chunk_size)
        chunk_size_tolerance = parse_bytes(ctol)

        if len(val.shape) > 0:
            rechunk = True
            if existing_chunks is not None:
                # Perform chunk optimization
                # 1. Get the chunk total from existing chunks
                chunk_total = np.prod(existing_chunks) * val.dtype.itemsize
                # 2. Get chunk size difference from the optimal chunk size
                chunk_diff = optimal_chunk_size - chunk_total
                # 3. Check difference from tolerance, if diff is less than
                #    tolerance then no need to rechunk
                if chunk_diff < chunk_size_tolerance:
                    rechunk = False
                    chunks = existing_chunks

            if rechunk:
                # Use dask auto chunk to determine the optimal chunk
                # spread for optimal chunk size
                chunks = _get_auto_chunk(val, chunk_size=chunk_size)

            encoding[name]["chunks"] = chunks
        if PREFERRED_CHUNKS in encoding[name]:
            # Remove 'preferred_chunks', use chunks only instead
            encoding[name].pop(PREFERRED_CHUNKS)

    return encoding


def set_netcdf_encodings(
    ds: xr.Dataset,
    compression_settings: Dict[str, Any] = {},
) -> Dict[str, Dict[str, Any]]:
    """
    Obtains all variables encodings based on netcdf default values

    Parameters
    ----------
    ds : xr.Dataset
        The dataset object to generate encoding for
    compression_settings : dict
        The compression settings dictionary

    Returns
    -------
    dict
        The final encoding values for dataset variables
    """
    encoding = dict()
    for name, val in ds.variables.items():
        encoding[name] = {**val.encoding}
        if np.issubdtype(val.dtype, np.str_):
            encoding[name].update(
                {
                    "zlib": False,
                }
            )
        elif compression_settings:
            encoding[name].update(compression_settings)
        else:
            encoding[name].update(COMPRESSION_SETTINGS["netcdf4"])

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
