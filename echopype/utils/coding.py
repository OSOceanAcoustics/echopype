from re import search

import numpy as np
import xarray as xr
import zarr
from xarray import coding

COMPRESSION_SETTINGS = {
    "netcdf4": {"zlib": True, "complevel": 4},
    "zarr": {"compressor": zarr.Blosc(cname="zstd", clevel=3, shuffle=2)},
}

DEFAULT_TIME_ENCODING = {
    "units": "seconds since 1900-01-01T00:00:00+00:00",
    "calendar": "gregorian",
    "_FillValue": np.nan,
    "dtype": np.dtype("float64"),
}


DEFAULT_ENCODINGS = {
    "ping_time": DEFAULT_TIME_ENCODING,
    "ping_time_burst": DEFAULT_TIME_ENCODING,
    "ping_time_average": DEFAULT_TIME_ENCODING,
    "ping_time_echosounder": DEFAULT_TIME_ENCODING,
    "ping_time_echosounder_raw": DEFAULT_TIME_ENCODING,
    "ping_time_echosounder_raw_transmit": DEFAULT_TIME_ENCODING,
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


def set_encodings(ds: xr.Dataset) -> xr.Dataset:
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
