import sys
from functools import reduce

import numpy as np
import pandas as pd
import psutil

COMPLEX_VAR = "complex"


def _get_power_dims(dgram_zarr_vars):
    return list(reduce(lambda x, y: {*x, *y}, dgram_zarr_vars.values()))


def _extract_datagram_dfs(zarr_datagrams, dgram_zarr_vars):
    data_keys = dgram_zarr_vars.keys()
    power_dims = _get_power_dims(dgram_zarr_vars)
    power_angle = [k for k in data_keys if k != COMPLEX_VAR]

    datagram_df = pd.DataFrame.from_dict(zarr_datagrams)

    pow_ang_df = datagram_df[power_dims + power_angle]

    complex_df = None
    if COMPLEX_VAR in datagram_df:
        # Not EK60
        complex_df = datagram_df[power_dims + [COMPLEX_VAR]]

    # Clean up nans if there's any
    if isinstance(pow_ang_df, pd.DataFrame):
        pow_ang_df = pow_ang_df.dropna().reset_index(drop=True)

    if isinstance(complex_df, pd.DataFrame):
        complex_df = complex_df.dropna().reset_index(drop=True)

    return pow_ang_df, complex_df


def get_req_mem(datagram_df, dgram_zarr_vars):
    total_req_mem = 0
    if datagram_df is not None:
        power_dims = _get_power_dims(dgram_zarr_vars)
        df_shapes = datagram_df.apply(
            lambda col: col.unique().shape
            if col.name in power_dims
            else col.apply(lambda row: row.shape).max(),
            result_type="reduce",
        )

        for k, v in dgram_zarr_vars.items():
            if k in df_shapes:
                cols = v + [k]
                expected_shape = reduce(lambda x, y: x + y, df_shapes[cols])
                itemsize = datagram_df[k].dtype.itemsize
                req_mem = np.prod(expected_shape) * itemsize
                total_req_mem += req_mem

    return total_req_mem


def _get_zarr_dgrams_size(zarr_datagrams) -> int:
    """
    Returns the size in bytes of the list of zarr
    datagrams.
    """

    size = 0
    for i in zarr_datagrams:
        size += sum([sys.getsizeof(val) for val in i.values()])

    return size


def should_use_swap(zarr_datagrams, dgram_zarr_vars, mem_mult: float = 0.3) -> bool:
    zdgrams_mem = _get_zarr_dgrams_size(zarr_datagrams)

    # Estimate expansion size
    pow_ang_df, complex_df = _extract_datagram_dfs(zarr_datagrams, dgram_zarr_vars)
    pow_ang_mem = get_req_mem(pow_ang_df, dgram_zarr_vars)
    complex_mem = get_req_mem(complex_df, dgram_zarr_vars)
    total_mem = pow_ang_mem + complex_mem

    # get statistics about system memory usage
    mem = psutil.virtual_memory()

    # approx. the amount of memory that will be used after expansion
    req_mem = mem.used - zdgrams_mem + total_mem

    return mem.total * mem_mult < req_mem
