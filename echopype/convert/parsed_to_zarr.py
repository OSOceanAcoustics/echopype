import pandas as pd
import numpy as np
import tempfile
import zarr
import more_itertools as miter
from typing import Tuple


def set_multi_index(df: pd.DataFrame, dims: list) -> pd.DataFrame:
    """
    Sets a multiindex on a copy of df and then
    returns it.

    Parameters
    ----------
    df : pd.Dataframe
        Dataframe with columns specified by ``dims``
    dims : list
        Names of the dimensions used to create the multiindex

    Notes
    -----
    By setting the multiindex, this method fills (or pads)
    missing dimension values.
    """

    # create multi index using the product of the unique dims
    unique_dims = [list(df[name].unique()) for name in dims]

    # set index to dims, so we can reindex with multiindex product
    df_multi = df.copy().set_index(dims)

    multi_index = pd.MultiIndex.from_product(unique_dims, names=dims)

    # set multiindex i.e. a preliminary padding of the df
    df_multi = df_multi.reindex(multi_index, fill_value=np.nan)

    return df_multi


def get_col_info(df: pd.DataFrame, time_name: str,
                 is_array: bool, num_mb: int) -> Tuple[dict, dict]:
    """
    Provides the maximum number of times needed to
    fill approximately `num_mb` Mb  of memory and the
    shape of each chunk.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each column has the same coordinates
    time_name : str
        The name of the index corresponding to time
    is_array : bool
        Specifies if we are working with a column that has arrays
    num_mb : int
        Approximately the amount of MB we want for each chunk

    Returns
    -------
    max_num_times : dict
        The key corresponds to the column name and the
        value is the number of times needed to fill
        approximately `num_mb` Mb  of memory.
    chunk_shape : dict
        The key corresponds to the column name and the
        value is the shape of the chunk. The shape of
        chunk is of the form: ``[None, num_index_2, max_dim]``
        if we have an array column and ``None`` if  we have
        a column that does not contain an array.

    Notes
    -----
    This function assumes that our df has at most 2 indices and
    ``time_name`` is one of them.

    For ``chunk_shape`` the first element corresponds to time
    and this element will be filled later, thus, it is set
    to None here.
    """

    multi_ind_names = list(df.index.names)

    if len(multi_ind_names) > 2:
        raise NotImplementedError("df contains more than 2 indices")

    multi_ind_names.remove(time_name)

    max_num_times = {}
    chunk_shape = {}
    for column in df:

        if is_array:
            # TODO: this only works for 1D arrays, generalize it
            max_dim = np.max(df[column].apply(lambda x: x.shape[0] if isinstance(x, np.ndarray) else 0))
        else:
            max_dim = 1

        # bytes required to hold one element of the column
        # TODO: this assumes we are holding floats (the 8 value), generalize it
        elem_bytes = max_dim * 8

        # the number of elements required to fill approximately `num_mb` MB  of memory
        num_elements = int(num_mb) * int(1e6 // elem_bytes)

        if multi_ind_names:

            index_2_name = multi_ind_names[0]
            # the number of unique elements in the second index
            num_index_2 = len(df[column].index.unique(index_2_name))
        else:
            num_index_2 = 1

        # The maximum number of times needed to fill approximately `num_mb` Mb  of memory
        max_num_times[column] = num_elements // num_index_2

        if max_dim != 1:
            chunk_shape[column] = [None, num_index_2, max_dim]
        else:
            chunk_shape[column] = None

    return max_num_times, chunk_shape


def get_np_chunk(series_chunk: pd.Series,
                 chunk_shape: list,
                 nan_array: np.array):

    # TODO: need to pad range_sample here too, if it is the same size as `nan_array`

    np_chunk = np.concatenate([elm if isinstance(elm, np.ndarray)
                               else nan_array for elm in series_chunk.to_list()],
                              axis=0)

    np_chunk = np_chunk.reshape(chunk_shape)

    return np_chunk


def write_chunks(pd_series, zarr_grp, chunks: list, chunk_shape: list):
    """
    pd_series -- pandas series representing a column of the datagram df
    num_chan -- number of unique channels
    size_elem -- size of element for the range_sample dimension
    nan_array -- an array filled with NaNs with the same size as the number of bins
    max_time_chunk -- the maximum number of indices of time for each chunk
    """

    if chunk_shape:
        # nan array used in padding of elements
        nan_array = np.empty(chunk_shape[2], dtype=np.float64)
        nan_array[:] = np.nan
    else:
        nan_array = None

    # obtain the number of times for each chunk
    chunk_len = [len(i) for i in chunks]

    max_chunk_len = max(chunk_len)

    if chunk_shape:
        zarr_chunk_shape = chunk_shape
        zarr_chunk_shape[0] = max_chunk_len
    else:
        zarr_chunk_shape = [max_chunk_len]

    # write initial chunk to the Zarr
    series_chunk = pd_series.loc[chunks[0]]

    if chunk_shape:
        chunk_shape[0] = chunk_len[0]
    else:
        chunk_shape = chunk_len[0]
    np_chunk = get_np_chunk(series_chunk, chunk_shape, nan_array)
    full_array = zarr_grp.array(name=pd_series.name,
                                data=np_chunk,
                                chunks=zarr_chunk_shape,
                                dtype='f8', fill_value='NaN')

    # append each chunk to full_array
    for i, chunk in enumerate(chunks[1:], start=1):

        series_chunk = pd_series.loc[chunk]
        if chunk_shape:
            chunk_shape[0] = chunk_len[i]
        else:
            chunk_shape = chunk_len[0]
        np_chunk = get_np_chunk(series_chunk, chunk_shape, nan_array)
        full_array.append(np_chunk)


def write_array_cols(df, zarr_grp, time_name: str = "timestamp", num_mb: int = 100):
    """
    This assumes that our df has at most 2 indices and
    ``time_name`` is one of them.

    """

    max_num_times, chunk_shape = get_col_info(df, time_name, is_array=True, num_mb=num_mb)

    unique_times = df.index.get_level_values(time_name).unique()

    for column in df:

        # evenly chunk unique times so that the smallest and largest
        # chunk differ by at most 1 element
        chunks = list(miter.chunked_even(unique_times,
                                         max_num_times[column]))

        write_chunks(df[column], zarr_grp, chunks, chunk_shape[column])


def write_df_to_zarr(df, array_grp, num_mb, time_name: str = "timestamp"):
    """
    This assumes that our df has at most 2 indices and
    ``time_name`` is one of them.

    """

    # TODO: this function may not be good enough for multiple freq, test it!
    #  Specifically the shapes for the numpy and zarr arrays

    # see if there is an array in any element of the column
    is_array = {}
    for column in df:
        is_array[column] = np.any(df[column].apply(lambda x: isinstance(x, np.ndarray)))

    array_cols = [key for key, val in is_array.items() if val]
    write_array_cols(df[array_cols], array_grp, time_name=time_name, num_mb=num_mb)

    # non_array_cols = [key for key, val in is_array.items() if not val]
    # write_array_cols(df[non_array_cols], array_grp, time_name=time_name, num_mb=num_mb)


def datagram_to_zarr(zarr_dgrams: list,
                     zarr_vars: dict,
                     temp_dir: tempfile.TemporaryDirectory,
                     num_mb: int):
    """
    Facilitates the conversion of a list of
    datagrams to a form that can be written
    to a zarr store.

    Parameters
    ----------
    zarr_dgrams : list
        A list of datagrams where each datagram contains
        at least one variable that should be written to
        a zarr file and any associated dimensions.
    zarr_vars : dict
        A dictionary where the keys represent the variable
        that should be written to a zarr file and the values
        are a list of the variable's dimensions.
    temp_dir: tempfile.TemporaryDirectory
        Temporary directory that will hold the Zarr Store
    num_mb : int
        The number of Mb to use for each chunk
    """

    # create zarr store and array_group
    zarr_file_name = temp_dir.name + '/temp.zarr'
    store = zarr.DirectoryStore(zarr_file_name)
    root = zarr.group(store=store, overwrite=True)
    array_grp = root.create_group('All_Arrays')

    datagram_df = pd.DataFrame.from_dict(zarr_dgrams)
    unique_dims = map(list, set(map(tuple, zarr_vars.values())))

    for dims in unique_dims:

        # get all variables with dimensions dims
        var_names = [key for key, val in zarr_vars.items() if val == dims]

        # columns needed to compute df_multi
        req_cols = var_names + dims

        df_multi = set_multi_index(datagram_df[req_cols], dims)
        write_df_to_zarr(df_multi, array_grp, num_mb, time_name="timestamp")  # TODO: change time_name, generalize

    # close zarr store
    zarr.consolidate_metadata(store)
    store.close()
