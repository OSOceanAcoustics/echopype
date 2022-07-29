import pandas as pd
import numpy as np
import tempfile
import zarr
import more_itertools as miter
from typing import Tuple


def unique_second_index(df: pd.DataFrame) -> None:
    """
    Raises an error if the second index has
    repeated values. All routines assume that
    the second index does not have repeated
    values.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with two indices.
    """

    for time in df.index.get_level_values(0).unique():
        val, cnts = np.unique(df.loc[time].index.get_level_values(0), return_counts=True)

        if max(cnts) > 1:
            raise NotImplementedError("write_df_to_zarr requires a unique second index.")


def set_multi_index(df: pd.DataFrame, dims: list) -> pd.DataFrame:
    """
    Sets a multiindex on a copy of df and then
    returns it.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns specified by ``dims``
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
                 is_array: bool, max_mb: int) -> Tuple[dict, dict]:
    """
    Provides the maximum number of times needed to
    fill at most `max_mb` MB  of memory and the
    shape of each chunk.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each column has the same coordinates
    time_name : str
        The name of the index corresponding to time
    is_array : bool
        Specifies if we are working with a column that has arrays
    max_mb : int
        Maximum MB allowed for each chunk

    Returns
    -------
    max_num_times : dict
        The key corresponds to the column name and the
        value is the number of times needed to fill
        at most `max_mb` MB  of memory.
    chunk_shape : dict
        The key corresponds to the column name and the
        value is the shape of the chunk.

    Notes
    -----
    This function assumes that our df has 2 indices and
    ``time_name`` is one of them.

    For ``chunk_shape`` the first element corresponds to time
    and this element will be filled later, thus, it is set
    to None here. The shape of chunk is of the form:
    ``[None, num_index_2, max_dim]`` if we have an array column
    and ``[None, num_index_2]`` if  we have a column that does
    not contain an array.
    """

    multi_ind_names = list(df.index.names)

    if len(multi_ind_names) > 2:
        raise NotImplementedError("df contains more than 2 indices!")

    multi_ind_names.remove(time_name)  # allows us to infer the other index name

    max_num_times = {}
    chunk_shape = {}
    for column in df:

        # get maximum dimension of column element
        if is_array:
            # TODO: this only works for 1D arrays, generalize it
            max_dim = np.max(df[column].apply(lambda x: x.shape[0] if isinstance(x, np.ndarray) else 0))
        else:
            max_dim = 1

        # bytes required to hold one element of the column
        # TODO: this assumes we are holding floats (the 8 value), generalize it
        elem_bytes = max_dim * 8

        # the number of unique elements in the second index
        index_2_name = multi_ind_names[0]
        num_index_2 = len(df[column].index.unique(index_2_name))

        bytes_per_time = num_index_2 * elem_bytes

        mb_per_time = bytes_per_time / 1e6

        # The maximum number of times needed to fill at most `max_mb` MB of memory
        max_num_times[column] = max_mb // mb_per_time

        # create form of chunk shape
        if max_dim != 1:
            chunk_shape[column] = [None, num_index_2, max_dim]
        else:
            chunk_shape[column] = [None, num_index_2]

    return max_num_times, chunk_shape


def get_np_chunk(series_chunk: pd.Series, chunk_shape: list,
                 nan_array: np.ndarray) -> np.ndarray:
    """
    Manipulates the ``series_chunk`` values into the
    correct shape that can then be written to a
    zarr array.

    Parameters
    ----------
    series_chunk : pd.Series
        A chunk of the dataframe column
    chunk_shape : list
        Specifies what shape the numpy chunk
        should be reshaped to
    nan_array : np.ndarray
        An array filled with NaNs that has the
        maximum length of a column's element.
        This value is used to pad empty elements.

    Returns
    -------
    np_chunk : np.ndarray
        Final form of series_chunk that can be
        written to a zarr array
    """

    if isinstance(nan_array, np.ndarray):

        # appropriately pad elements of series_chunk, if needed
        padded_elements = []
        for elm in series_chunk.to_list():

            if isinstance(elm, np.ndarray):

                # obtain the slice for the nan array
                nan_slice = slice(elm.shape[0], None)

                # pad elm array so its of size nan_array
                padded_array = np.concatenate([elm, nan_array[nan_slice]],
                                              axis=0, dtype=np.float64)

                padded_elements.append(padded_array)

            else:
                padded_elements.append(nan_array)

        np_chunk = np.concatenate(padded_elements, axis=0, dtype=np.float64)
        np_chunk = np_chunk.reshape(chunk_shape)

    else:
        np_chunk = series_chunk.to_numpy().reshape(chunk_shape)

    return np_chunk


def write_chunks(pd_series: pd.Series, zarr_grp: zarr.group,
                 is_array: bool, chunks: list,
                 chunk_shape: list) -> None:
    """
    Writes ``pd_series`` to ``zarr_grp`` as a zarr array
    with name ``pd_series.name``, using the specified chunks.

    Parameters
    ----------
    pd_series : pd:Series
        Series representing a column of the datagram df
    zarr_grp: zarr.group
        Zarr group that we should write the zarr array to
    is_array : bool
        True if ``pd_series`` has elements that are arrays,
        False otherwise
    chunks: list
        A list where each element corresponds to a list of
        index values that should be chosen for the chunk.
        For example, if we are chunking along time, ``chunks``
        would have the form:
        ``[['2004-09-09 16:19:06.059000', ..., '2004-09-09 16:19:06.746000'],
           ['2004-09-09 16:19:07.434000', ..., '2004-09-09 16:19:08.121000']]``.
    chunk_shape: list
        A list where each element specifies the shape of the
        zarr chunk for a given element of ``chunks``
    """

    if is_array:
        # nan array used in padding of elements
        nan_array = np.empty(chunk_shape[2], dtype=np.float64)
        nan_array[:] = np.nan
    else:
        nan_array = np.empty(1, dtype=np.float64)

    # obtain the number of times for each chunk
    chunk_len = [len(i) for i in chunks]

    max_chunk_len = max(chunk_len)

    zarr_chunk_shape = chunk_shape
    zarr_chunk_shape[0] = max_chunk_len

    # obtain initial chunk in the proper form
    series_chunk = pd_series.loc[chunks[0]]
    chunk_shape[0] = chunk_len[0]
    np_chunk = get_np_chunk(series_chunk, chunk_shape, nan_array)

    # create array in zarr_grp using initial chunk
    full_array = zarr_grp.array(name=pd_series.name,
                                data=np_chunk,
                                chunks=zarr_chunk_shape,
                                dtype='f8', fill_value='NaN')

    # append each chunk to full_array
    for i, chunk in enumerate(chunks[1:], start=1):
        series_chunk = pd_series.loc[chunk]
        chunk_shape[0] = chunk_len[i]
        np_chunk = get_np_chunk(series_chunk, chunk_shape, nan_array)
        full_array.append(np_chunk)


def write_df_cols(df: pd.DataFrame, zarr_grp: zarr.group,
                  is_array: bool, time_name: str = "timestamp",
                  max_mb: int = 100) -> None:
    """
    Obtains the appropriate information needed
    to determine the chunks of a column and
    then calls the function that writes a
    column to a zarr array, for each column in ``df``.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame where each column element is an array or
        none of the column elements are arrays.
    zarr_grp: zarr.group
        Zarr group that we should write the zarr array to
    is_array : bool
        True if ``df`` is such that the elements of every
        column are arrays, False otherwise
    time_name: str
        The name of the time index for ``df``
    max_mb : int
        Maximum MB allowed for each chunk

    Notes
    -----
    This assumes that our df has at most 2 indices and
    ``time_name`` is one of them.
    """

    # TODO: this function and subsequent ones may not be good enough
    #  for multiple freq, test it! Specifically the shapes for the
    #  numpy and zarr arrays

    if len(df.index.names) > 2:
        raise NotImplementedError("df contains more than 2 indices!")

    # For each column, obtain the maximum amount of times needed for
    # each chunk and the associated form for the shape of the chunks
    max_num_times, chunk_shape = get_col_info(df, time_name, is_array=is_array, max_mb=max_mb)

    unique_times = df.index.get_level_values(time_name).unique()

    # write each column of df to a zarr array
    for column in df:
        # evenly chunk unique times so that the smallest and largest
        # chunk differ by at most 1 element
        chunks = list(miter.chunked_even(unique_times,
                                         max_num_times[column]))

        write_chunks(df[column], zarr_grp, is_array, chunks, chunk_shape[column])


def write_df_to_zarr(df: pd.DataFrame, zarr_grp: zarr.group,
                     time_name: str = "timestamp", max_mb: int = 100) -> None:
    """
    Splits ``df`` into a DataFrame where all
    columns have elements that are arrays and
    a DataFrame where none of the columns have
    elements that are arrays, then calls a
    function that initiates the writing of these
    Dataframe columns to a zarr group.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing columns we want to write
        to a zarr group
    zarr_grp: zarr.group
        Zarr group that we should write the zarr array to
    time_name: str
        The name of the time index for ``df``
    max_mb : int
        Maximum MB allowed for each chunk
    """

    # see if there is an array in any element of the column
    is_array = {}
    for column in df:
        is_array[column] = np.any(df[column].apply(lambda x: isinstance(x, np.ndarray)))

    # write df columns whose elements are arrays
    array_cols = [key for key, val in is_array.items() if val]
    if array_cols:
        write_df_cols(df[array_cols], zarr_grp, is_array=True, time_name=time_name, max_mb=max_mb)

    # write df columns whose elements are not arrays
    non_array_cols = [key for key, val in is_array.items() if not val]
    if non_array_cols:
        write_df_cols(df[non_array_cols], zarr_grp, is_array=False, time_name=time_name, max_mb=max_mb)


def datagram_to_zarr(zarr_dgrams: list, zarr_vars: dict,
                     temp_dir: tempfile.TemporaryDirectory,
                     max_mb: int) -> None:
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
    max_mb : int
        Maximum MB allowed for each chunk

    Notes
    -----
    This function specifically writes chunks along the time
    index.

    The dimensions provided in ``zarr_vars`` must have the
    time dimension as the first element.

    The chunking routine evenly distributes the times such
    that each chunk differs by at most one time. This makes
    it so that the memory required for each chunk is approximately
    the same.
    """

    # create zarr store and zarr group we want to write to
    zarr_file_name = temp_dir.name + '/temp.zarr'
    store = zarr.DirectoryStore(zarr_file_name)
    array_grp = zarr.group(store=store, overwrite=True)

    datagram_df = pd.DataFrame.from_dict(zarr_dgrams)

    unique_dims = map(list, set(map(tuple, zarr_vars.values())))

    # write groups of variables with the same dimensions to zarr
    for dims in unique_dims:
        # get all variables with dimensions dims
        var_names = [key for key, val in zarr_vars.items() if val == dims]

        # columns needed to compute df_multi
        req_cols = var_names + dims

        df_multi = set_multi_index(datagram_df[req_cols], dims)

        # check to make sure the second index is unique
        unique_second_index(df_multi)

        write_df_to_zarr(df_multi, array_grp, time_name=dims[0], max_mb=max_mb)

    # consolidate metadata and close zarr store
    zarr.consolidate_metadata(store)
    store.close()
