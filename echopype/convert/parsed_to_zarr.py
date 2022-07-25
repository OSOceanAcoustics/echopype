import pandas as pd
import numpy as np
import tempfile
import zarr


def set_multi_index(df: pd.DataFrame, dims: list):
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


def get_np_chunk(df_chunk, time_chunk, num_chan, size_elem, nan_array):

    # TODO: need to pad range_sample here too, if is is not of size `size_elem`

    np_chunk = np.concatenate([elm if isinstance(elm, np.ndarray) else nan_array for elm in df_chunk.to_list()],
                              axis=0)

    np_chunk = np_chunk.reshape((time_chunk, num_chan, size_elem))

    return np_chunk


def write_chunks(pd_series, num_chan, size_elem, nan_array, time_chunk, zarr_grp):
    """
    pd_series -- pandas series representing a column of the datagram df
    num_chan -- number of unique channels
    size_elem -- size of element for the range_sample dimension
    nan_array -- an array filled with NaNs with the same size as the number of bins
    time_chunk -- the number of indices of time for each chunk


    """

    num_times = pd_series.index.get_level_values(0).unique().size

    num_chunks = (num_times // time_chunk)
    rem = num_times % time_chunk

    # write initial chunk to the Zarr
    df_chunk = pd_series.iloc[slice(0, num_chan * time_chunk)]
    np_chunk = get_np_chunk(df_chunk, time_chunk, num_chan, size_elem, nan_array)

    full_array = zarr_grp.array(name=pd_series.name,
                                data=np_chunk,
                                chunks=(time_chunk, num_chan, size_elem),  # TODO: round-robin -> change chunks
                                dtype='f8', fill_value='NaN')

    for i in range(1, num_chunks):
        i_start = i * time_chunk * num_chan
        i_end = (i + 1) * time_chunk * num_chan

        df_chunk = pd_series.iloc[slice(i_start, i_end)]

        np_chunk = get_np_chunk(df_chunk, time_chunk, num_chan, size_elem, nan_array)
        full_array.append(np_chunk)

    if num_chunks * time_chunk * num_chan < pd_series.shape[0]:
        # write the remaining chunk to zarr
        # TODO: there is a better way to do this (round robin)
        last_index = num_chunks * time_chunk * num_chan

        df_chunk = pd_series.iloc[slice(last_index, None)]
        np_chunk = get_np_chunk(df_chunk, rem, num_chan, size_elem, nan_array)
        full_array.append(np_chunk)


def array_col_to_zarr(pd_series, array_grp, num_mb):
    """
    This function specifically sets those columns
    in df that are arrays and have dims (timestamp, channel)
    in the future we can make this more general
    """

    max_dim = np.max(pd_series.apply(lambda x: x.shape[0] if isinstance(x, np.ndarray) else 0))

    # bytes required to hold one element of the column
    # TODO: this assumes we are holding floats, can generalize in the future
    elem_bytes = max_dim * 8

    # the number of elements required to fill approximately `num_mb` Mb  of memory
    num_elements = int(num_mb)*int(1e6 // elem_bytes)  # TODO: play around with this value

    num_chan = len(pd_series.index.unique('channel'))

    # The number of pings needed to fill approximately 1Mb of memory
    num_pings = num_elements // num_chan

    # nan array used in padding of elements
    nan_array = np.empty(max_dim, dtype=np.float64)
    nan_array[:] = np.nan

    # print(f"elem_bytes = {elem_bytes}")
    # print(f"num_elements = {num_elements}")
    # print(f"num_pings = {num_pings}")

    write_chunks(pd_series, num_chan, max_dim, nan_array, num_pings, array_grp)


def write_df_to_zarr(df, array_grp, num_mb):

    for column in df:

        pd_series = df[column]

        is_array = True  # TODO: create a function for this, check for np array and make sure it is 1D

        if is_array:
            # TODO: this may not be good enough for multiple freq
            if not (set(pd_series.index.names).difference({"timestamp", "channel"})):
                array_col_to_zarr(pd_series, array_grp, num_mb)
            else:
                raise NotImplementedError(f"variable arrays with dims {list(pd_series.index.names)} " +
                                          "have not been implemented yet.")


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
        write_df_to_zarr(df_multi, array_grp, num_mb)

    # close zarr store
    zarr.consolidate_metadata(store)
    store.close()

    return df_multi
