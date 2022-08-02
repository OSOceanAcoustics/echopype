import pandas as pd
import numpy as np
import tempfile
import zarr
import more_itertools as miter
from typing import Tuple, List


class Parsed2Zarr:
    """
    This class contains functions that facilitate
    the writing of a parsed file to a zarr file.
    Additionally, it contains useful information,
    such as names of array groups and their paths.
    """

    def __init__(self):

        # temporary directory that will hold the zarr file
        # TODO: will this work well in the cloud?
        self.temp_zarr_dir = tempfile.TemporaryDirectory()

        # create zarr store and zarr group we want to write to
        self.zarr_file_name = self.temp_zarr_dir.name + '/temp.zarr'
        self.store = zarr.DirectoryStore(self.zarr_file_name)
        self.zarr_root = zarr.group(store=self.store, overwrite=True)

    def _close_store(self):
        """properly closes zarr store"""

        # consolidate metadata and close zarr store
        zarr.consolidate_metadata(self.store)
        self.store.close()

    @staticmethod
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

    @staticmethod
    def set_multi_index(pd_series: pd.Series, unique_dims: List[pd.Index]) -> pd.Series:
        """
        Sets a multi-index from the product of the unique
        dimension values on a series and then
        returns it.

        Parameters
        ----------
        pd_series : pd.Series
            Series that needs its multi-index modified.
        unique_dims : List[pd.Index]
            List where the elements are the unique values
            of the index.

        Notes
        -----
        By setting the multiindex, this method fills (or pads)
        missing dimension values.
        """

        multi_index = pd.MultiIndex.from_product(unique_dims)

        # set product multi-index i.e. a preliminary padding of the df
        series_prod = pd_series.reindex(multi_index, fill_value=np.nan)

        return series_prod

    @staticmethod
    def get_col_info(pd_series: pd.Series, time_name: str,
                     is_array: bool, max_mb: int) -> Tuple[int, list]:
        """
        Provides the maximum number of times needed to
        fill at most `max_mb` MB  of memory and the
        shape of each chunk.

        Parameters
        ----------
        pd_series : pd.Series
            Series representing a column of the datagram df
        time_name : str
            The name of the index corresponding to time
        is_array : bool
            Specifies if we are working with a column that
            has arrays
        max_mb : int
            Maximum MB allowed for each chunk

        Returns
        -------
        max_num_times : int
            The number of times needed to fill at most
            `max_mb` MB  of memory.
        chunk_shape : list
            The shape of the chunk.

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

        multi_ind_names = list(pd_series.index.names)

        if len(multi_ind_names) > 2:
            raise NotImplementedError("series contains more than 2 indices!")

        multi_ind_names.remove(time_name)  # allows us to infer the other index name

        # get maximum dimension of column element
        if is_array:
            # TODO: this only works for 1D arrays, generalize it
            max_dim = np.max(pd_series.apply(lambda x: x.shape[0] if isinstance(x, np.ndarray) else 0))
        else:
            max_dim = 1

        # bytes required to hold one element of the column
        # TODO: this assumes we are holding floats (the 8 value), generalize it
        elem_bytes = max_dim * 8

        # the number of unique elements in the second index
        index_2_name = multi_ind_names[0]
        num_index_2 = len(pd_series.index.unique(index_2_name))

        bytes_per_time = num_index_2 * elem_bytes

        mb_per_time = bytes_per_time / 1e6

        # The maximum number of times needed to fill at most `max_mb` MB of memory
        max_num_times = max_mb // mb_per_time

        # create form of chunk shape
        if max_dim != 1:
            chunk_shape = [None, num_index_2, max_dim]
        else:
            chunk_shape = [None, num_index_2]

        return max_num_times, chunk_shape

    @staticmethod
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

    def write_chunks(self, pd_series: pd.Series, zarr_grp: zarr.group,
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
        np_chunk = self.get_np_chunk(series_chunk, chunk_shape, nan_array)

        # create array in zarr_grp using initial chunk
        full_array = zarr_grp.array(name=pd_series.name,
                                    data=np_chunk,
                                    chunks=zarr_chunk_shape,
                                    dtype='f8', fill_value='NaN')

        # append each chunk to full_array
        for i, chunk in enumerate(chunks[1:], start=1):
            series_chunk = pd_series.loc[chunk]
            chunk_shape[0] = chunk_len[i]
            np_chunk = self.get_np_chunk(series_chunk, chunk_shape, nan_array)
            full_array.append(np_chunk)

    def write_df_column(self, pd_series: pd.Series, zarr_grp: zarr.group,
                        is_array: bool, unique_time_ind: pd.Index,
                        max_mb: int = 100) -> None:
        """
        Obtains the appropriate information needed
        to determine the chunks of a column and
        then calls the function that writes a
        column to a zarr array.

        Parameters
        ----------
        pd_series: pd.Series
            Series with product multi-index and elements that
            are either an array or none of the elements are arrays.
        zarr_grp: zarr.group
            Zarr group that we should write the zarr array to
        is_array : bool
            True if ``pd_series`` is such that the elements of every
            column are arrays, False otherwise
        unique_time_ind : pd.Index
            The unique time index values of ``pd_series``
        max_mb : int
            Maximum MB allowed for each chunk

        Notes
        -----
        This assumes that our pd_series has at most 2 indices.
        """

        # TODO: this function and subsequent ones may not be good enough
        #  for multiple freq, test it! Specifically the shapes for the
        #  numpy and zarr arrays

        if len(pd_series.index.names) > 2:
            raise NotImplementedError("series contains more than 2 indices!")

        # For a column, obtain the maximum amount of times needed for
        # each chunk and the associated form for the shape of the chunks
        max_num_times, chunk_shape = self.get_col_info(pd_series, unique_time_ind.name,
                                                       is_array=is_array, max_mb=max_mb)

        # evenly chunk unique times so that the smallest and largest
        # chunk differ by at most 1 element
        chunks = list(miter.chunked_even(unique_time_ind,
                                         max_num_times))

        self.write_chunks(pd_series, zarr_grp, is_array, chunks, chunk_shape)

    def datagram_to_zarr(self, **kwargs) -> None:
        """
        Facilitates the conversion of a list of
        datagrams to a form that can be written
        to a zarr store.
        """

        pass
