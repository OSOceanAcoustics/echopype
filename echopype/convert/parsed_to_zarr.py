import sys
import tempfile
from typing import List, Tuple, Union

import more_itertools as miter
import numpy as np
import pandas as pd
import zarr


class Parsed2Zarr:
    """
    This class contains functions that facilitate
    the writing of a parsed file to a zarr file.
    Additionally, it contains useful information,
    such as names of array groups and their paths.
    """

    def __init__(self, parser_obj):

        self.temp_zarr_dir = None
        self.zarr_file_name = None
        self.store = None
        self.zarr_root = None
        self.parser_obj = parser_obj  # parser object ParseEK60/ParseEK80/etc.

    def _create_zarr_info(self):
        """
        Creates the temporary directory for zarr
        storage, zarr file name, zarr store, and
        the root group of the zarr store.
        """

        # temporary directory that will hold the zarr file
        # TODO: will this work well in the cloud?
        self.temp_zarr_dir = tempfile.TemporaryDirectory()

        # create zarr store and zarr group we want to write to
        self.zarr_file_name = self.temp_zarr_dir.name + "/temp.zarr"
        self.store = zarr.DirectoryStore(self.zarr_file_name)
        self.zarr_root = zarr.group(store=self.store, overwrite=True)

    def _close_store(self):
        """properly closes zarr store"""

        # consolidate metadata and close zarr store
        zarr.consolidate_metadata(self.store)
        self.store.close()

    @staticmethod
    def set_multi_index(
        pd_obj: Union[pd.Series, pd.DataFrame], unique_dims: List[pd.Index]
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Sets a multi-index from the product of the unique
        dimension values on a series and then
        returns it.

        Parameters
        ----------
        pd_obj : Union[pd.Series, pd.DataFrame]
            Series or DataFrame that needs its multi-index modified.
        unique_dims : List[pd.Index]
            List where the elements are the unique values
            of the index.

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            ``pd_obj`` with product multi-index

        Notes
        -----
        By setting the multiindex, this method fills (or pads)
        missing dimension values.
        """

        multi_index = pd.MultiIndex.from_product(unique_dims)

        # set product multi-index i.e. a preliminary padding of the df
        return pd_obj.reindex(multi_index, fill_value=np.nan)

    @staticmethod
    def get_max_elem_shape(pd_series: pd.Series) -> np.ndarray:
        """
        Returns the maximum element shape for a
        Series that has array elements

        Parameters
        ----------
        pd_series: pd.Series
            Series with array elements

        Returns
        -------
        np.ndarray
            The maximum element shape
        """

        all_shapes = pd_series.apply(
            lambda x: np.array(x.shape) if isinstance(x, np.ndarray) else None
        ).dropna()

        all_dims = np.vstack(all_shapes.to_list())

        return all_dims.max(axis=0)

    def get_col_info(
        self, pd_series: pd.Series, time_name: str, is_array: bool, max_mb: int
    ) -> Tuple[int, list]:
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
        ``[None, num_index_2, max_element_shape]`` if we have an
        array column and ``[None, num_index_2]`` if  we have a
        column that does not contain an array.
        """

        multi_ind_names = list(pd_series.index.names)

        if len(multi_ind_names) > 2:
            raise NotImplementedError("series contains more than 2 indices!")

        multi_ind_names.remove(time_name)  # allows us to infer the other index name

        # get maximum dimension of column element
        if is_array:
            max_element_shape = self.get_max_elem_shape(pd_series)
        else:
            max_element_shape = 1

        # bytes required to hold one element of the column
        # TODO: this assumes we are holding floats (the 8 value), generalize it
        elem_bytes = max_element_shape.prod(axis=0) * 8

        # the number of unique elements in the second index
        index_2_name = multi_ind_names[0]
        num_index_2 = len(pd_series.index.unique(index_2_name))

        bytes_per_time = num_index_2 * elem_bytes

        mb_per_time = bytes_per_time / 1e6

        # The maximum number of times needed to fill at most `max_mb` MB of memory
        max_num_times = max_mb // mb_per_time

        # create form of chunk shape
        if isinstance(max_element_shape, np.ndarray):
            chunk_shape = [None, num_index_2, max_element_shape]
        else:
            chunk_shape = [None, num_index_2]

        return max_num_times, chunk_shape

    @staticmethod
    def get_np_chunk(
        series_chunk: pd.Series, chunk_shape: list, nan_array: np.ndarray
    ) -> np.ndarray:
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

                    # TODO: ideally this would take place in the parser, do this
                    elm = elm.astype(np.float64)

                    # amount of padding to add to each axis
                    padding_amount = chunk_shape[2] - elm.shape

                    # create np.pad pad_width
                    pad_width = [(0, i) for i in padding_amount]

                    padded_array = np.pad(elm, pad_width, "constant", constant_values=np.nan)

                    padded_elements.append(padded_array)

                else:
                    padded_elements.append(nan_array)

            np_chunk = np.concatenate(padded_elements, axis=0, dtype=np.float64)

            # reshape chunk to the appropriate size
            full_shape = chunk_shape[:2] + list(chunk_shape[2])
            np_chunk = np_chunk.reshape(full_shape)

        else:
            np_chunk = series_chunk.to_numpy().reshape(chunk_shape)

        return np_chunk

    def write_chunks(
        self,
        pd_series: pd.Series,
        zarr_grp: zarr.group,
        is_array: bool,
        chunks: list,
        chunk_shape: list,
    ) -> None:
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

        zarr_chunk_shape = chunk_shape[:2] + list(chunk_shape[2])
        zarr_chunk_shape[0] = max_chunk_len

        # obtain initial chunk in the proper form
        series_chunk = pd_series.loc[chunks[0]]
        chunk_shape[0] = chunk_len[0]
        np_chunk = self.get_np_chunk(series_chunk, chunk_shape, nan_array)

        # create array in zarr_grp using initial chunk
        full_array = zarr_grp.array(
            name=pd_series.name,
            data=np_chunk,
            chunks=zarr_chunk_shape,
            dtype="f8",
            fill_value="NaN",
        )

        # append each chunk to full_array
        for i, chunk in enumerate(chunks[1:], start=1):
            series_chunk = pd_series.loc[chunk]
            chunk_shape[0] = chunk_len[i]
            np_chunk = self.get_np_chunk(series_chunk, chunk_shape, nan_array)
            full_array.append(np_chunk)

    def write_df_column(
        self,
        pd_series: pd.Series,
        zarr_grp: zarr.group,
        is_array: bool,
        unique_time_ind: pd.Index,
        max_mb: int = 100,
    ) -> None:
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

        if len(pd_series.index.names) > 2:
            raise NotImplementedError("series contains more than 2 indices!")

        # For a column, obtain the maximum amount of times needed for
        # each chunk and the associated form for the shape of the chunks
        max_num_times, chunk_shape = self.get_col_info(
            pd_series, unique_time_ind.name, is_array=is_array, max_mb=max_mb
        )

        # evenly chunk unique times so that the smallest and largest
        # chunk differ by at most 1 element
        chunks = list(miter.chunked_even(unique_time_ind, max_num_times))

        self.write_chunks(pd_series, zarr_grp, is_array, chunks, chunk_shape)

    def _get_zarr_dgrams_size(self) -> int:
        """
        Returns the size in bytes of the list of zarr
        datagrams.
        """

        size = 0
        for i in self.parser_obj.zarr_datagrams:

            size += sum([sys.getsizeof(val) for key, val in i.items()])

        return size

    def array_series_bytes(self, pd_series: pd.Series, n_rows: int) -> int:
        """
        Determines the amount of bytes required for a
        series with array elements, for ``n_rows``.

        Parameters
        ----------
        pd_series: pd.Series
            Series with array elements
        n_rows: int
            The number of rows with array elements

        Returns
        -------
        The amount of bytes required to hold data
        """

        # the number of bytes required to hold 1 element of series
        # Note: this assumes that we are holding floats
        pow_bytes = self.get_max_elem_shape(pd_series).prod(axis=0) * 8

        # total memory required for series data
        return n_rows * pow_bytes

    def write_to_zarr(self, **kwargs) -> None:
        """
        Determines if the zarr data provided will expand
        into a form that is larger than a percentage of
        the total physical RAM.
        """

        pass

    def datagram_to_zarr(self, **kwargs) -> None:
        """
        Facilitates the conversion of a list of
        datagrams to a form that can be written
        to a zarr store.
        """

        pass
