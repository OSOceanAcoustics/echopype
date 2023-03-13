from collections import defaultdict
from itertools import islice
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple

import dask
import dask.array
import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from dask.distributed import Lock

from ..utils.coding import COMPRESSION_SETTINGS, get_zarr_compression
from ..utils.io import env_indep_joinpath
from ..utils.prov import echopype_prov_attrs
from .api import open_converted
from .echodata import EchoData


class ZarrCombine:
    """
    A class that combines a list of EchoData objects by
    creating a Zarr store and appending each group's
    Dataset to the store.
    """

    def __init__(self):
        # all possible time dimensions
        self.possible_time_dims = {"time1", "time2", "time3", "ping_time"}

        # all possible dimensions that we will append to (mainly time dims)
        self.append_dims = {"filenames"}.union(self.possible_time_dims)

        # encodings associated with lazy loaded variables
        self.lazy_encodings = ["chunks", "preferred_chunks"]

        # defaultdict that holds every group's attributes
        self.group_attrs = defaultdict(list)

        # The sonar_model for the new combined EchoData object
        self.sonar_model = None

        # The maximum chunk length allowed for every append dimension
        # TODO: in the future we should investigate this value
        self.max_append_chunk_size = 1000

        # initialize variables created within class methods
        # descriptions of these variables can be found in _get_ds_info
        self.dims_df = None
        self.dims_sum = None
        self.dims_csum = None
        self.dims_max = None

    def _check_ascending_ds_times(self, ds_list: List[xr.Dataset], ed_name: str) -> None:
        """
        A minimal check that the first time value of each Dataset is less than
        the first time value of the subsequent Dataset. If each first time value
        is NaT, then this check is skipped.

        Parameters
        ----------
        ds_list: list of xr.Dataset
            List of Datasets to be combined
        ed_name: str
            The name of the ``EchoData`` group being combined
        """

        # get all time dimensions of the input Datasets
        ed_time_dim = set(ds_list[0].dims).intersection(self.possible_time_dims)

        for time in ed_time_dim:
            # gather the first time of each Dataset
            first_times = []
            for ds in ds_list:
                times = ds[time].values
                if isinstance(times, np.ndarray):
                    # store first time if we have an array
                    first_times.append(times[0])
                else:
                    # store first time if we have a single value
                    first_times.append(times)

            first_times = np.array(first_times)

            # skip check if all first times are NaT
            if not np.isnan(first_times).all():
                is_descending = (np.diff(first_times) < np.timedelta64(0, "ns")).any()

                if is_descending:
                    raise RuntimeError(
                        f"The coordinate {time} is not in ascending order for "
                        f"group {ed_name}, combine cannot be used!"
                    )

    @staticmethod
    def _compare_attrs(attr1: dict, attr2: dict) -> List[str]:
        """
        Compares two attribute dictionaries to ensure that they
        are acceptably identical.

        Parameters
        ----------
        attr1: dict
            Attributes from Dataset 1
        attr2: dict
            Attributes from Dataset 2

        Returns
        -------
        numpy_keys: list
            All keys that have numpy arrays as values

        Raises
        ------
        RuntimeError
            - If the keys are not the same
            - If the values are not identical
            - If the keys ``date_created`` or ``conversion_time``
            do not have the same types

        Notes
        -----
        For the keys ``date_created``, ``conversion_time`` the values
        are not required to be identical, rather their type must be identical.
        """

        # make sure all keys are identical (this should never be triggered)
        if attr1.keys() != attr2.keys():
            raise RuntimeError(
                "The attribute keys amongst the ds lists are not the same, combine cannot be used!"
            )

        # make sure that all values are identical
        numpy_keys = []
        for key in attr1.keys():
            if isinstance(attr1[key], np.ndarray):
                numpy_keys.append(key)

                if not np.allclose(attr1[key], attr2[key], rtol=1e-12, atol=1e-12, equal_nan=True):
                    raise RuntimeError(
                        f"The attribute {key}'s value amongst the ds lists are not the "
                        f"same, combine cannot be used!"
                    )
            elif key in ["date_created", "conversion_time"]:
                if not isinstance(attr1[key], type(attr2[key])):
                    raise RuntimeError(
                        f"The attribute {key}'s type amongst the ds lists "
                        f"are not the same, combine cannot be used!"
                    )

            else:
                if attr1[key] != attr2[key]:
                    raise RuntimeError(
                        f"The attribute {key}'s value amongst the ds lists are not the "
                        f"same, combine cannot be used!"
                    )

        return numpy_keys

    def _get_ds_info(self, ds_list: List[xr.Dataset], ed_name: str) -> None:
        """
        Constructs useful dictionaries that contain information
        about the dimensions of the Dataset. Additionally, collects
        the attributes from each Dataset in ``ds_list`` and saves
        this group specific information to the class variable
        ``group_attrs``.

        Parameters
        ----------
        ds_list: list of xr.Dataset
            The Datasets that will be combined
        ed_name: str
            The name of the EchoData group corresponding to the
            Datasets in ``ds_list``

        Notes
        -----
        This method creates the following class variables:
        dims_df: pd.DataFrame
            Dataframe with column as dim names, rows as the
            different Datasets, and values as the length of the dimension
        dims_sum: dict
            Keys as the dimension name and values as the corresponding
            sum of the lengths across all Datasets
        dims_csum: dict
            Keys as the dimension name and values as a dictionary of
            the corresponding cumulative sum of the lengths across
            all Datasets
        dims_max: dict
            Keys as the dimension name and values as the corresponding
            maximum length across all Datasets

        If attribute values are numpy arrays, then they will not be included
        in the ``self.group_attrs``. Instead, these values will only appear
        in the attributes of the combined ``EchoData`` object.
        """

        self._check_ascending_ds_times(ds_list, ed_name)

        # Dataframe with column as dim names and rows as the different Datasets
        self.dims_df = pd.DataFrame([ds.dims for ds in ds_list])

        # calculate useful information about the dimensions
        self.dims_sum = self.dims_df.sum(axis=0).to_dict()
        self.dims_csum = self.dims_df.cumsum(axis=0).to_dict()
        self.dims_max = self.dims_df.max(axis=0).to_dict()

        # format ed_name appropriately
        ed_name = ed_name.replace("-", "_").replace("/", "_").lower()

        if len(ds_list) == 1:
            # get numpy keys if we only have one Dataset
            numpy_keys = self._compare_attrs(ds_list[0].attrs, ds_list[0].attrs)
        else:
            # compare attributes and get numpy keys, if they exist
            for ind in range(len(ds_list) - 1):
                numpy_keys = self._compare_attrs(ds_list[ind].attrs, ds_list[ind + 1].attrs)

        # collect Dataset attributes
        for count, ds in enumerate(ds_list):
            # get reduced attributes that do not include numpy keys
            red_attrs = {key: val for key, val in ds.attrs.items() if key not in numpy_keys}

            if count == 0:
                self.group_attrs[ed_name + "_attr_key"].extend(red_attrs.keys())
            self.group_attrs[ed_name + "_attrs"].append(list(red_attrs.values()))

    def _get_temp_arr(self, dims: List[str], dtype: type) -> Tuple[type(dask.array), list]:
        """
        Constructs a temporary (or dummy) array representing a
        variable in its final combined form. This array will
        specify the shape, data type, and chunks of the final
        combined data.

        Parameters
        ----------
        dims: list of str
            A list of the dimension names
        dtype: type
            The data type of the variable

        Returns
        -------
        temp_arr: dask.array
            A temporary (or dummy) array representing a
            variable in its final combined form.
        chnk_shape: list of int
            The chunk shape used to construct ``temp_arr``

        Notes
        -----
        This array is never interacted with in a traditional sense.
        Its sole purpose is to construct metadata for the zarr store.
        """

        # Create the shape of the variable in its final combined form
        shape = [
            self.dims_sum[dim] if dim in self.append_dims else self.dims_max[dim] for dim in dims
        ]

        # Create the chunk shape of the variable
        # TODO: investigate if this is the best chunking
        chnk_shape = [
            min(self.dims_max[dim], self.max_append_chunk_size)
            if dim in self.append_dims
            else self.dims_max[dim]
            for dim in dims
        ]

        temp_arr = dask.array.zeros(shape=shape, dtype=dtype, chunks=chnk_shape)

        return temp_arr, chnk_shape

    def _get_encodings(self, name: str, val: xr.Variable, chnk_shape: List[int]) -> Dict[str, dict]:
        """
        Obtains the encodings for the variable ``name`` by including all
        encodings in ``val``, except those encodings that are specified by
        ``self.lazy_encodings``, such as ``chunks``  and ``preferred_chunks``
        here. Additionally, if a compressor is not found, a default compressor
        will be assigned.

        Parameters
        ----------
        name: str
            The name of the variable we are setting the encodings for
        val: xr.Variable
            The variable that contains the encodings we want to assign
            to ``name``
        chnk_shape: list of int
            The shape of the chunks for ``name`` (used in encodings)

        Returns
        -------
        var_encoding : dict
            All encodings associated with ``name``
        """

        # initialize dict that will hold all encodings for the variable name
        var_encoding = dict()

        # gather all encodings, except the lazy encodings
        var_encoding[name] = {
            key: encod for key, encod in val.encoding.items() if key not in self.lazy_encodings
        }

        # assign default compressor, if one does not exist
        if "compressor" not in var_encoding[name]:
            var_encoding[name].update(get_zarr_compression(val, COMPRESSION_SETTINGS["zarr"]))

        # set the chunk encoding
        # cast to int type just to be safe
        var_encoding[name]["chunks"] = [int(c) for c in chnk_shape]

        return var_encoding

    def _construct_lazy_ds_and_var_info(
        self, ds_model: xr.Dataset
    ) -> Tuple[xr.Dataset, List[str], Dict[str, dict]]:
        """
        Constructs a lazy Dataset representing the EchoData group
        Dataset in its final combined form. Additionally, collects
        all variable and dimension names that are constant across
        the Datasets to be combined, and collects the encodings for
        all variables and dimensions that will be written to the
        zarr store by regions

        Parameters
        ----------
        ds_model: xr.Dataset
            A Dataset that we will model our lazy Dataset after. In practice,
            this is the first element in the list of Datasets to be combined.

        Returns
        -------
        ds: xr.Dataset
            A lazy Dataset representing the EchoData group Dataset in
            its final combined form
        const_names: list of str
            The names of all variables and dimensions that are constant
            (with respect to chunking) across all Datasets to be combined
        encodings: dict
            The encodings for all variables and dimensions that will be
            written to the zarr store by regions

        Notes
        -----
        The sole purpose of the Dataset created is to construct metadata
        for the zarr store.
        """

        xr_vars_dict = dict()
        xr_coords_dict = dict()
        encodings = dict()
        const_names = []
        for name, val in ds_model.variables.items():
            # get all dimensions of val that are also append dimensions
            append_dims_in_val = set(val.dims).intersection(self.append_dims)

            if not append_dims_in_val:
                # collect the names of all constant variables/dimensions
                const_names.append(str(name))

            else:
                # create lazy DataArray representation
                temp_arr, chnk_shape = self._get_temp_arr(list(val.dims), val.dtype)

                if name not in ds_model.dims:
                    xr_vars_dict[name] = (val.dims, temp_arr, val.attrs)
                elif name in self.append_dims:
                    xr_coords_dict[name] = (val.dims, temp_arr, val.attrs)

                # add var encodings to the everything-in-one encodings dict
                encodings.update(self._get_encodings(str(name), val, chnk_shape))

        # construct lazy Dataset form
        ds = xr.Dataset(xr_vars_dict, coords=xr_coords_dict, attrs=ds_model.attrs)

        return ds, const_names, encodings

    def _get_region(self, ds_ind: int, ds_dims: Set[Hashable]) -> Dict[str, slice]:
        """
        Returns the region of the zarr file to write to. This region
        corresponds to the input set of dimensions that do not
        include append dimensions.

        Parameters
        ----------
        ds_ind: int
            The key of the values of ``self.dims_csum`` or index of
            ``self.dims_df`` to use for each dimension name
        ds_dims: set
            The names of the dimensions used in the region creation

        Returns
        -------
        region: dict
            Keys set as the dimension name and values as
            the slice of the zarr portion to write to
        """

        # get the initial region
        region = dict()
        for dim in ds_dims:
            if dim not in self.append_dims:
                region[dim] = slice(0, self.dims_df.loc[ds_ind][dim])

        return region

    @staticmethod
    def _uniform_chunks_as_np_array(array: np.ndarray, chunk_size: int) -> List[np.ndarray]:
        """
        Split ``array`` into chunks with size ``chunk_size``, where the
        last element in the split has length ``len(array) % chunk_size``.

        Parameters
        ----------
        array: np.ndarray
            Array to split up into chunks
        chunk_size: int
            The maximum chunk size

        Returns
        -------
        list of np.ndarray
            The chunked input ``array``

        Example
        -------
        >>> arr = np.array([1, 2, 3, 4, 5])
        >>> _uniform_chunks_as_np_array(arr, 2)
        [array([1, 2]), array([3, 4]), array([5])]
        """

        # get array iterable
        array_iter = iter(array)

        # construct chunks as an iterable of lists
        chunks_iter = iter(lambda: list(islice(array_iter, chunk_size)), list())

        # convert each element in the iterable to a numpy array
        return list(map(np.array, chunks_iter))

    def _get_chunk_dicts(self, dim: str) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Obtains dictionaries specifying the chunk index and the
        indices (with respect to the full combined length) that
        are contained in that chunk, for both the uniform and
        non-uniform chunks.

        Parameters
        ----------
        dim: str
            The name of the dimension to create the chunk dicts for

        Returns
        -------
        og_chunk_dict: dict
            The chunk dictionary corresponding to the original
            non-uniform chunks
        uniform_chunk_dict: dict
            The chunk dictionary corresponding to the uniform chunks
        """

        # an array specifying the indices of the final combined array
        x_no_chunk = np.arange(self.dims_sum[dim], dtype=np.int64)

        # get end indices for the non-uniform chunks
        csum_og_chunks = np.array(list(self.dims_csum[dim].values()))

        # obtain the indices of the final combined array that are in each non-uniform chunk
        og_chunk = np.split(x_no_chunk, csum_og_chunks)

        # construct a mapping between the non-uniform chunk and the indices
        og_chunk_dict = dict(zip(range(len(og_chunk)), og_chunk))

        # obtain the uniform chunk size
        # TODO: investigate if this if the best chunk size
        zarr_chunk_size = min(self.dims_max[dim], self.max_append_chunk_size)

        # get the indices of the final combined array that are in each uniform chunk
        uniform_chunk = self._uniform_chunks_as_np_array(x_no_chunk, int(zarr_chunk_size))

        # construct a mapping between the uniform chunk and the indices
        uniform_chunk_dict = dict(zip(range(len(uniform_chunk)), uniform_chunk))

        return og_chunk_dict, uniform_chunk_dict

    def _get_uniform_to_nonuniform_map(self, dim: str) -> Dict[int, dict]:
        """
        Constructs a uniform to non-uniform mapping of chunks
        for a dimension ``dim``.

        Parameters
        ----------
        dim: str
            The name of the dimension to create a mapping for

        Returns
        -------
        final_mapping: dict
            Uniform to non-uniform mapping where the keys are
            the chunk index in the uniform chunk and the values
            are dictionaries. The value dictionaries have keys
            which correspond to the index of the non-uniform chunk
            and the values are a tuple with the first element being
            a ``slice`` object for the non-uniform chunk values and
            the second element is a ``slice`` object for the region
            chunk values.
        """

        # obtains dictionaries specifying the indices contained in each chunk
        og_chunk_dict, uniform_chunk_dict = self._get_chunk_dicts(dim)

        # construct the uniform to non-uniform mapping
        final_mapping = defaultdict(dict)
        for u_key, u_val in uniform_chunk_dict.items():
            for og_key, og_val in og_chunk_dict.items():
                # find the intersection of uniform and non-uniform chunk indices
                intersect = np.intersect1d(u_val, og_val)

                if len(intersect) > 0:
                    # get min and max indices in intersect
                    min_val = intersect.min()
                    max_val = intersect.max()

                    # determine the start and end index for the og_val
                    start_og = np.argwhere(og_val == min_val)[0, 0]
                    end_og = np.argwhere(og_val == max_val)[0, 0] + 1

                    # determine the start and end index for the region
                    start_region = min_val
                    end_region = max_val + 1

                    # add non-uniform specific information to final mapping
                    final_mapping[u_key].update(
                        {og_key: (slice(start_og, end_og), slice(start_region, end_region))}
                    )

        return final_mapping

    @dask.delayed
    def write_to_file(
        self,
        ds_in: xr.Dataset,
        lock_name: str,
        zarr_path: str,
        zarr_group: str,
        region: Dict[str, slice],
        storage_options: Dict[str, Any] = {},
    ) -> None:
        """
        Constructs a delayed write of ``ds_in`` to the appropriate zarr
        store position using a unique lock name.

        Parameters
        ----------
        ds_in: xr.Dataset
            Dataset subset with only one append dimension containing
            variables with the append dimension in their dimensions
        lock_name: str
            A unique lock name for the chunk being written to
        zarr_path: str
            The full path of the final combined zarr store
        zarr_group: str
            The name of the group of the zarr store
            corresponding to the Datasets in ``ds_list``
        region: dict
            Keys set as the dimension name and values as
            the slice of the zarr portion to write to
        storage_options: dict
            Any additional parameters for the storage
            backend (ignored for local paths)
        """

        with Lock(lock_name):
            ds_in.to_zarr(
                zarr_path,
                group=zarr_group,
                region=region,
                compute=True,
                storage_options=storage_options,
                synchronizer=zarr.ThreadSynchronizer(),
                consolidated=False,
            )

            # TODO: put a check to make sure that the chunk has been written

    def _append_ds_list_to_zarr(
        self,
        zarr_path: str,
        ds_list: List[xr.Dataset],
        zarr_group: str,
        ed_name: str,
        storage_options: Dict[str, Any] = {},
    ) -> List[str]:
        """
        Creates a zarr store and then appends each Dataset
        in ``ds_list`` to it. The final result is a combined
        Dataset along the time dimensions.

        Parameters
        ----------
        zarr_path: str
            The full path of the final combined zarr store
        ds_list: list of xr.Dataset
            The Datasets that will be combined
        zarr_group: str
            The name of the group of the zarr store
            corresponding to the Datasets in ``ds_list``
        ed_name: str
            The name of the EchoData group corresponding to the
            Datasets in ``ds_list``
        storage_options: dict
            Any additional parameters for the storage
            backend (ignored for local paths)

        Returns
        -------
        const_names: list
            The names of all variables and dimensions that are constant
            (with respect to chunking) across all Datasets to be combined
        """

        self._get_ds_info(ds_list, ed_name)

        ds_lazy, const_names, encodings = self._construct_lazy_ds_and_var_info(ds_list[0])

        # create zarr file and all associated metadata (this is delayed)
        ds_lazy.to_zarr(
            zarr_path,
            compute=False,
            group=zarr_group,
            encoding=encodings,
            storage_options=storage_options,
            synchronizer=zarr.ThreadSynchronizer(),
            consolidated=False,
        )

        # get all dimensions in ds that are append dimensions
        ds_append_dims = set(ds_list[0].dims).intersection(self.append_dims)

        # collect delayed functions that write each non-constant variable
        # in ds_list to the zarr store
        delayed_to_zarr = []
        for dim in ds_append_dims:
            # collect all variables/coordinates that should be dropped
            drop_names = [
                var_name
                for var_name, var_val in ds_list[0].variables.items()
                if dim not in var_val.dims
            ]
            drop_names.append(dim)

            chunk_mapping = self._get_uniform_to_nonuniform_map(str(dim))

            for uniform_ind, non_uniform_dict in chunk_mapping.items():
                for ds_list_ind, dim_slice in non_uniform_dict.items():
                    # get ds containing only variables who have dim in their dims
                    cur_ds = ds_list[ds_list_ind]
                    # get all variable names from dataset
                    ds_keys = [k for k in cur_ds.variables.keys()]
                    # compare variable names from dataset with the first
                    # file's variable names and only grab the ones that exists
                    # in the current dataset
                    final_drop_names = [name for name in drop_names if name in ds_keys]
                    ds_drop = ds_list[ds_list_ind].drop_vars(final_drop_names)

                    # get xarray region for all dims, except dim
                    region = self._get_region(ds_list_ind, set(ds_drop.dims))

                    # get xarray region for dim
                    region[str(dim)] = dim_slice[1]

                    # select subset of dim corresponding to the region
                    ds_in = ds_drop.isel({dim: dim_slice[0]})

                    # construct the unique lock name for the uniform chunk
                    grp_name = zarr_group.replace("-", "_").replace("/", "_").lower()
                    lock_name = grp_name + "_" + str(dim) + "_" + str(uniform_ind)

                    # write the subset of each Dataset to a zarr file
                    delayed_to_zarr.append(
                        self.write_to_file(
                            ds_in,
                            lock_name,
                            zarr_path,
                            zarr_group,
                            region,
                            storage_options,
                        )
                    )

        # compute all delayed writes to the zarr store
        dask.compute(*delayed_to_zarr)

        return const_names

    def _append_const_to_zarr(
        self,
        const_vars: List[str],
        ds_list: List[xr.Dataset],
        zarr_path: str,
        zarr_group: str,
        storage_options: Dict[str, Any] = {},
    ) -> None:
        """
        Appends all constant (i.e. not chunked) variables and dimensions to the
        zarr group.

        Parameters
        ----------
        const_vars: list of str
            The names of all variables/dimensions that are not chunked
        ds_list: list of xr.Dataset
            The Datasets that will be combined
        zarr_path: str
            The full path of the final combined zarr store
        zarr_group: str
            The name of the group of the zarr store
            corresponding to the Datasets in ``ds_list``
        storage_options: dict
            Any additional parameters for the storage
            backend (ignored for local paths)

        Notes
        -----
        Those variables/dimensions that are in ``self.append_dims``
        should not be appended here.
        """

        # write constant vars to zarr using the first element of ds_list
        for var in const_vars:
            # make sure to choose the dataset with the largest size for variable
            if var in self.dims_df:
                ds_list_ind = int(self.dims_df[var].argmax())
            else:
                ds_list_ind = int(0)

            ds_list[ds_list_ind][[var]].to_zarr(
                zarr_path,
                group=zarr_group,
                mode="a",
                storage_options=storage_options,
                consolidated=False,
            )

    def _write_append_dims(
        self,
        ds_list: List[xr.Dataset],
        zarr_path: str,
        zarr_group: str,
        storage_options: Dict[str, Any] = {},
    ) -> None:
        """
        Sequentially writes each Dataset's append dimension in ``ds_list`` to
        the appropriate final combined zarr store.

        Parameters
        ----------
        ds_list: list of xr.Dataset
            The Datasets that will be combined
        zarr_path: str
            The full path of the final combined zarr store
        zarr_group: str
            The name of the group of the zarr store
            corresponding to the Datasets in ``ds_list``
        storage_options: dict
            Any additional parameters for the storage
            backend (ignored for local paths)
        """

        # get all dimensions in ds that are append dimensions
        ds_append_dims = set(ds_list[0].dims).intersection(self.append_dims)

        for dim in ds_append_dims:
            for count, ds in enumerate(ds_list):
                # obtain the appropriate region to write to
                if count == 0:
                    region = {str(dim): slice(0, self.dims_csum[dim][count])}
                else:
                    region = {
                        str(dim): slice(self.dims_csum[dim][count - 1], self.dims_csum[dim][count])
                    }

                ds[[dim]].to_zarr(
                    zarr_path,
                    group=zarr_group,
                    region=region,
                    compute=True,
                    storage_options=storage_options,
                    synchronizer=zarr.ThreadSynchronizer(),
                    consolidated=False,
                )

    def _append_provenance_attr_vars(
        self, zarr_path: str, storage_options: Dict[str, Any] = {}
    ) -> None:
        """
        Creates an xarray Dataset with variables set as the attributes
        from all groups before the combination. Additionally, appends
        this Dataset to the ``Provenance`` group located in the zarr
        store specified by ``zarr_path``.

        Parameters
        ----------
        zarr_path: str
            The full path of the final combined zarr store
        storage_options: dict
            Any additional parameters for the storage
            backend (ignored for local paths)
        """

        xr_dict = dict()
        for name, val in self.group_attrs.items():
            if "attrs" in name:
                # create Dataset variables
                coord_name = name[:-1] + "_key"
                xr_dict[name] = {"dims": ["echodata_filename", coord_name], "data": val}

            else:
                # create Dataset coordinates
                xr_dict[name] = {"dims": [name], "data": val}

        # construct the Provenance Dataset's attributes
        prov_attributes = echopype_prov_attrs("conversion")

        if "duplicate_ping_times" in self.group_attrs["provenance_attr_key"]:
            dup_pings_position = self.group_attrs["provenance_attr_key"].index(
                "duplicate_ping_times"
            )

            # see if the duplicate_ping_times value is equal to 1
            elem_is_one = [
                True if val[dup_pings_position] == 1 else False
                for val in self.group_attrs["provenance_attrs"]
            ]

            # set duplicate_ping_times = 1 if any file has 1
            prov_attributes["duplicate_ping_times"] = 1 if any(elem_is_one) else 0

        # construct Dataset and assign Provenance attributes
        all_ds_attrs = xr.Dataset.from_dict(xr_dict).assign_attrs(prov_attributes)

        # append Dataset to zarr
        all_ds_attrs.to_zarr(
            zarr_path,
            group="Provenance",
            mode="a",
            storage_options=storage_options,
            consolidated=False,
        )

    @staticmethod
    def _modify_prov_filenames(zarr_path: str, storage_options: Dict[str, Any] = {}) -> None:
        """
        After the ``Provenance`` group has been constructed, the
        coordinate ``filenames`` will be filled with zeros. This
        function fills ``filenames`` with the appropriate values
        by directly overwriting the zarr array.

        Parameters
        ----------
        zarr_path: str
            The full path of the final combined zarr store
        storage_options: dict
            Any additional parameters for the storage
            backend (ignored for local paths)
        """

        # obtain the filenames zarr array
        zarr_filenames = zarr.open_array(
            env_indep_joinpath(zarr_path, "Provenance", "filenames"),
            mode="r+",
            storage_options=storage_options,
        )
        # Assume that this is 1D so zarr_filenames.shape
        # is expected to return tuple such as (x,)
        zarr_filenames[:] = np.arange(*zarr_filenames.shape)

    def combine(
        self,
        zarr_path: str,
        eds: List[EchoData] = [],
        storage_options: Dict[str, Any] = {},
        sonar_model: str = None,
        echodata_filenames: List[str] = [],
        ed_group_chan_sel: Dict[str, Optional[List[str]]] = {},
        consolidated: bool = True,
    ) -> EchoData:
        """
        Combines all ``EchoData`` objects in ``eds`` by
        writing each element in parallel to the zarr store
        specified by ``zarr_path``.

        Parameters
        ----------
        zarr_path: str
            The full path of the final combined zarr store
        eds: list of EchoData object
            The list of ``EchoData`` objects to be combined
            The list of ``EchoData`` objects to be combined
        storage_options: dict
            Any additional parameters for the storage
            backend (ignored for local paths)
        sonar_model : str
            The sonar model used for all elements in ``eds``
        echodata_filenames : list of str
            The source files names for all elements in ``eds``
        ed_group_chan_sel: dict
            A dictionary with keys corresponding to the ``EchoData`` groups
            and values specify what channels should be selected within that
            group. If a value is ``None``, then a subset of channels should
            not be selected.
        consolidated: bool
            Flag to consolidate zarr metadata.
            Defaults to ``True``

        Returns
        -------
        ed_combined: EchoData
            The final combined form of the input ``eds`` before
            a reversed time check has been run

        Raises
        ------
        RuntimeError
            If the first time value of each Dataset is not less than
            the first time value of the subsequent Dataset
        RuntimeError
            If each Dataset in ``ds_list`` does not have the
            same number of channels and the same name for each
            of these channels.
        RuntimeError
            If any of the following attribute checks are not met
            amongst the combined Datasets
            - the keys are not the same
            - the values are not identical
            - the keys ``date_created`` or ``conversion_time``
            do not have the same types

        Notes
        -----
        All attributes that are not arrays will be made into
        variables and their result will be stored in the
        ``Provenance`` group.
        """

        # set class variables from input
        self.sonar_model = sonar_model
        self.group_attrs["echodata_filename"] = echodata_filenames

        # loop through all possible group and write them to a zarr store
        for grp_info in EchoData.group_map.values():
            # obtain the appropriate group name
            if grp_info["ep_group"]:
                ed_group = grp_info["ep_group"]
            else:
                ed_group = "Top-level"

            # collect the group Dataset from all eds that have their channels unselected
            all_chan_ds_list = [ed[ed_group] for ed in eds if ed_group in ed.group_paths]

            # select only the appropriate channels from each Dataset
            ds_list = [
                ds.sel(channel=ed_group_chan_sel[ed_group])
                if ed_group_chan_sel[ed_group] is not None
                else ds
                for ds in all_chan_ds_list
            ]

            if ds_list:  # necessary because a group may not be present
                const_names = self._append_ds_list_to_zarr(
                    zarr_path,
                    ds_list=ds_list,
                    zarr_group=grp_info["ep_group"],
                    ed_name=ed_group,
                    storage_options=storage_options,
                )

                self._append_const_to_zarr(
                    const_names,
                    ds_list,
                    zarr_path,
                    grp_info["ep_group"],
                    storage_options,
                )

                self._write_append_dims(
                    ds_list,
                    zarr_path,
                    grp_info["ep_group"],
                    storage_options,
                )

        # append all group attributes before combination to zarr store
        self._append_provenance_attr_vars(
            zarr_path,
            storage_options=storage_options,
        )

        # change filenames numbering to range(len(eds))
        self._modify_prov_filenames(zarr_path, storage_options=storage_options)

        if consolidated:
            # consolidate at the end if consolidated flag is True
            store = fsspec.get_mapper(zarr_path, **storage_options)
            zarr.consolidate_metadata(store=store)

        # open lazy loaded combined EchoData object
        ed_combined = open_converted(
            zarr_path,
            chunks={},
            synchronizer=zarr.ThreadSynchronizer(),
            storage_options=storage_options,
            backend_kwargs={"consolidated": consolidated},
        )  # TODO: is this appropriate for chunks?

        return ed_combined
