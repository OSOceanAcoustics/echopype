from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Set, Tuple

import dask
import dask.array
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from ..utils.coding import COMPRESSION_SETTINGS
from ..utils.io import get_zarr_compression
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

    def _check_ascending_ds_times(self, ds_list: List[xr.Dataset], ed_name: str) -> None:
        """
        A minimal check that the first time value of each Dataset is less than
        the first time value of the subsequent Dataset. If each first time value
        is NaT, then this check is skipped.

        Parameters
        ----------
        ds_list: List[xr.Dataset]
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
    def _check_channels(ds_list: List[xr.Dataset], ed_name: str) -> None:
        """
        Makes sure that each Dataset in ``ds_list`` has the
        same number of channels and the same name for each
        of these channels.

        Parameters
        ----------
        ds_list: List[xr.Dataset]
            List of Datasets to be combined
        ed_name: str
            The name of the ``EchoData`` group being combined
        """

        if "channel" in ds_list[0].dims:

            # check to make sure we have the same number of channels in each ds
            if np.unique([len(ds["channel"].values) for ds in ds_list]).size == 1:

                # make each array an element of a numpy array
                channel_arrays = np.array([ds["channel"].values for ds in ds_list])

                # check for unique rows
                if np.unique(channel_arrays, axis=0).shape[0] > 1:

                    raise RuntimeError(
                        f"All {ed_name} groups do not have that same channel coordinate, "
                        f"combine cannot be used!"
                    )

            else:
                raise RuntimeError(
                    f"All {ed_name} groups do not have that same number of channel coordinates, "
                    f"combine cannot be used!"
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
        numpy_keys: List[str]
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

    def _get_ds_info(self, ds_list: List[xr.Dataset], ed_name: Optional[str]) -> None:
        """
        Constructs useful dictionaries that contain information
        about the dimensions of the Dataset. Additionally, collects
        the attributes from each Dataset in ``ds_list`` and saves
        this group specific information to the class variable
        ``group_attrs``.

        Parameters
        ----------
        ds_list: List[xr.Dataset]
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
        self._check_channels(ds_list, ed_name)

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
        variable in its final combined form.

        Parameters
        ----------
        dims: List[str]
            A list of the dimension names
        dtype: type
            The data type of the variable

        Returns
        -------
        temp_arr: dask.array
            a temporary (or dummy) array representing a
            variable in its final combined form.
        chnk_shape: List[int]
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
        chnk_shape = [self.dims_max[dim] for dim in dims]

        temp_arr = dask.array.zeros(shape=shape, dtype=dtype, chunks=chnk_shape)

        return temp_arr, chnk_shape

    def _set_encodings(
        self, encodings: Dict[str, dict], name: Hashable, val: xr.Variable, chnk_shape: list
    ) -> None:
        """
        Sets the encodings for the variable ``name`` by including all
        encodings in ``val``, except those encodings that are deemed
        lazy encodings. Additionally, if a compressor is not found,
        a default compressor will be assigned.

        Parameters
        ----------
        encodings: Dict[str, dict]
            The dictionary to set the encodings for
        name: Hashable
            The name of the variable we are setting the encodings for
        val: xr.Variable
            The variable that contains the encodings we want to assign
            to ``name``
        chnk_shape: list
            The shape of the chunks for ``name`` (used in encodings)

        Notes
        -----
        The input ``encodings`` is directly modified
        """

        # gather all encodings, except the lazy encodings
        encodings[str(name)] = {
            key: encod for key, encod in val.encoding.items() if key not in self.lazy_encodings
        }

        # assign default compressor, if one does not exist
        if "compressor" not in encodings[str(name)]:
            encodings[str(name)].update(get_zarr_compression(val, COMPRESSION_SETTINGS["zarr"]))

        # set the chunk encoding
        encodings[str(name)]["chunks"] = chnk_shape

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
        const_names: List[str]
            The names of all variables and dimensions that are constant
            (with respect to chunking) across all Datasets to be combined
        encodings: Dict[str, dict]
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

            elif name not in ds_model.dims:

                # create lazy DataArray representations corresponding to the variables
                temp_arr, chnk_shape = self._get_temp_arr(list(val.dims), val.dtype)
                xr_vars_dict[name] = (val.dims, temp_arr, val.attrs)

                self._set_encodings(encodings, name, val, chnk_shape)

            elif name in self.append_dims:

                # create lazy DataArray for those coordinates that can be appended to
                temp_arr, chnk_shape = self._get_temp_arr(list(val.dims), val.dtype)
                xr_coords_dict[name] = (val.dims, temp_arr, val.attrs)

                self._set_encodings(encodings, name, val, chnk_shape)

        # construct lazy Dataset form
        ds = xr.Dataset(xr_vars_dict, coords=xr_coords_dict, attrs=ds_model.attrs)

        return ds, const_names, encodings

    def _get_region(self, ds_ind: int, ds_dims: Set[Hashable]) -> Dict[str, slice]:
        """
        Returns the region of the zarr file to write to. This region
        corresponds to the input set of dimensions.

        Parameters
        ----------
        ds_ind: int
            The key of the values of ``self.dims_csum`` or index of
            ``self.dims_df`` to use for each dimension name
        ds_dims: Set[Hashable]
            The names of the dimensions used in the region creation

        Returns
        -------
        region: Dict[str, slice]
            Keys set as the dimension name and values as
            the slice of the zarr portion to write to
        """

        # get the initial region
        region = dict()
        for dim in ds_dims:

            if dim in self.append_dims:

                if ds_ind == 0:
                    # get the initial region
                    region[dim] = slice(0, self.dims_csum[dim][ds_ind])
                else:
                    # get all other regions
                    region[dim] = slice(
                        self.dims_csum[dim][ds_ind - 1], self.dims_csum[dim][ds_ind]
                    )

            else:
                region[dim] = slice(0, self.dims_df.loc[ds_ind][dim])

        return region

    def _append_ds_list_to_zarr(
        self,
        zarr_path: str,
        ds_list: List[xr.Dataset],
        zarr_group: str,
        ed_name: str,
        storage_options: Optional[dict] = {},
    ) -> List[str]:
        """
        Creates a zarr store and then appends each Dataset
        in ``ds_list`` to it. The final result is a combined
        Dataset along the time dimensions.

        Parameters
        ----------
        zarr_path: str
            The full path of the final combined zarr store
        ds_list: List[xr.Dataset]
            The Datasets that will be combined
        zarr_group: str
            The name of the group of the zarr store
            corresponding to the Datasets in ``ds_list``
        ed_name: str
            The name of the EchoData group corresponding to the
            Datasets in ``ds_list``
        storage_options: Optional[dict]
            Any additional parameters for the storage
            backend (ignored for local paths)

        Returns
        -------
        const_names: List[str]
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
            consolidated=None,
            storage_options=storage_options,
            synchronizer=zarr.ThreadSynchronizer(),
        )

        # collect delayed functions that write each non-constant variable
        # in ds_list to the zarr store
        delayed_to_zarr = []
        for ind, ds in enumerate(ds_list):

            ds_drop = ds.drop(const_names)

            region = self._get_region(ind, set(ds_drop.dims))

            # TODO: below is an xarray delayed approach, however, data will be corrupted,
            #  we can remove data corruption by implementing a locking scheme
            delayed_to_zarr.append(
                ds_drop.to_zarr(
                    zarr_path,
                    group=zarr_group,
                    region=region,
                    compute=False,
                    storage_options=storage_options,
                    synchronizer=zarr.ThreadSynchronizer(),
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
        storage_options: Optional[dict],
    ) -> None:
        """
        Appends all constant (i.e. not chunked) variables and dimensions to the
        zarr group.

        Parameters
        ----------
        const_vars: List[str]
            The names of all variables/dimensions that are not chunked
        ds_list: List[xr.Dataset]
            The Datasets that will be combined
        zarr_path: str
            The full path of the final combined zarr store
        zarr_group: str
            The name of the group of the zarr store
            corresponding to the Datasets in ``ds_list``
        storage_options: Optional[dict]
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
                zarr_path, group=zarr_group, mode="a", storage_options=storage_options
            )

    def _append_provenance_attr_vars(
        self, zarr_path: str, storage_options: Optional[dict] = {}
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
        storage_options: Optional[dict]
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

        # construct Dataset and assign Provenance attributes
        all_ds_attrs = xr.Dataset.from_dict(xr_dict).assign_attrs(echopype_prov_attrs("conversion"))

        # append Dataset to zarr
        all_ds_attrs.to_zarr(
            zarr_path,
            group="Provenance",
            mode="a",
            storage_options=storage_options,
            consolidated=True,
        )

    @staticmethod
    def _modify_prov_filenames(zarr_path: str, len_eds: int) -> None:
        """
        After the ``Provenance`` group has been constructed, the
        coordinate ``filenames`` will be filled with zeros. This
        function fills ``filenames`` with the appropriate values
        by directly overwriting the zarr array.

        Parameters
        ----------
        zarr_path: str
            The full path of the final combined zarr store
        len_eds: int
            The number of ``EchoData`` objects being combined
        """

        # obtain the filenames zarr array
        zarr_filenames = zarr.open_array(zarr_path + "/Provenance/filenames", mode="r+")

        zarr_filenames[:] = np.arange(len_eds)

    def combine(
        self,
        zarr_path: str,
        eds: List[EchoData] = [],
        storage_options: Optional[dict] = {},
        sonar_model: str = None,
        echodata_filenames: List[str] = [],
    ) -> EchoData:
        """
        Combines all ``EchoData`` objects in ``eds`` by
        writing each element in parallel to the zarr store
        specified by ``zarr_path``.

        Parameters
        ----------
        zarr_path: str
            The full path of the final combined zarr store
        eds: List[EchoData]
            The list of ``EchoData`` objects to be combined
        storage_options: Optional[dict]
            Any additional parameters for the storage
            backend (ignored for local paths)
        sonar_model : str
            The sonar model used for all elements in ``eds``
        echodata_filenames : List[str]
            The source files names for all elements in ``eds``

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

        # TODO: the below line should be uncommented, if blosc issues persist
        # blosc.use_threads = False

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

            # collect the group Dataset from all eds
            ds_list = [ed[ed_group] for ed in eds if ed_group in ed.group_paths]

            if ds_list:  # necessary because a group may not be present

                const_names = self._append_ds_list_to_zarr(
                    zarr_path,
                    ds_list=ds_list,
                    zarr_group=grp_info["ep_group"],
                    ed_name=ed_group,
                    storage_options=storage_options,
                )

                self._append_const_to_zarr(
                    const_names, ds_list, zarr_path, grp_info["ep_group"], storage_options
                )

        # append all group attributes before combination to zarr store
        self._append_provenance_attr_vars(zarr_path, storage_options=storage_options)

        # change filenames numbering to range(len(eds))
        self._modify_prov_filenames(zarr_path, len_eds=len(eds))

        # TODO: the below line should be uncommented, if blosc issues persist
        # blosc.use_threads = None

        # open lazy loaded combined EchoData object
        ed_combined = open_converted(
            zarr_path, chunks={}, synchronizer=zarr.ThreadSynchronizer()
        )  # TODO: is this appropriate for chunks?

        return ed_combined
