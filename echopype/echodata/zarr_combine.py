from typing import Dict, Hashable, List, Optional, Set, Tuple, Any
from collections import defaultdict
import dask
import dask.array
import dask.distributed
import pandas as pd
import xarray as xr
from .echodata import EchoData
from .api import open_converted
import zarr
from ..utils.prov import echopype_prov_attrs
from warnings import warn


class ZarrCombine:
    """
    A class that combines a list of EchoData objects by
    creating a Zarr store and appending each group's
    Dataset to the store.
    """

    def __init__(self):

        # all possible dimensions that we will append to (mainly time dims)
        self.append_dims = {"time1", "time2", "time3", "ping_time", "filenames"}

        # encodings associated with lazy loaded variables
        self.lazy_encodings = ["chunks", "preferred_chunks"]

        # defaultdict that holds every group's attributes
        self.group_attrs = defaultdict(list)

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
        """

        # Dataframe with column as dim names and rows as the different Datasets
        self.dims_df = pd.DataFrame([ds.dims for ds in ds_list])

        # calculate useful information about the dimensions
        self.dims_sum = self.dims_df.sum(axis=0).to_dict()
        self.dims_csum = self.dims_df.cumsum(axis=0).to_dict()
        self.dims_max = self.dims_df.max(axis=0).to_dict()

        # format ed_name appropriately
        ed_name = ed_name.replace('-', '_').replace('/', '_').lower()

        # collect Dataset attributes
        for count, ds in enumerate(ds_list):
            if count == 0:
                self.group_attrs[ed_name + '_attr_key'].extend(ds.attrs.keys())
            self.group_attrs[ed_name + '_attrs'].append(list(ds.attrs.values()))

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

        # Create the shape of the variable in its final combined
        # form (padding occurs here)  # TODO: make sure this is true
        shape = [
            self.dims_sum[dim] if dim in self.append_dims else self.dims_max[dim] for dim in dims
        ]

        # Create the chunk shape of the variable
        chnk_shape = [self.dims_max[dim] for dim in dims]

        temp_arr = dask.array.zeros(shape=shape, chunks=chnk_shape, dtype=dtype)

        return temp_arr, chnk_shape

    def _set_encodings(self, encodings: Dict[str, dict], name: Hashable,
                       val: xr.Variable, chnk_shape: list) -> None:
        """
        Sets the encodings for the variable ``name`` by including all
        encodings in ``val``, except those encodings that are deemed
        lazy encodings.

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

        # set the chunk encoding
        encodings[str(name)]["chunks"] = chnk_shape

    def _construct_lazy_ds_and_var_info(self, ds_model: xr.Dataset) -> Tuple[xr.Dataset, List[str], Dict[str, dict]]:
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
            across all Datasets to be combined
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
            The key of the values of ``dims_csum`` to use for each
            dimension name
        ds_dims: Set[Hashable]
            The names of the dimensions used in the region creation

        Returns
        -------
        region: Dict[str, slice]
            Keys set as the dimension name and values as
            the slice of the zarr portion to write to

        Notes
        -----
        Only append dimensions should show up in the region result.
        """

        if ds_ind == 0:

            # get the initial region
            region = {dim: slice(0, self.dims_csum[dim][ds_ind]) for dim in ds_dims if dim in self.append_dims}

        else:

            # get all other regions
            region = {
                dim: slice(self.dims_csum[dim][ds_ind - 1], self.dims_csum[dim][ds_ind])
                for dim in ds_dims if dim in self.append_dims
            }

        return region

    def _append_const_to_zarr(self, const_vars: List[str], ds_list: List[xr.Dataset],
                              path: str, zarr_group: str, storage_options: dict):
        """
        Appends all constant (i.e. not chunked) variables and dimensions to the
        zarr group.

        Parameters
        ----------
        const_vars: List[str]
            The names of all variables/dimensions that are not chunked
        ds_list: List[xr.Dataset]
            The Datasets that will be combined
        path: str
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

            # dims will be automatically filled when they occur in a variable
            if (var not in list(ds_list[0].dims)) or (var in ["beam", "range_sample"]):

                # TODO: when range_sample needs to be padded, here we will
                #  need to pick the dataset with the max size for range_sample

                # make sure to choose the dataset with the largest size for variable
                if var in self.dims_df:
                    ds_list_ind = int(self.dims_df[var].argmax())
                else:
                    ds_list_ind = int(0)

                ds_list[ds_list_ind][[var]].to_zarr(
                    path, group=zarr_group, mode='a', storage_options=storage_options
                )

    def _append_ds_list_to_zarr(
        self, path: str, ds_list: List[xr.Dataset], zarr_group: str, ed_name: str,
            storage_options: Optional[dict] = {}, to_zarr_compute: bool = True
    ) -> List[str]:
        """
        Creates a zarr store and then appends each Dataset
        in ``ds_list`` to it. The final result is a combined
        Dataset along the time dimensions.

        Parameters
        ----------
        path: str
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
        """

        self._get_ds_info(ds_list, ed_name)

        # TODO: Check that all of the channels are the same and times
        #  don't overlap and they increase may have an issue with time1 and NaT

        # TODO: check for and correct reversed time

        ds_lazy, const_names, encodings = self._construct_lazy_ds_and_var_info(ds_list[0])

        # create zarr file and all associated metadata (this is delayed)
        ds_lazy.to_zarr(
            path,
            compute=False,
            group=zarr_group,
            encoding=encodings,
            consolidated=True,
            storage_options=storage_options, synchronizer=zarr.ThreadSynchronizer()
        )

        # write each non-constant variable in ds_list to the zarr store
        delayed_to_zarr = []
        for ind, ds in enumerate(ds_list):

            region = self._get_region(ind, set(ds.dims))

            ds_drop = ds.drop(const_names)

            delayed_to_zarr.append(ds_drop.to_zarr(
                path, group=zarr_group, region=region, storage_options=storage_options, compute=to_zarr_compute,
                synchronizer=zarr.ThreadSynchronizer()
            ))

        if not to_zarr_compute:
            dask.compute(*delayed_to_zarr, retries=1)  # TODO: maybe use persist in the future?

        # TODO: need to consider the case where range_sample needs to be padded?

        return const_names

    def _append_provenance_attr_vars(self, path: str, storage_options: Optional[dict] = {}) -> None:
        """
        Creates an xarray Dataset with variables set as the attributes
        from all groups before the combination. Additionally, appends
        this Dataset to the ``Provenance`` group located in the zarr
        store specified by ``path``.

        Parameters
        ----------
        path: str
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
        all_ds_attrs.to_zarr(path, group="Provenance", mode="a",
                             storage_options=storage_options, consolidated=True)

    def combine(self, path: str, eds: List[EchoData] = [],
                storage_options: Optional[dict] = {}) -> EchoData:

        if not isinstance(eds, list):
            raise TypeError("The input, eds, must be a list of EchoData objects!")

        if not isinstance(path, str):
            raise TypeError("The input, path, must be a string!")

        # return empty EchoData object, if no EchoData objects are provided
        if not eds:
            warn("No EchoData objects were provided, returning an empty EchoData object.")
            return EchoData()

        # collect filenames associated with EchoData objects
        self.group_attrs["echodata_filename"].extend([str(ed.source_file) if ed.source_file is not None else str(ed.converted_raw_path) for ed in eds])

        to_zarr_compute = False

        for grp_info in EchoData.group_map.values():

            if grp_info['ep_group']:
                ed_group = grp_info['ep_group']
            else:
                ed_group = "Top-level"

            ds_list = [ed[ed_group] for ed in eds if ed_group in ed.group_paths]

            if ds_list:

                print(f"ed_group = {ed_group}")

                const_names = self._append_ds_list_to_zarr(path, ds_list=ds_list, zarr_group=grp_info['ep_group'],
                                                           ed_name=ed_group, storage_options=storage_options,
                                                           to_zarr_compute=to_zarr_compute)

                self._append_const_to_zarr(const_names, ds_list,
                                           path, grp_info['ep_group'], storage_options)

        # append all group attributes before combination to zarr store
        self._append_provenance_attr_vars(path, storage_options=storage_options)

        # open lazy loaded combined EchoData object
        ed_combined = open_converted(path)
        #
        return ed_combined
