from typing import Dict, Hashable, List, Optional, Set
from collections import defaultdict
import dask
import dask.array
import pandas as pd
import xarray as xr
from .echodata import EchoData
from .api import open_converted
import zarr
from numcodecs import blosc
from ..utils.prov import echopype_prov_attrs
from warnings import warn


class ZarrCombine:
    """
    A class that combines a list of EchoData objects by
    creating a Zarr store and appending each group's
    Dataset to the store.
    """

    def __init__(self):

        # those dimensions that should not be chunked
        self.const_dims = ["channel", "beam_group", "beam", "range_sample", "pulse_length_bin"]

        # those dimensions associated with time
        self.time_dims = ["time1", "time2", "time3", "ping_time"]

        # all possible dimensions we can encounter
        self.possible_dims = self.const_dims + self.time_dims

        # encodings associated with lazy loaded variables
        self.lazy_encodings = ["chunks", "preferred_chunks", "compressor"]

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
        dims_df = pd.DataFrame([ds.dims for ds in ds_list])

        # calculate useful information about the dimensions
        self.dims_sum = dims_df.sum(axis=0).to_dict()
        self.dims_csum = dims_df.cumsum(axis=0).to_dict()
        self.dims_max = dims_df.max(axis=0).to_dict()

        # format ed_name appropriately
        ed_name = ed_name.replace('-', '_').replace('/', '_').lower()

        # collect Dataset attributes
        for count, ds in enumerate(ds_list):
            if count == 0:
                self.group_attrs[ed_name + '_attr_key'].extend(ds.attrs.keys())
            self.group_attrs[ed_name + '_attrs'].append(list(ds.attrs.values()))

    def _get_temp_arr(self, dims: List[str], dtype: type) -> dask.array:
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
        dask.array
            a temporary (or dummy) array representing a
            variable in its final combined form.

        Notes
        -----
        This array is never interacted with in a traditional sense.
        Its sole purpose is to construct metadata for the zarr store.
        """

        # Create the shape of the variable in its final combined
        # form (padding occurs here)  # TODO: make sure this is true
        shape = [
            self.dims_max[dim] if dim in self.const_dims else self.dims_sum[dim] for dim in dims
        ]

        # Create the chunk shape of the variable
        chnk_shape = [self.dims_max[dim] for dim in dims]

        return dask.array.zeros(shape=shape, chunks=chnk_shape, dtype=dtype)

    def _construct_lazy_ds(self, ds_model: xr.Dataset) -> xr.Dataset:
        """
        Constructs a lazy Dataset representing the EchoData group
        Dataset in its final combined form.

        Parameters
        ----------
        ds_model: xr.Dataset
            A Dataset that we will model our lazy Dataset after. In practice,
            this is the first element in the list of Datasets to be combined.

        Returns
        -------
        xr.Dataset
            A lazy Dataset representing the EchoData group Dataset in
            its final combined form

        Notes
        -----
        The sole purpose of the Dataset created is to construct metadata
        for the zarr store.
        """

        xr_vars_dict = dict()
        xr_coords_dict = dict()
        for name, val in ds_model.variables.items():
            if name not in self.possible_dims:

                # create lazy DataArray representations corresponding to the variables
                temp_arr = self._get_temp_arr(list(val.dims), val.dtype)
                xr_vars_dict[name] = (val.dims, temp_arr, val.attrs)

            else:

                # create lazy DataArray representations corresponding to the coordinates
                temp_arr = self._get_temp_arr(list(val.dims), val.dtype)
                xr_coords_dict[name] = (val.dims, temp_arr, val.attrs)

        # construct lazy Dataset form
        ds = xr.Dataset(xr_vars_dict, coords=xr_coords_dict, attrs=ds_model.attrs)

        return ds

    def _get_ds_encodings(self, ds_model: xr.Dataset) -> Dict[Hashable, dict]:
        """
        Obtains the encodings needed for each variable
        of the lazy Dataset form.

        Parameters
        ----------
        ds_model: xr.Dataset
            The Dataset that we modelled our lazy Dataset after. In practice,
            this is the first element in the list of Datasets to be combined.

        Returns
        -------
        encodings: Dict[Hashable, dict]
            The keys are a string representing the variable name and the
            values are a dictionary of the corresponding encodings

        Notes
        -----
        The encodings corresponding to the lazy encodings (e.g. compressor)
        should not be included here, these will be generated by `to_zarr`.
        """

        encodings = dict()
        for name, val in ds_model.variables.items():

            # get all encodings except the lazy encodings
            encodings[name] = {
                key: encod for key, encod in val.encoding.items() if key not in self.lazy_encodings
            }

        return encodings

    def _get_constant_vars(self, ds_model: xr.Dataset) -> list:
        """
        Obtains all variable and dimension names that will
        be the same across all Datasets that will be combined.

        Parameters
        ----------
        ds_model: xr.Dataset
            The Dataset that we modelled our lazy Dataset after. In practice,
            this is the first element in the list of Datasets to be combined.

        Returns
        -------
        const_vars: list
            Variable and dimension names that will be the same across all
            Datasets that will be combined.
        """

        # obtain the form of the dimensions for each constant variable
        dim_form = [(dim,) for dim in self.const_dims]

        # account for Vendor_specific variables
        dim_form.append(("channel", "pulse_length_bin"))  # TODO: is there a better way?

        # obtain all constant variables and dimensions
        const_vars = []
        for name, val in ds_model.variables.items():
            if val.dims in dim_form:
                const_vars.append(name)

        return const_vars

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
        """

        if ds_ind == 0:

            # get the initial region
            region = {dim: slice(0, self.dims_csum[dim][ds_ind]) for dim in ds_dims}

        else:

            # get all other regions
            region = {
                dim: slice(self.dims_csum[dim][ds_ind - 1], self.dims_csum[dim][ds_ind])
                for dim in ds_dims
            }

        return region

    def _append_ds_list_to_zarr(
        self, path: str, ds_list: List[xr.Dataset], zarr_group: str, ed_name: str,
            storage_options: Optional[dict] = {}, to_zarr_compute: bool = True
    ) -> None:
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

        ds_lazy = self._construct_lazy_ds(ds_list[0])

        encodings = self._get_ds_encodings(ds_list[0])

        # create zarr file and all associated metadata (this is delayed)
        ds_lazy.to_zarr(
            path,
            compute=False,
            group=zarr_group,
            encoding=encodings,
            consolidated=True,
            storage_options=storage_options, synchronizer=zarr.ThreadSynchronizer()
        )

        # constant variables that will be written later
        const_vars = self._get_constant_vars(ds_list[0])

        # write each non-constant variable in ds_list to the zarr store
        delayed_to_zarr = []
        for ind, ds in enumerate(ds_list):

            # obtain the names of all ds dimensions that are not constant
            ds_dims = set(ds.dims) - set(const_vars)

            region = self._get_region(ind, ds_dims)

            delayed_to_zarr.append(ds.drop(const_vars).to_zarr(
                path, group=zarr_group, region=region, storage_options=storage_options, compute=to_zarr_compute,
                synchronizer=zarr.ThreadSynchronizer()
            ))
            # TODO: see if compression is occurring, maybe mess with encoding.

        if not to_zarr_compute:
            dask.compute(*delayed_to_zarr)  # TODO: maybe use persist in the future?

        # write constant vars to zarr using the first element of ds_list
        for var in const_vars:  # TODO: one should not parallelize this loop??

            # dims will be automatically filled when they occur in a variable
            if (var not in self.possible_dims) or (var in ["beam", "range_sample"]):

                region = self._get_region(0, set(ds_list[0][var].dims))

                ds_list[0][[var]].to_zarr(
                    path, group=zarr_group, region=region, storage_options=storage_options,
                    synchronizer=zarr.ThreadSynchronizer()
                )

        # TODO: need to consider the case where range_sample needs to be padded?

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

    def combine(self, path: str, eds: List[EchoData] = None,
                storage_options: Optional[dict] = {}) -> EchoData:

        # return empty EchoData object, if no EchoData objects are provided
        if (isinstance(eds, list) and len(eds) == 0) or (not eds):
            warn("No EchoData objects were provided, returning an empty EchoData object.")
            return EchoData()

        # collect filenames associated with EchoData objects
        self.group_attrs["echodata_filename"].extend([str(ed.source_file) if ed.source_file is not None else str(ed.converted_raw_path) for ed in eds])

        to_zarr_compute = False

        print(f"to_zarr_compute = {to_zarr_compute}")

        # tell Blosc to runs in single-threaded contextual mode (necessary for parallel)
        blosc.use_threads = False

        for grp_info in EchoData.group_map.values():

            if grp_info['ep_group']:
                ed_group = grp_info['ep_group']
            else:
                ed_group = "Top-level"

            ds_list = [ed[ed_group] for ed in eds if ed_group in ed.group_paths]

            if ds_list:

                print(f"ed_group = {ed_group}")

                self._append_ds_list_to_zarr(path, ds_list=ds_list, zarr_group=grp_info['ep_group'],
                                             ed_name=ed_group, storage_options=storage_options,
                                             to_zarr_compute=to_zarr_compute)

        # append all group attributes before combination to zarr store
        self._append_provenance_attr_vars(path, storage_options=storage_options)

        # TODO: re-chunk the zarr store after everything has been added?

        # re-enable automatic switching (the default behavior)
        blosc.use_threads = None

        # open lazy loaded combined EchoData object
        ed_combined = open_converted(path)

        return ed_combined
