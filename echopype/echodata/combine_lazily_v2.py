from typing import Dict, Hashable, List, Optional, Set
from collections import defaultdict
import dask
import dask.array
import pandas as pd
import xarray as xr
from .echodata import EchoData

# TODO: make this a class and have dims info/below lists as a class variable


class LazyCombine:
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

        # TODO: document/bring up that I changed naming scheme of attributes

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

        # TODO: add ds attributes here?

        # TODO: do special case for Provenance, where we create attr variables

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

    def direct_write(
        self, path: str, ds_list: List[xr.Dataset], zarr_group: str, ed_name: str,
            storage_options: Optional[dict] = {}
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
            storage_options=storage_options,
        )

        # constant variables that will be written later
        const_vars = self._get_constant_vars(ds_list[0])

        to_zarr_compute = True

        print(f"to_zarr_compute = {to_zarr_compute}")

        # write each non-constant variable in ds_list to the zarr store
        delayed_to_zarr = []
        for ind, ds in enumerate(ds_list):

            # obtain the names of all ds dimensions that are not constant
            ds_dims = set(ds.dims) - set(const_vars)

            region = self._get_region(ind, ds_dims)

            delayed_to_zarr.append(ds.drop(const_vars).to_zarr(
                path, group=zarr_group, region=region, storage_options=storage_options, compute=to_zarr_compute
            ))
            # TODO: see if compression is occurring, maybe mess with encoding.

        if not to_zarr_compute:
            dask.compute(*delayed_to_zarr)

        # write constant vars to zarr using the first element of ds_list
        for var in const_vars:  # TODO: one should not parallelize this loop??

            # dims will be automatically filled when they occur in a variable
            if (var not in self.possible_dims) or (var in ["beam", "range_sample"]):

                region = self._get_region(0, set(ds_list[0][var].dims))

                ds_list[0][[var]].to_zarr(
                    path, group=zarr_group, region=region, storage_options=storage_options
                )

        # TODO: need to consider the case where range_sample needs to be padded?
        # TODO: is there a way we can preserve order in variables with writing?

    def combine(self, path: str, eds: List[EchoData], storage_options: Optional[dict] = {}):

        for grp_info in EchoData.group_map.values():

            # print(grp_info)

            if grp_info['ep_group']:
                ed_group = grp_info['ep_group']
            else:
                ed_group = "Top-level"

            ds_list = [ed[ed_group] for ed in eds if ed_group in ed.group_paths]

            if ds_list:

                print(f"ed_group = {ed_group}")

                self.direct_write(path,
                                  ds_list=ds_list,
                                  zarr_group=grp_info['ep_group'],
                                  ed_name=ed_group, storage_options=storage_options)

        # TODO: add back in attributes for dataset
        # TODO: correctly add attribute keys for Provenance group
        # TODO: re-chunk the zarr store after everything has been added?

        # TODO: do provenance group last
        # temp = {key: {"dims": ["echodata_filename"], "data": val} for key, val in self.group_attrs.items()}
        # xr.Dataset.from_dict(temp)

        # TODO: do direct_write(path, ds_list) for each group in eds
        #  then do open_converted(path) --> here we could re-chunk?
