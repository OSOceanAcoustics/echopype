import xarray as xr
import pandas as pd
import dask.array
import dask
from typing import List, Tuple, Dict, Hashable, Optional, Set


# TODO: make this a class and have dims info/below lists as a class variable


# those dimensions that should not be chunked
const_dims = ['channel', 'beam_group', 'beam', 'range_sample', 'pulse_length_bin']

# those dimensions associated with time
time_dims = ['time1', 'time2', 'time3', 'ping_time']

# all possible dimensions we can encounter
possible_dims = const_dims + time_dims

# encodings associated with lazy loaded variables
lazy_encodings = ["chunks", "preferred_chunks", "compressor"]


def get_ds_dims_info(ds_list: List[xr.Dataset]) -> Tuple[dict, dict, dict]:
    """
    Constructs useful dictionaries that contain information
    about the dimensions of the Dataset

    Parameters
    ----------
    ds_list: List[xr.Dataset]
        The Datasets that will be combined

    Returns
    -------
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
    dims_sum = dims_df.sum(axis=0).to_dict()
    dims_csum = dims_df.cumsum(axis=0).to_dict()
    dims_max = dims_df.max(axis=0).to_dict()

    return dims_sum, dims_csum, dims_max


def get_temp_arr(dims: List[str], dtype: type,
                 dims_max: dict, dims_sum: dict) -> dask.array:
    """
    Constructs a temporary (or dummy) array representing a
    variable in its final combined form.

    Parameters
    ----------
    dims: List[str]
        A list of the dimension names
    dtype: type
        The data type of the variable
    dims_max: dict
        Keys as the dimension name and values as the corresponding
        maximum length across all Datasets
    dims_sum: dict
        Keys as the dimension name and values as the corresponding
        sum of the lengths across all Datasets

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

    # Create the shape of the variable in its final combined form (padding occurs here)  # TODO: make sure this is true
    shape = [dims_max[dim] if dim in const_dims else dims_sum[dim] for dim in dims]

    # Create the chunk shape of the variable
    chnk_shape = [dims_max[dim] for dim in dims]

    return dask.array.zeros(shape=shape, chunks=chnk_shape, dtype=dtype)


def construct_lazy_ds(ds_model: xr.Dataset, dims_sum: dict,
                      dims_max: dict) -> xr.Dataset:
    """
    Constructs a lazy Dataset representing the EchoData group
    Dataset in its final combined form.

    Parameters
    ----------
    ds_model: xr.Dataset
        A Dataset that we will model our lazy Dataset after. In practice,
        this is the first element in the list of Datasets to be combined.
    dims_sum: dict
        Keys as the dimension name and values as the corresponding
        sum of the lengths across all Datasets
    dims_max: dict
        Keys as the dimension name and values as the corresponding
        maximum length across all Datasets

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
        if name not in possible_dims:

            # create lazy DataArray representations corresponding to the variables
            temp_arr = get_temp_arr(list(val.dims), val.dtype, dims_max, dims_sum)
            xr_vars_dict[name] = (val.dims, temp_arr, val.attrs)

        else:

            # create lazy DataArray representations corresponding to the coordinates
            temp_arr = get_temp_arr(list(val.dims), val.dtype, dims_max, dims_sum)
            xr_coords_dict[name] = (val.dims, temp_arr, val.attrs)

    # construct lazy Dataset form
    ds = xr.Dataset(xr_vars_dict, coords=xr_coords_dict)

    # TODO: add ds attributes here and store all dataset attributes?

    return ds


def get_ds_encodings(ds_model: xr.Dataset) -> Dict[Hashable, dict]:
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
        encodings[name] = {key: encod for key, encod in val.encoding.items() if
                           key not in lazy_encodings}

    return encodings


def get_constant_vars(ds_model: xr.Dataset) -> list:
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
    dim_form = [(dim,) for dim in const_dims]

    # account for Vendor_specific variables
    dim_form.append(('channel', 'pulse_length_bin'))  # TODO: is there a better way?

    # obtain all constant variables and dimensions
    const_vars = []
    for name, val in ds_model.variables.items():
        if val.dims in dim_form:
            const_vars.append(name)

    return const_vars


def get_region(ds_ind: int, dims_csum: dict,
               ds_dims: Set[Hashable]) -> Dict[str, slice]:
    """
    Returns the region of the zarr file to write to. This region
    corresponds to the input set of dimensions.

    Parameters
    ----------
    ds_ind: int
        The key of the values of ``dims_csum`` to use for each
        dimension name
    dims_csum: dict
        Keys as the dimension name and values as a dictionary of
        the corresponding cumulative sum of the lengths across
        all Datasets
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
        region = {dim: slice(0, dims_csum[dim][ds_ind]) for dim in ds_dims}

    else:

        # get all other regions
        region = {dim: slice(dims_csum[dim][ds_ind - 1], dims_csum[dim][ds_ind]) for dim in ds_dims}

    return region


def direct_write(path: str, ds_list: List[xr.Dataset],
                 group: str, storage_options: Optional[dict] = {}) -> None:
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
    group: str
        The name of the group of the zarr store
        corresponding to the Datasets in ``ds_list``
    storage_options: Optional[dict]
        Any additional parameters for the storage
        backend (ignored for local paths)
    """

    dims_sum, dims_csum, dims_max = get_ds_dims_info(ds_list)

    # TODO: Check that all of the channels are the same and times don't overlap and they increase
    #  may have an issue with time1 and NaT

    ds_lazy = construct_lazy_ds(ds_list[0], dims_sum, dims_max)

    encodings = get_ds_encodings(ds_list[0])

    # create zarr file and all associated metadata (this is delayed)
    ds_lazy.to_zarr(path, compute=False, group=group, encoding=encodings,
                    consolidated=True, storage_options=storage_options)

    # constant variables that will be written later
    const_vars = get_constant_vars(ds_list[0])

    print(f"const_vars = {const_vars}")

    # write each non-constant variable in ds_list to the zarr store
    for i in range(len(ds_list)):  # TODO: parallelize this loop

        # obtain the names of all ds dimensions that are not constant
        ds_dims = set(ds_list[i].dims) - set(const_vars)

        region = get_region(i, dims_csum, ds_dims)

        ds_list[i].drop(const_vars).to_zarr(path, group=group, region=region,
                                            storage_options=storage_options)

    # write constant vars to zarr using the first element of ds_list
    for var in const_vars:   # TODO: one should not parallelize this loop??

        # dims will be automatically filled when they occur in a variable
        if (var not in possible_dims) or (var in ['beam', 'range_sample']):

            region = get_region(0, dims_csum, set(ds_list[0][var].dims))

            ds_list[0][[var]].to_zarr(path, group=group, region=region,
                                      storage_options=storage_options)


    # TODO: add back in attributes for dataset
    # TODO: correctly add attribute keys for Provenance group

    # TODO: need to consider the case where range_sample needs to be padded?

    # TODO: re-chunk the zarr store after everything has been added?

    # TODO: is there a way we can preserve order in variables with writing?

# def lazy_combine(path, eds):
#

# TODO: do direct_write(path, ds_list) for each group in eds
#  then do open_converted(path) --> here we could re-chunk?


