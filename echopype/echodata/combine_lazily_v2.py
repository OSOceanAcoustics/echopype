import xarray as xr
import pandas as pd
import dask.array
import dask


# those dimensions that should not be chunked
const_dims = ['channel', 'beam_group', 'beam', 'range_sample', 'pulse_length_bin']

# those dimensions associated with time
time_dims = ['time1', 'time2', 'time3', 'ping_time']

# all possible dimensions we can encounter
possible_dims = const_dims + time_dims

# encodings associated with lazy loaded variables
lazy_encodings = ["chunks", "preferred_chunks", "compressor"]


def get_ds_dims_info(ds_list):

    ds_dims = []
    for ds in ds_list:
        ds_dims.append(ds.dims)

    dims_df = pd.DataFrame(ds_dims)
    dims_sum = dims_df.sum(axis=0).to_dict()
    dims_max = dims_df.max(axis=0).to_dict()
    dims_csum = dims_df.cumsum(axis=0).to_dict()

    return dims_sum, dims_csum, dims_max, dims_df


def get_temp_arr(dims, dtype, dims_max, dims_sum):

    shape = [dims_max[dim] if dim in const_dims else dims_sum[dim] for dim in dims]

    chnk_shape = [dims_max[dim] for dim in dims]

    return dask.array.zeros(shape=shape, chunks=chnk_shape, dtype=dtype)


def construct_lazy_ds(ds_model, dims_sum, dims_max):

    xr_vars_dict = dict()
    xr_coords_dict = dict()
    for name, val in ds_model.variables.items():
        if name not in possible_dims:
            temp_arr = get_temp_arr(list(val.dims), val.dtype, dims_max, dims_sum)
            xr_vars_dict[name] = (val.dims, temp_arr, val.attrs)

        else:
            temp_arr = get_temp_arr(list(val.dims), val.dtype, dims_max, dims_sum)
            xr_coords_dict[name] = (val.dims, temp_arr, val.attrs)

    ds = xr.Dataset(xr_vars_dict, coords=xr_coords_dict)

    # TODO: add ds attributes here?

    return ds


def get_region(ds_ind, dims_csum, ds_dims):

    if ds_ind == 0:
        region = {dim: slice(0, dims_csum[dim][ds_ind]) for dim in ds_dims}

    else:
        region = {dim: slice(dims_csum[dim][ds_ind-1], dims_csum[dim][ds_ind]) for dim in ds_dims}

    return region


def get_ds_encodings(ds_model):

    encodings = dict()
    for name, val in ds_model.variables.items():
        encodings[name] = {key: encod for key, encod in val.encoding.items() if
                           key not in lazy_encodings}

    return encodings


def get_constant_vars(ds_model):

    dim_form = [(dim,) for dim in const_dims]

    # account for Vendor_specific vars
    dim_form.append(('channel', 'pulse_length_bin'))  # TODO: is there a better way?

    const_vars = []
    for name, val in ds_model.variables.items():

        if val.dims in dim_form:
            const_vars.append(name)

    return const_vars


def direct_write(path, ds_list, group, storage_options):

    dims_sum, dims_csum, dims_max, dims_df = get_ds_dims_info(ds_list)

    # TODO: Do check that all of the channels are the same and times don't overlap and they increase
    #  may have an issue with time1 and NaT

    ds_lazy = construct_lazy_ds(ds_list[0], dims_sum, dims_max)

    # get encodings for each of the arrays
    encodings = get_ds_encodings(ds_list[0])

    ds_lazy.to_zarr(path, compute=False, group=group, encoding=encodings,
                    consolidated=True, storage_options=storage_options)

    # constant variables that will be written in later
    const_vars = get_constant_vars(ds_list[0])

    print(f"const_vars = {const_vars}")

    for i in range(len(ds_list)):  # TODO: parallelize this loop

        ds_dims = set(ds_list[i].dims) - set(const_vars)

        region = get_region(i, dims_csum, ds_dims)
        ds_list[i].drop(const_vars).to_zarr(path, group=group, region=region,
                                            storage_options=storage_options)

    # write constant vars to zarr using the first element of ds_list
    for var in const_vars:   # TODO: one should not parallelize this loop??

        # dims will be automatically filled when they occur in a variable
        if (var not in possible_dims) or (var in ['beam', 'range_sample']):

            region = get_region(0, dims_csum, list(ds_list[0][var].dims))
            ds_list[0][[var]].to_zarr(path, group=group, region=region,
                                      storage_options=storage_options)


    # TODO: add back in attributes for dataset
    # TODO: correctly add attribute keys for Provenance group

    # TODO: need to consider the case where range_sample needs to be padded

    # TODO: re-chunk the zarr store after everything has been added?

    # TODO: is there a way we can preserve order in variables with writing?

# def lazy_combine(path, eds):
#
#     # TODO: do direct_write(path, ds_list) for each group in eds

