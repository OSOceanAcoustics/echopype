import xarray as xr
import pandas as pd
import dask.array
import dask
import numpy as np

const_dims = ['channel']

def get_ds_dims_info(ds_list):

    ds_dims = []
    for ds in ds_list:
        ds_dims.append(ds.dims)

    dims_df = pd.DataFrame(ds_dims)
    dims_sum = dims_df.sum(axis=0).to_dict()
    dims_max = dims_df.max(axis=0).to_dict()
    dims_csum = dims_df.cumsum(axis=0).to_dict()

    return dims_sum, dims_csum, dims_max, dims_df


def get_temp_arr_vals(dims, dims_max, dims_sum, dims_df):

    shape = [dims_max[dim] if dim in const_dims else dims_sum[dim] for dim in dims]

    chnk_shape = [None if dim in const_dims else tuple(dims_df[dim].to_list()) for dim in dims]

    return shape, chnk_shape


def constuct_lazy_ds(ds_model, dims_sum, dims_max, dims_df):

    xr_dict = dict()

    unwritten_vars = []
    for name, val in ds_model.variables.items():

        if ('channel',) != val.dims:
            shape, chnk_shape = get_temp_arr_vals(val.dims, dims_max, dims_sum, dims_df)

            temp_arr = dask.array.zeros(shape=shape, chunks=chnk_shape, dtype=val.dtype)

            xr_dict[name] = (val.dims, temp_arr, val.attrs)

        else:
            unwritten_vars.append(name)

    ds = xr.Dataset(xr_dict)

    return ds, unwritten_vars


def get_region(ds_ind, dims_csum):

    if ds_ind == 0:
        region = {dim: slice(0, csum[ds_ind]) for dim, csum in dims_csum.items() if dim not in const_dims}

    else:
        region = {dim: slice(csum[ds_ind-1], csum[ds_ind]) for dim, csum in dims_csum.items() if dim not in const_dims}

    return region


def get_fill_dict(ds_lazy):

    fill_vals = dict()
    for var, val in ds_lazy.variables.items():

        if val.dtype == np.float64:
            fill_vals[var] = {'_FillValue': np.nan}
        elif val.dtype == np.dtype('<M8[ns]'):
            fill_vals[var] = {'_FillValue': np.datetime64("NaT")}
        else:
            raise NotImplementedError("Setting fill value for dtype not implemented!")

    return fill_vals


def direct_write(path, ds_list):

    dims_sum, dims_csum, dims_max, dims_df = get_ds_dims_info(ds_list)

    ds_lazy, unwritten_vars = constuct_lazy_ds(ds_list[0], dims_sum, dims_max, dims_df)

    # set fill value for each of the arrays
    fill_vals = get_fill_dict(ds_lazy)

    ds_lazy.to_zarr(path, compute=False, encoding=fill_vals)

    for i in range(len(ds_list)):

        ds_list[i] = ds_list[i].drop(unwritten_vars)

        ds_list[i].to_zarr(path, region=get_region(i, dims_csum))
        #TODO: figure out why time1 is not being correctly written to zarr



# def lazy_combine(path, eds):
#
#     # TODO: do direct_write(path, ds_list) for each group in eds

