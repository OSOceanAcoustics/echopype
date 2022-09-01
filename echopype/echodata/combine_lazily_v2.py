import xarray as xr
import pandas as pd
import dask.array
import dask
import numpy as np

const_dims = ['channel']  # those dimensions that should not be chunked
time_dims = ['time1', 'time2', 'time3']  # those dimensions associated with time
possible_dims = [] #const_dims + time_dims  # all possible dimensions we can encounter


def get_ds_dims_info(ds_list):

    ds_dims = []
    for ds in ds_list:
        ds_dims.append(ds.dims)

    dims_df = pd.DataFrame(ds_dims)
    dims_sum = dims_df.sum(axis=0).to_dict()
    dims_max = dims_df.max(axis=0).to_dict()
    dims_csum = dims_df.cumsum(axis=0).to_dict()

    return dims_sum, dims_csum, dims_max, dims_df


def get_temp_arr_vals(dims, dims_max, dims_sum):

    shape = [dims_max[dim] if dim in const_dims else dims_sum[dim] for dim in dims]

    chnk_shape = [None if dim in const_dims else dims_max[dim] for dim in dims]

    return shape, chnk_shape


def construct_lazy_ds(ds_model, dims_sum, dims_max):

    xr_dict = dict()

    unwritten_dict = dict()
    for name, val in ds_model.variables.items():

        if (name not in possible_dims) and (val.dims != ('channel',)):  # TODO: hard coded, can we avoid it?
            shape, chnk_shape = get_temp_arr_vals(val.dims, dims_max, dims_sum)

            temp_arr = dask.array.zeros(shape=shape, chunks=chnk_shape, dtype=val.dtype)

            xr_dict[name] = (val.dims, temp_arr, val.attrs)

        else:
            unwritten_dict[name] = val

    ds = xr.Dataset(xr_dict)
    ds_unwritten = xr.Dataset(unwritten_dict)

    return ds, ds_unwritten


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


def direct_write(path, ds_list, group):

    dims_sum, dims_csum, dims_max, dims_df = get_ds_dims_info(ds_list)

    # TODO: Do check that all of the channels are the same and times don't overlap and they increase

    ds_lazy, ds_unwritten = construct_lazy_ds(ds_list[0], dims_sum, dims_max)

    # set fill value for each of the arrays
    fill_vals = get_fill_dict(ds_lazy)

    print("group")
    ds_lazy.to_zarr(path, compute=False, group=group, encoding=fill_vals, consolidated=True)

    # variables to drop from each ds and write in later
    drop_vars = list(ds_unwritten) + list(ds_unwritten.dims)

    for i in range(len(ds_list)):  # TODO: parallelize this loop

        region = get_region(i, dims_csum)
        ds_list[i].drop(drop_vars).to_zarr(path, group=group, region=region)


    # TODO: maybe this will work for time:
    # ds_lazy[0][["time1"]].to_zarr(path, group=grp_name, region={'time1': slice(0, 5923)})

    # ds_opened = xr.open_zarr(path, group=group)
    #
    # dims_drop = set(ds_unwritten.dims).intersection(set(time_dims))
    # for name, val in ds_unwritten.drop(dims_drop).items():
    #     ds_opened[name] = val
    #
    # def func(ds):
    #
    #     return ds[time_dims]
    #
    # times = xr.concat(list(map(func, ds_lazy)), dim=time_dims, coords='all').drop("concat_dim")
    #
    # for time, val in times.coords.items():
    #     ds_opened[time] = val



    # TODO: add back in coordinates and attributes for dataset

    # TODO: re-chunk the zarr store after everything has been added

# def lazy_combine(path, eds):
#
#     # TODO: do direct_write(path, ds_list) for each group in eds

