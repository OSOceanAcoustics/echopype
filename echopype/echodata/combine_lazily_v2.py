import xarray as xr
import pandas as pd
import dask.array
import dask

const_dims = ['channel']

def get_ds_dims_info(ds_list):

    ds_dims = []
    for ds in ds_list:
        ds_dims.append(ds.dims)

    dims_df = pd.DataFrame(ds_dims)
    dims_sum = dims_df.sum(axis=0).to_dict()
    dims_max = dims_df.max(axis=0).to_dict()
    dims_csum = dims_df.cumsum(axis=0).to_dict()

    return dims_sum, dims_csum, dims_max


def get_temp_arr_vals(dims, dims_max, dims_sum):

    shape = [dims_max[dim] if dim in const_dims else dims_sum[dim] for dim in dims]

    chnk_shape = [None if dim in const_dims else dims_max[dim] for dim in dims]

    return shape, chnk_shape


def constuct_lazy_ds(ds_model, dims_sum, dims_max):

    xr_dict = dict()

    unwritten_vars = []
    for name, val in ds_model.variables.items():

        if ('channel',) != val.dims:
            shape, chnk_shape = get_temp_arr_vals(val.dims, dims_max, dims_sum)
            temp_arr = dask.array.zeros(shape=shape, chunks=chnk_shape)

            xr_dict[name] = (val.dims, temp_arr, val.attrs)

        else:
            unwritten_vars.append(name)

    ds = xr.Dataset(xr_dict)

    return ds, unwritten_vars


def get_region(ds_ind, dims_csum):

    print([csum[ds_ind] for dim, csum in dims_csum.items()])

    if ds_ind == 0:
        region = {dim: slice(0, csum[ds_ind]) for dim, csum in dims_csum.items() if dim not in const_dims}

    else:
        region = {dim: slice(csum[ds_ind-1], csum[ds_ind]) for dim, csum in dims_csum.items() if dim not in const_dims}

    return region



def direct_write(path, ds_list):

    dims_sum, dims_csum, dims_max = get_ds_dims_info(ds_list)

    ds_lazy, unwritten_vars = constuct_lazy_ds(ds_list[0], dims_sum, dims_max)

    # ds_lazy.to_zarr(path, compute=False)

    for i in range(len(ds_list)):

        print(get_region(i, dims_csum))


    #
    # eds_lazy[0] = eds_lazy[0].drop(['time1', 'channel', 'frequency_nominal'])
    # eds_lazy[0].to_zarr(path, region={"time1": slice(0, var_cumulative_sum["time1"].loc[0])})
    #
    # for i in range(1, len(eds_lazy)):
    #     print(i)
    #     eds_lazy[i] = eds_lazy[i].drop(['time1', 'channel', 'frequency_nominal'])
    #
    #     print(var_cumulative_sum["time1"].loc[i - 1], var_cumulative_sum["time1"].loc[i])
    #     slc = slice(var_cumulative_sum["time1"].loc[i - 1], var_cumulative_sum["time1"].loc[i])
    #     eds_lazy[i].to_zarr(path, region={"time1": slc})


# def lazy_combine(path, eds):
#
#     # TODO: do direct_write(path, ds_list) for each group in eds

