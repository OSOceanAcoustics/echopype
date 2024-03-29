from typing import Literal

import iris  # noqa
import iris.cube  # noqa
import numpy as np
import xarray as xr
from iris.coords import DimCoord  # noqa

from ..utils.prov import echopype_prov_attrs, insert_input_processing_level

FNAME = "filenames"
CHANNEL = "channel"
PING_TIME = "ping_time"
RANGE_SAMPLE = "range_sample"
ECHO_RANGE = "echo_range"
DEPTH = "depth"
Sv_var = "Sv"

# Iris dims
PROJECTION_X = "projection_x_coordinate"
PROJECTION_Y = "projection_y_coordinate"


def regrid_Sv(
    input_ds: xr.Dataset,
    target_grid: xr.Dataset,
    range_var: Literal["echo_range", "depth"] = ECHO_RANGE,
) -> xr.Dataset:
    """
    Regrid Sv data to a desired grid

    Parameters
    ----------
    input_ds : xr.Dataset
        The input dataset containing Sv data
    target_grid : xr.Dataset
        The target grid to regrid the data to,
        this dataset should only contain coordinates
    range_var : {'echo_range', 'depth'}
        The name of the range variable, by default "echo_range"

    Returns
    -------
    xr.Dataset
        The regridded dataset
    """
    if FNAME in input_ds.dims:
        input_ds = input_ds.drop_dims(FNAME)

    # Get target dims
    target_dims = _get_iris_dims(target_grid, range_var)

    # Regrid each channel separately
    ds_list = []
    for chan in input_ds[CHANNEL]:
        channel_Sv = input_ds.sel(channel=chan)
        original_dims = _get_iris_dims(channel_Sv, range_var)
        regrid_ds = _regrid_data(channel_Sv[Sv_var].data, original_dims, target_dims)
        ds_list.append(regrid_ds)

    # Convert back to match input dataset
    result_ds = xr.concat(ds_list, dim=CHANNEL).to_dataset(name=Sv_var)
    result_ds[Sv_var].attrs = {
        **input_ds[Sv_var].attrs,
        "actual_range": [
            round(float(input_ds[Sv_var].min().values), 2),
            round(float(input_ds[Sv_var].max().values), 2),
        ],
    }

    # Assign original coordinates back
    result_ds = result_ds.assign_coords(
        {
            CHANNEL: input_ds[CHANNEL],
            PING_TIME: (PROJECTION_X, target_grid[PING_TIME].data, input_ds[PING_TIME].attrs),
            RANGE_SAMPLE: (
                PROJECTION_Y,
                np.arange(0, len(target_grid[range_var])),
                input_ds[RANGE_SAMPLE].attrs,
            ),
        }
    )

    # Swap dims back to original
    result_ds = result_ds.swap_dims(
        {
            PROJECTION_Y: RANGE_SAMPLE,
            PROJECTION_X: PING_TIME,
        }
    ).drop([PROJECTION_Y, PROJECTION_X])

    # Re-attach some variables
    result_ds["frequency_nominal"] = input_ds["frequency_nominal"]  # re-attach frequency_nominal
    result_ds[range_var] = (
        (CHANNEL, PING_TIME, RANGE_SAMPLE),
        np.array(
            [[target_grid[range_var].data] * len(target_grid[PING_TIME])] * len(result_ds[CHANNEL])
        ),
        input_ds[range_var].attrs,
    )
    # Add water level if it exists in Sv dataset
    water_level = "water_level"
    if range_var == ECHO_RANGE and water_level in input_ds.data_vars:
        result_ds[water_level] = input_ds[water_level]

    # Add provenance related attributes
    prov_dict = echopype_prov_attrs(process_type="processing")
    prov_dict["processing_function"] = "commongrid.regrid_Sv"
    result_ds = result_ds.assign_attrs(prov_dict)
    result_ds = insert_input_processing_level(result_ds, input_ds=input_ds)

    return result_ds


def _get_iris_dims(ds, range_var: Literal["echo_range", "depth"] = "echo_range"):
    range_dim = "range"

    # Original grid
    original_dims = []
    for idx, dim in enumerate(ds.dims):
        data_array = ds[dim]
        kwargs = {}
        if dim == PING_TIME:
            if not np.issubdtype(data_array.dtype, np.datetime64):
                raise TypeError(f"Expected time dimension to be datetime64, got {data_array.dtype}")
            data_array = data_array.astype("float64")
            standard_name = PROJECTION_X
        elif (dim.startswith(range_dim) or dim.endswith(range_dim)) or dim == "depth":
            data_array = ds[range_var]
            if PING_TIME in data_array.dims:
                data_array = data_array.isel({PING_TIME: 0})
            standard_name = PROJECTION_Y
        else:
            raise ValueError(f"Unknown dimension {dim}")
        kwargs.update(
            {
                "standard_name": standard_name,
                "long_name": data_array.attrs.get("long_name", None),
                # "units": data_array.attrs.get("units", None),
            }
        )
        iris_dim = (DimCoord(data_array, **kwargs), idx)
        original_dims.append(iris_dim)
    return original_dims


def _regrid_data(data, old_dims, new_dims, regridder=None):
    """
    Regrid data with iris regridder

    Original code: https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-classifiers-unet/blob/master/crimac_unet/data_preprocessing/regridding.py#L35-L57
    :param data: data to be regridded, 2D or 3D
    :param old_dims: old data dimensions (list of Iris DimCoord)
    :param new_dims: new data dimensions (list of Iris DimCoord)
    :param regridder: iris regrid algorithm
    :return:
    """  # noqa
    orig_cube = iris.cube.Cube(data, dim_coords_and_dims=old_dims)
    grid_cube = iris.cube.Cube(
        np.zeros([coord[0].shape[0] for coord in new_dims]), dim_coords_and_dims=new_dims
    )

    try:
        orig_cube.coord("projection_y_coordinate").guess_bounds()
        orig_cube.coord("projection_x_coordinate").guess_bounds()
        grid_cube.coord("projection_y_coordinate").guess_bounds()
        grid_cube.coord("projection_x_coordinate").guess_bounds()
    except ValueError:
        pass

    if regridder is None:
        regridder = iris.analysis.AreaWeighted(mdtol=1)
    regrid = orig_cube.regrid(grid_cube, regridder)
    regrid_ds = xr.DataArray.from_iris(regrid)
    return regrid_ds
