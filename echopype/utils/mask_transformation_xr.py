import numpy as np
import xarray as xr


def lin(db: xr.DataArray) -> xr.DataArray:
    """Convert decibel to linear scale, handling NaN values."""
    linear = xr.where(db.isnull(), np.nan, 10 ** (db / 10))
    return linear


def log(linear: xr.DataArray) -> xr.DataArray:
    """
    Turn variable into the logarithmic domain. This function will return -999
    in the case of values less or equal to zero (undefined logarithm). -999 is
    the convention for empty water or vacant sample in fisheries acoustics.

    Args:
        variable (float): array of elements to be transformed.

    Returns:
        float: array of elements transformed
    """
    back_list = False
    back_single = False
    if not isinstance(linear, xr.DataArray):
        if isinstance(linear, list):
            linear = xr.DataArray(linear)
            back_list = True
        else:
            linear = xr.DataArray([linear])
            back_single = True

    db = xr.apply_ufunc(lambda x: 10 * np.log10(x), linear)
    db = xr.where(db.isnull(), -999, db)
    db = xr.where(linear == 0, -999, db)
    if back_list:
        db = db.values
    if back_single:
        db = db.values[0]
    return db


def downsample(dataset, coordinates: {str: int}, operation: str = "mean", is_log: bool = False):
    """
    Given a dataset, downsamples it on the specified coordinates

    Args:
        dataset (xr.DataArray)  : the dataset to resample
        coordinates({str: int}  : a mapping of dimensions to the windows to use
        operation (str)         : the downsample operation to use
        is_log (bool)           : True if the data is logarithmic and should be
                                    converted to linear

    Returns:
        xr.DataArray            : the resampled dataset
    """
    operation_list = ["mean", "sum"]
    if operation not in operation_list:
        raise Exception("Operation not in approved list")
    for k in coordinates.keys():
        if k not in dataset.dims:
            raise Exception("Coordinate " + k + " not in dataset coordinates")
    if is_log:
        dataset = lin(dataset)
    if operation == "mean":
        dataset = dataset.coarsen(coordinates, boundary="pad").mean()
    elif operation == "sum":
        dataset = dataset.coarsen(coordinates, boundary="pad").sum()
    else:
        raise Exception("Operation not in approved list")
    # print(dataset)
    if is_log:
        dataset = log(dataset)
    # mask = dataset.isnull()
    return dataset


def upsample(dataset: xr.DataArray, dataset_size: xr.DataArray):
    """
    Given a data dataset and an example dataset, upsamples the data dataset
    to the example dataset's dimensions by repeating values

    Args:
        dataset (xr.DataArray)      : data
        dataset_size (xr.DataArray) : dataset of the right size

    Returns
        xr.DataArray: the input dataset, with the same coords as dataset_size
        and the values repeated to fill it up.
    """

    interpolated = dataset.interp_like(dataset_size, method="nearest")
    return interpolated


def line_to_square(one: xr.DataArray, two: xr.DataArray, dim: str):
    """
    Given a single dimension dataset and an example dataset with 2 dimensions,
    returns a two-dimensional dataset that is the single dimension dataset
    repeated as often as needed

    Args:
        one (xr.DataArray): data
        two (xr.DataArray): shape dataset
        dim (str): name of dimension to concat against

    Returns:
        xr.DataArray: the input dataset, with the same coords as dataset_size and
        the values repeated to fill it up
    """
    length = len(two[dim])
    array_list = [one for _ in range(0, length)]
    array = xr.concat(array_list, dim=dim)
    # return_data = xr.DataArray(data=array.values, dims=two.dims, coords=two.coords)
    return array.values
