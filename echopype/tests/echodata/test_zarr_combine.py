from echopype.echodata.zarr_combine import ZarrCombine
from dask.distributed import Client
import numpy as np
import xarray as xr
from echopype.utils.coding import set_encodings
from typing import List, Tuple, Dict
import tempfile


def get_ranges(chunks: np.ndarray) -> List[Tuple[int, int]]:
    """
    Obtains a list of ranges using the provided
    ``chunks`` input

    Parameters
    ----------
    chunks: np.ndarray
        The chunk size for each range

    Returns
    -------
    test_ds_ranges: List[Tuple[int, int]]
        A list of tuples specifying ranges, where
        each element has a chunk size corresponding to ``chunks``.

    Examples
    --------
    get_ranges(np.array([6,7,3,9])) -> [(0, 6), (6, 13), (13, 16), (16, 25)]
    """
    # TODO: for some reason, Pycharm debugger wont allow properly formatted Examples

    cummulative_chunks = np.cumsum(chunks)

    test_ds_ranges = []
    for ind, end_val in enumerate(cummulative_chunks):

        if ind == 0:
            test_ds_ranges.append((0, end_val))
        else:
            test_ds_ranges.append((cummulative_chunks[ind - 1], end_val))

    return test_ds_ranges


def generate_test_ds(append_dims_ranges: Dict[str, Tuple[int, int]],
                     var_names_dims: Dict[str, str]) -> xr.Dataset:
    """
    Constructs a test Dataset.

    Parameters
    ----------
    append_dims_ranges: Dict[str, Tuple[int, int]]
        Dictionary specifying the dimensions/coordinates
        of the generated test Dataset. The keys correspond
        to the name of the dimension and the values are a
        tuple of length two where the elements are the start
        and end of the range values, respectively.
    var_names_dims: Dict[str, str]
        Dictionary specifying the variables of the Dataset.
        The keys correspond to the variable name and the
        values are the name of the dimension of the variable.

    Returns
    -------
    ds: xr.Dataset
        A test Dataset where the values of each variable and
        coordinate are between the start and end ranges
        specified by ``append_dims_ranges`` with step size 1.

    Examples
    --------
    generate_test_ds(append_dims_ranges: {'time1': (0, 21), 'time2': (0, 14)},
                     var_names_dims: {'var1': 'time1', 'var2': 'time2'})

    <xarray.Dataset>
    Dimensions:  (time1: 21, time2: 14)
    Coordinates:
        * time1    (time1) datetime64[ns] 1900-01-01 ... 1900-01-01T00:00:20
        * time2    (time2) datetime64[ns] 1900-01-01 ... 1900-01-01T00:00:13
    Data variables:
        var1     (time1) float64 0.0 1.0 2.0 3.0 4.0 ... 16.0 17.0 18.0 19.0 20.0
        var2     (time2) float64 0.0 1.0 2.0 3.0 4.0 5.0 ... 9.0 10.0 11.0 12.0 13.0
    """

    xr_coords_dict = dict()
    for dim, dim_range in append_dims_ranges.items():
        dim_array = np.arange(dim_range[0], dim_range[1])

        xr_coords_dict[dim] = ([dim], dim_array)

    xr_vars_dict = dict()
    for var, var_dim in var_names_dims.items():
        var_range = append_dims_ranges[var_dim]
        var_array = np.arange(var_range[0], var_range[1], dtype=np.float64)

        xr_vars_dict[var] = (var_dim, var_array)

    ds = xr.Dataset(xr_vars_dict, coords=xr_coords_dict)

    ds = set_encodings(ds)

    return ds


def get_all_test_data(num_files, randint_low, randint_high):
    """
    Generates a list of Datasets with variable and
    dimensions of length specified by ``chunks``.
    Additionally, obtains the true combined form
    of all elements in the afformentioned generated
    list of Datasets.


    TODO: finish documenting
    """

    time1_chunks = np.random.randint(low=randint_low, high=randint_high, size=num_files)
    time2_chunks = np.random.randint(low=randint_low, high=randint_high, size=num_files)

    time1_ranges = get_ranges(time1_chunks)
    time2_ranges = get_ranges(time2_chunks)

    time1_true_range = (0, int(np.sum(time1_chunks, axis=0)))
    time2_true_range = (0, int(np.sum(time2_chunks, axis=0)))

    var_names_dims = {"var1": "time1", "var2": "time2"}
    true_comb = generate_test_ds(append_dims_ranges={"time1": time1_true_range,
                                                     "time2": time2_true_range},
                                 var_names_dims=var_names_dims)

    ds_list = []
    for chunk in range(num_files):
        ds_list.append(generate_test_ds(append_dims_ranges={"time1": time1_ranges[chunk],
                                                            "time2": time2_ranges[chunk]},
                                        var_names_dims=var_names_dims))

    return true_comb, ds_list


def test_in_memory_ds_append_ds_list_to_zarr():
    """
    This is a minimal test for ``_append_ds_list_to_zarr`` that
    ensures that we are

    """

    comb = ZarrCombine()

    # create temporary directory for zarr store
    temp_zarr_dir = tempfile.TemporaryDirectory()
    zarr_path = temp_zarr_dir.name + "/combined_echodatas.zarr"

    group = "test_group"

    client = Client()

    # true_comb, ds_list = get_all_test_data(randint_low=10, randint_high=5000, num_files=20)
    true_comb, ds_list = get_all_test_data(randint_low=1, randint_high=10, num_files=4)

    _ = comb._append_ds_list_to_zarr(
        zarr_path,
        ds_list=ds_list,
        zarr_group=group,
        ed_name=group,
        storage_options={},
    )

    # TODO: uncomment the below lines once PR #824 has been merged
    # comb._write_append_dims(ds_list, zarr_path, group, storage_options={})

    final_comb = xr.open_zarr(zarr_path, group=group)

    assert true_comb.identical(final_comb)

    client.close()
    temp_zarr_dir.cleanup()


def test_lazy_ds_append_ds_list_to_zarr():
    """
    This is a minimal test for ``_append_ds_list_to_zarr`` that
    ensures that we are

    """

    comb = ZarrCombine()

    # create temporary directory for zarr store
    temp_zarr_dir = tempfile.TemporaryDirectory()
    zarr_path = temp_zarr_dir.name + "/combined_echodatas.zarr"

    group = "test_group"

    client = Client()

    true_comb, ds_list = get_all_test_data(randint_low=10, randint_high=5000, num_files=20)

    # write ds_list to zarr
    for count, ds in enumerate(ds_list):

        ds_zarr_path = temp_zarr_dir.name + "/ds_sets/file_" + str(count) + ".zarr"
        ds.to_zarr(ds_zarr_path)

    # get lazy ds_list
    ds_list_lazy = []
    for count, ds in enumerate(ds_list):

        ds_zarr_path = temp_zarr_dir.name + "/ds_sets/file_" + str(count) + ".zarr"

        ds_list_lazy.append(xr.open_zarr(ds_zarr_path))

    _ = comb._append_ds_list_to_zarr(
        zarr_path,
        ds_list=ds_list_lazy,
        zarr_group=group,
        ed_name=group,
        storage_options={},
    )

    # TODO: uncomment the below lines once PR #824 has been merged
    # comb._write_append_dims(ds_list_lazy, zarr_path, group, storage_options={})

    final_comb = xr.open_zarr(zarr_path, group=group)

    assert true_comb.identical(final_comb)

    client.close()
    temp_zarr_dir.cleanup()

