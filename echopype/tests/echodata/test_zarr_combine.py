from echopype.echodata.zarr_combine import ZarrCombine
from dask.distributed import Client
import numpy as np
import xarray as xr
from echopype.utils.coding import set_time_encodings
from typing import List, Tuple, Dict
import tempfile
import pytest
import zarr
from pathlib import Path


@pytest.fixture(
    params=[
        (
                {
                    "randint_low": 10,
                    "randint_high": 5000,
                    "num_datasets": 20,
                    "group": "test_group",
                    "zarr_name": "combined_echodatas.zarr",
                    "delayed_ds_list": False
                }
        ),
        (
                {
                    "randint_low": 10,
                    "randint_high": 5000,
                    "num_datasets": 20,
                    "group": "test_group",
                    "zarr_name": "combined_echodatas.zarr",
                    "delayed_ds_list": True
                }
        )
    ],
    ids=["in-memory-ds_list", "lazy-ds_list"],
    scope="module"
)
def append_ds_list_params(request):
    return list(request.param.values())


def get_ranges(lengths: np.ndarray) -> List[Tuple[int, int]]:
    """
    Obtains a list of ranges using the provided
    ``lengths`` input.

    Parameters
    ----------
    lengths: np.ndarray
        The length for each range

    Returns
    -------
    test_ds_ranges: List[Tuple[int, int]]
        A list of tuples specifying ranges, where
        each element has a length corresponding to ``lengths``.

    Examples
    --------
    get_ranges(np.array([6,7,3,9])) -> [(0, 6), (6, 13), (13, 16), (16, 25)]
    """
    # TODO: for some reason, Pycharm debugger won't allow properly formatted Examples

    cummulative_lengths = np.cumsum(lengths)

    test_ds_ranges = []
    for ind, end_val in enumerate(cummulative_lengths):

        if ind == 0:
            test_ds_ranges.append((0, end_val))
        else:
            test_ds_ranges.append((cummulative_lengths[ind - 1], end_val))

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

    Notes
    -----
    If a coordinate is a time value (i.e., having variable names like "*_time" or "time_*"),
    then ``set_time_encodings`` will change the integer values of the coordinates to datetime stamps.

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

    # gather the coordinates of the Dataset
    xr_coords_dict = dict()
    for dim, dim_range in append_dims_ranges.items():
        dim_array = np.arange(dim_range[0], dim_range[1])
        xr_coords_dict[dim] = ([dim], dim_array)

    # gather the variables of the Dataset
    xr_vars_dict = dict()
    for var, var_dim in var_names_dims.items():
        var_range = append_dims_ranges[var_dim]
        var_array = np.arange(var_range[0], var_range[1], dtype=np.float64)
        xr_vars_dict[var] = (var_dim, var_array)

    # construct Dataset
    ds = xr.Dataset(xr_vars_dict, coords=xr_coords_dict)

    # set time encodings for time coordinates
    ds = set_time_encodings(ds)

    return ds


def get_all_test_data(num_datasets: int, randint_low: int,
                      randint_high: int) -> Tuple[List[xr.Dataset],
                                                  xr.Dataset, int, int]:
    """
    Generates a list of ``num_datasets`` Datasets with variable and
    coordinate lengths between ``[randint_low, randint_high)``.
    Additionally, obtains the true combined form of all elements in
    the aforementioned generated list of Datasets.

    Parameters
    ----------
    num_datasets: int
        The number of Datasets to generate
    randint_low: int
        The smallest length allowed for variables/coordinates
    randint_high: int
        The largest length allowed for variables/coordinates

    Returns
    -------
    ds_list: List[xr.Dataset]
        The generated list of Datasets
    true_comb: xr.Dataset
        The true combined form of the datasets in ``ds_list``
    max_time1_len: int
        The max length for all ``time1`` dims that will be combined
    max_time2_len: int
        The max length for all ``time2`` dims that will be combined
    """

    # generate differing time1 and time2 lengths for each Dataset
    time1_lengths = np.random.randint(low=randint_low, high=randint_high, size=num_datasets)
    time2_lengths = np.random.randint(low=randint_low, high=randint_high, size=num_datasets)

    # get time1 and time2 value ranges based off of lengths
    time1_ranges = get_ranges(time1_lengths)
    time2_ranges = get_ranges(time2_lengths)

    # get the true combined ranges for time1 and time2
    time1_true_range = (0, int(np.sum(time1_lengths, axis=0)))
    time2_true_range = (0, int(np.sum(time2_lengths, axis=0)))

    # assign variable names and their dimension
    var_names_dims = {"var1": "time1", "var2": "time2"}

    # generate the expected final combined Dataset
    true_comb = generate_test_ds(append_dims_ranges={"time1": time1_true_range,
                                                     "time2": time2_true_range},
                                 var_names_dims=var_names_dims)

    # generate the list of Datasets that will be combined
    ds_list = []
    for ind in range(num_datasets):
        ds_list.append(generate_test_ds(append_dims_ranges={"time1": time1_ranges[ind],
                                                            "time2": time2_ranges[ind]},
                                        var_names_dims=var_names_dims))

    # collect max length for time1 and time2, so we can determine the appropriate chunk shape
    max_time1_len = np.max(time1_lengths)
    max_time2_len = np.max(time2_lengths)

    return ds_list, true_comb, max_time1_len, max_time2_len


def test_append_ds_list_to_zarr(append_ds_list_params):
    """
    This is a minimal test for ``_append_ds_list_to_zarr`` that
    ensures we are properly combining variables and coordinates
    in a list of Datasets. This is done by creating a list of mock
    Datasets, which will take on a known form when combined. We then
    combine these Datasets using ``ZarrCombine._append_ds_list_to_zarr``
    and compare the generated Dataset against the known form.

    Notes
    -----
    Testing the combining of a list of delayed Datasets is possible
    by setting the variable ``delayed_ds_list=True``

    The variable ``randint_low`` should always be greater than or equal to 1.
    """

    # gather test parameters
    (
        randint_low,
        randint_high,
        num_datasets,
        group,
        zarr_name,
        delayed_ds_list
    ) = append_ds_list_params

    # initialize ZarrCombine
    comb = ZarrCombine()

    # create temporary directory for zarr store
    temp_zarr_dir = tempfile.TemporaryDirectory()
    zarr_path = Path(temp_zarr_dir.name + "/" + zarr_name)

    # obtain a client with a local scheduler
    client = Client()

    # generate the ds_list and the known combined form of the list
    ds_list, true_comb, max_time1_len, max_time2_len = get_all_test_data(randint_low=randint_low,
                                                                         randint_high=randint_high,
                                                                         num_datasets=num_datasets)

    if delayed_ds_list:

        # write ds_list to zarr
        for count, ds in enumerate(ds_list):
            ds_zarr_path = Path(temp_zarr_dir.name + "/ds_sets/file_" + str(count) + ".zarr")
            ds.to_zarr(ds_zarr_path)

        # get lazy ds_list
        ds_list_lazy = []
        for count, ds in enumerate(ds_list):
            ds_zarr_path = Path(temp_zarr_dir.name + "/ds_sets/file_" + str(count) + ".zarr")

            ds_list_lazy.append(xr.open_zarr(ds_zarr_path))

        ds_list = ds_list_lazy

    # combine ds_list using ZarrCombine method
    _ = comb._append_ds_list_to_zarr(
        zarr_path,
        ds_list=ds_list,
        zarr_group=group,
        ed_name=group,
        storage_options={},
    )

    # write the time1 and time2 coordinates to final combined zarr
    comb._write_append_dims(ds_list, zarr_path, group, storage_options={})

    # open combined Dataset produced by ZarrCombine method
    final_comb = xr.open_zarr(zarr_path, group=group)

    # ensure that the ZarrCombine method correctly combines ds_list
    assert true_comb.identical(final_comb)

    # ensure that the final combined file has the correct chunk shapes
    for var_name in final_comb.variables:

        z1 = zarr.open_array(zarr_path / f"{group}/{var_name}")

        if var_name in ["var1", "time1"]:
            assert z1.chunks == (min(comb.max_append_chunk_size, max_time1_len),)
        else:
            assert z1.chunks == (min(comb.max_append_chunk_size, max_time2_len),)

    # close client and scheduler
    client.close()

    # remove temporary directory
    temp_zarr_dir.cleanup()


