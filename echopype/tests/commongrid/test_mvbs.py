import dask.array
import numpy as np
import pandas as pd
import pytest
from typing import Tuple, Iterable, Union
from echopype.commongrid.mvbs import get_MVBS_along_channels
from echopype.consolidate.api import POSITION_VARIABLES
from flox.xarray import xarray_reduce


def create_bins(csum_array: np.ndarray) -> Iterable:
    """
    Constructs bin ranges based off of a cumulative
    sum array.

    Parameters
    ----------
    csum_array: np.ndarray
        1D array representing a cumulative sum

    Returns
    -------
    bins: list
        A list whose elements are the lower and upper bin ranges
    """

    bins = []

    # construct bins
    for count, csum in enumerate(csum_array):

        if count == 0:

            bins.append([0, csum])

        else:

            # add 0.01 so that left bins don't overlap
            bins.append([csum_array[count-1] + 0.01, csum])

    return bins


def create_echo_range_related_data(ping_bins: Iterable,
                                   num_pings_in_bin: np.ndarray,
                                   er_range: list, er_bins: Iterable,
                                   final_num_er_bins: int,
                                   create_dask: bool,
                                   rng: np.random.Generator,
                                   ping_bin_nan_ind: int) -> Tuple[list, list, list]:
    """
    Creates ``echo_range`` values and associated bin information.

    Parameters
    ----------
    ping_bins: list
        A list whose elements are the lower and upper ping time bin ranges
    num_pings_in_bin: np.ndarray
        Specifies the number of pings in each ping time bin
    er_range: list
        A list whose first element is the lowest and second element is
        the highest possible number of echo range values in a given bin
    er_bins:  list
        A list whose elements are the lower and upper echo range bin ranges
    final_num_er_bins: int
        The total number of echo range bins
    create_dask: bool
        If True ``final_arrays`` values will be
        dask arrays, else they will be numpy arrays
    rng: np.random.Generator
        The generator for random values
    ping_bin_nan_ind: int
        The ping bin index to fill with NaNs

    Returns
    -------
    all_er_bin_nums: list of np.ndarray
        A list whose elements are the number of values in each echo_range
        bin, for each ping bin
    ping_times_in_bin: list of np.ndarray
        A list whose elements are the ping_time values for each corresponding bin
    final_arrays: list of np.ndarray or dask.array.Array
        A list whose elements are the echo_range values for a given ping and
        echo range bin block
    """

    final_arrays = []
    all_er_bin_nums = []
    ping_times_in_bin = []

    # build echo_range array
    for ping_ind, ping_bin in enumerate(ping_bins):

        # create the ping times associated with each ping bin
        ping_times_in_bin.append(rng.uniform(ping_bin[0], ping_bin[1], (num_pings_in_bin[ping_ind],)))

        # randomly determine the number of values in each echo_range bin
        num_er_in_bin = rng.integers(low=er_range[0], high=er_range[1], size=final_num_er_bins)

        # store the number of values in each echo_range bin
        all_er_bin_nums.append(num_er_in_bin)

        er_row_block = []
        for count, bin_val in enumerate(er_bins):

            # create a block of echo_range values
            if create_dask:
                a = dask.array.random.uniform(bin_val[0], bin_val[1], (num_pings_in_bin[ping_ind],
                                                                       num_er_in_bin[count]))
            else:
                a = rng.uniform(bin_val[0], bin_val[1], (num_pings_in_bin[ping_ind],
                                                         num_er_in_bin[count]))

            # store the block of echo_range values
            er_row_block.append(a)

            # set all echo_range values at ping index to NaN
            if ping_ind == ping_bin_nan_ind:
                a[:, :] = np.nan

        # collect and construct echo_range row block
        final_arrays.append(np.concatenate(er_row_block, axis=1))

    return all_er_bin_nums, ping_times_in_bin, final_arrays


def construct_2d_echo_range_array(final_arrays: Iterable[np.ndarray],
                                  ping_csum: np.ndarray,
                                  create_dask: bool) -> Tuple[Union[np.ndarray, dask.array.Array], int]:
    """
    Creates the final 2D ``echo_range`` array with appropriate padding.

    Parameters
    ----------
    final_arrays: list of np.ndarray
        A list whose elements are the echo_range values for a given ping and
        echo range bin block
    ping_csum: np.ndarray
        1D array representing the cumulative sum for the number of ping times
        in each ping bin
    create_dask: bool
        If True ``final_er`` will be a dask array, else it will be a numpy array

    Returns
    -------
    final_er: np.ndarray or dask.array.Array
        The final 2D ``echo_range`` array
    max_num_er_elem: int
        The maximum number of ``echo_range`` elements amongst all times
    """

    # get maximum number of echo_range elements amongst all times
    max_num_er_elem = max([arr.shape[1] for arr in final_arrays])

    # total number of ping times
    tot_num_times = ping_csum[-1]

    # pad echo_range dimension with nans and create final echo_range
    if create_dask:
        final_er = dask.array.ones(shape=(tot_num_times, max_num_er_elem)) * np.nan
    else:
        final_er = np.empty((tot_num_times, max_num_er_elem))
        final_er[:] = np.nan

    for count, arr in enumerate(final_arrays):

        if count == 0:
            final_er[0:ping_csum[count], 0:arr.shape[1]] = arr
        else:
            final_er[ping_csum[count - 1]:ping_csum[count], 0:arr.shape[1]] = arr

    return final_er, max_num_er_elem


def construct_2d_sv_array(max_num_er_elem: int, ping_csum: np.ndarray,
                          all_er_bin_nums: Iterable[np.ndarray],
                          num_pings_in_bin: np.ndarray,
                          create_dask: bool,
                          ping_bin_nan_ind: int) -> Tuple[Union[np.ndarray, dask.array.Array],
                                                          np.ndarray]:
    """
    Creates the final 2D Sv array with appropriate padding.

    Parameters
    ----------
    max_num_er_elem: int
        The maximum number of ``echo_range`` elements amongst all times
    ping_csum: np.ndarray
        1D array representing the cumulative sum for the number of ping times
        in each ping bin
    all_er_bin_nums: list of np.ndarray
        A list whose elements are the number of values in each echo_range
        bin, for each ping bin
    num_pings_in_bin: np.ndarray
        Specifies the number of pings in each ping time bin
    create_dask: bool
        If True ``final_sv`` will be a dask array, else it will be a numpy array
    ping_bin_nan_ind: int
        The ping bin index to fill with NaNs

    Returns
    -------
    final_sv: np.ndarray or dask.array.Array
        The final 2D Sv array
    final_MVBS: np.ndarray
        The final 2D known MVBS array
    """

    # total number of ping times
    tot_num_times = ping_csum[-1]

    # pad echo_range dimension with nans and create final sv
    if create_dask:
        final_sv = dask.array.ones(shape=(tot_num_times, max_num_er_elem)) * np.nan
    else:
        final_sv = np.empty((tot_num_times, max_num_er_elem))
        final_sv[:] = np.nan

    final_means = []
    for count, arr in enumerate(all_er_bin_nums):

        # create sv row values using natural numbers
        sv_row_list = [np.arange(1, num_elem + 1, 1, dtype=np.float64) for num_elem in arr]

        # create final sv row
        sv_row = np.concatenate(sv_row_list)

        # get final mean which is n+1/2 (since we are using natural numbers)
        ping_mean = [(len(elem) + 1) / 2.0 for elem in sv_row_list]

        # create sv row block
        sv_row_block = np.tile(sv_row, (num_pings_in_bin[count], 1))

        if count == ping_bin_nan_ind:

            # fill values with NaNs
            ping_mean = [np.nan]*len(sv_row_list)
            sv_row_block[:, :] = np.nan

        # store means for ping
        final_means.append(ping_mean)

        if count == 0:
            final_sv[0:ping_csum[count], 0:sv_row_block.shape[1]] = sv_row_block
        else:
            final_sv[ping_csum[count - 1]:ping_csum[count], 0:sv_row_block.shape[1]] = sv_row_block

    # create final sv MVBS
    final_MVBS = np.vstack(final_means)

    return final_sv, final_MVBS


def create_known_mean_data(final_num_ping_bins: int,
                           final_num_er_bins: int,
                           ping_range: list,
                           er_range: list, create_dask: bool,
                           rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Iterable,
                                                              Iterable, np.ndarray, np.ndarray]:
    """
    Orchestrates the creation of ``echo_range``, ``ping_time``, and ``Sv`` arrays
    where the MVBS is known.

    Parameters
    ----------
    final_num_ping_bins: int
        The total number of ping time bins
    final_num_er_bins: int
        The total number of echo range bins
    ping_range: list
        A list whose first element is the lowest and second element is
        the highest possible number of ping time values in a given bin
    er_range: list
        A list whose first element is the lowest and second element is
        the highest possible number of echo range values in a given bin
    create_dask: bool
        If True the ``Sv`` and ``echo_range`` values produced will be
        dask arrays, else they will be numpy arrays.
    rng: np.random.Generator
        generator for random integers

    Returns
    -------
    final_MVBS: np.ndarray
        The final 2D known MVBS array
    final_sv: np.ndarray
        The final 2D Sv array
    ping_bins: Iterable
        A list whose elements are the lower and upper ping time bin ranges
    er_bins: Iterable
        A list whose elements are the lower and upper echo range bin ranges
    final_er: np.ndarray
        The final 2D ``echo_range`` array
    final_ping_time: np.ndarray
        The final 1D ``ping_time`` array
    """

    # randomly generate the number of pings in each ping bin
    num_pings_in_bin = rng.integers(low=ping_range[0], high=ping_range[1], size=final_num_ping_bins)

    # create bins for ping_time dimension
    ping_csum = np.cumsum(num_pings_in_bin)
    ping_bins = create_bins(ping_csum)

    # create bins for echo_range dimension
    num_er_in_bin = rng.integers(low=er_range[0], high=er_range[1], size=final_num_er_bins)
    er_csum = np.cumsum(num_er_in_bin)
    er_bins = create_bins(er_csum)

    # randomly select one ping bin to fill with NaNs
    ping_bin_nan_ind = rng.choice(len(ping_bins))

    # create the echo_range data and associated bin information
    all_er_bin_nums, ping_times_in_bin, final_er_arrays = create_echo_range_related_data(ping_bins, num_pings_in_bin,
                                                                                         er_range, er_bins,
                                                                                         final_num_er_bins,
                                                                                         create_dask,
                                                                                         rng,
                                                                                         ping_bin_nan_ind)

    # create the final echo_range array using created data and padding
    final_er, max_num_er_elem = construct_2d_echo_range_array(final_er_arrays, ping_csum, create_dask)

    # get final ping_time dimension
    final_ping_time = np.concatenate(ping_times_in_bin).astype('datetime64[ns]')

    # create the final sv array
    final_sv, final_MVBS = construct_2d_sv_array(max_num_er_elem, ping_csum,
                                                 all_er_bin_nums, num_pings_in_bin,
                                                 create_dask, ping_bin_nan_ind)

    return final_MVBS, final_sv, ping_bins, er_bins, final_er, final_ping_time


@pytest.fixture(
    params=[
        {
            "create_dask": True,
            "final_num_ping_bins": 10,
            "final_num_er_bins": 10,
            "ping_range": [10, 1000],
            "er_range": [10, 1000]
        },
        {
            "create_dask": False,
            "final_num_ping_bins": 10,
            "final_num_er_bins": 10,
            "ping_range": [10, 1000],
            "er_range": [10, 1000]
        },
    ],
    ids=[
        "delayed_data",
        "in_memory_data"
    ],
)
def bin_and_mean_2d_params(request):
    """
    Obtains all necessary parameters for ``test_bin_and_mean_2d``.
    """

    return list(request.param.values())

@pytest.mark.unit
@pytest.mark.parametrize(["range_var", "lat_lon"], [("depth", False), ("echo_range", True), ("echo_range", False)])
def test_get_MVBS_along_channels(request, range_var, lat_lon):
    """Testing the underlying function of compute_MVBS"""
    range_bin = 20
    ping_time_bin = "20S"
    method = "map-reduce"
    
    flox_kwargs = {
        "reindex": True
    }
    
    # Retrieve the correct dataset
    if range_var == "depth":
        ds_Sv = request.getfixturevalue("ds_Sv_er_regular_w_depth")
    elif range_var == "echo_range" and lat_lon is True:
        ds_Sv = request.getfixturevalue("ds_Sv_er_regular_w_latlon")
    else:
        ds_Sv = request.getfixturevalue("ds_Sv_er_regular")
    
    # compute range interval
    echo_range_max = ds_Sv[range_var].max()
    range_interval = np.arange(0, echo_range_max + range_bin, range_bin)
    
    # create bin information needed for ping_time
    d_index = (
        ds_Sv["ping_time"]
        .resample(ping_time=ping_time_bin, skipna=True)
        .asfreq()
        .indexes["ping_time"]
    )
    ping_interval = d_index.union([d_index[-1] + pd.Timedelta(ping_time_bin)])
    
    raw_MVBS = get_MVBS_along_channels(
        ds_Sv, range_interval, ping_interval,
        range_var=range_var, method=method, **flox_kwargs
    )
    
    # Check that the range_var is in the dimension
    assert f"{range_var}_bins" in raw_MVBS.dims
    
    # When it's echo_range and lat_lon, the dataset should have positions
    if range_var == "echo_range" and lat_lon is True:
        assert raw_MVBS.attrs["has_positions"] is True
        assert all(v in raw_MVBS for v in POSITION_VARIABLES)

        # Compute xarray reduce manually for this
        expected_Pos = xarray_reduce(
            ds_Sv[POSITION_VARIABLES],
            ds_Sv["ping_time"],
            func="nanmean",
            expected_groups=(ping_interval),
            isbin=True,
            method=method,
        )
        
        for v in POSITION_VARIABLES:
            assert np.array_equal(raw_MVBS[v].data, expected_Pos[v].data)
    else:
        assert raw_MVBS.attrs["has_positions"] is False
