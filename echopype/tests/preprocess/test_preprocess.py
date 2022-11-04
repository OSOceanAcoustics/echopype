import numpy as np
import pandas as pd
import xarray as xr
import echopype as ep
import pytest
import dask.array

from echopype.preprocess.api import bin_and_mean_2d
from typing import List, Tuple, Iterable, Union


@pytest.fixture(
    params=[
        (
            ("EK60", "ncei-wcsd", "Summer2017-D20170719-T211347.raw"),
            "EK60",
            None,
            {},
        ),
        (
            ("EK80_NEW", "echopype-test-D20211004-T235930.raw"),
            "EK80",
            None,
            {'waveform_mode': 'BB', 'encode_mode': 'complex'},
        ),
        (
            ("EK80_NEW", "D20211004-T233354.raw"),
            "EK80",
            None,
            {'waveform_mode': 'CW', 'encode_mode': 'power'},
        ),
        (
            ("EK80_NEW", "D20211004-T233115.raw"),
            "EK80",
            None,
            {'waveform_mode': 'CW', 'encode_mode': 'complex'},
        ),
        (("ES70", "D20151202-T020259.raw"), "ES70", None, {}),
        (("AZFP", "17082117.01A"), "AZFP", ("AZFP", "17041823.XML"), {}),
        (
            ("AD2CP", "raw", "090", "rawtest.090.00001.ad2cp"),
            "AD2CP",
            None,
            {},
        ),
    ],
    ids=[
        "ek60_cw_power",
        "ek80_bb_complex",
        "ek80_cw_power",
        "ek80_cw_complex",
        "es70",
        "azfp",
        "ad2cp",
    ],
)
def test_data_samples(request, test_path):
    (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    ) = request.param
    if sonar_model.lower() in ['es70', 'ad2cp']:
        pytest.xfail(
            reason="Not supported at the moment",
        )
    path_model, *paths = filepath
    filepath = test_path[path_model].joinpath(*paths)

    if azfp_xml_path is not None:
        path_model, *paths = azfp_xml_path
        azfp_xml_path = test_path[path_model].joinpath(*paths)
    return (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    )


def test_remove_noise():
    """Test remove_noise on toy data"""

    # Parameters for fake data
    nchan, npings, nrange_samples = 1, 10, 100
    chan = np.arange(nchan).astype(str)
    ping_index = np.arange(npings)
    range_sample = np.arange(nrange_samples)
    data = np.ones(nrange_samples)

    # Insert noise points
    np.put(data, 30, -30)
    np.put(data, 60, -30)
    # Add more pings
    data = np.array([data] * npings)
    # Make DataArray
    Sv = xr.DataArray(
        [data],
        coords=[
            ('channel', chan),
            ('ping_time', ping_index),
            ('range_sample', range_sample),
        ],
    )
    Sv.name = "Sv"
    ds_Sv = Sv.to_dataset()

    ds_Sv = ds_Sv.assign(
        echo_range=xr.DataArray(
            np.array([[np.linspace(0, 10, nrange_samples)] * npings]),
            coords=Sv.coords,
        )
    )
    ds_Sv = ds_Sv.assign(sound_absorption=0.001)
    # Run noise removal
    ds_Sv = ep.preprocess.remove_noise(
        ds_Sv, ping_num=2, range_sample_num=5, SNR_threshold=0
    )

    # Test if noise points are are nan
    assert np.isnan(
        ds_Sv.Sv_corrected.isel(channel=0, ping_time=0, range_sample=30)
    )
    assert np.isnan(
        ds_Sv.Sv_corrected.isel(channel=0, ping_time=0, range_sample=60)
    )

    # Test remove noise on a normal distribution
    np.random.seed(1)
    data = np.random.normal(
        loc=-100, scale=2, size=(nchan, npings, nrange_samples)
    )
    # Make Dataset to pass into remove_noise
    Sv = xr.DataArray(
        data,
        coords=[
            ('channel', chan),
            ('ping_time', ping_index),
            ('range_sample', range_sample),
        ],
    )
    Sv.name = "Sv"
    ds_Sv = Sv.to_dataset()
    # Attach required echo_range and sound_absorption values
    ds_Sv = ds_Sv.assign(
        echo_range=xr.DataArray(
            np.array([[np.linspace(0, 3, nrange_samples)] * npings]),
            coords=Sv.coords,
        )
    )
    ds_Sv = ds_Sv.assign(sound_absorption=0.001)
    # Run noise removal
    ds_Sv = ep.preprocess.remove_noise(
        ds_Sv, ping_num=2, range_sample_num=5, SNR_threshold=0
    )
    null = ds_Sv.Sv_corrected.isnull()
    # Test to see if the right number of points are removed before the range gets too large
    assert (
        np.count_nonzero(null.isel(channel=0, range_sample=slice(None, 50)))
        == 6
    )


def test_remove_noise_no_sound_absorption():
    """
    Tests remove_noise on toy data that does
    not have sound absorption as a variable.
    """

    pytest.xfail(f"Tests for remove_noise have not been implemented" +
                 " when no sound absorption is provided!")


def _construct_MVBS_toy_data(
    nchan, npings, nrange_samples, ping_size, range_sample_size
):
    """Construct data with values that increase every ping_num and ``range_sample_num``
    so that the result of computing MVBS is a smaller array
    that increases regularly for each resampled ``ping_time`` and ``range_sample``

    Parameters
    ----------
    nchan : int
        number of channels
    npings : int
        number of pings
    nrange_samples : int
        number of range samples
    ping_size : int
        number of pings with the same value
    range_sample_size : int
        number of range samples with the same value

    Returns
    -------
    np.ndarray
        Array with blocks of ``ping_time`` and ``range_sample`` with the same value,
        so that computing the MVBS will result in regularly increasing values
        every row and column
    """
    data = np.ones((nchan, npings, nrange_samples))
    for p_i, ping in enumerate(range(0, npings, ping_size)):
        for r_i, rb in enumerate(range(0, nrange_samples, range_sample_size)):
            data[0, ping : ping + ping_size, rb : rb + range_sample_size] += (
                r_i + p_i
            )
    # First channel increases by 1 each row and column, second increases by 2, third by 3, etc.
    for f in range(nchan):
        data[f] = data[0] * (f + 1)

    return data


def _construct_MVBS_test_data(nchan, npings, nrange_samples):
    """Construct data for testing the toy data from
    `_construct_MVBS_toy_data` after it has gone through the
    MVBS calculation.

    Parameters
    ----------
    nchan : int
        number of channels
    npings : int
        number of pings
    nrange_samples : int
        number of range samples

    Returns
    -------
    np.ndarray
        Array with values that increases regularly
        every ping and range sample
    """

    # Construct test array
    test_array = np.add(*np.indices((npings, nrange_samples)))
    return np.array([(test_array + 1) * (f + 1) for f in range(nchan)])


def test_compute_MVBS_index_binning():
    """Test compute_MVBS_index_binning on toy data"""

    # Parameters for toy data
    nchan, npings, nrange_samples = 4, 40, 400
    ping_num = 3  # number of pings to average over
    range_sample_num = 7  # number of range_samples to average over

    # Construct toy data that increases regularly every ping_num and range_sample_num
    data = _construct_MVBS_toy_data(
        nchan=nchan,
        npings=npings,
        nrange_samples=nrange_samples,
        ping_size=ping_num,
        range_sample_size=range_sample_num,
    )

    data_log = 10 * np.log10(data)  # Convert to log domain
    chan_index = np.arange(nchan).astype(str)
    ping_index = np.arange(npings)
    range_sample = np.arange(nrange_samples)
    Sv = xr.DataArray(
        data_log,
        coords=[
            ('channel', chan_index),
            ('ping_time', ping_index),
            ('range_sample', range_sample),
        ],
    )
    Sv.name = "Sv"
    ds_Sv = Sv.to_dataset()
    ds_Sv["frequency_nominal"] = chan_index  # just so there's value in freq_nominal
    ds_Sv = ds_Sv.assign(
        echo_range=xr.DataArray(
            np.array([[np.linspace(0, 10, nrange_samples)] * npings] * nchan),
            coords=Sv.coords,
        )
    )

    # Binned MVBS test
    ds_MVBS = ep.preprocess.compute_MVBS_index_binning(
        ds_Sv, range_sample_num=range_sample_num, ping_num=ping_num
    )
    data_test = 10 ** (ds_MVBS.Sv / 10)  # Convert to linear domain

    # Shape test
    data_binned_shape = np.ceil(
        (nchan, npings / ping_num, nrange_samples / range_sample_num)
    ).astype(int)
    assert np.all(data_test.shape == data_binned_shape)

    # Construct test array that increases by 1 for each range_sample and ping_time
    test_array = _construct_MVBS_test_data(
        nchan, data_binned_shape[1], data_binned_shape[2]
    )

    # Test all values in MVBS
    assert np.allclose(data_test, test_array, rtol=0, atol=1e-12)


def _coll_test_comp_MVBS(ds_Sv, nchan, ping_num,
                         range_sample_num, ping_time_bin,
                         total_range, range_meter_bin):
    """A collection of tests for test_compute_MVBS"""

    ds_MVBS = ep.preprocess.compute_MVBS(
        ds_Sv,
        range_meter_bin=range_meter_bin,
        ping_time_bin=f'{ping_time_bin}S',
    )

    data_test = 10 ** (ds_MVBS.Sv / 10)  # Convert to linear domain

    # Shape test
    data_binned_shape = np.ceil((nchan, ping_num, range_sample_num)).astype(int)
    assert np.all(data_test.shape == data_binned_shape)

    # Construct test array that increases by 1 for each range_sample and ping_time
    test_array = _construct_MVBS_test_data(
        nchan, data_binned_shape[1], data_binned_shape[2]
    )

    # Test all values in MVBS
    assert np.allclose(data_test, test_array, rtol=0, atol=1e-12)

    # Test to see if ping_time was resampled correctly
    test_ping_time = pd.date_range(
        '1/1/2020', periods=np.ceil(ping_num), freq=f'{ping_time_bin}S'
    )
    assert np.array_equal(data_test.ping_time, test_ping_time)

    # Test to see if range was resampled correctly
    test_range = np.arange(0, total_range, range_meter_bin)
    assert np.array_equal(data_test.echo_range, test_range)


def _fill_w_nans(narr, nan_ping_time, nan_range_sample):
    """
    A routine that fills a numpy array with nans.

    Parameters
    ----------
    narr : numpy array
        Array of dimensions (ping_time, range_sample)
    nan_ping_time : list
        ping times to fill with nans
    nan_range_sample: list
        range samples to fill with nans
    """
    if len(nan_ping_time) != len(nan_range_sample):
        raise ValueError('These lists must be the same size!')

    # fill in nans according to the provided lists
    for i, j in zip(nan_ping_time, nan_range_sample):
        narr[i, j] = np.nan

    return narr


def _nan_cases_comp_MVBS(ds_Sv, chan):
    """
    For a single channel, obtains numpy array
    filled with nans for various cases
    """

    # get echo_range values for a single channel
    one_chan_er = ds_Sv.echo_range.sel(channel=chan).copy().values

    # ping times to fill with NaNs
    nan_ping_time_1 = [slice(None), slice(None)]
    # range samples to fill with NaNs
    nan_range_sample_1 = [3, 4]
    # pad all ping_times with nans for a certain range_sample
    case_1 = _fill_w_nans(one_chan_er, nan_ping_time_1, nan_range_sample_1)

    # get echo_range values for a single channel
    one_chan_er = ds_Sv.echo_range.sel(channel=chan).copy().values
    # ping times to fill with NaNs
    nan_ping_time_2 = [1, 3, 5, 9]
    # range samples to fill with NaNs
    nan_range_sample_2 = [slice(None), slice(None), slice(None), slice(None)]
    # pad all range_samples of certain ping_times
    case_2 = _fill_w_nans(one_chan_er, nan_ping_time_2, nan_range_sample_2)

    # get echo_range values for a single channel
    one_chan_er = ds_Sv.echo_range.sel(channel=chan).copy().values
    # ping times to fill with NaNs
    nan_ping_time_3 = [0, 2, 5, 7]
    # range samples to fill with NaNs
    nan_range_sample_3 = [slice(0, 2), slice(None), slice(None), slice(0, 3)]
    # pad all range_samples of certain ping_times and
    # pad some ping_times with nans for a certain range_sample
    case_3 = _fill_w_nans(one_chan_er, nan_ping_time_3, nan_range_sample_3)

    return case_1, case_2, case_3


def test_compute_MVBS():
    """Test compute_MVBS on toy data"""

    # Parameters for fake data
    nchan, npings, nrange_samples = 4, 100, 4000
    range_meter_bin = 7  # range in meters to average over
    ping_time_bin = 3  # number of seconds to average over
    ping_rate = 2  # Number of pings per second
    range_sample_per_meter = 30  # Number of range_samples per meter

    # Useful conversions
    ping_num = (
        npings / ping_rate / ping_time_bin
    )  # number of pings to average over
    range_sample_num = (
        nrange_samples / range_sample_per_meter / range_meter_bin
    )  # number of range_samples to average over
    total_range = nrange_samples / range_sample_per_meter  # total range in meters

    # Construct data with values that increase with range and time
    # so that when compute_MVBS is performed, the result is a smaller array
    # that increases by a constant for each meter_bin and time_bin
    data = _construct_MVBS_toy_data(
        nchan=nchan,
        npings=npings,
        nrange_samples=nrange_samples,
        ping_size=ping_rate * ping_time_bin,
        range_sample_size=range_sample_per_meter * range_meter_bin,
    )

    data_log = 10 * np.log10(data)  # Convert to log domain
    chan_index = np.arange(nchan).astype(str)
    freq_nom = np.arange(nchan)
    # Generate a date range with `npings` number of pings with the frequency of the ping_rate
    ping_time = pd.date_range(
        '1/1/2020', periods=npings, freq=f'{1/ping_rate}S'
    )
    range_sample = np.arange(nrange_samples)
    Sv = xr.DataArray(
        data_log,
        coords=[
            ('channel', chan_index),
            ('ping_time', ping_time),
            ('range_sample', range_sample),
        ],
    )
    Sv.name = "Sv"
    ds_Sv = Sv.to_dataset()
    ds_Sv = ds_Sv.assign(
        frequency_nominal=xr.DataArray(freq_nom, coords={'channel': chan_index}),
        echo_range=xr.DataArray(
            np.array(
                [[np.linspace(0, total_range, nrange_samples)] * npings] * nchan
            ),
            coords=Sv.coords,
        )
    )

    # initial test of compute_MVBS
    _coll_test_comp_MVBS(ds_Sv, nchan, ping_num,
                         range_sample_num, ping_time_bin,
                         total_range, range_meter_bin)

    # TODO: use @pytest.fixture params/ids
    # for multiple similar tests using the same set of parameters
    # different nan cases for a single channel
    case_1, case_2, case_3 = _nan_cases_comp_MVBS(ds_Sv, chan='0')

    # pad all ping_times with nans for a certain range_sample
    ds_Sv['echo_range'].loc[{'channel': '0'}] = case_1

    _coll_test_comp_MVBS(ds_Sv, nchan, ping_num,
                         range_sample_num, ping_time_bin,
                         total_range, range_meter_bin)

    # pad all range_samples of certain ping_times
    ds_Sv['echo_range'].loc[{'channel': '0'}] = case_2

    _coll_test_comp_MVBS(ds_Sv, nchan, ping_num,
                         range_sample_num, ping_time_bin,
                         total_range, range_meter_bin)

    # pad all range_samples of certain ping_times and
    # pad some ping_times with nans for a certain range_sample
    ds_Sv['echo_range'].loc[{'channel': '0'}] = case_3

    _coll_test_comp_MVBS(ds_Sv, nchan, ping_num,
                         range_sample_num, ping_time_bin,
                         total_range, range_meter_bin)


def test_preprocess_mvbs(test_data_samples):
    """
    Test running through from open_raw to compute_MVBS.
    """
    (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    ) = test_data_samples
    ed = ep.open_raw(filepath, sonar_model, azfp_xml_path)
    if ed.sonar_model.lower() == 'azfp':
        avg_temperature = (
            ed["Environment"]['temperature'].mean('time1').values
        )
        env_params = {
            'temperature': avg_temperature,
            'salinity': 27.9,
            'pressure': 59,
        }
        range_kwargs['env_params'] = env_params
        if 'azfp_cal_type' in range_kwargs:
            range_kwargs.pop('azfp_cal_type')
    Sv = ep.calibrate.compute_Sv(ed, **range_kwargs)
    assert ep.preprocess.compute_MVBS(Sv) is not None


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
                                   create_dask: bool) -> Tuple[list, list, list]:
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
        ping_times_in_bin.append(np.random.uniform(ping_bin[0], ping_bin[1], (num_pings_in_bin[ping_ind],)))

        # randomly determine the number of values in each echo_range bin
        num_er_in_bin = np.random.randint(low=er_range[0], high=er_range[1], size=final_num_er_bins)

        # store the number of values in each echo_range bin
        all_er_bin_nums.append(num_er_in_bin)

        er_row_block = []
        for count, bin_val in enumerate(er_bins):

            # create a block of echo_range values
            if create_dask:
                a = dask.array.random.uniform(bin_val[0], bin_val[1], (num_pings_in_bin[ping_ind],
                                                                       num_er_in_bin[count]))
            else:
                a = np.random.uniform(bin_val[0], bin_val[1], (num_pings_in_bin[ping_ind],
                                                               num_er_in_bin[count]))

            # store the block of echo_range values
            er_row_block.append(a)

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
                          create_dask: bool) -> Tuple[Union[np.ndarray, dask.array.Array],
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
        sv_row_list = [np.arange(1, num_elem + 1, 1) for num_elem in arr]

        # create final sv row
        sv_row = np.concatenate(sv_row_list)

        # get final mean which is n+1/2 (since we are using natural numbers)
        final_means.append([(len(elem) + 1) / 2.0 for elem in sv_row_list])

        # create sv row block
        sv_row_block = np.tile(sv_row, (num_pings_in_bin[count], 1))

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
                           er_range: list, create_dask: bool) -> Tuple[np.ndarray, np.ndarray, Iterable,
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
    num_pings_in_bin = np.random.randint(low=ping_range[0], high=ping_range[1], size=final_num_ping_bins)

    # create bins for ping_time dimension
    ping_csum = np.cumsum(num_pings_in_bin)
    ping_bins = create_bins(ping_csum)

    # create bins for echo_range dimension
    num_er_in_bin = np.random.randint(low=er_range[0], high=er_range[1], size=final_num_er_bins)
    er_csum = np.cumsum(num_er_in_bin)
    er_bins = create_bins(er_csum)

    # create the echo_range data and associated bin information
    all_er_bin_nums, ping_times_in_bin, final_er_arrays = create_echo_range_related_data(ping_bins, num_pings_in_bin,
                                                                                         er_range, er_bins,
                                                                                         final_num_er_bins,
                                                                                         create_dask)

    # create the final echo_range array using created data and padding
    final_er, max_num_er_elem = construct_2d_echo_range_array(final_er_arrays, ping_csum, create_dask)

    # get final ping_time dimension
    final_ping_time = np.concatenate(ping_times_in_bin).astype('datetime64[ns]')

    # create the final sv array
    final_sv, final_MVBS = construct_2d_sv_array(max_num_er_elem, ping_csum,
                                                 all_er_bin_nums, num_pings_in_bin, create_dask)

    return final_MVBS, final_sv, ping_bins, er_bins, final_er, final_ping_time


def test_bin_and_mean_2d() -> None:
    """
    Tests the function ``bin_and_mean_2d``, which is the core
    method for ``compute_MVBS``. This is done by creating mock
    data (which can have varying number of ``echo_range`` values
    for each ``ping_time``) with known means.

    Parameters
    ----------
    create_dask: bool
        If True the ``Sv`` and ``echo_range`` values produced will be
        dask arrays, else they will be numpy arrays.
    """

    # TODO: document and create a fixture with input of create_dask

    create_dask = True

    final_num_ping_bins = 2
    final_num_er_bins = 5

    ping_range = [1, 5]
    er_range = [1, 5]

    # create echo_range, ping_time, and Sv arrays where the MVBS is known
    known_MVBS, final_sv, ping_bins, er_bins, final_er, final_ping_time = create_known_mean_data(final_num_ping_bins,
                                                                                                 final_num_er_bins,
                                                                                                 ping_range, er_range,
                                                                                                 create_dask)

    # put the created ping bins into a form that works with bin_and_mean_2d
    digitize_ping_bin = np.array([*ping_bins[0]] + [bin_val[1] for bin_val in ping_bins[1:-1]])
    digitize_ping_bin = digitize_ping_bin.astype('datetime64[ns]')

    # put the created echo range bins into a form that works with bin_and_mean_2d
    digitize_er_bin = np.array([*er_bins[0]] + [bin_val[1] for bin_val in er_bins[1:]])

    # calculate MVBS for mock data set
    calc_MVBS = bin_and_mean_2d(arr=final_sv, bins_time=digitize_ping_bin,
                                bins_er=digitize_er_bin, times=final_ping_time, echo_range=final_er)

    # compare known MVBS solution against its calculated counterpart
    assert np.allclose(calc_MVBS, known_MVBS, atol=1e-10, rtol=1e-10)
