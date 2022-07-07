import numpy as np
import pandas as pd
import xarray as xr
import echopype as ep
import pytest


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
            ed.environment['temperature'].mean('time1').values
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
