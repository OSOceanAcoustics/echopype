import numpy as np
import pandas as pd
import xarray as xr
import echopype as ep


def test_remove_noise():
    """Test remove_noise on toy data"""

    # Parameters for fake data
    nfreq, npings, nrange_bins = 1, 10, 100
    freq = np.arange(nfreq)
    ping_index = np.arange(npings)
    range_bin = np.arange(nrange_bins)
    data = np.ones(nrange_bins)

    # Insert noise points
    np.put(data, 30, -30)
    np.put(data, 60, -30)
    # Add more pings
    data = np.array([data] * npings)
    # Make DataArray
    Sv = xr.DataArray([data], coords=[('frequency', freq),
                                      ('ping_time', ping_index),
                                      ('range_bin', range_bin)])
    Sv.name = "Sv"
    ds_Sv = Sv.to_dataset()

    ds_Sv = ds_Sv.assign(range=xr.DataArray(np.array([[np.linspace(0, 10, nrange_bins)] * npings]), coords=Sv.coords))
    ds_Sv = ds_Sv.assign(sound_absorption=0.001)
    # Run noise removal
    ds_Sv = ep.preprocess.remove_noise(ds_Sv, ping_num=2, range_bin_num=5, SNR_threshold=0)

    # Test if noise points are are nan
    assert np.isnan(ds_Sv.Sv_corrected.isel(frequency=0, ping_time=0, range_bin=30))
    assert np.isnan(ds_Sv.Sv_corrected.isel(frequency=0, ping_time=0, range_bin=60))

    # Test remove noise on a normal distribution
    np.random.seed(1)
    data = np.random.normal(loc=-100, scale=2, size=(nfreq, npings, nrange_bins))
    # Make Dataset to pass into remove_noise
    Sv = xr.DataArray(data, coords=[('frequency', freq),
                                    ('ping_time', ping_index),
                                    ('range_bin', range_bin)])
    Sv.name = "Sv"
    ds_Sv = Sv.to_dataset()
    # Attach required range and sound_absorption values
    ds_Sv = ds_Sv.assign(range=xr.DataArray(np.array([[np.linspace(0, 3, nrange_bins)] * npings]), coords=Sv.coords))
    ds_Sv = ds_Sv.assign(sound_absorption=0.001)
    # Run noise removal
    ds_Sv = ep.preprocess.remove_noise(ds_Sv, ping_num=2, range_bin_num=5, SNR_threshold=0)
    null = ds_Sv.Sv_corrected.isnull()
    # Test to see if the right number of points are removed before the range gets too large
    assert np.count_nonzero(null.isel(frequency=0, range_bin=slice(None, 50))) == 6


def _construct_MVBS_toy_data(nfreq, npings, nrange_bins, ping_size, range_bin_size):
    """Construct data with values that increase every ping_num and range_bin_num
    so that the result of computing MVBS is a smaller array
    that increases regularly for each resampled ping_time and range_bin

    Parameters
    ----------
    nfreq : int
        number of frequencies
    npings : int
        number of pings
    nrange_bins : int
        number of range_bins
    ping_size : int
        number of pings with the same value
    range_bin_size : int
        number of range_bins with the same value

    Returns
    -------
    np.ndarray
        Array with blocks of ping_times and range_bins with the same value,
        so that computing the MVBS will result in regularly increasing values
        every row and column
    """
    data = np.ones((nfreq, npings, nrange_bins))
    for p_i, ping in enumerate(range(0, npings, ping_size)):
        for r_i, rb in enumerate(range(0, nrange_bins, range_bin_size)):
            data[0, ping:ping + ping_size, rb:rb + range_bin_size] += r_i + p_i
    # First frequency increases by 1 each row and column, second increases by 2, third by 3, etc.
    for f in range(nfreq):
        data[f] = data[0] * (f + 1)

    return data


def _construct_MVBS_test_data(nfreq, npings, nrange_bins):
    """Construct data for testing the toy data from
    `_construct_MVBS_toy_data` after it has gone through the
    MVBS calculation.

    Parameters
    ----------
    nfreq : int
        number of frequencies
    npings : int
        number of pings
    nrange_bins : int
        number of range_bins

    Returns
    -------
    np.ndarray
        Array with values that increases regularly
        every ping and range_bin
    """

    # Construct test array
    test_array = np.add(*np.indices((npings, nrange_bins)))
    return np.array([(test_array + 1) * (f + 1) for f in range(nfreq)])


def test_compute_MVBS_index_binning():
    """Test compute_MVBS_index_binning on toy data"""

    # Parameters for toy data
    nfreq, npings, nrange_bins = 4, 40, 400
    ping_num = 3             # number of pings to average over
    range_bin_num = 7        # number of range_bins to average over

    # Construct toy data that increases regularly every ping_num and range_bin_num
    data = _construct_MVBS_toy_data(
        nfreq=nfreq,
        npings=npings,
        nrange_bins=nrange_bins,
        ping_size=ping_num,
        range_bin_size=range_bin_num
    )

    data_log = 10 * np.log10(data)      # Convert to log domain
    freq_index = np.arange(nfreq)
    ping_index = np.arange(npings)
    range_bin = np.arange(nrange_bins)
    Sv = xr.DataArray(data_log, coords=[('frequency', freq_index),
                                        ('ping_time', ping_index),
                                        ('range_bin', range_bin)])
    Sv.name = "Sv"
    ds_Sv = Sv.to_dataset()
    ds_Sv = ds_Sv.assign(
        range=xr.DataArray(np.array([[np.linspace(0, 10, nrange_bins)] * npings] * nfreq),
                           coords=Sv.coords)
    )

    # Binned MVBS test
    ds_MVBS = ep.preprocess.compute_MVBS_index_binning(
        ds_Sv,
        range_bin_num=range_bin_num,
        ping_num=ping_num
    )
    data_test = (10 ** (ds_MVBS.Sv / 10))    # Convert to linear domain

    # Shape test
    data_binned_shape = np.ceil((nfreq, npings / ping_num, nrange_bins / range_bin_num)).astype(int)
    assert np.all(data_test.shape == data_binned_shape)

    # Construct test array that increases by 1 for each range_bin and ping_time
    test_array = _construct_MVBS_test_data(nfreq, data_binned_shape[1], data_binned_shape[2])

    # Test all values in MVBS
    assert np.allclose(data_test, test_array, rtol=0, atol=1e-12)


def test_compute_MVBS():
    """Test compute_MVBS on toy data"""

    # Parameters for fake data
    nfreq, npings, nrange_bins = 4, 100, 4000
    range_meter_bin = 7          # range in meters to average over
    ping_time_bin = 3            # number of seconds to average over
    ping_rate = 2                # Number of pings per second
    range_bin_per_meter = 30     # Number of range_bins per meter

    # Useful conversions
    ping_num = npings / ping_rate / ping_time_bin                           # number of pings to average over
    range_bin_num = nrange_bins / range_bin_per_meter / range_meter_bin     # number of range_bins to average over
    total_range = nrange_bins / range_bin_per_meter                         # total range in meters

    # Construct data with values that increase with range and time
    # so that when compute_MVBS is performed, the result is a smaller array
    # that increases by a constant for each meter_bin and time_bin
    data = _construct_MVBS_toy_data(
        nfreq=nfreq,
        npings=npings,
        nrange_bins=nrange_bins,
        ping_size=ping_rate * ping_time_bin,
        range_bin_size=range_bin_per_meter * range_meter_bin
    )

    data_log = 10 * np.log10(data)      # Convert to log domain
    freq_index = np.arange(nfreq)
    # Generate a date range with `npings` number of pings with the frequency of the ping_rate
    ping_time = pd.date_range('1/1/2020', periods=npings, freq=f'{1/ping_rate}S')
    range_bin = np.arange(nrange_bins)
    Sv = xr.DataArray(data_log, coords=[('frequency', freq_index),
                                        ('ping_time', ping_time),
                                        ('range_bin', range_bin)])
    Sv.name = "Sv"
    ds_Sv = Sv.to_dataset()
    ds_Sv = ds_Sv.assign(
        range=xr.DataArray(np.array([[np.linspace(0, total_range, nrange_bins)] * npings] * nfreq),
                           coords=Sv.coords)
    )
    ds_MVBS = ep.preprocess.compute_MVBS(
        ds_Sv,
        range_meter_bin=range_meter_bin,
        ping_time_bin=f'{ping_time_bin}S'
    )
    data_test = (10 ** (ds_MVBS.Sv / 10))    # Convert to linear domain

    # Shape test
    data_binned_shape = np.ceil((nfreq, ping_num, range_bin_num)).astype(int)
    assert np.all(data_test.shape == data_binned_shape)

    # Construct test array that increases by 1 for each range_bin and ping_time
    test_array = _construct_MVBS_test_data(nfreq, data_binned_shape[1], data_binned_shape[2])

    # Test all values in MVBS
    assert np.allclose(data_test, test_array, rtol=0, atol=1e-12)

    # Test to see if ping_time was resampled correctly
    test_ping_time = pd.date_range('1/1/2020', periods=np.ceil(ping_num), freq=f'{ping_time_bin}S')
    assert np.array_equal(data_test.ping_time, test_ping_time)

    # Test to see if range was resampled correctly
    test_range = np.arange(0, total_range, range_meter_bin)
    assert np.array_equal(data_test.range, test_range)
