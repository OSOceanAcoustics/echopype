import numpy as np
import pandas as pd
import xarray as xr
import echopype as ep


def test_remove_noise():
    """Test remove_noise on toy data"""

    # Parameters for fake data
    nfreq, npings, nrange = 1, 10, 100
    freq = np.arange(nfreq)
    ping_index = np.arange(npings)
    range_bin = np.arange(nrange)
    data = np.ones(nrange)

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

    ds_Sv = ds_Sv.assign(range=xr.DataArray(np.array([[np.linspace(0, 10, nrange)] * npings]), coords=Sv.coords))
    ds_Sv = ds_Sv.assign(sound_absorption=0.001)
    # Run noise removal
    ds_Sv = ep.preprocess.remove_noise(ds_Sv, ping_num=2, range_bin_num=5, SNR_threshold=0)

    # Test if noise points are are nan
    assert np.isnan(ds_Sv.Sv_corrected.isel(frequency=0, ping_time=0, range_bin=30))
    assert np.isnan(ds_Sv.Sv_corrected.isel(frequency=0, ping_time=0, range_bin=60))

    # Test remove noise on a normal distribution
    np.random.seed(1)
    data = np.random.normal(loc=-100, scale=2, size=(nfreq, npings, nrange))
    # Make Dataset to pass into remove_noise
    Sv = xr.DataArray(data, coords=[('frequency', freq),
                                    ('ping_time', ping_index),
                                    ('range_bin', range_bin)])
    Sv.name = "Sv"
    ds_Sv = Sv.to_dataset()
    # Attach required range and sound_absorption values
    ds_Sv = ds_Sv.assign(range=xr.DataArray(np.array([[np.linspace(0, 3, nrange)] * npings]), coords=Sv.coords))
    ds_Sv = ds_Sv.assign(sound_absorption=0.001)
    # Run noise removal
    ds_Sv = ep.preprocess.remove_noise(ds_Sv, ping_num=2, range_bin_num=5, SNR_threshold=0)
    null = ds_Sv.Sv_corrected.isnull()
    # Test to see if the right number of points are removed before the range gets too large
    assert np.count_nonzero(null.isel(frequency=0, range_bin=slice(None, 50))) == 6


def test_compute_MVBS_index_binning():
    """Test compute_MVBS_index_binning on toy data"""

    # Parameters for fake data
    nfreq, npings, nrange = 4, 40, 400
    ping_num = 3             # number of pings to average over
    range_bin_num = 7        # number of range_bins to average over

    # Construct data with values that increase every ping_num and range_bin_num
    # so that when compute_MVBS_index_binning is performed, the result is a smaller array
    # that increases regularly for each row and column
    data = np.ones((nfreq, npings, nrange))
    for p_i, ping in enumerate(range(0, npings, ping_num)):
        for r_i, rb in enumerate(range(0, nrange, range_bin_num)):
            data[0, ping:ping + ping_num, rb:rb + range_bin_num] += r_i + p_i
    # First frequency increases by 1 each row and column, second increses by 2, third by 3, etc.
    for f in range(nfreq):
        data[f] = data[0] * (f + 1)

    data_log = 10 * np.log10(data)      # Convert to log domain
    freq_index = np.arange(nfreq)
    ping_index = np.arange(npings)
    range_bin = np.arange(nrange)
    Sv = xr.DataArray(data_log, coords=[('frequency', freq_index),
                                        ('ping_time', ping_index),
                                        ('range_bin', range_bin)])
    Sv.name = "Sv"
    ds_Sv = Sv.to_dataset()
    ds_Sv = ds_Sv.assign(
        range=xr.DataArray(np.array([[np.linspace(0, 10, nrange)] * npings] * nfreq),
                           coords=Sv.coords)
    )

    # Binned MVBS test
    ds_MVBS = ep.preprocess.compute_MVBS_index_binning(
        ds_Sv,
        range_bin_interval=range_bin_num,
        ping_num_interval=ping_num
    )
    data_test = (10 ** (ds_MVBS.Sv / 10))    # Convert to linear domain

    # Shape test
    shape = np.ceil((nfreq, npings / ping_num, nrange / range_bin_num)).astype(int)
    assert np.all(ds_MVBS.Sv.shape == shape)

    # Construct test array that increases by 1 for each range_bin and ping_time
    test_array = np.repeat(np.add(*np.indices((shape[1], shape[2])))[None, ...] + 1, nfreq, axis=0)
    # Increase test data by a multiple each frequency
    for f in range(nfreq):
        test_array[f] = (test_array[0] * (f + 1))

    # Test values along range_bin
    np.allclose(data_test, test_array, rtol=0, atol=1e-12)


def test_compute_MVBS():
    """Test compute_MVBS on toy data"""

    # Parameters for fake data
    nfreq, npings, nrange = 4, 100, 4000
    range_meter_bin = 7          # range in meters to average over
    ping_time_bin = 3            # number of seconds to average over
    ping_rate = 2                # Number of pings per second
    range_bin_per_meter = 30     # Number of range_bins per meter

    # Useful conversions
    ping_num = npings / ping_rate / ping_time_bin                      # number of pings to average over
    range_bin_num = nrange / range_bin_per_meter / range_meter_bin     # number of range_bins to average over
    total_range = nrange / range_bin_per_meter                         # total range in meters

    # Construct data with values that increase with range and time
    # so that when compute_MVBS is performed, the result is a smaller array
    # that increases by 1 for each meter_bin and time_bin
    data = np.ones((nfreq, npings, nrange))
    for p_i, ping in enumerate(range(0, npings, ping_rate * ping_time_bin)):
        for r_i, rb in enumerate(range(0, nrange, range_bin_per_meter * range_meter_bin)):
            data[0, ping:ping + ping_rate * ping_time_bin, rb:rb + range_bin_per_meter * range_meter_bin] += r_i + p_i
    for f in range(nfreq):
        data[f] = data[0]

    data_log = 10 * np.log10(data)      # Convert to log domain
    freq_index = np.arange(nfreq)
    # Generate a date range with `npings` number of pings with the frequency of the ping_rate
    ping_time = pd.date_range('1/1/2020', periods=npings, freq=f'{1/ping_rate}S')
    range_bin = np.arange(nrange)
    Sv = xr.DataArray(data_log, coords=[('frequency', freq_index),
                                        ('ping_time', ping_time),
                                        ('range_bin', range_bin)])
    Sv.name = "Sv"
    ds_Sv = Sv.to_dataset()
    ds_Sv = ds_Sv.assign(
        range=xr.DataArray(np.array([[np.linspace(0, total_range, nrange)] * npings] * nfreq),
                           coords=Sv.coords)
    )
    ds_MVBS = ep.preprocess.compute_MVBS(
        ds_Sv,
        range_meter_bin=range_meter_bin,
        ping_time_bin=f'{ping_time_bin}S'
    )

    # Shape test
    shape = (nfreq, ping_num, range_bin_num)
    assert np.all(ds_MVBS.Sv.shape == np.ceil(shape))

    data_test = (10 ** (ds_MVBS.Sv / 10)).round().astype(int)    # Convert to linear domain
    # Test values along range_bin
    assert np.all(data_test.isel(frequency=0, ping_time=0) == np.arange(range_bin_num) + 1)
    # Test values along ping time
    assert np.all(data_test.isel(frequency=0, range=0) == np.arange(ping_num) + 1)
