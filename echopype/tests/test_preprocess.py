import numpy as np
import xarray as xr
import echopype as ep


def test_remove_noise():
    # Create process object
    nfreq, npings, nrange = 1, 10, 100
    freq = np.arange(nfreq)
    ping_index = np.arange(npings)
    range_bin = np.arange(nrange)

    # Test noise removal on toy data
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
    # Make DataArray
    Sv = xr.DataArray(data, coords=[('frequency', freq),
                                    ('ping_time', ping_index),
                                    ('range_bin', range_bin)])
    Sv.name = "Sv"
    ds_Sv = Sv.to_dataset()

    ds_Sv = ds_Sv.assign(range=xr.DataArray(np.array([[np.linspace(0, 3, nrange)] * npings]), coords=Sv.coords))
    ds_Sv = ds_Sv.assign(sound_absorption=0.001)
    # Run noise removal
    ds_Sv = ep.preprocess.remove_noise(ds_Sv, ping_num=2, range_bin_num=5, SNR_threshold=0)
    null = ds_Sv.Sv_corrected.isnull()
    # Test to see if the right number of points are removed before the range gets too large
    assert np.count_nonzero(null.isel(frequency=0, range_bin=slice(None, 50))) == 6


def test_compute_MVBS_index_binning():
    """Test get_MVBS on a normal distribution"""

    # Parameters for fake data
    nfreq, npings, nrange = 4, 40, 400
    ping_num = 2             # number of pings to average over
    range_bin_num = 3        # number of range_bins to average over

    # Construct data with values that increase every ping_num and range_bin_num
    # so that when binned get_MVBS is performed, the result is a smaller array
    # that increases by 1 for each row and column
    data = np.zeros((nfreq, npings, nrange))
    for p_i, ping in enumerate(range(0, npings, ping_num)):
        for r_i, rb in enumerate(range(0, nrange, range_bin_num)):
            data[0, ping:ping + ping_num, rb:rb + range_bin_num] += r_i + p_i
    for f in range(nfreq):
        data[f] = data[0]

    data_log = 10 * np.log10(data)      # Convert to log domain
    freq_index = np.arange(nfreq)
    ping_index = np.arange(npings)
    range_bin = np.arange(nrange)
    Sv = xr.DataArray(data_log, coords=[('frequency', freq_index),
                                        ('ping_time', ping_index),
                                        ('range_bin', range_bin)])
    Sv.name = "Sv"
    ds_Sv = Sv.to_dataset()
    ds_Sv = ds_Sv.assign(range=xr.DataArray(np.array([[np.linspace(0, 10, nrange)] * npings] * nfreq), coords=Sv.coords))

    # Binned MVBS test
    ds_MVBS = ep.preprocess.compute_MVBS_index_binning(
        ds_Sv,
        range_bin_interval=range_bin_num,
        ping_num_interval=ping_num
    )
    shape = (nfreq, npings / ping_num, nrange / range_bin_num)
    assert np.all(ds_MVBS.Sv.shape == np.ceil(shape))  # Shape test

    data_test = (10 ** (ds_MVBS.Sv / 10)).round().astype(int)    # Convert to linear domain
    # Test values along range_bin
    assert np.all(data_test.isel(frequency=0, ping_time=0) == np.arange(nrange / range_bin_num))
    # Test values along ping time
    assert np.all(data_test.isel(frequency=0, range_bin=0) == np.arange(npings / ping_num))
