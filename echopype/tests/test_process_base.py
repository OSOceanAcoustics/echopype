import os
import numpy as np
import xarray as xr
from ..convert import Convert
from ..process import Process, EchoDataOld, ProcessBase
ek60_raw_path = './echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw'     # Standard test


def test_validate_proc_path():
    # Create process object
    ed = EchoDataOld()
    ed._raw_path = [ek60_raw_path]
    pb = ProcessBase()

    # ed, postfix, save_path=None, save_format='zarr'
    # Create save folder
    orig_dir = './echopype/test_data/ek60'
    save_path = orig_dir + '/test_folder'

    file = pb.validate_proc_path(ed, save_path=save_path, postfix='_Sv', save_format='netcdf4')
    assert '_Sv' in file

    filename = os.path.basename(ed.raw_path[0])
    split = os.path.splitext(filename)
    new_filename = split[0] + '_Sv' + '.nc'
    test_file = os.path.join(save_path, new_filename)
    # Check if filename is correct
    assert file == test_file
    # Check if folder was created
    assert os.path.exists(save_path)
    # Check if base path is used when save_path is not provided
    assert pb.validate_proc_path(ed, postfix='_Sv', save_format='netcdf4') == os.path.join(orig_dir, new_filename)
    # Delete created folder and nc file
    os.rmdir(save_path)


def test_remove_noise():
    # Create process object
    ed = EchoDataOld()
    ed._raw_path = [ek60_raw_path]
    pb = ProcessBase()

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
    ds = Sv.to_dataset()
    ed._Sv = ds
    # Create range, and seawater_absorption needed for noise_removal
    env_params = {'absorption': 0.001}
    cal_params = {}
    proc_params = {
        'noise_est': {'ping_num': 2,
                      'range_bin_num': 5,
                      'SNR': 0}
    }
    ed.range = xr.DataArray(np.array([[np.linspace(0, 10, nrange)] * npings]), coords=Sv.coords)
    # Run noise removal
    pb.remove_noise(ed, env_params, cal_params, proc_params=proc_params)

    # Test if noise points are are nan
    assert np.isnan(ed.Sv_clean.Sv_clean.isel(frequency=0, ping_time=0, range_bin=30))
    assert np.isnan(ed.Sv_clean.Sv_clean.isel(frequency=0, ping_time=0, range_bin=60))

    # Test remove noise on a normal distribution
    np.random.seed(1)
    data = np.random.normal(loc=-100, scale=2, size=(nfreq, npings, nrange))
    # Make DataArray
    Sv = xr.DataArray(data, coords=[('frequency', freq),
                                    ('ping_time', ping_index),
                                    ('range_bin', range_bin)])
    Sv.name = "Sv"
    ds = Sv.to_dataset()
    ed._Sv = ds

    ed.range = xr.DataArray(np.array([[np.linspace(0, 3, nrange)] * npings]), coords=Sv.coords)
    # Run noise removal
    pb.remove_noise(ed, env_params, cal_params, proc_params=proc_params)
    null = ed.Sv_clean.Sv_clean.isnull()
    # Test to see if the right number of points are removed before the range gets too large
    assert np.count_nonzero(null.isel(frequency=0, range_bin=slice(None, 50))) == 6


def test_get_MVBS():
    """Test get_MVBS on a normal distribution"""

    # Parameters for fake data
    nfreq, npings, nrange = 4, 40, 400
    ping_num = 2             # number of pings to average over
    range_bin_num = 3        # number of range_bins to average over

    # Create process object
    ed = EchoDataOld()
    ed._raw_path = [ek60_raw_path]
    pb = ProcessBase()

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
    ds = Sv.to_dataset()
    ed._Sv = ds
    proc_params = {
        'MVBS': {
            'source': 'Sv',
            'ping_num': ping_num,
            'range_bin_num': range_bin_num,
            'type': 'binned'}
    }
    # Binned MVBS test
    pb.get_MVBS(ed, proc_params=proc_params)
    shape = (nfreq, npings / proc_params['MVBS']['ping_num'], nrange / proc_params['MVBS']['range_bin_num'])
    assert np.all(ed.MVBS.MVBS.shape == np.ceil(shape))  # Shape test

    data_test = (10 ** (ed.MVBS.MVBS / 10)).round().astype(int)    # Convert to linear domain
    # Test values along range_bin
    assert np.all(data_test.isel(frequency=0, ping_time=0) == np.arange(nrange / range_bin_num))
    # Test values along ping time
    assert np.all(data_test.isel(frequency=0, range_bin=0) == np.arange(npings / ping_num))

    # Rolling MVBS test
    proc_params['MVBS']['type'] = 'rolling'
    pb.get_MVBS(ed, proc_params=proc_params)
    assert ed.MVBS.MVBS.shape == (nfreq, npings, nrange)  # Shape test

    data_test = (10 ** (ed.MVBS.MVBS / 10))

    # Value test along range_bin
    roll_avg_range = []
    # Manually calculate moving average along the range_bin dimension
    for idx in range(len(data[0, 0, :])):
        if idx < range_bin_num - 1:
            roll_avg_range.append(np.nan)
            continue
        roll_avg_range.append(data[0, 0, idx - range_bin_num + 1:idx + 1].mean())
    assert np.allclose(roll_avg_range,
                       data_test.dropna('ping_time', 'all').isel(frequency=0, ping_time=0), equal_nan=True)

    # Value test along ping_time
    roll_avg_ping = []
    # Manually calculate moving average along the ping_time dimension
    for idx in range(len(data[0, :, 0])):
        if idx < ping_num - 1:
            roll_avg_ping.append(np.nan)
            continue
        roll_avg_ping.append(data[0, idx - ping_num + 1:idx + 1, 0].mean())
    assert np.allclose(roll_avg_ping,
                       data_test.dropna('range_bin', 'all').isel(frequency=0, range_bin=0), equal_nan=True)
