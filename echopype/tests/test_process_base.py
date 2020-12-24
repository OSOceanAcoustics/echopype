import os
import numpy as np
import xarray as xr
from ..convert import Convert
from ..process import Process, EchoData, ProcessBase
ek60_raw_path = './echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw'     # Standard test


def test_validate_proc_path():
    # Create process object
    ed = EchoData()
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
    ed = EchoData()
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
    assert np.isnan(ed.Sv_clean.Sv_clean[0][0][30])
    assert np.isnan(ed.Sv_clean.Sv_clean[0][0][60])

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
    assert np.count_nonzero(null[0, :, :50])


def test_get_MVBS():
    """Test get_MVBS on a normal distribution"""

    # Parameters for fake data
    nfreq, npings, nrange = 5, 100, 1000
    normal_loc = 100
    pn = 12         # number of pings to average over
    rbn = 66        # number of range_bins to average over

    # Create process object
    ed = EchoData()
    ed._raw_path = [ek60_raw_path]
    pb = ProcessBase()

    np.random.seed(0)
    data = np.random.normal(loc=normal_loc, scale=10, size=(nfreq, npings, nrange))
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
            'ping_num': pn,
            'range_bin_num': rbn,
            'type': 'binned'}
    }
    # Binned MVBS test
    pb.get_MVBS(ed, proc_params=proc_params)
    shape = (nfreq, npings / proc_params['MVBS']['ping_num'], nrange / proc_params['MVBS']['range_bin_num'])
    assert np.all(ed.MVBS.MVBS.shape == np.ceil(shape))  # Shape test

    data = 10 ** (ed.MVBS.MVBS / 10)                     # Convert to linear domain
    assert data.mean().round().values == normal_loc      # Value test

    # Rolling MVBS test
    proc_params['MVBS']['type'] = 'rolling'
    pb.get_MVBS(ed, proc_params=proc_params)
    assert ed.MVBS.MVBS.shape == (nfreq, npings, nrange)  # Shape test

    data = 10 ** (ed.MVBS.MVBS / 10)                     # Convert to linear domain
    assert data.mean().round().values == normal_loc      # Value test
