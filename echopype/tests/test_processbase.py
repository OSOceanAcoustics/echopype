import os
import numpy as np
import xarray as xr
from echopype.convert.ek60 import ConvertEK60
from echopype.process import Process
ek60_raw_path = './echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw'     # Standard test


def test_validate_path():
    # Create process object
    tmp = ConvertEK60(ek60_raw_path)
    tmp.raw2nc(overwrite=True)
    e_data = Process(tmp.nc_path)

    # Create save folder
    orig_dir = './echopype/test_data/ek60'
    save_path = orig_dir + '/test_folder'

    file = e_data.validate_path(save_path=save_path, save_postfix='_Sv')
    assert '_Sv' in file
    filename = os.path.basename(tmp.nc_path)
    split = os.path.splitext(filename)
    new_filename = split[0] + '_Sv' + split[1]
    test_file = os.path.join(save_path, new_filename)
    # Check if filename is correct
    assert file == test_file
    # Check if folder was created
    assert os.path.exists(save_path)
    # Check if base path is used when save_path is not provided
    assert e_data.validate_path() == os.path.join(orig_dir, new_filename)
    # Delete created folder and nc file
    os.rmdir(save_path)
    os.remove(tmp.nc_path)


def test_get_tile_params():
    # Create process object
    tmp = ConvertEK60(ek60_raw_path)
    tmp.raw2nc()
    e_data = Process(tmp.nc_path)

    # Create sample DataArray
    nfreq, npings, nrange = 2, 10, 50
    ping_index = np.arange(npings)
    range_bin = np.arange(nrange)
    freq = np.arange(1, nfreq + 1) * 10000
    data = np.random.normal(size=(nfreq, npings, nrange))
    Sv = xr.DataArray(data, coords=[('frequency', freq),
                                    ('ping_time', ping_index),
                                    ('range_bin', range_bin)])
    sample_thickness = xr.DataArray([0.1] * nfreq, coords=[('frequency', freq)])
    r_tile_size = 5
    p_tile_size = 5
    r_tile_sz, range_bin_tile_bin_edge, ping_tile_bin_edge =\
        e_data.get_tile_params(r_data_sz=Sv.range_bin.size,
                               p_data_sz=Sv.ping_time.size,
                               r_tile_sz=r_tile_size,
                               p_tile_sz=p_tile_size,
                               sample_thickness=sample_thickness)
    r_tile_sz_test = [5, 5]
    r_tile_bin_edge_test = [-1, 49]
    p_tile_bin_edge_test = [-1, 4, 9, 14]
    assert np.array_equal(r_tile_sz, r_tile_sz_test)
    assert np.array_equal(range_bin_tile_bin_edge[0], r_tile_bin_edge_test)
    assert np.array_equal(ping_tile_bin_edge, p_tile_bin_edge_test)

    # Delete created nc file
    os.remove(tmp.nc_path)


def test_get_proc_Sv():
    # Create process object
    tmp = ConvertEK60(ek60_raw_path)
    tmp.raw2nc()
    e_data = Process(tmp.nc_path)

    e_data.calibrate(save=True)
    ds = xr.open_dataset(e_data.Sv_path)
    Sv = ds.Sv
    # Test if _get_proc_Sv() returns the same Sv (from memory)
    assert np.array_equal(e_data._get_proc_Sv().Sv, Sv)

    # Test if _get_proc_Sv() returns the same Sv (from saved Sv)
    e_data.Sv = None
    assert np.array_equal(e_data._get_proc_Sv().Sv, Sv)

    # Test if _get_proc_Sv() returns the same Sv (from new calibration)
    ds.close()
    os.remove(e_data.Sv_path)
    e_data.Sv = None
    assert np.array_equal(e_data._get_proc_Sv().Sv, Sv)

    # Closed opened file and remove all paths created
    os.remove(tmp.nc_path)


def test_remove_noise():
    # Create process object
    tmp = ConvertEK60(ek60_raw_path)
    tmp.raw2nc()
    e_data = Process(tmp.nc_path)
    freq, npings, nrange = [100000], 2, 100
    ping_index = np.arange(npings)
    range_bin = np.arange(nrange)
    # Test noise rsemoval on upside-down parabola
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
    e_data.Sv = ds
    # Create range, and seawater_absorption needed for noise_removal
    e_data.sample_thickness = xr.DataArray([0.1], coords=[('frequency', freq)])
    e_data._range = e_data.sample_thickness * \
        xr.DataArray([np.arange(nrange)], coords=[('frequency', freq),
                                                  ('range_bin', np.arange(nrange))])

    e_data._seawater_absorption = xr.DataArray([0.1], coords=[('frequency', freq)])
    # Run noise removal
    e_data.remove_noise()
    # Test if noise points are are nan
    assert np.isnan(e_data.Sv_clean.Sv[0][0][30])
    assert np.isnan(e_data.Sv_clean.Sv[0][0][60])

    # delete created nc file
    os.remove(tmp.nc_path)


def test_get_MVBS():
    # Create process object
    tmp = ConvertEK60(ek60_raw_path)
    tmp.raw2nc()
    e_data = Process(tmp.nc_path)
    nfreq, npings, nrange = 2, 10, 100
    data = np.ones((nfreq, npings, nrange))
    freq_index = np.arange(nfreq)
    ping_index = np.arange(npings)
    range_bin = np.arange(nrange)
    Sv = xr.DataArray(data, coords=[('frequency', freq_index),
                                    ('ping_time', ping_index),
                                    ('range_bin', range_bin)])
    Sv.name = "Sv"
    ds = Sv.to_dataset()
    e_data.Sv = ds
    e_data.sample_thickness = xr.DataArray([1] * nfreq, coords=[('frequency', freq_index)])
    MVBS_ping_size, MVBS_range_bin_size = 2, 10
    # Calculate MVBS
    e_data.get_MVBS(MVBS_ping_size=MVBS_ping_size, MVBS_range_bin_size=MVBS_range_bin_size)
    # MVBS should be 2 x 5 x 10 array
    # pings = npings / MVBS_ping_size, ranges = nrange / MVBS_range_bin_size
    shape = (nfreq, npings / MVBS_ping_size, nrange / MVBS_range_bin_size)
    assert e_data.MVBS.MVBS.shape == shape
    assert np.all(e_data.MVBS.MVBS.round() == 1)

    # Try new ping and range_bin tile size
    # reset the necessary variables
    e_data.MVBS = None
    e_data.MVBS_range_bin_size = None
    # Choose new tile sizes
    MVBS_ping_size, MVBS_range_bin_size = 3, 23
    # Calculate MVBS with same test Sv
    e_data.get_MVBS(MVBS_ping_size=MVBS_ping_size, MVBS_range_bin_size=MVBS_range_bin_size)
    # Test if indivisible tile sizes are rounded up property
    shape = (nfreq, np.ceil(npings / MVBS_ping_size), np.ceil(nrange / MVBS_range_bin_size))
    assert e_data.MVBS.MVBS.shape == shape
    assert np.all(e_data.MVBS.MVBS.round() == 1)

    # Test averaging on random data
    np.random.seed(1)
    e_data.Sv = (Sv[:1, :1, :nrange] * np.random.random(nrange)).to_dataset()
    e_data.sample_thickness = xr.DataArray([1], coords=[('frequency', [0])])
    e_data.MVBS = None
    e_data.MVBS_range_bin_size = None
    # Output will be a single tile
    MVBS_ping_size, MVBS_range_bin_size = 1, nrange
    e_data.get_MVBS(MVBS_ping_size=MVBS_ping_size, MVBS_range_bin_size=MVBS_range_bin_size)
    # Tests averaging. Sv is avearged in linear domain then converted back
    assert e_data.MVBS.MVBS == (10 * np.log10((10 ** (e_data.Sv.Sv / 10)).mean('range_bin')))

    # delete created nc file
    os.remove(tmp.nc_path)
