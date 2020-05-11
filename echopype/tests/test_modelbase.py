import os
import numpy as np
import xarray as xr
from echopype.convert.ek60 import ConvertEK60
from echopype.model import EchoData
ek60_raw_path = './echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw'     # Standard test


def test_validate_path():
    # Create model object
    tmp = ConvertEK60(ek60_raw_path)
    tmp.raw2nc()
    e_data = EchoData(tmp.nc_path)

    # Create save folder
    save_path = './echopype/test_data/ek60/test_folder/'

    file = e_data.validate_path(save_path=save_path, save_postfix='_Sv')
    filename = os.path.basename(tmp.nc_path)
    split = os.path.splitext(filename)
    test_file = os.path.join(save_path, split[0] + '_Sv' + split[1])
    # Check if filename is correct
    assert file == test_file
    # Check if folder was created
    assert os.path.exists(save_path)

    # Delete created folder and nc file
    os.rmdir(save_path)
    os.remove(tmp.nc_path)


def test_get_tile_params():
    # Create model object
    tmp = ConvertEK60(ek60_raw_path)
    tmp.raw2nc()
    e_data = EchoData(tmp.nc_path)

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
    # Create model object
    tmp = ConvertEK60(ek60_raw_path)
    tmp.raw2nc()
    e_data = EchoData(tmp.nc_path)

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
    # Create model object
    tmp = ConvertEK60(ek60_raw_path)
    tmp.raw2nc()
    e_data = EchoData(tmp.nc_path)

    nfreq, npings, nrange = 5, 10, 1000
    with xr.open_dataset(tmp.nc_path, group='Beam') as ds_beam:
        freq = ds_beam.frequency
    ping_index = np.arange(npings)
    range_bin = np.arange(nrange)
    data = np.random.normal(size=(nfreq, npings, nrange))
    Sv = xr.DataArray(data, coords=[('frequency', freq),
                                    ('ping_time', ping_index),
                                    ('range_bin', range_bin)])
    Sv.name = "Sv"
    e_data.sample_thickness = xr.DataArray([0.1] * nfreq, coords=[('frequency', freq)])
    e_data.Sv = Sv.to_dataset()
    e_data.remove_noise()

    # delete created nc file
    os.remove(tmp.nc_path)
