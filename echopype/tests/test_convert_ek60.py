import os
import shutil
import numpy as np
import xarray as xr
import pandas as pd
from echopype.convert import Convert

raw_path = './echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw'     # Standard test
test_path = './echopype/test_data/ek60/from_matlab/DY1801_EK60-D20180211-T164025.nc'
csv_paths = ['./echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Power18.csv',
                  './echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Power38.csv',
                  './echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Power70.csv',
                  './echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Power120.csv',
                  './echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Power200.csv']
# raw_path = './echopype/test_data/ek60/2015843-D20151023-T190636.raw'     # Different ranges
# raw_path = ['./echopype/test_data/ek60/OOI-D20170821-T063618.raw',
                #  './echopype/test_data/ek60/OOI-D20170821-T081522.raw']       # Multiple files
# raw_path = ['./echopype/test_data/ek60/set1/' + file
#                  for file in os.listdir('./echopype/test_data/ek60/set1')]    # 2 range lengths
# raw_path = ['./echopype/test_data/ek60/set2/' + file
#                  for file in os.listdir('./echopype/test_data/ek60/set2')]    # 3 range lengths
# Other data files
# raw_filename = 'data_zplsc/OceanStarr_2017-D20170725-T004612.raw'  # OceanStarr 2 channel EK60
# raw_filename = '../data/DY1801_EK60-D20180211-T164025.raw'  # Dyson 5 channel EK60
# raw_filename = 'data_zplsc/D20180206-T000625.raw   # EK80


def test_convert_matlab():
    """Test converting power and angle data"""
    tmp = Convert(raw_path)

    # Test saving nc file and perform checks
    tmp.raw2nc(overwrite=True)

    # Read .nc file into an xarray DataArray
    ds_beam = xr.open_dataset(tmp.nc_path, group='Beam')

    # Test dataset was created by exporting values from MATLAB EK60 parsing code
    with xr.open_dataset(test_path) as ds_test:
        assert np.allclose(ds_test.power, ds_beam.backscatter_r)    # Identical to MATLAB output to 1e-6
        athwartship = (ds_beam['angle_athwartship'] * 1.40625 / ds_beam['angle_sensitivity_athwartship'] -
                       ds_beam['angle_offset_athwartship'])
        alongship = (ds_beam['angle_alongship'] * 1.40625 / ds_beam['angle_sensitivity_alongship'] -
                     ds_beam['angle_offset_alongship'])
        assert np.allclose(ds_test.athwartship, athwartship)    # Identical to MATLAB output to 1e-7
        assert np.allclose(ds_test.alongship, alongship)        # Identical to MATLAB output to 1e-7

    # Check if backscatter data from all channels are identical to those directly unpacked
    for idx in tmp.config_datagram['transceivers'].keys():
        # idx is channel index assigned by instrument, starting from 1
        assert np.any(tmp.power_dict_split[0][idx - 1, :, :] ==  # idx-1 because power_dict_split[0] has a numpy array
                      ds_beam.backscatter_r.sel(frequency=tmp.config_datagram['transceivers'][idx]['frequency']).data)
    ds_beam.close()
    os.remove(tmp.nc_path)
    del tmp


def test_convert_power_echoview():
    tmp = Convert(raw_path)
    tmp.raw2nc()

    channels = []
    for file in csv_paths:
        channels.append(pd.read_csv(file, header=None, skiprows=[0]).iloc[:, 13:])
    test_power = np.stack(channels)
    with xr.open_dataset(tmp.nc_path, group='Beam') as ds_beam:
        assert np.allclose(test_power, ds_beam.backscatter_r[:, :10, 1:], atol=1e-10)

    os.remove(tmp.nc_path)


def test_convert_zarr():
    tmp = Convert(raw_path)
    tmp.raw2zarr()
    ds_beam = xr.open_zarr(tmp.zarr_path, group='Beam')
    with xr.open_dataset(test_path) as ds_test:
        assert np.allclose(ds_test.power, ds_beam.backscatter_r)

    shutil.rmtree(tmp.zarr_path, ignore_errors=True)    # Delete non-empty folder
