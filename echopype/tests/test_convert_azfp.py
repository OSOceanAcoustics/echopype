import os
import shutil
import numpy as np
import xarray as xr
import pandas as pd
from echopype.convert import Convert

# azfp_01a_path = './echopype/data/azfp/17031001.01A'     # Canada (Different ranges)
# azfp_xml_path = './echopype/data/azfp/17030815.XML'     # Canada (Different ranges)
azfp_01a_path = './echopype/test_data/azfp/17082117.01A'     # Standard test
azfp_xml_path = './echopype/test_data/azfp/17041823.XML'     # Standard test
azfp_test_path = './echopype/test_data/azfp/from_matlab/17082117.nc'
# azfp_01a_path = ['./echopype/test_data/azfp/set1/17033000.01A',     # Multiple files
#                  './echopype/test_data/azfp/set1/17033001.01A',
#                  './echopype/test_data/azfp/set1/17033002.01A',
#                  './echopype/test_data/azfp/set1/17033003.01A']
# azfp_xml_path = './echopype/test_data/azfp/set1/17033000.XML'       # Multiple files

def test_convert_AZFP():
    # Read in the dataset that will be used to confirm working conversions. Generated from MATLAB code
    ds_test = xr.open_dataset(azfp_test_path)

    # Unpacking data
    tmp = Convert(azfp_01a_path, azfp_xml_path)

    # Test saving zarr file
    tmp.raw2zarr()
    shutil.rmtree(tmp.zarr_path, ignore_errors=True)

    # Test saving nc file and perform checks
    tmp.raw2nc()

    # Test beam group
    with xr.open_dataset(tmp.nc_path, group='Beam') as ds_beam:
        # Test frequency
        assert np.array_equal(ds_test.frequency, ds_beam.frequency)
        # Test sea absorption
        # assert np.array_equal(ds_test.sea_abs, ds_beam.sea_abs)
        # Test ping time
        assert np.array_equal(ds_test.ping_time, ds_beam.ping_time)
        # Test tilt x and y
        assert np.array_equal(ds_test.tilt_x, ds_beam.tilt_x)
        assert np.array_equal(ds_test.tilt_y, ds_beam.tilt_y)
        # Test backscatter_r
        assert np.array_equal(ds_test.backscatter, ds_beam.backscatter_r)

    # Test environment group
    with xr.open_dataset(tmp.nc_path, group='Environment') as ds_env:
        # Test temperature
        assert np.array_equal(ds_test.temperature, ds_env.temperature)
        # Test sound speed. 1 value is used because sound speed is the same across frequencies
        # assert ds_test.sound_speed == ds_env.sound_speed_indicative.values[0]

    # with xr.open_dataset(tmp.nc_path, group="Vendor") as ds_vend:
    #     # Test battery values
    #     assert np.array_equal(ds_test.battery_main, ds_vend.battery_main)
    #     assert np.array_equal(ds_test.battery_tx, ds_vend.battery_tx)

    ds_test.close()
    os.remove(tmp.nc_path)
    del tmp
