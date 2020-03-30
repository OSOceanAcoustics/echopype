import os
import shutil
import numpy as np
import xarray as xr
import pandas as pd
from echopype.convert import Convert
from echopype.convert.ek80 import ConvertEK80

ek60_raw_path = './echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw'     # Standard test
ek60_test_path = './echopype/test_data/ek60/from_matlab/DY1801_EK60-D20180211-T164025.nc'
# ek60_raw_path = './echopype/test_data/ek60/2015843-D20151023-T190636.raw'     # Different ranges
# ek60_raw_path = ['./echopype/test_data/ek60/OOI-D20170821-T063618.raw',
#                  './echopype/test_data/ek60/OOI-D20170821-T081522.raw']       # Multiple files
# Other data files
# raw_filename = 'data_zplsc/OceanStarr_2017-D20170725-T004612.raw'  # OceanStarr 2 channel EK60
# raw_filename = '../data/DY1801_EK60-D20180211-T164025.raw'  # Dyson 5 channel EK60
# raw_filename = 'data_zplsc/D20180206-T000625.raw   # EK80
# ek80_raw_path = './echopype/test_data/ek80/D20170912-T234910.raw'     # Large file (BB and CW)
ek80_raw_path = './echopype/test_data/ek80/D20190822-T161221.raw'       # Small file (Standard test) (CW)
ek80_power_test_path = ['./echopype/test_data/ek80/from_echoview/18kHz.power.csv',
                        './echopype/test_data/ek80/from_echoview/38kHz.power.csv',
                        './echopype/test_data/ek80/from_echoview/70kHz.power.csv',
                        './echopype/test_data/ek80/from_echoview/120kHz.power.csv',
                        './echopype/test_data/ek80/from_echoview/200kHz.power.csv']
ek80_angle_test_path = './echopype/test_data/ek80/from_echoview/EK80_test_angles.csv'
# ek80_raw_path = ['./echopype/test_data/ek80/Summer2018--D20180905-T033113.raw',
#                  './echopype/test_data/ek80/Summer2018--D20180905-T033258.raw']  # Multiple files
# azfp_01a_path = './echopype/data/azfp/17031001.01A'     # Canada (Different ranges)
# azfp_xml_path = './echopype/data/azfp/17030815.XML'     # Canada (Different ranges)
azfp_01a_path = './echopype/test_data/azfp/17082117.01A'     # Standard test
azfp_xml_path = './echopype/test_data/azfp/17041823.XML'     # Standard test
azfp_test_path = './echopype/test_data/azfp/from_matlab/17082117.nc'
# azfp_01a_path = ['./echopype/test_data/azfp/17033000.01A',     # Multiple files
#                  './echopype/test_data/azfp/17033001.01A']
# azfp_xml_path = './echopype/test_data/azfp/17033000.XML'       # Multiple files


def test_convert_ek60():
    """Test converting """
    # Unpacking data
    # tmp = ConvertEK60(ek60_raw_path)
    # tmp.load_ek60_raw()

    # # Convert to .nc file
    # tmp.raw2nc()
    tmp = Convert(ek60_raw_path)

    # Test saving zarr file
    # tmp.raw2zarr()
    # shutil.rmtree(tmp.zarr_path, ignore_errors=True)  # delete non-empty folder
                                                      # consider alternative using os.walk() if have os-specific errors

    # Test saving nc file and perform checks
    tmp.raw2nc(overwrite=True)

    # Read .nc file into an xarray DataArray
    ds_beam = xr.open_dataset(tmp.nc_path, group='Beam')

    with xr.open_dataset(ek60_test_path) as ds_test:
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
        assert np.any(tmp.power_dict_split[0][idx-1, :, :] ==  # idx-1 because power_dict_split[0] has a numpy array
                      ds_beam.backscatter_r.sel(frequency=tmp.config_datagram['transceivers'][idx]['frequency']).data)
    ds_beam.close()
    os.remove(tmp.nc_path)
    del tmp


def test_convert_ek80():
    tmp = ConvertEK80(ek80_raw_path)
    tmp.raw2nc()

    # Perform angle and power tests. Only 3 pings are tested in order to reduce the size of test datasets
    with xr.open_dataset(tmp.nc_path, group='Beam') as ds_beam:
        # Angle test data was origininaly exported from EchoView as 1 csv file for each frequency.
        # These files were combined and 3 pings were taken and saved as a single small csv file.
        df = pd.read_csv(ek80_angle_test_path, compression='gzip')
        # Test angles
        # Convert from electrical angles to degrees.
        major = (ds_beam['angle_athwartship'] * 1.40625 / ds_beam['angle_sensitivity_athwartship'] -
                 ds_beam['angle_offset_athwartship'])[:, 1:4, :]
        minor = (ds_beam['angle_alongship'] * 1.40625 / ds_beam['angle_sensitivity_alongship'] -
                 ds_beam['angle_offset_alongship'])[:, 1:4, :]
        for f in np.unique(df['frequency']):
            major_test = []
            minor_test = []
            df_freq = df[df['frequency'] == f]
            for i in np.unique(df_freq['ping_index']):
                val_maj = df_freq[df_freq['ping_index'] == i]['major']
                val_min = df_freq[df_freq['ping_index'] == i]['minor']
                major_test.append(xr.DataArray(val_maj, coords=[('range_bin', np.arange(val_maj.size))]))
                minor_test.append(xr.DataArray(val_min, coords=[('range_bin', np.arange(val_min.size))]))
            assert np.allclose(xr.concat(major_test, 'ping_time').dropna('range_bin'),
                               major.sel(frequency=f).dropna('range_bin'))
            assert np.allclose(xr.concat(minor_test, 'ping_time').dropna('range_bin'),
                               minor.sel(frequency=f).dropna('range_bin'))
        # Test power
        # Echoview power data is exported with the following constant multiplied to it
        POWER_FACTOR = 0.011758984205624266  # 10*log10(2)/256
        power = ds_beam.backscatter_r * POWER_FACTOR
        # single point error in original raw data. Read as -2000 by echopype and -999 by Echoview
        power[3][4][13174] = -999
        for i, f in enumerate(ek80_power_test_path):
            test_power = pd.read_csv(f, delimiter=';').iloc[:, 13:].values
            assert np.allclose(test_power, power[i].dropna('range_bin'))
    # Test saving zarr file
    tmp.raw2zarr()
    np_path = tmp.nc_path
    zarr_path = tmp.zarr_path
    del tmp
    # Remove generated files
    os.remove(np_path)
    shutil.rmtree(zarr_path, ignore_errors=True)


def test_convert_AZFP():
    # Read in the dataset that will be used to confirm working conversions. Generated from MATLAB code
    ds_test = xr.open_dataset(azfp_test_path)

    # Unpacking data
    # tmp = ConvertAZFP(azfp_01a_path, azfp_xml_path)
    # tmp.parse_raw()
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

test_convert_ek60()