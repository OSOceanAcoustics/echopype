import os
import shutil
import numpy as np
import xarray as xr
import pandas as pd
from ..convert import Convert

raw_path_bb = './echopype/test_data/ek80/D20170912-T234910.raw'       # Large file (BB)
raw_path_cw = './echopype/test_data/ek80/D20190822-T161221.raw'       # Small file (CW) (Standard test)
power_test_path = ['./echopype/test_data/ek80/from_echoview/18kHz.power.csv',
                   './echopype/test_data/ek80/from_echoview/38kHz.power.csv',
                   './echopype/test_data/ek80/from_echoview/70kHz.power.csv',
                   './echopype/test_data/ek80/from_echoview/120kHz.power.csv',
                   './echopype/test_data/ek80/from_echoview/200kHz.power.csv']
angle_test_path = './echopype/test_data/ek80/from_echoview/EK80_test_angles.csv'
bb_power_test_path = './echopype/test_data/ek80/from_echoview/70 kHz raw power.complex.csv'
# raw_path = ['./echopype/test_data/ek80/Summer2018--D20180905-T033113.raw',
#             './echopype/test_data/ek80/Summer2018--D20180905-T033258.raw']  # Multiple files (CW and BB)
raw_path_bb_cw = './echopype/test_data/ek80/Summer2018--D20180905-T033113.raw'
raw_path_2_f = './echopype/test_data/ek80/2019118 group2survey-D20191214-T081342.raw'


def test_cw():
    # Test conversion of EK80 continuous wave data file
    tmp = Convert()
    tmp.set_source(file=raw_path_cw,
                   model='EK80')
    tmp.to_netcdf()

    # Perform angle and power tests. Only 3 pings are tested in order to reduce the size of test datasets
    with xr.open_dataset(tmp.nc_path, group='Beam') as ds_beam:
        # Angle test data was origininaly exported from EchoView as 1 csv file for each frequency.
        # These files were combined and 3 pings were taken and saved as a single small csv file.
        df = pd.read_csv(angle_test_path, compression='gzip')
        # Test angles
        # Convert from electrical angles to degrees.
        major = (ds_beam['angle_athwartship'] * 1.40625 / ds_beam['angle_sensitivity_athwartship'] -
                 ds_beam['angle_offset_athwartship'])[:, 1:4, :]
        minor = (ds_beam['angle_alongship'] * 1.40625 / ds_beam['angle_sensitivity_alongship'] -
                 ds_beam['angle_offset_alongship'])[:, 1:4, :]
        # Loop over the 5 frequencies
        for f in np.unique(df['frequency']):
            major_test = []
            minor_test = []
            df_freq = df[df['frequency'] == f]
            # Loop over the 3 pings
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
        for i, f in enumerate(power_test_path):
            test_power = pd.read_csv(f, delimiter=';').iloc[:, 13:].values
            assert np.allclose(test_power, power[i].dropna('range_bin'))

    # Test saving zarr cw file
    tmp.to_zarr()

    # Remove generated files
    os.remove(tmp.nc_path)
    shutil.rmtree(tmp.zarr_path, ignore_errors=True)


def test_bb():
    # Test conversion of EK80 broadband data file
    tmp = Convert(file=raw_path_bb, model='EK80')
    tmp.to_netcdf()

    # Compare with EchoView exported data
    bb_test_df = pd.read_csv(bb_power_test_path, header=None, skiprows=[0])
    bb_test_df_r = bb_test_df.iloc[::2, 14:]
    bb_test_df_i = bb_test_df.iloc[1::2, 14:]
    with xr.open_dataset(tmp.nc_path, group='Beam') as ds_beam:
        # Select 70 kHz channel and averaged across the quadrants
        backscatter_r = ds_beam.backscatter_r[0].dropna('range_bin').mean(axis=0)
        backscatter_i = ds_beam.backscatter_i[0].dropna('range_bin').mean(axis=0)
        assert np.allclose(backscatter_r, bb_test_df_r)
        assert np.allclose(backscatter_i, bb_test_df_i)

    # Test saving zarr bb file
    tmp.to_zarr()

    # Remove generated files
    os.remove(tmp.nc_path)
    shutil.rmtree(tmp.zarr_path, ignore_errors=True)


def test_cw_bb():
    # Test converting file that contains both cw and bb channels
    tmp = Convert(file=raw_path_bb_cw, model='EK80')
    tmp.to_netcdf()

    cw_path = './echopype/test_data/ek80/Summer2018--D20180905-T033113_cw.nc'
    nc_path = './echopype/test_data/ek80/Summer2018--D20180905-T033113.nc'
    assert os.path.exists(cw_path)
    assert os.path.exists(nc_path)
    os.remove(cw_path)
    os.remove(nc_path)


def test_freq_subset():
    # Test converting file with multiple frequencies that do not record power data
    tmp = Convert(file=raw_path_2_f, model='EK80')
    tmp.to_netcdf(overwrite=True)
    # Check if parsed output has the correct shape
    with xr.open_dataset(tmp.nc_path, group='Beam') as ds_beam:
        assert ds_beam.backscatter_r.shape == (2, 4, 1, 191327)
    os.remove(tmp.nc_path)


def test_xml():
    # Tests the exporting of the configuration xml as well as the environment xml
    tmp = Convert(file=raw_path_bb_cw, model='EK80')
    tmp.to_xml(data_type='CONFIG_XML')
    assert os.path.exists(tmp.output_file)
    os.remove(tmp.output_file)

    tmp.to_xml(save_path='env.xml', data_type='ENV_XML')
    assert os.path.exists(tmp.output_file)
    os.remove(tmp.output_file)
