import os
import shutil
import numpy as np
import xarray as xr
import pandas as pd
from ..convert import Convert, ConvertEK80
from ..process import ProcessEK60
from ..process import Process

# EK60 PATHS
# ek60_raw_path = './echopype/test_data/ek60/2015843-D20151023-T190636.raw'   # Varying ranges
ek60_raw_path = './echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw'     # Constant ranges
ek60_test_path = './echopype/test_data/ek60/from_matlab/DY1801_EK60-D20180211-T164025_Sv_TS.nc'
# Volume backscattering strength aqcuired from EchoView
ek60_csv_paths = ['./echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Sv18.csv',
                  './echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Sv38.csv',
                  './echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Sv70.csv',
                  './echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Sv120.csv',
                  './echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Sv200.csv']
nc_path = os.path.join(os.path.dirname(ek60_raw_path),
                       os.path.splitext(os.path.basename(ek60_raw_path))[0] + '.nc')
Sv_path = os.path.join(os.path.dirname(ek60_raw_path),
                       os.path.splitext(os.path.basename(ek60_raw_path))[0] + '_Sv.nc')

# AZFP PATHS
azfp_xml_path = './echopype/test_data/azfp/17041823.XML'
azfp_01a_path = './echopype/test_data/azfp/17082117.01A'
azfp_test_Sv_path = './echopype/test_data/azfp/from_matlab/17082117_Sv.nc'
azfp_test_TS_path = './echopype/test_data/azfp/from_matlab/17082117_TS.nc'
azfp_test_path = './echopype/test_data/azfp/from_matlab/17082117.nc'
azfp_echoview_Sv_paths = ['./echopype/test_data/azfp/from_echoview/17082117-Sv38.csv',
                          './echopype/test_data/azfp/from_echoview/17082117-Sv125.csv',
                          './echopype/test_data/azfp/from_echoview/17082117-Sv200.csv',
                          './echopype/test_data/azfp/from_echoview/17082117-Sv455.csv']
azfp_echoview_TS_paths = ['./echopype/test_data/azfp/from_echoview/17082117-TS38.csv',
                          './echopype/test_data/azfp/from_echoview/17082117-TS125.csv',
                          './echopype/test_data/azfp/from_echoview/17082117-TS200.csv',
                          './echopype/test_data/azfp/from_echoview/17082117-TS455.csv']


# EK80 PATHS
raw_path_bb = './echopype/test_data/ek80/D20170912-T234910.raw'       # Large file (BB)
raw_path_cw = './echopype/test_data/ek80/D20190822-T161221.raw'       # Small file (CW) (Standard test)
bb_power_test_path = './echopype/test_data/ek80/from_echoview/70 kHz raw power.complex.csv'
ek80_raw_path = './echopype/test_data/ek80/Summer2018--D20180905-T033113.raw'   # BB and CW


def test_bb():
    # Test conversion of EK80 broadband data file
    tmp = ConvertEK80(raw_path_bb)
    tmp.raw2nc()

    # Compare with EchoView exported data
    bb_test_df = pd.read_csv(bb_power_test_path, header=None, skiprows=[0])
    bb_test_df_r = bb_test_df.iloc[::2,14:]
    bb_test_df_i = bb_test_df.iloc[1::2,14:]
    with xr.open_dataset(tmp.nc_path, group='Beam') as ds_beam:
        # Select 70 kHz channel and averaged across the quadrants
        backscatter_r = ds_beam.backscatter_r[0].dropna('range_bin').mean('quadrant')
        backscatter_i = ds_beam.backscatter_i[0].dropna('range_bin').mean('quadrant')
        assert np.allclose(backscatter_r, bb_test_df_r)
        assert np.allclose(backscatter_i, bb_test_df_i)

    # Test saving zarr cw file
    tmp.raw2zarr()

    # Remove generated files
    os.remove(tmp.nc_path)
    shutil.rmtree(tmp.zarr_path, ignore_errors=True)


def test_calibrate_ek80_cw():
    """Check noise estimation and noise removal using xarray and brute force using numpy.
    """

    # Noise estimation via Process method =========
    # Unpack data and convert to .nc file
    tmp = Convert(raw_path_cw, model="EK80")
    tmp.raw2nc()

    # Read .nc file into an Process object and calibrate
    e_data = Process(tmp.nc_path)
    e_data.calibrate(save=True)
    e_data._temp_ed.close()
    os.remove('./echopype/test_data/ek80/D20190822-T161221_Sv.nc')


def test_calibration_ek60_echoview():
    tmp = Convert(ek60_raw_path)
    tmp.raw2nc(overwrite=True)

    # Read .nc file into an Process object and calibrate
    e_data = Process(nc_path)
    e_data.calibrate(save=True)

    channels = []
    for file in ek60_csv_paths:
        channels.append(pd.read_csv(file, header=None, skiprows=[0]).iloc[:, 13:])
    test_Sv = np.stack(channels)
    # Echoview data is missing 1 range. Also the first few ranges are handled differently
    assert np.allclose(test_Sv[:, :, 7:], e_data.Sv.Sv[:, :10, 8:], atol=1e-8)


def test_process_AZFP_matlab():
    # Read in the dataset that will be used to confirm working conversions. Generated from MATLAB code.
    Sv_test = xr.open_dataset(azfp_test_Sv_path)
    TS_test = xr.open_dataset(azfp_test_TS_path)

    # Convert to .nc file
    tmp_convert = Convert(azfp_01a_path, azfp_xml_path)
    tmp_convert.raw2nc()

    tmp_echo = Process(tmp_convert.nc_path)
    tmp_echo.calibrate(save=True)
    tmp_echo.calibrate_TS(save=True)
    tmp_echo.remove_noise()
    tmp_echo.get_MVBS()

    # Tolerance lowered due to temperature not being averaged as is the case in the matlab code
    # Test Sv data
    with xr.open_dataset(tmp_echo.Sv_path) as ds_Sv:
        assert np.allclose(Sv_test.Sv, ds_Sv.Sv, atol=1e-03)

    # Test TS data
    with xr.open_dataset(tmp_echo.TS_path) as ds_TS:
        assert np.allclose(TS_test.TS, ds_TS.Sp, atol=1e-03)

    Sv_test.close()
    TS_test.close()
    os.remove(tmp_echo.Sv_path)
    os.remove(tmp_echo.TS_path)
    os.remove(tmp_convert.nc_path)
    del tmp_convert
    del tmp_echo
