import os
import shutil
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from echopype.convert import Convert, ConvertEK80
from echopype.process import Process

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

# AZFP PATHS
azfp_path = Path('./echopype/test_data/azfp')


# EK80 PATHS
raw_path_bb = './echopype/test_data/ek80/D20170912-T234910.raw'       # Large file (BB)
raw_path_cw = './echopype/test_data/ek80/D20190822-T161221.raw'       # Small file (CW) (Standard test)
bb_power_test_path = './echopype/test_data/ek80/from_echoview/D20170912-T234910/70 kHz raw power.complex.csv'
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
    e_data = Process(tmp.nc_path)
    e_data.calibrate(save=True)

    channels = []
    for file in ek60_csv_paths:
        channels.append(pd.read_csv(file, header=None, skiprows=[0]).iloc[:, 13:])
    test_Sv = np.stack(channels)
    # Echoview data is missing 1 range. Also the first few ranges are handled differently
    assert np.allclose(test_Sv[:, :, 7:], e_data.Sv.Sv[:, :10, 8:], atol=1e-8)


def test_process_AZFP_matlab():
    # Read in the dataset that will be used to confirm working conversions. Generated from MATLAB code.
    Sv_test = loadmat(str(azfp_path.joinpath('from_matlab/17082117_matlab_Output_Sv.mat')))
    TS_test = loadmat(str(azfp_path.joinpath('from_matlab/17082117_matlab_Output_TS.mat')))

    # Convert to .nc file
    tmp_convert = Convert(str(azfp_path.joinpath('17082117.01A')), str(azfp_path.joinpath('17041823.XML')))
    tmp_convert.raw2nc()

    tmp_echo = Process(tmp_convert.nc_path)
    tmp_echo.calibrate(save=True)
    tmp_echo.calibrate_TS(save=True)
    tmp_echo.remove_noise()
    tmp_echo.get_MVBS()

    # Tolerance lowered due to temperature not being averaged as is the case in the matlab code
    # Test Sv data
    with xr.open_dataset(tmp_echo.Sv_path) as ds_Sv:
        assert np.allclose(
            np.array([Sv_test['Output']['Sv'][0][fidx] for fidx in range(4)]),
            ds_Sv.Sv,
            atol=1e-03
        )

    # Test TS data
    with xr.open_dataset(tmp_echo.TS_path) as ds_TS:
        assert np.allclose(
            np.array([TS_test['Output']['TS'][0][fidx] for fidx in range(4)]),
            ds_TS.TS,
            atol=1e-03
        )

    Sv_test.close()
    TS_test.close()
    os.remove(tmp_echo.Sv_path)
    os.remove(tmp_echo.TS_path)
    os.remove(tmp_convert.nc_path)
