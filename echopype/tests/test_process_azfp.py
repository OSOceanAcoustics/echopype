import os
import numpy as np
import pandas as pd
import xarray as xr
from echopype.convert import Convert
from echopype.process import Process

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

    # Check setters
    tmp_echo.pressure = 10
    tmp_echo.salinity = 20
    tmp_echo.temperature = 12

    with xr.open_dataset(tmp_echo.Sv_path) as ds_Sv:
        assert np.allclose(Sv_test.Sv, ds_Sv.Sv, atol=1e-15)

    # Test TS data
    with xr.open_dataset(tmp_echo.TS_path) as ds_TS:
        assert np.allclose(TS_test.TS, ds_TS.TS, atol=1e-15)

    Sv_test.close()
    TS_test.close()
    os.remove(tmp_echo.Sv_path)
    os.remove(tmp_echo.TS_path)
    os.remove(tmp_convert.nc_path)
    del tmp_convert
    del tmp_echo


# def test_process_AZFP_echoview():
#     # Convert to .nc file
#     tmp_convert = Convert(azfp_01a_path, azfp_xml_path)
#     tmp_convert.raw2nc()

#     tmp_echo = Process(tmp_convert.nc_path)
#     tmp_echo.calibrate()
#     tmp_echo.calibrate_TS()

#     Sv_channels = []
#     for file in azfp_echoview_Sv_paths:
#         Sv_channels.append(pd.read_csv(file, header=None, skiprows=[0]).iloc[:, 13:])
#     test_Sv = np.stack(Sv_channels)
#     assert np.allclose(test_Sv, tmp_echo.TS.TS[:, :10, :], atol=1e-2)

#     TS_channels = []
#     for file in azfp_echoview_TS_paths:
#         TS_channels.append(pd.read_csv(file, header=None, skiprows=[0]).iloc[:, 13:])
#     test_Sv = np.stack(TS_channels)
#     assert np.allclose(test_Sv, tmp_echo.TS.TS[:, :10, :], atol=1e-2)
