import os
import numpy as np
import xarray as xr
import shutil
from ..convert import Convert
from ..process import Process, EchoData

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
azfp_multi_paths = ['./echopype/test_data/azfp/set1/' + file
                    for file in os.listdir('./echopype/test_data/azfp/set1')]  # Multiple files (first is xml file)


def test_process_AZFP_matlab():
    # Read in the dataset that will be used to confirm working conversions. Generated from MATLAB code.

    # Convert to .nc file
    tmp_convert = Convert(file=azfp_01a_path, model='AZFP', xml_path=azfp_xml_path)
    tmp_convert.to_netcdf(overwrite=False)

    ed = EchoData(raw_path=tmp_convert.output_file)
    proc = Process(model='AZFP', ed=ed)
    # Matlab uses averaged temperature
    proc.env_params['water_temperature'] = proc.env_params['water_temperature'].mean('ping_time')
    proc.recalculate_environment(ed, ss=True)
    proc.get_Sv(ed, save=True, save_format='netcdf4')
    proc.get_Sp(ed, save=True, save_format='netcdf4')
    proc.remove_noise(ed, save=True, save_format='netcdf4')
    proc.get_MVBS(ed, save=True, save_format='netcdf4')

    Sv_test = xr.open_dataset(azfp_test_Sv_path)
    Sp_test = xr.open_dataset(azfp_test_TS_path)

    # Test Sv data
    assert np.allclose(Sv_test.Sv, ed.Sv.Sv, atol=1e-15)

    # Test TS data
    assert np.allclose(Sp_test.TS, ed.Sp.Sp, atol=1e-15)

    Sv_test.close()
    Sp_test.close()
    ed.close()
    os.remove(tmp_convert.output_file)
    os.remove(ed.Sv_path)
    os.remove(ed.Sp_path)
    os.remove(ed.Sv_clean_path)
    os.remove(ed.MVBS_path)


def test_multiple_raw():
    """Test calibration on multiple raw files"""
    export_folder = './echopype/test_data/azfp/export/'

    # Test combining while converting
    tmp = Convert(file=azfp_multi_paths[1:5], model='AZFP', xml_path=azfp_multi_paths[0])
    tmp.to_netcdf(save_path=export_folder)

    ed = EchoData(raw_path=tmp.output_file)
    proc = Process(model='AZFP', ed=ed)
    proc.get_Sv(ed, save=True, save_format='netcdf4')

    with xr.open_dataset(ed.Sv_path) as ds_sv:
        assert len(ds_sv.ping_time) == 960

    ed.close()
    shutil.rmtree(export_folder)


# def test_process_AZFP_echoview():
#     # Does not work because EchoView output does not match MATLAB output
#     # Convert to .nc file
#     tmp_convert = Convert(file=azfp_01a_path, model='AZFP', xml_path=azfp_xml_path)
#     tmp_convert.to_netcdf(overwrite=False)

#     ed = EchoData(raw_path=tmp_convert.output_path)
#     proc = Process(model='AZFP', ed=ed)
#     # Matlab uses averaged temperature
#     proc.env_params['water_temperature'] = proc.env_params['water_temperature'].mean('ping_time')
#     proc.recalculate_environment(ed, ss=True)
#     proc.get_Sv(ed, save=True, save_format='netcdf4')
#     proc.get_Sp(ed, save=True, save_format='netcdf4')

#     import pandas as pd
#     Sv_channels = []
#     for file in azfp_echoview_Sv_paths:
#         Sv_channels.append(pd.read_csv(file, header=None, skiprows=[0]).iloc[:, 13:])
#     test_Sv = np.stack(Sv_channels)
#     assert np.allclose(test_Sv, ed.Sv.Sv[:, :10, :], atol=1e-2)

#     Sp_channels = []
#     for file in azfp_echoview_TS_paths:
#         Sp_channels.append(pd.read_csv(file, header=None, skiprows=[0]).iloc[:, 13:])
#     test_Sp = np.stack(Sp_channels)
#     assert np.allclose(test_Sp, proc.Sp.Sp[:, :10, :], atol=1e-2)
