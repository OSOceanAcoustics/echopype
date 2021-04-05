from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat
from ... import echopype as ep


def test_get_Sv_ek60_echoview():
    ek60_raw_path = './echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw'  # Constant ranges
    ek60_csv_path = Path('./echopype/test_data/ek60/from_echoview/')

    # Convert file
    c = ep.Convert(ek60_raw_path, model='EK60')
    c.to_netcdf(overwrite=True)

    # Calibrate to get Sv
    echodata = ep.EchoData(raw_path=c.output_file)
    ds_Sv = ep.calibrate.compute_Sv(echodata)

    # Compare with EchoView outputs
    channels = []
    for freq in [18, 38, 70, 120, 200]:
        fname = str(ek60_csv_path.joinpath('DY1801_EK60-D20180211-T164025-Sv%d.csv' % freq))
        channels.append(pd.read_csv(fname, header=None, skiprows=[0]).iloc[:, 13:])
    test_Sv = np.stack(channels)

    # Echoview data is shifted by 1 sample along range (missing the first sample)
    assert np.allclose(test_Sv[:, :, 7:],
                       ds_Sv.Sv.isel(ping_time=slice(None, 10), range_bin=slice(8, None)), atol=1e-8)

    Path(c.output_file).unlink()


def test_get_Sv_azfp():
    azfp_xml_path = './echopype/test_data/azfp/17041823.XML'
    azfp_01a_path = './echopype/test_data/azfp/17082117.01A'

    # Test data generated from AZFP Matlab code
    azfp_test_Sv_path = './echopype/test_data/azfp/from_matlab/17082117_Sv.nc'
    # azfp_test_TS_path = './echopype/test_data/azfp/from_matlab/17082117_TS.nc'

    # Convert to .nc file
    c = ep.Convert(file=azfp_01a_path, model='AZFP', xml_path=azfp_xml_path)
    c.to_netcdf(overwrite=True)

    # Calibrate using identical env params as in Matlab ParametersAZFP.m
    with xr.open_dataset(c.output_file, group='Environment') as ds_env:
        avg_temperature = ds_env['temperature'].mean('ping_time').values  # AZFP Matlab code uses average temperature
    env_params = {'temperature': avg_temperature, 'salinity': 27.9, 'pressure': 59}
    echodata = ep.EchoData(raw_path=c.output_file)
    ds_Sv = ep.calibrate.compute_Sv(echodata=echodata, env_params=env_params)
    ds_Sp = ep.calibrate.compute_Sp(echodata=echodata, env_params=env_params)

    # Load Matlab outputs and test
    # Matlab outputs were saved using
    #   save('from_matlab/17082117_matlab_Output.mat', 'Output')  # data variables
    #   save('from_matlab/17082117_matlab_Par.mat', 'Par')  # parameters
    base_Sv = loadmat('./echopype/test_data/azfp/from_matlab/17082117_matlab_Output.mat')
    for fidx in range(4):  # loop through all freq
        assert np.alltrue(ds_Sv.range.isel(frequency=fidx).values == base_Sv['Output'][0]['Range'][fidx])
        assert np.allclose(ds_Sv.Sv.isel(frequency=fidx).values, base_Sv['Output'][0]['Sv'][fidx],
                           atol=1e-13, rtol=0)

    base_Sp = loadmat('./echopype/test_data/azfp/from_matlab/17082117_matlab_Output_TS.mat')
    for fidx in range(4):  # loop through all freq
        assert np.alltrue(ds_Sp.range.isel(frequency=fidx).values == base_Sp['Output'][0]['Range'][fidx])
        assert np.allclose(ds_Sp.Sp.isel(frequency=fidx).values, base_Sp['Output'][0]['TS'][fidx],
                           atol=1e-13, rtol=0)

    Path(c.output_file).unlink()


def test_compute_Sv_EK80_CW_complex():
    fname_zarr = '/Volumes/MURI_4TB/MURI/spheroid_echoes/Data_zarr/ar2.0-D20201210-T000409.zarr'  # CW complex
    echodata = ep.EchoData(fname_zarr)
    ds_Sv = ep.calibrate.compute_Sv(echodata, waveform_mode='CW', encode_mode='complex')


def test_compute_Sv_EK80_BB_complex():
    fname_zarr = '/Volumes/MURI_4TB/MURI/spheroid_echoes/Data_zarr/ar2.0-D20201209-T235955.zarr'
    echodata = ep.EchoData(fname_zarr)
    Sv = ep.calibrate.compute_Sv(echodata, waveform_mode='BB', encode_mode='complex')
    Sp = ep.calibrate.compute_Sp(echodata, waveform_mode='BB', encode_mode='complex')