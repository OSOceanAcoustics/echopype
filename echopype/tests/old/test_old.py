import os
import shutil
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from echopype.convert import Convert, ConvertEK80
from echopype.process import Process

ek60_path = Path('./echopype/test_data/ek60')
azfp_path = Path('./echopype/test_data/azfp')
ek80_path = Path('./echopype/test_data/ek80')


def test_calibrate_ek80_bb():
    ek80_raw_path = ek80_path.joinpath('D20170912-T234910.raw')
    # Test conversion of EK80 broadband data file
    tmp = ConvertEK80(ek80_raw_path)
    tmp.raw2nc()

    # Compare with EchoView exported data
    bb_test_df = pd.read_csv(ek80_path.joinpath('from_echoview/D20170912-T234910/70 kHz raw power.complex.csv'),
                             header=None, skiprows=[0])
    bb_test_df_r = bb_test_df.iloc[::2, 14:]
    bb_test_df_i = bb_test_df.iloc[1::2, 14:]
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
    ek80_raw_path = ek80_path.joinpath('D20190822-T161221.raw')
    # Unpack data and convert to .nc file
    tmp = Convert(ek80_raw_path, model="EK80")
    tmp.raw2nc()

    # Read .nc file into an Process object and calibrate
    e_data = Process(tmp.nc_path)
    e_data.calibrate(save=True)
    os.remove(e_data.Sv_path)


def test_calibration_ek60_echoview():
    ek60_raw_path = str(ek60_path.joinpath('DY1801_EK60-D20180211-T164025.raw'))  # constant range_bin
    ek60_echoview_path = ek60_path.joinpath('from_echoview')

    tmp = Convert(ek60_raw_path)
    tmp.raw2nc(overwrite=True)

    # Read .nc file into an Process object and calibrate
    e_data = Process(tmp.nc_path)
    e_data.calibrate(save=True)

    # Compare with EchoView outputs
    channels = []
    for freq in [18, 38, 70, 120, 200]:
        fname = str(ek60_echoview_path.joinpath('DY1801_EK60-D20180211-T164025-Sv%d.csv' % freq))
        channels.append(pd.read_csv(fname, header=None, skiprows=[0]).iloc[:, 13:])
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

    tmp_echo = Process(tmp_convert.nc_path, salinity=27.9, pressure=59, temperature=None)
    tmp_echo.calibrate(save=True)
    tmp_echo.calibrate_TS(save=True)
    tmp_echo.remove_noise()
    tmp_echo.get_MVBS()

    # Tolerance lowered due to temperature not being averaged as is the case in the matlab code
    # Test Sv data
    def check_output(ds_base, ds_cmp, cal_type):
        for fidx in range(4):  # loop through all freq
            assert np.alltrue(
                ds_cmp.range.isel(frequency=fidx).values == ds_base['Output'][0]['Range'][fidx]
            )
            assert np.allclose(
                ds_cmp[cal_type].isel(frequency=fidx).values,
                ds_base['Output'][0][cal_type][fidx],
                atol=1e-13, rtol=0
            )
    # Check Sv
    check_output(ds_base=Sv_test, ds_cmp=tmp_echo.Sv, cal_type='Sv')

    # Check Sp
    check_output(ds_base=TS_test, ds_cmp=tmp_echo.TS, cal_type='TS')

    os.remove(tmp_echo.Sv_path)
    os.remove(tmp_echo.TS_path)
    del tmp_echo
    os.remove(tmp_convert.nc_path)
