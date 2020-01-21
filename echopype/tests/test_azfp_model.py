import os
import numpy as np
import xarray as xr
from echopype.convert import Convert
from echopype.model import EchoData

azfp_xml_path = './echopype/test_data/azfp/17041823.XML'
azfp_01a_path = './echopype/test_data/azfp/17082117.01A'
azfp_test_Sv_path = './echopype/test_data/azfp/from_matlab/17082117_Sv.nc'
azfp_test_TS_path = './echopype/test_data/azfp/from_matlab/17082117_TS.nc'
azfp_test_path = './echopype/test_data/azfp/from_matlab/17082117.nc'


def test_model_AZFP():
    # Read in the dataset that will be used to confirm working conversions. Generated from MATLAB code.
    Sv_test = xr.open_dataset(azfp_test_Sv_path)
    TS_test = xr.open_dataset(azfp_test_TS_path)

    # Convert to .nc file
    tmp_convert = Convert(azfp_01a_path, azfp_xml_path)
    tmp_convert.raw2nc()

    tmp_echo = EchoData(tmp_convert.nc_path)
    tmp_echo.calibrate(save=True)
    tmp_echo.calibrate_TS(save=True)
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
