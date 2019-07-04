import os
import numpy as np
import xarray as xr
from echopype.convert.azfp import ConvertAZFP
from echopype.model.azfp import EchoDataAZFP

azfp_xml_path = './echopype/data/17041823.XML'
azfp_01a_path = './echopype/data/17082117.01A'
azfp_test_Sv_path = './echopype/data/azfp_test/17082117_Sv.nc'
azfp_test_TS_path = './echopype/data/azfp_test/17082117_TS.nc'


def test_model_AZFP():
    # Read in the dataset that will be used to confirm working conversions. Generated from MATLAB code.
    Sv_test = xr.open_dataset(azfp_test_Sv_path)
    TS_test = xr.open_dataset(azfp_test_TS_path)

    # Unpacking data
    tmp_convert = ConvertAZFP(azfp_01a_path, azfp_xml_path)
    tmp_convert.parse_raw()

    # Convert to .nc file
    tmp_convert.raw2nc()

    tmp_echo = EchoDataAZFP(tmp_convert.nc_path)
    tmp_echo.calibrate()
    # TODO: add this back after adding TS component back in to data model
    # tmp_echo.calibrate_ts()

    # Test Sv data
    with xr.open_dataset(tmp_echo.Sv_path) as ds_Sv:
        assert np.allclose(Sv_test.Sv, ds_Sv.Sv, atol=1e-11)

    # Test TS data
    # TODO: add this back after adding TS component back in to data model
    # with xr.open_dataset(tmp_echo.TS_path) as ds_TS:
    #     assert np.allclose(TS_test.TS, ds_TS.TS, atol=1e-11)

    Sv_test.close()
    TS_test.close()
    os.remove(tmp_echo.Sv_path)
    # TODO: add this back after adding TS component back in to data model
    # os.remove(tmp_echo.TS_path)
    os.remove(tmp_convert.nc_path)
    del tmp_convert
    del tmp_echo
