from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from ..convert import Convert
from ..process import EchoDataNew
from ..calibrate import calibrate


def test_get_Sv_ek60_echoview():
    ek60_raw_path = './echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw'  # Constant ranges
    ek60_csv_path = Path('./echopype/test_data/ek60/from_echoview/')

    # Convert file
    c = Convert(ek60_raw_path, model='EK60')
    c.to_netcdf(overwrite=True)

    # Calibrate to get Sv
    echodata = EchoDataNew(raw_path=c.output_file)
    ds_Sv = calibrate(echodata)

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
    c = Convert(file=azfp_01a_path, model='AZFP', xml_path=azfp_xml_path)
    c.to_netcdf(overwrite=True)

    # Calibrate to get Sv and TS
    with xr.open_dataset(c.output_file, group='Environment') as ds_env:
        avg_temperature = ds_env['temperature'].mean('ping_time').values  # AZFP Matlab code uses average temperature
    echodata = EchoDataNew(raw_path=c.output_file)
    ds_Sv = calibrate(echodata, env_params={'temperature': avg_temperature, 'salinity': 29.6, 'pressure': 60})

    # Load Matlab outputs and test
    Sv_test = xr.open_dataset(azfp_test_Sv_path)
    # Sp_test = xr.open_dataset(azfp_test_TS_path)

    assert np.allclose(Sv_test.Sv, ds_Sv.Sv, atol=1e-15)

    Path(c.output_file).unlink()
