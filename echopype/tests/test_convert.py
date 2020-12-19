import os
import shutil
import numpy as np
import xarray as xr
import pandas as pd
from ..convert import Convert

def test_http_azfp_convert():
    # http
    azfp_path = 'https://rawdata.oceanobservatories.org/files/CE01ISSM/R00007/instrmts/dcl37/ZPLSC_sn55075/ce01issm_zplsc_55075_recovered_2017-10-27/DATA/201703/17032923.01A'
    xml_path = 'https://rawdata.oceanobservatories.org/files/CE01ISSM/R00007/instrmts/dcl37/ZPLSC_sn55075/ce01issm_zplsc_55075_recovered_2017-10-27/DATA/201703/17032922.XML'

    out_file = "./test.nc"
    ec = Convert(azfp_path, model='AZFP', xml_path=xml_path)

    assert ec.source_file[0] == azfp_path
    assert ec.xml_path == xml_path

    ec.to_netcdf(out_file, overwrite=True)
    groups = ['Environment', 'Platform', 'Provenance', 'Sonar']

    for g in groups:
        ds = xr.open_dataset(out_file, engine='netcdf4', group=g)

        assert isinstance(ds, xr.Dataset) is True

    os.unlink(out_file)

def test_http_ek60_convert():
    # http
    ek60_path = 'https://rawdata.oceanobservatories.org/files/CE04OSPS/PC01B/ZPLSCB102_10.33.10.143/2017/08/21/OOI-D20170821-T031816.raw'

    out_file = "./test.nc"
    ec = Convert(ek60_path, model='EK60')
    ec.to_netcdf(out_file, overwrite=True)

    assert ec.source_file[0] == ek60_path

    os.unlink(out_file)

# TODO: Need ek80 http example
# def test_http_ek80_convert():
#     # http
#     ek80_path = 'https://rawdata.oceanobservatories.org/files/CE01ISSM/R00007/instrmts/dcl37/ZPLSC_sn55075/ce01issm_zplsc_55075_recovered_2017-10-27/DATA/201703/17032923.01A'

#     out_file = "./test.nc"
#     ec = Convert(ek80_path, model='EK80')
#     ec.to_netcdf(out_file, overwrite=True)

#     assert ec.source_file[0] == ek80_path

#     os.unlink(out_file)
