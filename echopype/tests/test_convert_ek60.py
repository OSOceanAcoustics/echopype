import os
import shutil
import numpy as np
import xarray as xr
import pandas as pd
import pytest
from ..convert import Convert

raw_path = './echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw'     # Standard test
test_path = './echopype/test_data/ek60/from_matlab/DY1801_EK60-D20180211-T164025.nc'
csv_paths = ['./echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Power18.csv',
             './echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Power38.csv',
             './echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Power70.csv',
             './echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Power120.csv',
             './echopype/test_data/ek60/from_echoview/DY1801_EK60-D20180211-T164025-Power200.csv']
# raw_paths = ['./echopype/test_data/ek60/set1/' + file
            #  for file in os.listdir('./echopype/test_data/ek60/set1')]    # 2 range lengths
# raw_path = ['./echopype/test_data/ek60/set2/' + file
#                  for file in os.listdir('./echopype/test_data/ek60/set2')]    # 3 range lengths
# Other data files
# raw_filename = 'data_zplsc/OceanStarr_2017-D20170725-T004612.raw'  # OceanStarr 2 channel EK60
# raw_filename = '../data/DY1801_EK60-D20180211-T164025.raw'  # Dyson 5 channel EK60
# raw_filename = 'data_zplsc/D20180206-T000625.raw   # EK80


def test_convert_matlab():
    """Test converting power and angle data and compare output to MATLAB conversion output"""
    tmp = Convert(file=raw_path, model='EK60')

    # Test saving nc file and perform checks
    tmp.to_netcdf()

    # Read .nc file into an xarray DataArray
    ds_beam = xr.open_dataset(tmp.output_path, group='Beam')

    # Test dataset was created by exporting values from MATLAB EK60 parsing code
    with xr.open_dataset(test_path) as ds_test:
        assert np.allclose(ds_test.power, ds_beam.backscatter_r)    # Identical to MATLAB output to 1e-6
        athwartship = (ds_beam['angle_athwartship'] * 1.40625 / ds_beam['angle_sensitivity_athwartship'] -
                       ds_beam['angle_offset_athwartship'])
        alongship = (ds_beam['angle_alongship'] * 1.40625 / ds_beam['angle_sensitivity_alongship'] -
                     ds_beam['angle_offset_alongship'])
        assert np.allclose(ds_test.athwartship, athwartship)    # Identical to MATLAB output to 1e-7
        assert np.allclose(ds_test.alongship, alongship)        # Identical to MATLAB output to 1e-7

    ds_beam.close()
    os.remove(tmp.output_path)
    del tmp


def test_convert_power_echoview():
    """Test converting power and compare it to echoview output"""
    tmp = Convert(file=raw_path, model='EK60')
    tmp.to_netcdf()

    channels = []
    for file in csv_paths:
        channels.append(pd.read_csv(file, header=None, skiprows=[0]).iloc[:, 13:])
    test_power = np.stack(channels)
    with xr.open_dataset(tmp.output_path, group='Beam') as ds_beam:
        assert np.allclose(
            test_power,
            ds_beam.backscatter_r.isel(ping_time=slice(None, 10), range_bin=slice(1, None)),
            atol=1e-10)

    os.remove(tmp.output_path)


def test_convert_zarr():
    """Test saving a zarr file"""
    tmp = Convert(file=raw_path, model='EK60')
    tmp.to_zarr()

    ds_beam = xr.open_zarr(tmp.output_path, group='Beam')
    with xr.open_dataset(test_path) as ds_test:
        assert np.allclose(ds_test.power, ds_beam.backscatter_r)

    shutil.rmtree(tmp.output_path, ignore_errors=True)    # Delete non-empty folder


@pytest.mark.skip(reason='Too many raw files needed')
def test_combine():
    """Test combing converted files"""
    export_folder = './echopype/test_data/ek60/export/'

    # Test combining while converting
    tmp = Convert(file=raw_paths[:4], model='EK60')
    tmp.to_netcdf(save_path=export_folder, overwrite=True, combine=True)

    # Test combining after converting
    tmp = Convert(file=raw_paths[4:8], model='EK60')
    tmp.to_netcdf(save_path=export_folder, overwrite=True)
    tmp.combine_files(tmp.output_path, remove_orig=False)

    shutil.rmtree(export_folder)


@pytest.mark.skip(reason='Do not use cloud resources for auto test')
def test_save_to_s3():
    # Test saving a zarr file to bucket
    import s3fs
    raw_paths = ['./echopype/test_data/ek60/set1/Winter2017-D20170115-T131932.raw',
                 './echopype/test_data/ek60/set1/Winter2017-D20170115-T134426.raw']
    path = 's3://"bucket name"/'
    fs = s3fs.S3FileSystem()
    store = s3fs.S3Map(root=path, s3=fs, check=False)
    tmp = Convert(file=raw_paths, model='EK60')
    tmp.to_zarr(save_path=store, overwrite=True, combine=True)


@pytest.mark.skip(reason='Do not use cloud resources for auto test')
def test_save_to_gcloud():
    import gcsfs
    raw_paths = ['./echopype/test_data/ek60/set1/Winter2017-D20170115-T131932.raw',
                 './echopype/test_data/ek60/set1/Winter2017-D20170115-T134426.raw']
    path = 'gcs://"bucket name"/'
    fs = gcsfs.GCSFileSystem(token='C:/Users/"user"/AppData/Roaming/gcloud/application_default_credentials.json')
    store = gcsfs.GCSMap(root=path, gcs=fs)
    tmp = Convert(file=raw_paths, model='EK60')
    tmp.to_zarr(save_path=store, overwrite=True, combine=True)


@pytest.mark.skip(reason='Do not use cloud resources for auto test')
def test_save_to_azure():
    # adlfs with zarr still in development
    import adlfs
    raw_paths = ['./echopype/test_data/ek60/set1/Winter2017-D20170115-T131932.raw',
                 './echopype/test_data/ek60/set1/Winter2017-D20170115-T134426.raw']
    path = '"blob name"'
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')      # Save azure keys in environment variable first
    fs = adlfs.AzureBlobFileSystem(account_name='"account name"', connection_string=connect_str)
    store = fs.get_mapper(path)
    tmp = Convert(file=raw_paths, model='EK60')
    tmp.to_zarr(save_path=store, overwrite=True, combine=True)
