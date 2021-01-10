import os
imort glob
import fsspec
import shutil
import numpy as np
import xarray as xr
import pandas as pd
from ..convert import Convert


# TODO: add test for Convert._validate_path
def test_validate_path_single_source():
    single_path = './echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw'
    fsmap = fsspec.get_mapper(single_path)
    single_dir = os.path.dirname(fsmap.root)
    single_fname = os.path.splitext(os.path.basename(fsmap.root))[0]
    tmp_single = Convert(single_path, model='EK60')

    # if no output path is given
    tmp_single._validate_path(file_format='.zarr')
    assert tmp_single.output_file == [os.path.join(single_dir, single_fname+'.zarr')]

    # if an output folder is given, below works with and without the slash at the end
    test_save_path = './echopype/test_data/dump/'
    tmp_single._validate_path(file_format='.zarr', save_path=test_save_path)
    fsmap_tmp = fsspec.get_mapper(test_save_path)
    assert tmp_single.output_file == [os.path.join(fsmap_tmp.root, single_fname+'.zarr')]
    os.rmdir(os.path.dirname(tmp_single.output_file[0]))

    # if an output filename is given
    test_save_path = './echopype/test_data/dump/tmp.zarr'
    tmp_single._validate_path(file_format='.zarr', save_path=test_save_path)
    fsmap_tmp = fsspec.get_mapper(test_save_path)
    assert tmp_single.output_file == [fsmap_tmp.root]
    os.rmdir(os.path.dirname(tmp_single.output_file[0]))

    # force output file extension to the called type (here .zarr)
    test_save_path = './echopype/test_data/dump/tmp.nc'
    tmp_single._validate_path(file_format='.zarr', save_path=test_save_path)
    fsmap_tmp = fsspec.get_mapper(test_save_path)
    assert tmp_single.output_file == [os.path.splitext(fsmap_tmp.root)[0] + '.zarr']
    os.rmdir(os.path.dirname(tmp_single.output_file[0]))


def test_validate_path_multiple_source():
    mult_path = glob.glob('./echopype/test_data/ek60/*.raw')
    fsmap = fsspec.get_mapper(mult_path[0])
    mult_dir = os.path.dirname(fsmap.root)
    tmp_mult = Convert(mult_path, model='EK60')

    # if no output path is given
    tmp_mult._validate_path(file_format='.zarr')
    assert tmp_mult.output_file == [os.path.join(mult_dir, os.path.splitext(os.path.basename(f))[0]+'.zarr')
                                    for f in mult_path]

    # if an output folder is given, below works with and without the slash at the end
    test_save_path = './echopype/test_data/dump/'
    tmp_mult._validate_path(file_format='.zarr', save_path=test_save_path)
    fsmap_tmp = fsspec.get_mapper(test_save_path)
    assert tmp_mult.output_file == [os.path.join(fsmap_tmp.root, os.path.splitext(os.path.basename(f))[0]+'.zarr')
                                    for f in mult_path]
    os.rmdir(os.path.dirname(tmp_mult.output_file[0]))

    # if an output filename is given: only use the directory
    test_save_path = './echopype/test_data/dump/tmp.zarr'
    tmp_mult._validate_path(file_format='.zarr', save_path=test_save_path)
    fsmap_tmp = fsspec.get_mapper(test_save_path)
    assert tmp_mult.output_file == [os.path.join(os.path.dirname(fsmap_tmp.root),
                                                 os.path.splitext(os.path.basename(f))[0]+'.zarr')
                                    for f in mult_path]
    os.rmdir(os.path.dirname(tmp_mult.output_file[0]))

    # force output file extension to the called type (here .zarr)
    test_save_path = './echopype/test_data/dump/tmp.nc'
    tmp_mult._validate_path(file_format='.zarr', save_path=test_save_path)
    fsmap_tmp = fsspec.get_mapper(test_save_path)
    assert tmp_mult.output_file == [os.path.join(os.path.dirname(fsmap_tmp.root),
                                                 os.path.splitext(os.path.basename(f))[0] + '.zarr')
                                    for f in mult_path]
    os.rmdir(os.path.dirname(tmp_mult.output_file[0]))


def _converted_group_checker(model, engine, out_file):
    groups = ['Environment', 'Platform', 'Provenance', 'Sonar']
    if model in ['EK60', 'EK80']:
        groups = groups + ['Beam', 'Vendor']

    for g in groups:
        ds = xr.open_dataset(out_file, engine=engine, group=g)

        assert isinstance(ds, xr.Dataset) is True


def _file_export_checks(ec, model):
    file_map = {
        '.nc': {
            'engine': 'netcdf4',
            'export_func': ec.to_netcdf,
            'cleaner': os.unlink,
        },
        '.zarr': {
            'engine': 'zarr',
            'export_func': ec.to_zarr,
            'cleaner': shutil.rmtree,
        },
    }
    for k, v in file_map.items():
        out_file = f"./test_{model.lower()}{k}"
        func = v['export_func']
        clean = v['cleaner']
        func(out_file, overwrite=True)
        _converted_group_checker(
            model=model, engine=v['engine'], out_file=out_file
        )
        clean(out_file)


def test_http_azfp_convert():
    model = 'AZFP'
    # http
    azfp_path = 'https://rawdata.oceanobservatories.org/files/CE01ISSM/R00007/instrmts/dcl37/ZPLSC_sn55075/ce01issm_zplsc_55075_recovered_2017-10-27/DATA/201703/17032923.01A'
    xml_path = 'https://rawdata.oceanobservatories.org/files/CE01ISSM/R00007/instrmts/dcl37/ZPLSC_sn55075/ce01issm_zplsc_55075_recovered_2017-10-27/DATA/201703/17032922.XML'

    ec = Convert(azfp_path, model=model, xml_path=xml_path)

    assert ec.source_file[0] == azfp_path
    assert ec.xml_path == xml_path

    _file_export_checks(ec, model)


def test_http_ek60_convert():
    model = 'EK60'
    # http
    ek60_path = 'https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK60/Summer2017-D20170615-T190214.raw'

    ec = Convert(ek60_path, model=model)

    assert ec.source_file[0] == ek60_path

    _file_export_checks(ec, model)


def test_http_ek80_convert():
    model = 'EK80'
    # http
    ek80_path = 'https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK80/D20170826-T205615.raw'

    ec = Convert(ek80_path, model=model)

    assert ec.source_file[0] == ek80_path

    _file_export_checks(ec, model)


def test_s3_ek60_convert():
    model = 'EK60'
    # http
    ek60_path = 's3://ncei-wcsd-archive/data/raw/Bell_M._Shimada/SH1707/EK60/Summer2017-D20170615-T190214.raw'

    ec = Convert(ek60_path, model=model, storage_options={'anon': True})

    assert ec.source_file[0] == ek60_path

    _file_export_checks(ec, model)


def test_s3_ek80_convert():
    model = 'EK80'
    # http
    ek80_path = 's3://ncei-wcsd-archive/data/raw/Bell_M._Shimada/SH1707/EK80/D20170826-T205615.raw'

    ec = Convert(ek80_path, model=model, storage_options={'anon': True})

    assert ec.source_file[0] == ek80_path

    _file_export_checks(ec, model)
