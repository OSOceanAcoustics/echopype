"""test_convert.py

This module contain all the various tests for echopype conversion 
from a raw data to standard compliant zarr or netcdf file(s).

**Note that in order to run this test, minio server is required for s3
output tests.**
"""


import os
import glob
import fsspec
import xarray as xr
import pytest
from pathlib import Path
from ..convert import Convert


def _check_file_group(data_file, engine, groups):
    for g in groups:
        ds = xr.open_dataset(data_file, engine=engine, group=g)

        assert isinstance(ds, xr.Dataset) is True


def _check_output_files(engine, output_files, storage_options):
    groups = [
        'Provenance',
        'Environment',
        'Beam',
        'Sonar',
        'Vendor',
        'Platform',
    ]
    if isinstance(output_files, list):
        fs = fsspec.get_mapper(output_files[0], **storage_options).fs
        for f in output_files:
            if engine == 'zarr':
                _check_file_group(fs.get_mapper(f), engine, groups)
                fs.delete(f, recursive=True)
            else:
                _check_file_group(f, engine, groups)
                fs.delete(f)
    else:
        fs = fsspec.get_mapper(output_files, **storage_options).fs
        if engine == 'zarr':
            _check_file_group(fs.get_mapper(output_files), engine, groups)
            fs.delete(output_files, recursive=True)
        else:
            _check_file_group(output_files, engine, groups)
            fs.delete(output_files)


def _download_file(source_url, target_url):
    fs = fsspec.filesystem('file')
    if not fs.exists(os.path.dirname(target_url)):
        fs.mkdir(os.path.dirname(target_url))

    if not fs.exists(target_url):
        with fsspec.open(source_url, mode="rb") as source:
            with fs.open(target_url, mode="wb") as target:
                target.write(source.read())


@pytest.fixture(scope="session")
def minio_bucket():
    bucket_name = 'ooi-raw-data'
    fs = fsspec.filesystem(
        's3',
        **dict(
            client_kwargs=dict(endpoint_url='http://localhost:9000/'),
            key='minioadmin',
            secret='minioadmin',
        ),
    )
    if not fs.exists(bucket_name):
        fs.mkdir(bucket_name)


@pytest.fixture(scope="session")
def download_files():
    ek60_source = 'https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK60/Summer2017-D20170615-T190214.raw'
    ek80_source = 'https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK80/D20170826-T205615.raw'
    azfp_source = 'https://rawdata.oceanobservatories.org/files/CE01ISSM/R00007/instrmts/dcl37/ZPLSC_sn55075/ce01issm_zplsc_55075_recovered_2017-10-27/DATA/201703/17032923.01A'
    azfp_xml_source = 'https://rawdata.oceanobservatories.org/files/CE01ISSM/R00007/instrmts/dcl37/ZPLSC_sn55075/ce01issm_zplsc_55075_recovered_2017-10-27/DATA/201703/17032922.XML'

    ek60_path = os.path.join(
        './echopype/test_data/ek60/ncei-wcsd',
        os.path.basename(ek60_source),
    )
    ek80_path = os.path.join(
        './echopype/test_data/ek80/ncei-wcsd',
        os.path.basename(ek80_source),
    )
    azfp_path = os.path.join(
        './echopype/test_data/azfp/ooi', os.path.basename(azfp_source)
    )
    azfp_xml_path = os.path.join(
        './echopype/test_data/azfp/ooi', os.path.basename(azfp_xml_source)
    )
    download_paths = [
        (ek60_source, ek60_path),
        (ek80_source, ek80_path),
        (azfp_source, azfp_path),
        (azfp_xml_source, azfp_xml_path),
    ]

    for p in download_paths:
        _download_file(*p)


@pytest.mark.parametrize("model", ["EK60"])
@pytest.mark.parametrize("file_format", [".zarr"])
@pytest.mark.parametrize(
    "input_path",
    [
        "./echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw",
        "https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK60/Summer2017-D20170615-T190214.raw",
    ],
)
@pytest.mark.parametrize(
    "output_save_path",
    [
        None,
        "./echopype/test_data/dump/",
        "./echopype/test_data/dump/tmp.zarr",
        "./echopype/test_data/dump/tmp.nc",
        "s3://ooi-raw-data/dump/",
        "s3://ooi-raw-data/dump/tmp.zarr",
        "s3://ooi-raw-data/dump/tmp.nc",
    ],
)
def test_validate_path_single_source(
    model, file_format, input_path, output_save_path, minio_bucket
):

    output_storage_options = {}
    if output_save_path and output_save_path.startswith('s3://'):
        output_storage_options = dict(
            client_kwargs=dict(endpoint_url='http://localhost:9000/'),
            key='minioadmin',
            secret='minioadmin',
        )
    fsmap = fsspec.get_mapper(input_path)
    single_dir = os.path.dirname(fsmap.root)
    single_fname = os.path.splitext(os.path.basename(fsmap.root))[0]
    tmp_single = Convert(input_path, model=model)
    tmp_single._output_storage_options = output_storage_options

    tmp_single._validate_path(
        file_format=file_format, save_path=output_save_path
    )

    if output_save_path is not None:
        fsmap_tmp = fsspec.get_mapper(
            output_save_path, **output_storage_options
        )
        fs = fsmap_tmp.fs
        if not output_save_path.startswith('s3'):
            if output_save_path.endswith('/'):
                # if an output folder is given, below works with and without the slash at the end
                assert tmp_single.output_file == [
                    os.path.join(fsmap_tmp.root, single_fname + '.zarr')
                ]
            elif output_save_path.endswith('.zarr'):
                # if an output filename is given
                assert tmp_single.output_file == [fsmap_tmp.root]
            else:
                # force output file extension to the called type (here .zarr)
                assert tmp_single.output_file == [
                    os.path.splitext(fsmap_tmp.root)[0] + '.zarr'
                ]
            os.rmdir(os.path.dirname(tmp_single.output_file[0]))
        else:
            if output_save_path.endswith('/'):
                # if an output folder is given, below works with and without the slash at the end
                assert tmp_single.output_file == [
                    os.path.join(output_save_path, single_fname + '.zarr')
                ]
            elif output_save_path.endswith('.zarr'):
                # if an output filename is given
                assert tmp_single.output_file == [output_save_path]
            else:
                # force output file extension to the called type (here .zarr)
                assert tmp_single.output_file == [
                    os.path.splitext(output_save_path)[0] + '.zarr'
                ]
            fs.delete(tmp_single.output_file[0])
    else:
        if input_path.startswith('https') or input_path.startswith('s3'):
            current_dir = Path.cwd()
            temp_dir = current_dir.joinpath(Path('temp_echopype_output'))
            assert tmp_single.output_file == [
                str(temp_dir.joinpath(Path(single_fname + '.zarr')))
            ]
            os.rmdir(os.path.dirname(tmp_single.output_file[0]))
        else:
            # if no output path is given
            assert tmp_single.output_file == [
                os.path.join(single_dir, single_fname + '.zarr')
            ]


@pytest.mark.parametrize("model", ["EK60"])
@pytest.mark.parametrize("file_format", [".zarr"])
@pytest.mark.parametrize(
    "input_path",
    [
        "./echopype/test_data/ek60/*.raw",
        [
            'https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK60/Summer2017-D20170615-T190214.raw',
        ],
    ],
)
@pytest.mark.parametrize(
    "output_save_path",
    [
        None,
        "./echopype/test_data/dump/",
        "./echopype/test_data/dump/tmp.zarr",
        "./echopype/test_data/dump/tmp.nc",
        "s3://ooi-raw-data/dump/",
        "s3://ooi-raw-data/dump/tmp.zarr",
        "s3://ooi-raw-data/dump/tmp.nc",
    ],
)
def test_validate_path_multiple_source(
    model, file_format, input_path, output_save_path, minio_bucket
):
    output_storage_options = {}
    if output_save_path and output_save_path.startswith('s3://'):
        output_storage_options = dict(
            client_kwargs=dict(endpoint_url='http://localhost:9000/'),
            key='minioadmin',
            secret='minioadmin',
        )

    if isinstance(input_path, str):
        mult_path = glob.glob(input_path)
    else:
        mult_path = input_path
    fsmap = fsspec.get_mapper(mult_path[0])
    mult_dir = os.path.dirname(fsmap.root)
    tmp_mult = Convert(mult_path, model='EK60')
    tmp_mult._output_storage_options = output_storage_options

    tmp_mult._validate_path(
        file_format=file_format, save_path=output_save_path
    )

    if output_save_path is not None:
        fsmap_tmp = fsspec.get_mapper(output_save_path)
        fs = fsmap_tmp.fs
        if not output_save_path.startswith('s3'):
            if output_save_path.endswith('/'):
                # if an output folder is given, below works with and without the slash at the end
                assert tmp_mult.output_file == [
                    os.path.join(
                        fsmap_tmp.root,
                        os.path.splitext(os.path.basename(f))[0] + '.zarr',
                    )
                    for f in mult_path
                ]
            elif output_save_path.endswith('.zarr'):
                # if an output filename is given: only use the directory
                assert tmp_mult.output_file == [
                    os.path.abspath(output_save_path)
                ]
            elif output_save_path.endswith('.nc'):
                # force output file extension to the called type (here .zarr)
                assert tmp_mult.output_file == [
                    os.path.abspath(output_save_path.replace('.nc', '.zarr'))
                ]
            os.rmdir(os.path.dirname(tmp_mult.output_file[0]))
        else:
            if output_save_path.endswith('/'):
                # if an output folder is given, below works with and without the slash at the end
                assert tmp_mult.output_file == [
                    os.path.join(
                        output_save_path,
                        os.path.splitext(os.path.basename(f))[0] + '.zarr',
                    )
                    for f in mult_path
                ]
            elif output_save_path.endswith('.zarr'):
                # if an output filename is given: only use the directory
                assert tmp_mult.output_file == [output_save_path]
            elif output_save_path.endswith('.nc'):
                # force output file extension to the called type (here .zarr)
                assert tmp_mult.output_file == [
                    output_save_path.replace('.nc', '.zarr')
                ]
            fs.delete(tmp_mult.output_file[0])
    else:
        if input_path[0].startswith('https') or input_path[0].startswith('s3'):
            current_dir = Path.cwd()
            temp_dir = current_dir.joinpath(Path('temp_echopype_output'))
            assert tmp_mult.output_file == [
                str(
                    temp_dir.joinpath(
                        Path(
                            os.path.splitext(os.path.basename(f))[0] + '.zarr'
                        )
                    )
                )
                for f in mult_path
            ]
            os.rmdir(os.path.dirname(tmp_mult.output_file[0]))
        else:
            # if no output path is given
            assert tmp_mult.output_file == [
                os.path.join(
                    mult_dir,
                    os.path.splitext(os.path.basename(f))[0] + '.zarr',
                )
                for f in mult_path
            ]


@pytest.mark.parametrize("model", ["EK60"])
@pytest.mark.parametrize(
    "input_path",
    [
        "./echopype/test_data/ek60/ncei-wcsd/Summer2017-D20170615-T190214.raw",
        "https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK60/Summer2017-D20170615-T190214.raw",
        "s3://ncei-wcsd-archive/data/raw/Bell_M._Shimada/SH1707/EK60/Summer2017-D20170615-T190214.raw",
        [
            'https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK60/Summer2017-D20170615-T190214.raw',
            'https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK60/Summer2017-D20170615-T190843.raw',
        ],
    ],
)
@pytest.mark.parametrize("export_engine", ["zarr", "netcdf4"])
@pytest.mark.parametrize(
    "output_save_path",
    [
        None,
        "./echopype/test_data/dump/",
        "./echopype/test_data/dump/tmp.zarr",
        "./echopype/test_data/dump/tmp.nc",
        "s3://ooi-raw-data/dump/",
        "s3://ooi-raw-data/dump/tmp.zarr",
        "s3://ooi-raw-data/dump/tmp.nc",
    ],
)
@pytest.mark.parametrize("combine_files", [False])
def test_convert_ek60(
    model,
    input_path,
    export_engine,
    output_save_path,
    combine_files,
    minio_bucket,
    download_files,
):
    output_storage_options = {}
    ipath = input_path
    if isinstance(input_path, list):
        ipath = input_path[0]

    input_storage_options = {'anon': True} if ipath.startswith('s3://') else {}
    if output_save_path and output_save_path.startswith('s3://'):
        output_storage_options = dict(
            client_kwargs=dict(endpoint_url='http://localhost:9000/'),
            key='minioadmin',
            secret='minioadmin',
        )

    ec = Convert(
        file=input_path, model=model, storage_options=input_storage_options
    )

    if (
        export_engine == 'netcdf4'
        and output_save_path is not None
        and output_save_path.startswith('s3://')
    ):
        return
    ec._to_file(
        convert_type=export_engine,
        save_path=output_save_path,
        overwrite=True,
        combine=combine_files,
        storage_options=output_storage_options,
    )

    _check_output_files(export_engine, ec.output_file, output_storage_options)


@pytest.mark.parametrize("model", ["azfp"])
@pytest.mark.parametrize(
    "input_path",
    [
        "./echopype/test_data/azfp/ooi/17032923.01A",
        "https://rawdata.oceanobservatories.org/files/CE01ISSM/R00007/instrmts/dcl37/ZPLSC_sn55075/ce01issm_zplsc_55075_recovered_2017-10-27/DATA/201703/17032923.01A",
    ],
)
@pytest.mark.parametrize(
    "xml_path",
    [
        "./echopype/test_data/azfp/ooi/17032922.XML",
        "https://rawdata.oceanobservatories.org/files/CE01ISSM/R00007/instrmts/dcl37/ZPLSC_sn55075/ce01issm_zplsc_55075_recovered_2017-10-27/DATA/201703/17032922.XML",
    ],
)
@pytest.mark.parametrize("export_engine", ["zarr", "netcdf4"])
@pytest.mark.parametrize(
    "output_save_path",
    [
        None,
        "./echopype/test_data/dump/",
        "./echopype/test_data/dump/tmp.zarr",
        "./echopype/test_data/dump/tmp.nc",
        "s3://ooi-raw-data/dump/",
        "s3://ooi-raw-data/dump/tmp.zarr",
        "s3://ooi-raw-data/dump/tmp.nc",
    ],
)
@pytest.mark.parametrize("combine_files", [False])
def test_convert_azfp(
    model,
    input_path,
    xml_path,
    export_engine,
    output_save_path,
    combine_files,
    minio_bucket,
    download_files,
):
    output_storage_options = {}
    ipath = input_path
    if isinstance(input_path, list):
        ipath = input_path[0]

    input_storage_options = {'anon': True} if ipath.startswith('s3://') else {}
    if output_save_path and output_save_path.startswith('s3://'):
        output_storage_options = dict(
            client_kwargs=dict(endpoint_url='http://localhost:9000/'),
            key='minioadmin',
            secret='minioadmin',
        )

    ec = Convert(
        file=input_path,
        xml_path=xml_path,
        model=model,
        storage_options=input_storage_options,
    )

    assert ec.xml_path == xml_path

    if (
        export_engine == 'netcdf4'
        and output_save_path is not None
        and output_save_path.startswith('s3://')
    ):
        return
    ec._to_file(
        convert_type=export_engine,
        save_path=output_save_path,
        overwrite=True,
        combine=combine_files,
        storage_options=output_storage_options,
    )

    _check_output_files(export_engine, ec.output_file, output_storage_options)


@pytest.mark.parametrize("model", ["EK80"])
@pytest.mark.parametrize(
    "input_path",
    [
        "./echopype/test_data/ek80/ncei-wcsd/D20170826-T205615.raw",
        "https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK80/D20170826-T205615.raw",
        "s3://ncei-wcsd-archive/data/raw/Bell_M._Shimada/SH1707/EK80/D20170826-T205615.raw",
    ],
)
@pytest.mark.parametrize("export_engine", ["zarr", "netcdf4"])
@pytest.mark.parametrize(
    "output_save_path",
    [
        None,
        "./echopype/test_data/dump/",
        "./echopype/test_data/dump/tmp.zarr",
        "./echopype/test_data/dump/tmp.nc",
        "s3://ooi-raw-data/dump/",
        "s3://ooi-raw-data/dump/tmp.zarr",
        "s3://ooi-raw-data/dump/tmp.nc",
    ],
)
@pytest.mark.parametrize("combine_files", [False])
def test_convert_ek80(
    model,
    input_path,
    export_engine,
    output_save_path,
    combine_files,
    minio_bucket,
    download_files,
):
    output_storage_options = {}
    ipath = input_path
    if isinstance(input_path, list):
        ipath = input_path[0]

    input_storage_options = {'anon': True} if ipath.startswith('s3://') else {}
    if output_save_path and output_save_path.startswith('s3://'):
        output_storage_options = dict(
            client_kwargs=dict(endpoint_url='http://localhost:9000/'),
            key='minioadmin',
            secret='minioadmin',
        )

    ec = Convert(
        file=input_path, model=model, storage_options=input_storage_options
    )

    if (
        export_engine == 'netcdf4'
        and output_save_path is not None
        and output_save_path.startswith('s3://')
    ):
        return
    ec._to_file(
        convert_type=export_engine,
        save_path=output_save_path,
        overwrite=True,
        combine=combine_files,
        storage_options=output_storage_options,
    )

    _check_output_files(export_engine, ec.output_file, output_storage_options)
