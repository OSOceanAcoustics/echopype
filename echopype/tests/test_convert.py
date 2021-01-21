import os
import glob
import fsspec
import shutil
import xarray as xr
import pytest
from pathlib import Path
from ..convert import Convert


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
    if isinstance(input_path, str):
        mult_path = glob.glob(input_path)
    else:
        mult_path = input_path
    fsmap = fsspec.get_mapper(mult_path[0])
    mult_dir = os.path.dirname(fsmap.root)
    tmp_mult = Convert(mult_path, model='EK60')

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
                assert tmp_mult.output_file == [os.path.abspath(output_save_path)]
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


def _check_file_group(data_file, engine, groups):
    for g in groups:
        ds = xr.open_dataset(data_file, engine=engine, group=g)

        assert isinstance(ds, xr.Dataset) is True


def _converted_group_checker(model, engine, out_file, multiple_files):
    groups = ['Environment', 'Platform', 'Provenance', 'Sonar']
    if model in ['EK60', 'EK80']:
        groups = groups + ['Beam', 'Vendor']

    if multiple_files:
        dirname = os.path.abspath(out_file)
        out_files = [os.path.join(dirname, f) for f in os.listdir(dirname)]
        for data_file in out_files:
            _check_file_group(data_file, engine, groups)
    else:
        _check_file_group(out_file, engine, groups)


def _file_export_checks(ec, model, export_engine, multiple_files):
    if export_engine == "netcdf4":
        out_file = f"./test_{model.lower()}.nc"
        if multiple_files:
            out_file = out_file.replace(".nc", "")
        ec.to_netcdf(save_path=out_file, overwrite=True)
    elif export_engine == "zarr":
        out_file = f"./test_{model.lower()}.zarr"
        if multiple_files:
            out_file = out_file.replace(".zarr", "")
        ec.to_zarr(save_path=out_file, overwrite=True)

    _converted_group_checker(
        model=model,
        engine=export_engine,
        out_file=out_file,
        multiple_files=multiple_files,
    )

    # Cleanup
    if os.path.isfile(out_file):
        os.unlink(out_file)
    else:
        shutil.rmtree(out_file)


@pytest.mark.skip()
@pytest.mark.parametrize("model", ["AZFP"])
@pytest.mark.parametrize(
    "file",
    [
        "https://rawdata.oceanobservatories.org/files/CE01ISSM/R00007/instrmts/dcl37/ZPLSC_sn55075/ce01issm_zplsc_55075_recovered_2017-10-27/DATA/201703/17032923.01A"
    ],
)
@pytest.mark.parametrize(
    "xml_path",
    [
        "https://rawdata.oceanobservatories.org/files/CE01ISSM/R00007/instrmts/dcl37/ZPLSC_sn55075/ce01issm_zplsc_55075_recovered_2017-10-27/DATA/201703/17032922.XML"
    ],
)
@pytest.mark.parametrize("storage_options", [{'anon': True}])
@pytest.mark.parametrize("export_engine", ["netcdf4", "zarr"])
def test_convert_azfp(model, file, xml_path, storage_options, export_engine):
    if isinstance(file, str):
        multiple_files = False
        if not file.startswith("s3://"):
            storage_options = {}
    else:
        multiple_files = True
        if not file[0].startswith("s3://"):
            storage_options = {}

    ec = Convert(
        file=file,
        model=model,
        xml_path=xml_path,
        storage_options=storage_options,
    )

    if multiple_files:
        assert sorted(ec.source_file) == sorted(file)
    else:
        assert ec.source_file[0] == file
    assert ec.xml_path == xml_path

    _file_export_checks(ec, model, export_engine, multiple_files)


@pytest.mark.skip()
@pytest.mark.parametrize("model", ["EK60"])
@pytest.mark.parametrize(
    "file",
    [
        "https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK60/Summer2017-D20170615-T190214.raw",
        "s3://ncei-wcsd-archive/data/raw/Bell_M._Shimada/SH1707/EK60/Summer2017-D20170615-T190214.raw",
        [
            'https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK60/Summer2017-D20170615-T190214.raw',
            'https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK60/Summer2017-D20170615-T190843.raw',
            'https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK60/Summer2017-D20170615-T212409.raw',
        ],
    ],
)
@pytest.mark.parametrize("storage_options", [{'anon': True}])
@pytest.mark.parametrize("export_engine", ["netcdf4", "zarr"])
def test_convert_ek60(model, file, storage_options, export_engine):
    if isinstance(file, str):
        multiple_files = False
        if not file.startswith("s3://"):
            storage_options = {}
    else:
        multiple_files = True
        if not file[0].startswith("s3://"):
            storage_options = {}

    ec = Convert(file=file, model=model, storage_options=storage_options)

    if multiple_files:
        assert sorted(ec.source_file) == sorted(file)
    else:
        assert ec.source_file[0] == file

    _file_export_checks(ec, model, export_engine, multiple_files)


@pytest.mark.skip()
@pytest.mark.parametrize("model", ["EK80"])
@pytest.mark.parametrize(
    "file",
    [
        "https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK80/D20170826-T205615.raw",
        "s3://ncei-wcsd-archive/data/raw/Bell_M._Shimada/SH1707/EK80/D20170826-T205615.raw",
    ],
)
@pytest.mark.parametrize("storage_options", [{'anon': True}])
@pytest.mark.parametrize("export_engine", ["netcdf4", "zarr"])
def test_convert_ek80(model, file, storage_options, export_engine):
    if isinstance(file, str):
        multiple_files = False
        if not file.startswith("s3://"):
            storage_options = {}
    else:
        multiple_files = True
        if not file[0].startswith("s3://"):
            storage_options = {}

    ec = Convert(file=file, model=model, storage_options=storage_options)

    if multiple_files:
        assert sorted(ec.source_file) == sorted(file)
    else:
        assert ec.source_file[0] == file

    _file_export_checks(ec, model, export_engine, multiple_files)
