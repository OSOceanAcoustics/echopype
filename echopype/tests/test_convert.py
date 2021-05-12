"""test_convert.py

This module contain all the various tests for echopype conversion
from a raw data to standard compliant zarr or netcdf file(s).

**Note that in order to run this test, minio server is required for s3
output tests.**
"""


import os
import shutil
import glob
import fsspec
import xarray as xr
import pytest
from pathlib import Path
from echopype import open_raw
from echopype.convert.api import _validate_path
from echopype.testing import TEST_DATA_FOLDER
from echopype.convert.set_groups_base import DEFAULT_ENCODINGS


def _check_file_group(data_file, engine, groups):
    for g in groups:
        ds = xr.open_dataset(data_file, engine=engine, group=g)

        assert isinstance(ds, xr.Dataset) is True


def _check_output_files(engine, output_files, storage_options):
    groups = [
        "Provenance",
        "Environment",
        "Beam",
        "Sonar",
        "Vendor",
        "Platform",
    ]
    if isinstance(output_files, list):
        fs = fsspec.get_mapper(output_files[0], **storage_options).fs
        for f in output_files:
            if engine == "zarr":
                _check_file_group(fs.get_mapper(f), engine, groups)
                fs.delete(f, recursive=True)
            else:
                _check_file_group(f, engine, groups)
                fs.delete(f)
    else:
        fs = fsspec.get_mapper(output_files, **storage_options).fs
        if engine == "zarr":
            _check_file_group(fs.get_mapper(output_files), engine, groups)
            fs.delete(output_files, recursive=True)
        else:
            _check_file_group(output_files, engine, groups)
            fs.delete(output_files)


@pytest.fixture(scope="session")
def minio_bucket():
    common_storage_options = dict(
        client_kwargs=dict(endpoint_url="http://localhost:9000/"),
        key="minioadmin",
        secret="minioadmin",
    )
    bucket_name = "ooi-raw-data"
    fs = fsspec.filesystem(
        "s3",
        **common_storage_options,
    )
    test_data = "data"
    if not fs.exists(test_data):
        fs.mkdir(test_data)

    if not fs.exists(bucket_name):
        fs.mkdir(bucket_name)

    # Load test data into bucket
    test_data_path = Path(__file__).parent.parent.joinpath(Path("test_data"))
    for d in test_data_path.iterdir():
        source_path = f'echopype/test_data/{d.name}'
        fs.put(source_path, f'{test_data}/{d.name}', recursive=True)

    return common_storage_options


@pytest.mark.parametrize("model", ["EK60"])
@pytest.mark.parametrize("file_format", [".zarr"])
@pytest.mark.parametrize(
    "input_path",
    [
        "./echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw",
        "http://localhost:8080/data/ek60/ncei-wcsd/Summer2017-D20170615-T190214.raw",
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
    if output_save_path and output_save_path.startswith("s3://"):
        output_storage_options = dict(
            client_kwargs=dict(endpoint_url="http://localhost:9000/"),
            key="minioadmin",
            secret="minioadmin",
        )
    fsmap = fsspec.get_mapper(input_path)
    single_fname = os.path.splitext(os.path.basename(fsmap.root))[0]

    converted_raw_path = _validate_path(
        source_file=single_fname,
        file_format=file_format,
        output_storage_options=output_storage_options,
        save_path=output_save_path
    )
    # Used for cross-platform path comparisons
    output_path = Path(converted_raw_path)

    if output_save_path is not None:
        fsmap_tmp = fsspec.get_mapper(output_save_path, **output_storage_options)
        fs = fsmap_tmp.fs
        if not output_save_path.startswith("s3"):
            if output_save_path.endswith("/"):
                # if an output folder is given, below works with and without the slash at the end
                assert output_path == Path(os.path.join(fsmap_tmp.root, single_fname + ".zarr"))
            elif output_save_path.endswith(".zarr"):
                # if an output filename is given
                assert output_path == Path(fsmap_tmp.root)
            else:
                # force output file extension to the called type (here .zarr)
                assert output_path == Path(os.path.splitext(fsmap_tmp.root)[0] + ".zarr")
            shutil.rmtree(os.path.dirname(converted_raw_path))
        else:
            if output_save_path.endswith("/"):
                # if an output folder is given, below works with and without the slash at the end
                assert output_path == Path(os.path.join(output_save_path, single_fname + ".zarr"))
            elif output_save_path.endswith(".zarr"):
                # if an output filename is given
                assert output_path == Path(output_save_path)
            else:
                # force output file extension to the called type (here .zarr)
                assert output_path == Path(os.path.splitext(output_save_path)[0] + ".zarr")
            fs.delete(converted_raw_path)
    else:
        current_dir = Path.cwd()
        temp_dir = current_dir.joinpath(Path("temp_echopype_output"))
        assert output_path == Path(str(temp_dir.joinpath(Path(single_fname + ".zarr"))))
        shutil.rmtree(os.path.dirname(converted_raw_path))


@pytest.mark.parametrize("model", ["EK60"])
@pytest.mark.parametrize("file_format", [".zarr"])
@pytest.mark.parametrize(
    "input_path",
    [
        "./echopype/test_data/ek60/*.raw",
        [
            "http://localhost:8080/data/ek60/ncei-wcsd/Summer2017-D20170615-T190214.raw",
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
@pytest.mark.skip(reason='_validate_path only takes single files')
def test_validate_path_multiple_source(
    model, file_format, input_path, output_save_path, minio_bucket
):
    output_storage_options = {}
    if output_save_path and output_save_path.startswith("s3://"):
        output_storage_options = dict(
            client_kwargs=dict(endpoint_url="http://localhost:9000/"),
            key="minioadmin",
            secret="minioadmin",
        )

    if isinstance(input_path, str):
        mult_path = glob.glob(input_path)
    else:
        mult_path = input_path
    echodata_mult = open_raw(mult_path, sonar_model="EK60")
    echodata_mult._output_storage_options = output_storage_options

    echodata_mult._validate_path(file_format=file_format, save_path=output_save_path)

    if output_save_path is not None:
        fsmap_tmp = fsspec.get_mapper(output_save_path, **output_storage_options)
        fs = fsmap_tmp.fs
        if not output_save_path.startswith("s3"):
            if output_save_path.endswith("/"):
                # if an output folder is given, below works with and without the slash at the end
                assert echodata_mult.converted_raw_path == [
                    os.path.join(
                        fsmap_tmp.root,
                        os.path.splitext(os.path.basename(f))[0] + ".zarr",
                    )
                    for f in mult_path
                ]
            elif output_save_path.endswith(".zarr"):
                # if an output filename is given: only use the directory
                assert echodata_mult.converted_raw_path == [os.path.abspath(output_save_path)]
            elif output_save_path.endswith(".nc"):
                # force output file extension to the called type (here .zarr)
                assert echodata_mult.converted_raw_path == [
                    os.path.abspath(output_save_path.replace(".nc", ".zarr"))
                ]
            shutil.rmtree(os.path.dirname(echodata_mult.converted_raw_path[0]))
        else:
            if output_save_path.endswith("/"):
                # if an output folder is given, below works with and without the slash at the end
                assert echodata_mult.converted_raw_path == [
                    os.path.join(
                        output_save_path,
                        os.path.splitext(os.path.basename(f))[0] + ".zarr",
                    )
                    for f in mult_path
                ]
            elif output_save_path.endswith(".zarr"):
                # if an output filename is given: only use the directory
                assert echodata_mult.converted_raw_path == [output_save_path]
            elif output_save_path.endswith(".nc"):
                # force output file extension to the called type (here .zarr)
                assert echodata_mult.converted_raw_path == [output_save_path.replace(".nc", ".zarr")]
            fs.delete(echodata_mult.converted_raw_path[0])
    else:
        current_dir = Path.cwd()
        temp_dir = current_dir.joinpath(Path("temp_echopype_output"))
        assert echodata_mult.converted_raw_path == [
            str(temp_dir.joinpath(Path(os.path.splitext(os.path.basename(f))[0] + ".zarr")))
            for f in mult_path
        ]
        shutil.rmtree(os.path.dirname(echodata_mult.converted_raw_path[0]))


@pytest.mark.parametrize(
    "sonar_model, raw_file, xml_path",
    [
        (
            "azfp",
            TEST_DATA_FOLDER / "azfp/ooi/17032923.01A",
            TEST_DATA_FOLDER / "azfp/ooi/17032922.XML"
        ),
        (
            "ek60",
            TEST_DATA_FOLDER / "ek60/DY1801_EK60-D20180211-T164025.raw",
            None
        ),
        (
            "ek80",
            TEST_DATA_FOLDER / "ek80/ncei-wcsd/D20170826-T205615.raw",
            None
        )
    ]
)
def test_convert_time_encodings(sonar_model, raw_file, xml_path):
    ed = open_raw(
        sonar_model=sonar_model,
        raw_file=raw_file,
        xml_path=xml_path
    )
    ed.to_netcdf(overwrite=True)
    for group, details in ed._EchoData__group_map.items():
        if hasattr(ed, group):
            group_ds = getattr(ed, group)
            if isinstance(group_ds, xr.Dataset):
                for var, encoding in DEFAULT_ENCODINGS.items():
                    if var in group_ds:
                        da = group_ds[var]
                        assert da.encoding == encoding

                        # Combine encoding and attributes since this
                        # is what is shown when using decode_cf=False
                        # without dtype attribute
                        total_attrs = dict(**da.attrs, **da.encoding)
                        total_attrs.pop('dtype')

                        # Read converted file back in
                        file_da = xr.open_dataset(
                            ed.converted_raw_path,
                            group=details['ep_group'],
                            decode_cf=False
                        )[var]
                        assert file_da.dtype == encoding['dtype']

                        # Read converted file back in
                        decoded_da = xr.open_dataset(
                            ed.converted_raw_path,
                            group=details['ep_group'],
                        )[var]
                        assert da.equals(decoded_da) is True
    os.unlink(ed.converted_raw_path)


@pytest.mark.parametrize("model", ["EK60"])
@pytest.mark.parametrize(
    "input_path",
    [
        "./echopype/test_data/ek60/ncei-wcsd/Summer2017-D20170615-T190214.raw",
        "s3://data/ek60/ncei-wcsd/Summer2017-D20170615-T190214.raw",
        [
            "http://localhost:8080/data/ek60/ncei-wcsd/Summer2017-D20170615-T190214.raw",
            "http://localhost:8080/data/ek60/ncei-wcsd/Summer2017-D20170615-T190843.raw",
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
def test_convert_ek60(
    model,
    input_path,
    export_engine,
    output_save_path,
    minio_bucket,
):
    common_storage_options = minio_bucket
    output_storage_options = {}
    ipath = input_path
    if isinstance(input_path, list):
        ipath = input_path[0]

    input_storage_options = common_storage_options if ipath.startswith("s3://") else {}
    if output_save_path and output_save_path.startswith("s3://"):
        output_storage_options = common_storage_options

    # Only using one file
    ec = open_raw(raw_file=ipath, sonar_model=model, storage_options=input_storage_options)

    if (
        export_engine == "netcdf4"
        and output_save_path is not None
        and output_save_path.startswith("s3://")
    ):
        return

    if export_engine == "netcdf4":
        to_file = getattr(ec, "to_netcdf")
    elif export_engine == "zarr":
        to_file = getattr(ec, "to_zarr")
    else:
        return

    to_file(
        save_path=output_save_path,
        overwrite=True,
        output_storage_options=output_storage_options,
    )

    _check_output_files(export_engine, ec.converted_raw_path, output_storage_options)


@pytest.mark.parametrize("model", ["azfp"])
@pytest.mark.parametrize(
    "input_path",
    [
        "./echopype/test_data/azfp/ooi/17032923.01A",
        "http://localhost:8080/data/azfp/ooi/17032923.01A",
    ],
)
@pytest.mark.parametrize(
    "xml_path",
    [
        "./echopype/test_data/azfp/ooi/17032922.XML",
        "http://localhost:8080/data/azfp/ooi/17032922.XML",
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
):
    common_storage_options = minio_bucket
    output_storage_options = {}

    input_storage_options = common_storage_options if input_path.startswith("s3://") else {}
    if output_save_path and output_save_path.startswith("s3://"):
        output_storage_options = common_storage_options

    ec = open_raw(
        raw_file=input_path,
        xml_path=xml_path,
        sonar_model=model,
        storage_options=input_storage_options,
    )

    assert ec.xml_path == xml_path

    if (
        export_engine == "netcdf4"
        and output_save_path is not None
        and output_save_path.startswith("s3://")
    ):
        return

    if export_engine == "netcdf4":
        to_file = getattr(ec, "to_netcdf")
    elif export_engine == "zarr":
        to_file = getattr(ec, "to_zarr")
    else:
        return

    to_file(
        save_path=output_save_path,
        overwrite=True,
        output_storage_options=output_storage_options,
    )

    _check_output_files(export_engine, ec.converted_raw_path, output_storage_options)


@pytest.mark.parametrize("model", ["EK80"])
@pytest.mark.parametrize(
    "input_path",
    [
        "./echopype/test_data/ek80/ncei-wcsd/D20170826-T205615.raw",
        "http://localhost:8080/data/ek80/ncei-wcsd/D20170826-T205615.raw",
        "s3://data/ek80/ncei-wcsd/D20170826-T205615.raw",
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
):
    common_storage_options = minio_bucket
    output_storage_options = {}

    input_storage_options = common_storage_options if input_path.startswith("s3://") else {}
    if output_save_path and output_save_path.startswith("s3://"):
        output_storage_options = common_storage_options

    ec = open_raw(raw_file=input_path, sonar_model=model, storage_options=input_storage_options)

    if (
        export_engine == "netcdf4"
        and output_save_path is not None
        and output_save_path.startswith("s3://")
    ):
        return

    if export_engine == "netcdf4":
        to_file = getattr(ec, "to_netcdf")
    elif export_engine == "zarr":
        to_file = getattr(ec, "to_zarr")
    else:
        return

    to_file(
        save_path=output_save_path,
        overwrite=True,
        combine=combine_files,
        output_storage_options=output_storage_options,
    )

    _check_output_files(export_engine, ec.converted_raw_path, output_storage_options)
