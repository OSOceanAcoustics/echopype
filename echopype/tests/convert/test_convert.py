"""test_convert.py

This module contain all the various tests for echopype conversion
from a raw data to standard compliant zarr or netcdf file(s).

**Note that in order to run this test, minio server is required for s3
output tests.**
"""


import os
import fsspec
import xarray as xr
import pytest
from echopype import open_raw
from echopype.utils.coding import DEFAULT_ENCODINGS


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


def _create_path_str(test_folder, paths):
    return str(test_folder.joinpath(*paths).absolute())


@pytest.fixture(
    params=[
        None,
        "./echopype/test_data/dump/",
        "./echopype/test_data/dump/tmp.zarr",
        "./echopype/test_data/dump/tmp.nc",
        "s3://ooi-raw-data/dump/",
        "s3://ooi-raw-data/dump/tmp.zarr",
        "s3://ooi-raw-data/dump/tmp.nc",
    ],
    ids=[
        "None",
        "folder_string",
        "zarr_file_string",
        "netcdf_file_string",
        "s3_folder_string",
        "s3_zarr_file_string",
        "s3_netcdf_file_string",
    ],
)
def output_save_path(request):
    return request.param


@pytest.fixture(params=["zarr", "netcdf4"])
def export_engine(request):
    return request.param


@pytest.fixture(
    params=[
        ("ncei-wcsd", "Summer2017-D20170615-T190214.raw"),
        "s3://data/ek60/ncei-wcsd/Summer2017-D20170615-T190214.raw",
        [
            "http://localhost:8080/data/ek60/ncei-wcsd/Summer2017-D20170615-T190214.raw",
            "http://localhost:8080/data/ek60/ncei-wcsd/Summer2017-D20170615-T190843.raw",
        ],
    ],
    ids=["file_path_string", "s3_file_string", "multiple_http_file_string"],
)
def ek60_input_paths(request, test_path):
    if isinstance(request.param, tuple):
        return _create_path_str(test_path["EK60"], request.param)
    return request.param


@pytest.fixture(
    params=[
        ("D20151202-T020259.raw",),
        "s3://data/es70/D20151202-T020259.raw",
        [
            "http://localhost:8080/data/es70/D20151202-T020259.raw",
        ],
    ],
    ids=["file_path_string", "s3_file_string", "multiple_http_file_string"],
)
def es70_input_paths(request, test_path):
    if isinstance(request.param, tuple):
        return _create_path_str(test_path["ES70"], request.param)
    return request.param


@pytest.fixture(
    params=[
        ("WBT-D20210620-T012250.raw",),
        ("WBT-but-internally-marked-as-EK80-D20210710-T204029.raw",),
        "s3://data/es80/WBT-D20210620-T012250.raw",
        [
            "http://localhost:8080/data/es80/WBT-D20210620-T012250.raw",
        ],
    ],
    ids=[
        "file_path_string_WBT",
        "file_path_string_WBT_EK80",
        "s3_file_string",
        "multiple_http_file_string",
    ],
)
def es80_input_paths(request, test_path):
    if isinstance(request.param, tuple):
        return _create_path_str(test_path["ES80"], request.param)
    return request.param


@pytest.fixture(
    params=[
        ("ea640_test.raw",),
        "s3://data/ea640/ea640_test.raw",
        [
            "http://localhost:8080/data/ea640/ea640_test.raw",
        ],
    ],
    ids=[
        "file_path_string",
        "s3_file_string",
        "multiple_http_file_string",
    ],
)
def ea640_input_paths(request, test_path):
    if isinstance(request.param, tuple):
        return _create_path_str(test_path["EA640"], request.param)
    return request.param


@pytest.fixture(
    params=[
        ("ooi", "17032923.01A"),
        "http://localhost:8080/data/azfp/ooi/17032923.01A",
    ],
    ids=["file_path_string", "http_file_string"],
)
def azfp_input_paths(request, test_path):
    if isinstance(request.param, tuple):
        return _create_path_str(test_path["AZFP"], request.param)
    return request.param


@pytest.fixture(
    params=[
        ("ooi", "17032922.XML"),
        "http://localhost:8080/data/azfp/ooi/17032922.XML",
    ],
    ids=["xml_file_path_string", "xml_http_file_string"],
)
def azfp_xml_paths(request, test_path):
    if isinstance(request.param, tuple):
        return _create_path_str(test_path["AZFP"], request.param)
    return request.param


@pytest.fixture(
    params=[
        ("echopype-test-D20211005-T001135.raw",),
        "http://localhost:8080/data/ek80_new/echopype-test-D20211005-T001135.raw",
        "s3://data/ek80_new/echopype-test-D20211005-T001135.raw",
    ],
    ids=["file_path_string", "http_file_string", "s3_file_string"],
)
def ek80_input_paths(request, test_path):
    if isinstance(request.param, tuple):
        return _create_path_str(test_path["EK80_NEW"], request.param)
    return request.param


@pytest.mark.parametrize(
    "sonar_model, raw_file, xml_path",
    [
        ("azfp", ("ooi", "17032923.01A"), ("ooi", "17032922.XML")),
        (
            "ek60",
            ("DY1801_EK60-D20180211-T164025.raw",),
            None,
        ),
        (
            "es70",
            ("D20151202-T020259.raw",),
            None,
        ),
        (
            "es80",
            ("WBT-D20210620-T012250.raw",),
            None,
        ),
        (
            "ea640",
            ("ea640_test.raw",),
            None,
        ),
        (
            "ek80",
            ("echopype-test-D20211004-T235757.raw",),
            None,
        ),
        (
            "ad2cp",
            ("raw", "076", "rawtest.076.00000.ad2cp"),
            None,
        ),
    ],
    ids=["azfp", "ek60", "es70", "es80", "ea640", "ek80", "ad2cp"],
)
def test_convert_time_encodings(sonar_model, raw_file, xml_path, test_path):
    path_model = sonar_model.upper()
    if path_model == "EK80":
        path_model = path_model + "_NEW"

    raw_file = str(test_path[path_model].joinpath(*raw_file).absolute())
    if xml_path is not None:
        xml_path = str(test_path[path_model].joinpath(*xml_path).absolute())

    ed = open_raw(
        sonar_model=sonar_model, raw_file=raw_file, xml_path=xml_path
    )
    ed.to_netcdf(overwrite=True)
    for group, details in ed.group_map.items():
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
                            decode_cf=False,
                        )[var]
                        assert file_da.dtype == encoding['dtype']

                        # Read converted file back in
                        decoded_da = xr.open_dataset(
                            ed.converted_raw_path,
                            group=details['ep_group'],
                        )[var]
                        assert da.equals(decoded_da) is True
    os.unlink(ed.converted_raw_path)


def test_convert_ek60(
    ek60_input_paths,
    export_engine,
    output_save_path,
    minio_bucket,
    model="EK60",
):
    common_storage_options = minio_bucket
    output_storage_options = {}
    ipath = ek60_input_paths
    if isinstance(ek60_input_paths, list):
        ipath = ek60_input_paths[0]

    input_storage_options = (
        common_storage_options if ipath.startswith("s3://") else {}
    )
    if output_save_path and output_save_path.startswith("s3://"):
        output_storage_options = common_storage_options

    # Only using one file
    ec = open_raw(
        raw_file=ipath,
        sonar_model=model,
        storage_options=input_storage_options,
    )

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
    try:
        to_file(
            save_path=output_save_path,
            overwrite=True,
            output_storage_options=output_storage_options,
        )

        _check_output_files(
            export_engine, ec.converted_raw_path, output_storage_options
        )
    except Exception as e:
        if export_engine == 'netcdf4' and output_save_path.startswith("s3://"):
            assert isinstance(e, ValueError) is True
            assert str(e) == 'Only local netcdf4 is supported.'


def test_convert_azfp(
    azfp_input_paths,
    azfp_xml_paths,
    export_engine,
    output_save_path,
    minio_bucket,
    model="AZFP",
):
    common_storage_options = minio_bucket
    output_storage_options = {}

    input_storage_options = (
        common_storage_options if azfp_input_paths.startswith("s3://") else {}
    )
    if output_save_path and output_save_path.startswith("s3://"):
        output_storage_options = common_storage_options

    ec = open_raw(
        raw_file=azfp_input_paths,
        xml_path=azfp_xml_paths,
        sonar_model=model,
        storage_options=input_storage_options,
    )

    assert ec.xml_path == azfp_xml_paths

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
    try:
        to_file(
            save_path=output_save_path,
            overwrite=True,
            output_storage_options=output_storage_options,
        )

        _check_output_files(
            export_engine, ec.converted_raw_path, output_storage_options
        )
    except Exception as e:
        if export_engine == 'netcdf4' and output_save_path.startswith("s3://"):
            assert isinstance(e, ValueError) is True
            assert str(e) == 'Only local netcdf4 is supported.'


def test_convert_ek80(
    ek80_input_paths,
    export_engine,
    output_save_path,
    minio_bucket,
    model="EK80",
    combine_files=False,
):
    common_storage_options = minio_bucket
    output_storage_options = {}

    input_storage_options = (
        common_storage_options if ek80_input_paths.startswith("s3://") else {}
    )
    if output_save_path and output_save_path.startswith("s3://"):
        output_storage_options = common_storage_options

    ec = open_raw(
        raw_file=ek80_input_paths,
        sonar_model=model,
        storage_options=input_storage_options,
    )

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

    try:
        to_file(
            save_path=output_save_path,
            overwrite=True,
            combine=combine_files,
            output_storage_options=output_storage_options,
        )

        _check_output_files(
            export_engine, ec.converted_raw_path, output_storage_options
        )
    except Exception as e:
        if export_engine == 'netcdf4' and output_save_path.startswith("s3://"):
            assert isinstance(e, ValueError) is True
            assert str(e) == 'Only local netcdf4 is supported.'


def test_convert_es70(
    es70_input_paths,
    export_engine,
    output_save_path,
    minio_bucket,
    model="ES70",
):
    common_storage_options = minio_bucket
    output_storage_options = {}
    ipath = es70_input_paths
    if isinstance(es70_input_paths, list):
        ipath = es70_input_paths[0]

    input_storage_options = (
        common_storage_options if ipath.startswith("s3://") else {}
    )
    if output_save_path and output_save_path.startswith("s3://"):
        output_storage_options = common_storage_options

    # Only using one file
    ec = open_raw(
        raw_file=ipath,
        sonar_model=model,
        storage_options=input_storage_options,
    )

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
    try:
        to_file(
            save_path=output_save_path,
            overwrite=True,
            output_storage_options=output_storage_options,
        )

        _check_output_files(
            export_engine, ec.converted_raw_path, output_storage_options
        )
    except Exception as e:
        if export_engine == 'netcdf4' and output_save_path.startswith("s3://"):
            assert isinstance(e, ValueError) is True
            assert str(e) == 'Only local netcdf4 is supported.'


def test_convert_es80(
    es80_input_paths,
    export_engine,
    output_save_path,
    minio_bucket,
    model="ES80",
):
    common_storage_options = minio_bucket
    output_storage_options = {}
    ipath = es80_input_paths
    if isinstance(es80_input_paths, list):
        ipath = es80_input_paths[0]

    input_storage_options = (
        common_storage_options if ipath.startswith("s3://") else {}
    )
    if output_save_path and output_save_path.startswith("s3://"):
        output_storage_options = common_storage_options

    # Only using one file
    ec = open_raw(
        raw_file=ipath,
        sonar_model=model,
        storage_options=input_storage_options,
    )

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
    try:
        to_file(
            save_path=output_save_path,
            overwrite=True,
            output_storage_options=output_storage_options,
        )

        _check_output_files(
            export_engine, ec.converted_raw_path, output_storage_options
        )
    except Exception as e:
        if export_engine == 'netcdf4' and output_save_path.startswith("s3://"):
            assert isinstance(e, ValueError) is True
            assert str(e) == 'Only local netcdf4 is supported.'


def test_convert_ea640(
    ea640_input_paths,
    export_engine,
    output_save_path,
    minio_bucket,
    model="EA640",
):
    common_storage_options = minio_bucket
    output_storage_options = {}
    ipath = ea640_input_paths
    if isinstance(ea640_input_paths, list):
        ipath = ea640_input_paths[0]

    input_storage_options = (
        common_storage_options if ipath.startswith("s3://") else {}
    )
    if output_save_path and output_save_path.startswith("s3://"):
        output_storage_options = common_storage_options

    # Only using one file
    ec = open_raw(
        raw_file=ipath,
        sonar_model=model,
        storage_options=input_storage_options,
    )

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
    try:
        to_file(
            save_path=output_save_path,
            overwrite=True,
            output_storage_options=output_storage_options,
        )

        _check_output_files(
            export_engine, ec.converted_raw_path, output_storage_options
        )
    except Exception as e:
        if export_engine == 'netcdf4' and output_save_path.startswith("s3://"):
            assert isinstance(e, ValueError) is True
            assert str(e) == 'Only local netcdf4 is supported.'
