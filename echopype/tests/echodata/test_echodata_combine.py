from typing import Any, List, Dict
from textwrap import dedent
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import echopype
from echopype.utils.coding import DEFAULT_ENCODINGS
from echopype.qc import exist_reversed_time
from echopype.core import SONAR_MODELS

import tempfile
from dask.distributed import Client


@pytest.fixture
def ek60_test_data(test_path):
    files = [
        ("ncei-wcsd", "Summer2017-D20170620-T011027.raw"),
        ("ncei-wcsd", "Summer2017-D20170620-T014302.raw"),
        ("ncei-wcsd", "Summer2017-D20170620-T021537.raw"),
    ]
    return [test_path["EK60"].joinpath(*f) for f in files]


@pytest.fixture
def ek60_reversed_ping_time_test_data(test_path):
    files = [
        ("ncei-wcsd", "Summer2017-D20170719-T203615.raw"),
        ("ncei-wcsd", "Summer2017-D20170719-T205415.raw"),
        ("ncei-wcsd", "Summer2017-D20170719-T211347.raw"),
    ]
    return [test_path["EK60"].joinpath(*f) for f in files]


@pytest.fixture
def ek80_test_data(test_path):
    files = [
        ("echopype-test-D20211005-T000706.raw",),
        ("echopype-test-D20211005-T000737.raw",),
        ("echopype-test-D20211005-T000810.raw",),
        ("echopype-test-D20211005-T000843.raw",),
    ]
    return [test_path["EK80_NEW"].joinpath(*f) for f in files]


@pytest.fixture
def azfp_test_data(test_path):
    files = [
        ("ooi", "18100407.01A"),
        ("ooi", "18100408.01A"),
        ("ooi", "18100409.01A"),
    ]
    return [test_path["AZFP"].joinpath(*f) for f in files]


@pytest.fixture
def azfp_test_xml(test_path):
    return test_path["AZFP"].joinpath(*("ooi", "18092920.XML"))


@pytest.fixture(
    params=[
        {
        "sonar_model": "EK60",
        "xml_file": None,
        "files": "ek60_test_data",
    },
    #     {
    #     "sonar_model": "EK80",
    #     "xml_file": None,
    #     "files": "ek80_test_data",
    # },
        {
        "sonar_model": "AZFP",
        "xml_file": "azfp_test_xml",
        "files": "azfp_test_data",
    }
    ],
    ids=["ek60", "azfp"] #["ek60", "ek80", "azfp"]
)
def raw_datasets(request):
    files = request.param["files"]
    xml_file = request.param["xml_file"]
    if xml_file is not None:
        xml_file = request.getfixturevalue(xml_file)

    files = request.getfixturevalue(files)

    return (
        files,
        request.param['sonar_model'],
        xml_file,
        SONAR_MODELS[request.param['sonar_model']]["concat_dims"],
        SONAR_MODELS[request.param['sonar_model']]["concat_data_vars"],
    )


def test_combine_echodata(raw_datasets):
    (
        files,
        sonar_model,
        xml_file,
        concat_dims,
        concat_data_vars,
    ) = raw_datasets

    eds = [echopype.open_raw(file, sonar_model, xml_file) for file in files]

    append_dims = {"filenames", "time1", "time2", "time3", "ping_time"}

    # create temporary directory for zarr store
    temp_zarr_dir = tempfile.TemporaryDirectory()
    zarr_file_name = temp_zarr_dir.name + "/combined_echodatas.zarr"

    # create dask client
    client = Client()

    combined = echopype.combine_echodata(eds, zarr_file_name, client=client)

    # get all possible dimensions that should be dropped
    # these correspond to the attribute arrays created
    all_drop_dims = []
    for grp in combined.group_paths:
        # format group name appropriately
        ed_name = grp.replace("-", "_").replace("/", "_").lower()

        # create and append attribute array dimension
        all_drop_dims.append(ed_name + "_attr_key")

    # add dimension for Provenance group
    all_drop_dims.append("echodata_filename")

    for group_name in combined.group_paths:

        # get all Datasets to be combined
        combined_group: xr.Dataset = combined[group_name]
        eds_groups = [
            ed[group_name]
            for ed in eds
            if ed[group_name] is not None
        ]

        # all grp dimensions that are in all_drop_dims
        if combined_group is None:
            grp_drop_dims = []
            concat_dims = []
        else:
            grp_drop_dims = list(set(combined_group.dims).intersection(set(all_drop_dims)))
            concat_dims = list(set(combined_group.dims).intersection(append_dims))

        # concat all Datasets along each concat dimension
        diff_concats = []
        for dim in concat_dims:

            drop_dims = [c_dim for c_dim in concat_dims if c_dim != dim]

            diff_concats.append(xr.concat([ed_subset.drop_dims(drop_dims) for ed_subset in eds_groups], dim=dim,
                                coords="minimal", data_vars="minimal"))

        if len(diff_concats) < 1:
            test_ds = eds_groups[0]  # needed for groups that do not have append dims
        else:
            # create the full combined Dataset
            test_ds = xr.merge(diff_concats, compat="override")

            # correctly set filenames values for constructed combined Dataset
            if "filenames" in test_ds:
                test_ds.filenames.values[:] = np.arange(len(test_ds.filenames), dtype=int)

            # correctly modify Provenance attributes so we can do a direct compare
            if group_name == "Provenance":
                test_ds.attrs["reversed_ping_times"] = 0

                del test_ds.attrs["conversion_time"]
                del combined_group.attrs["conversion_time"]

        if (combined_group is not None) and (test_ds is not None):
            assert test_ds.identical(combined_group.drop_dims(grp_drop_dims))

    temp_zarr_dir.cleanup()

    # close client
    client.close()


def test_ping_time_reversal(ek60_reversed_ping_time_test_data):

    eds = [
        echopype.open_raw(file, "EK60")
        for file in ek60_reversed_ping_time_test_data
    ]

    # create temporary directory for zarr store
    temp_zarr_dir = tempfile.TemporaryDirectory()
    zarr_file_name = temp_zarr_dir.name + "/combined_echodatas.zarr"

    # create dask client
    client = Client()

    combined = echopype.combine_echodata(eds, zarr_file_name, client=client)

    for group_name, value in combined.group_map.items():
        if value['ep_group'] is None:
            combined_group: xr.Dataset = combined['Top-level']
        else:
            combined_group: xr.Dataset = combined[value['ep_group']]

        if combined_group is not None:
            if "ping_time" in combined_group and group_name != "provenance":
                assert not exist_reversed_time(combined_group, "ping_time")
            if "old_ping_time" in combined_group:
                assert exist_reversed_time(combined_group, "old_ping_time")
            if "time1" in combined_group and group_name not in (
                "provenance",
                "nmea",
            ):
                assert not exist_reversed_time(combined_group, "time1")
            if "old_time1" in combined_group:
                assert exist_reversed_time(combined_group, "old_time1")
            if "time2" in combined_group and group_name != "provenance":
                assert not exist_reversed_time(combined_group, "time2")
            if "old_time2" in combined_group:
                assert exist_reversed_time(combined_group, "old_time2")

    temp_zarr_dir.cleanup()

    # close client
    client.close()


def test_attr_storage(ek60_test_data):
    # check storage of attributes before combination in provenance group
    eds = [echopype.open_raw(file, "EK60") for file in ek60_test_data]

    # create temporary directory for zarr store
    temp_zarr_dir = tempfile.TemporaryDirectory()
    zarr_file_name = temp_zarr_dir.name + "/combined_echodatas.zarr"

    # create dask client
    client = Client()

    combined = echopype.combine_echodata(eds, zarr_file_name, client=client)

    for group, value in combined.group_map.items():
        if value['ep_group'] is None:
            group_path = 'Top-level'
        else:
            group_path = value['ep_group']
        if f"{group}_attrs" in combined["Provenance"]:
            group_attrs = combined["Provenance"][f"{group}_attrs"]
            for i, ed in enumerate(eds):
                for attr, value in ed[group_path].attrs.items():
                    assert str(
                        group_attrs.isel(echodata_filename=i)
                        .sel({f"{group}_attr_key": attr})
                        .values[()]
                    ) == str(value)

    # check selection by echodata_filename
    for file in ek60_test_data:
        assert Path(file).name in combined["Provenance"]["echodata_filename"]
    for group in combined.group_map:
        if f"{group}_attrs" in combined["Provenance"]:
            group_attrs = combined["Provenance"][f"{group}_attrs"]
            assert np.array_equal(
                group_attrs.sel(
                    echodata_filename=Path(ek60_test_data[0]).name
                ),
                group_attrs.isel(echodata_filename=0),
            )

    temp_zarr_dir.cleanup()

    # close client
    client.close()


def test_combined_encodings(ek60_test_data):
    eds = [echopype.open_raw(file, "EK60") for file in ek60_test_data]

    # create temporary directory for zarr store
    temp_zarr_dir = tempfile.TemporaryDirectory()
    zarr_file_name = temp_zarr_dir.name + "/combined_echodatas.zarr"

    # create dask client
    client = Client()

    combined = echopype.combine_echodata(eds, zarr_file_name, client=client)

    encodings_to_drop = {'chunks', 'preferred_chunks', 'compressor', 'filters'}

    group_checks = []
    for group, value in combined.group_map.items():
        if value['ep_group'] is None:
            ds = combined['Top-level']
        else:
            ds = combined[value['ep_group']]

        if ds is not None:
            for k, v in ds.variables.items():
                if k in DEFAULT_ENCODINGS:
                    encoding = ds[k].encoding

                    # remove any encoding relating to lazy loading
                    lazy_encodings = set(encoding.keys()).intersection(encodings_to_drop)
                    for encod_name in lazy_encodings:
                        del encoding[encod_name]

                    if encoding != DEFAULT_ENCODINGS[k]:
                        group_checks.append(
                            f"  {value['name']}::{k}"
                        )

    temp_zarr_dir.cleanup()

    # close client
    client.close()

    if len(group_checks) > 0:
        all_messages = ['Encoding mismatch found!'] + group_checks
        message_text = '\n'.join(all_messages)
        raise AssertionError(message_text)


def test_combined_echodata_repr(ek60_test_data):
    eds = [echopype.open_raw(file, "EK60") for file in ek60_test_data]

    # create temporary directory for zarr store
    temp_zarr_dir = tempfile.TemporaryDirectory()
    zarr_file_name = temp_zarr_dir.name + "/combined_echodatas.zarr"

    # create dask client
    client = Client()

    combined = echopype.combine_echodata(eds, zarr_file_name, client=client)

    expected_repr = dedent(
        f"""\
        <EchoData: standardized raw data from {zarr_file_name}>
        Top-level: contains metadata about the SONAR-netCDF4 file format.
        ├── Environment: contains information relevant to acoustic propagation through water.
        ├── Platform: contains information about the platform on which the sonar is installed.
        │   └── NMEA: contains information specific to the NMEA protocol.
        ├── Provenance: contains metadata about how the SONAR-netCDF4 version of the data were obtained.
        ├── Sonar: contains sonar system metadata and sonar beam groups.
        │   └── Beam_group1: contains backscatter data (either complex samples or uncalibrated power samples) and other beam or channel-specific data, including split-beam angle data when they exist.
        └── Vendor_specific: contains vendor-specific information about the sonar and the data."""
    )

    assert isinstance(repr(combined), str) is True

    actual = "\n".join(x.rstrip() for x in repr(combined).split("\n"))
    assert actual == expected_repr

    temp_zarr_dir.cleanup()

    # close client
    client.close()
