from typing import Any, List, Dict
from textwrap import dedent
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from xarray.core.merge import MergeError

import echopype
from echopype.utils.coding import DEFAULT_ENCODINGS
from echopype.qc import exist_reversed_time
from echopype.core import SONAR_MODELS


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
        ("ooi", "18100409.01A"),
        ("ooi", "18100408.01A"),
    ]
    return [test_path["AZFP"].joinpath(*f) for f in files]


@pytest.fixture
def azfp_test_xml(test_path):
    return test_path["AZFP"].joinpath(*("ooi", "18092920.XML"))


@pytest.fixture(
    params=[{
        "sonar_model": "EK60",
        "xml_file": None,
        "files": "ek60_test_data",
    }, {
        "sonar_model": "EK60",
        "xml_file": None,
        "files": "ek60_reversed_ping_time_test_data",
    }, {
        "sonar_model": "EK80",
        "xml_file": None,
        "files": "ek80_test_data",
    }, {
        "sonar_model": "AZFP",
        "xml_file": "azfp_test_xml",
        "files": "azfp_test_data",
    }],
    ids=["ek60", "ek60_reversed_ping_time", "ek80", "azfp"]
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
    combined = echopype.combine_echodata(eds, "overwrite_conflicts")  # type: ignore

    for group_name in combined.group_map:
        if group_name in ("top", "sonar", "provenance"):
            continue
        combined_group: xr.Dataset = getattr(combined, group_name)
        eds_groups = [
            getattr(ed, group_name)
            for ed in eds
            if getattr(ed, group_name) is not None
        ]

        def union_attrs(datasets: List[xr.Dataset]) -> Dict[str, Any]:
            """
            Merges attrs from a list of datasets.
            Prioritizes keys from later datasets.
            """

            total_attrs = {}
            for ds in datasets:
                total_attrs.update(ds.attrs)
            return total_attrs

        test_ds = xr.combine_nested(
            eds_groups,
            [concat_dims.get(group_name, concat_dims["default"])],
            data_vars=concat_data_vars.get(
                group_name, concat_data_vars["default"]
            ),
            coords="minimal",
            combine_attrs="drop",
        )
        test_ds.attrs.update(union_attrs(eds_groups))
        test_ds = test_ds.drop_dims(
            [
                "concat_dim",
                "old_ping_time",
                "ping_time",
                "old_time1",
                "time1",
            ],
            errors="ignore",
        ).drop_dims(
            [f"{group}_attrs" for group in combined.group_map], errors="ignore"
        )
        assert combined_group is None or test_ds.identical(
            combined_group.drop_dims(
                [
                    "old_ping_time",
                    "ping_time",
                    "old_time1",
                    "time1",
                ],
                errors="ignore",
            )
        )


def test_ping_time_reversal(ek60_reversed_ping_time_test_data):
    eds = [
        echopype.open_raw(file, "EK60")
        for file in ek60_reversed_ping_time_test_data
    ]
    combined = echopype.combine_echodata(eds, "overwrite_conflicts")  # type: ignore

    for group_name in combined.group_map:
        combined_group: xr.Dataset = getattr(combined, group_name)

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


def test_attr_storage(ek60_test_data):
    # check storage of attributes before combination in provenance group
    eds = [echopype.open_raw(file, "EK60") for file in ek60_test_data]
    combined = echopype.combine_echodata(eds, "overwrite_conflicts")  # type: ignore
    for group in combined.group_map:
        if f"{group}_attrs" in combined.provenance:
            group_attrs = combined.provenance[f"{group}_attrs"]
            for i, ed in enumerate(eds):
                for attr, value in getattr(ed, group).attrs.items():
                    assert str(
                        group_attrs.isel(echodata_filename=i)
                        .sel({f"{group}_attr_key": attr})
                        .data[()]
                    ) == str(value)

    # check selection by echodata_filename
    for file in ek60_test_data:
        assert Path(file).name in combined.provenance["echodata_filename"]
    for group in combined.group_map:
        if f"{group}_attrs" in combined.provenance:
            group_attrs = combined.provenance[f"{group}_attrs"]
            assert np.array_equal(
                group_attrs.sel(
                    echodata_filename=Path(ek60_test_data[0]).name
                ),
                group_attrs.isel(echodata_filename=0),
            )


def test_combine_attrs(ek60_test_data):
    # check parameter passed to combine_echodata that controls behavior of attribute combination
    eds = [echopype.open_raw(file, "EK60") for file in ek60_test_data]
    eds[0].beam.attrs.update({"foo": 1})
    eds[1].beam.attrs.update({"foo": 2})
    eds[2].beam.attrs.update({"foo": 3})

    combined = echopype.combine_echodata(eds, "override")  # type: ignore
    assert combined.beam.attrs["foo"] == 1

    combined = echopype.combine_echodata(eds, "drop")  # type: ignore
    assert "foo" not in combined.beam.attrs

    try:
        combined = echopype.combine_echodata(eds, "identical")  # type: ignore
    except MergeError:
        pass
    else:
        raise AssertionError

    try:
        combined = echopype.combine_echodata(eds, "no_conflicts")  # type: ignore
    except MergeError:
        pass
    else:
        raise AssertionError

    combined = echopype.combine_echodata(eds, "overwrite_conflicts")  # type: ignore
    assert combined.beam.attrs["foo"] == 3

    eds[0].beam.attrs.update({"foo": 1})
    eds[1].beam.attrs.update({"foo": 1})
    eds[2].beam.attrs.update({"foo": 1})

    combined = echopype.combine_echodata(eds, "identical")  # type: ignore
    assert combined.beam.attrs["foo"] == 1

    combined = echopype.combine_echodata(eds, "no_conflicts")  # type: ignore
    assert combined.beam.attrs["foo"] == 1


def test_combined_encodings(ek60_test_data):
    eds = [echopype.open_raw(file, "EK60") for file in ek60_test_data]
    combined = echopype.combine_echodata(eds, "overwrite_conflicts")  # type: ignore

    group_checks = []
    for group in combined.group_map:
        ds = getattr(combined, group)
        if ds is not None:
            for k, v in ds.variables.items():
                if k in DEFAULT_ENCODINGS:
                    encoding = ds[k].encoding
                    if encoding != DEFAULT_ENCODINGS[k]:
                        group_checks.append(
                            f"  {combined.group_map[group]['name']}::{k}"
                        )

    if len(group_checks) > 0:
        all_messages = ['Encoding mismatch found!'] + group_checks
        message_text = '\n'.join(all_messages)
        raise AssertionError(message_text)


def test_combined_echodata_repr(ek60_test_data):
    eds = [echopype.open_raw(file, "EK60") for file in ek60_test_data]
    combined = echopype.combine_echodata(eds, "overwrite_conflicts")  # type: ignore
    expected_repr = dedent(
        """\
        <EchoData: standardized raw data from Internal Memory>
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
