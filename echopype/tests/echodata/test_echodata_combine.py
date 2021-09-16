from typing import Any, List, Optional, TYPE_CHECKING, Dict, Union
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from xarray.core.merge import MergeError

import echopype
from echopype.testing import TEST_DATA_FOLDER
from echopype.qc import exist_reversed_time
from echopype.core import SONAR_MODELS

if TYPE_CHECKING:
    from echopype.core import SonarModelsHint

azfp_ooi_folder = TEST_DATA_FOLDER / "azfp" / "ooi"
azfp_test_data = [
    azfp_ooi_folder / "18100407.01A",
    azfp_ooi_folder / "18100409.01A",
    azfp_ooi_folder / "18100408.01A",
]
azfp_xml_file = azfp_ooi_folder / "18092920.XML"
ek60_ncei_wcsd_folder = TEST_DATA_FOLDER / "ek60" / "ncei-wcsd"
ek60_test_data = [
    ek60_ncei_wcsd_folder / "Summer2017-D20170620-T011027.raw",
    ek60_ncei_wcsd_folder / "Summer2017-D20170620-T014302.raw",
    ek60_ncei_wcsd_folder / "Summer2017-D20170620-T021537.raw",
]
ek60_reversed_ping_time_test_data = [
    ek60_ncei_wcsd_folder / "Summer2017-D20170719-T203615.raw",
    ek60_ncei_wcsd_folder / "Summer2017-D20170719-T205415.raw",
    ek60_ncei_wcsd_folder / "Summer2017-D20170719-T211347.raw",
]


@pytest.mark.parametrize(
    "files, sonar_model, xml_file, concat_dims, concat_data_vars",
    [
        (
            azfp_test_data,
            "AZFP",
            azfp_xml_file,
            SONAR_MODELS["AZFP"]["concat_dims"],
            SONAR_MODELS["AZFP"]["concat_data_vars"],
        ),
        (
            ek60_test_data,
            "EK60",
            None,
            SONAR_MODELS["EK60"]["concat_dims"],
            SONAR_MODELS["EK60"]["concat_data_vars"],
        ),
        (
            ek60_reversed_ping_time_test_data,
            "EK60",
            None,
            SONAR_MODELS["EK60"]["concat_dims"],
            SONAR_MODELS["EK60"]["concat_data_vars"],
        ),
    ],
)
def test_combine_echodata(
    files: List[Path],
    sonar_model: "SonarModelsHint",
    xml_file: Optional[Path],
    concat_dims: Dict[str, Optional[Union[str, List[str]]]],
    concat_data_vars: Dict[str, str],
):
    eds = [echopype.open_raw(file, sonar_model, xml_file) for file in files]
    combined = echopype.combine_echodata(eds, "overwrite_conflicts")  # type: ignore

    for group_name in combined.group_map:
        if group_name in ("top", "sonar", "provenance"):
            continue
        combined_group: xr.Dataset = getattr(combined, group_name)
        eds_groups = [
            getattr(ed, group_name) for ed in eds if getattr(ed, group_name) is not None
        ]

        def union_attrs(datasets: List[xr.Dataset]) -> Dict[str, Any]:
            """
            Merges attrs from a list of datasets.
            Prioritizes keys from later datsets.
            """

            total_attrs = dict()
            for ds in datasets:
                total_attrs.update(ds.attrs)
            return total_attrs

        test_ds = xr.combine_nested(
            eds_groups,
            [concat_dims.get(group_name, concat_dims["default"])],
            data_vars=concat_data_vars.get(group_name, concat_data_vars["default"]),
            coords="minimal",
            combine_attrs="drop",
        )
        test_ds.attrs.update(union_attrs(eds_groups))
        test_ds = test_ds.drop_dims(
            [
                "concat_dim",
                "old_ping_time",
                "ping_time",
                "old_location_time",
                "location_time",
            ],
            errors="ignore",
        ).drop_dims([f"{group}_attrs" for group in combined.group_map], errors="ignore")
        assert combined_group is None or test_ds.identical(
            combined_group.drop_dims(
                ["old_ping_time", "ping_time", "old_location_time", "location_time"],
                errors="ignore",
            )
        )


def test_ping_time_reversal():
    eds = [
        echopype.open_raw(file, "EK60") for file in ek60_reversed_ping_time_test_data
    ]
    combined = echopype.combine_echodata(eds, "overwrite_conflicts")  # type: ignore

    for group_name in combined.group_map:
        combined_group: xr.Dataset = getattr(combined, group_name)

        if combined_group is not None:
            if "ping_time" in combined_group and group_name != "provenance":
                assert not exist_reversed_time(combined_group, "ping_time")
            if "old_ping_time" in combined_group:
                assert exist_reversed_time(combined_group, "old_ping_time")
            if "location_time" in combined_group and group_name not in ("provenance", "nmea"):
                assert not exist_reversed_time(combined_group, "location_time")
            if "old_location_time" in combined_group:
                assert exist_reversed_time(combined_group, "old_location_time")
            if "mru_time" in combined_group and group_name != "provenance":
                assert not exist_reversed_time(combined_group, "mru_time")
            if "old_mru_time" in combined_group:
                assert exist_reversed_time(combined_group, "old_mru_time")


def test_attr_storage():
    # check storage of attributes before combination in provenance group
    eds = [
        echopype.open_raw(file, "EK60") for file in ek60_test_data
    ]
    combined = echopype.combine_echodata(eds, "overwrite_conflicts")  # type: ignore
    for group in combined.group_map:
        if f"{group}_attrs" in combined.provenance:
            group_attrs = combined.provenance[f"{group}_attrs"]
            for i, ed in enumerate(eds):
                for attr, value in getattr(ed, group).attrs.items():
                    assert str(group_attrs.isel(echodata_filename=i).sel({f"{group}_attr_key": attr}).data[()]) == str(value)

    # check selection by echodata_filename
    for file in ek60_test_data:
        assert Path(file).name in combined.provenance["echodata_filename"]
    for group in combined.group_map:
        if f"{group}_attrs" in combined.provenance:
            group_attrs = combined.provenance[f"{group}_attrs"]
            assert np.array_equal(group_attrs.sel(echodata_filename=Path(ek60_test_data[0]).name), group_attrs.isel(echodata_filename=0))


def test_combine_attrs():
    # check parameter passed to combine_echodata that controls behavior of attribute combination
    eds = [
        echopype.open_raw(file, "EK60") for file in ek60_test_data
    ]
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
