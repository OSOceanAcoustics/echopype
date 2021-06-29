from typing import List, Optional, TYPE_CHECKING, Dict, Union
from pathlib import Path
import pytest
import xarray as xr
import echopype
from echopype.testing import TEST_DATA_FOLDER
from echopype.qc import exist_reversed_time
from echopype.core import SONAR_MODELS

if TYPE_CHECKING:
    from echopype.core import SonarModelsHint

azfp_ooi_folder = TEST_DATA_FOLDER / "azfp" / "ooi" / "combine-echodata"
azfp_test_data = [
    azfp_ooi_folder / "18100407.01A",
    azfp_ooi_folder / "18100409.01A",
    azfp_ooi_folder / "18100408.01A",
]
azfp_xml_file = azfp_ooi_folder / "18092920.XML"
ek60_ncei_wcsd_folder = TEST_DATA_FOLDER / "ek60" / "ncei-wcsd" / "combine-echodata"
ek60_test_data = [
    ek60_ncei_wcsd_folder / "Summer2017-D20170620-T011027.raw",
    ek60_ncei_wcsd_folder / "Summer2017-D20170620-T014302.raw",
    ek60_ncei_wcsd_folder / "Summer2017-D20170620-T021537.raw",
]
ek60_ooi_folder = TEST_DATA_FOLDER / "ek60" / "ooi" / "combine-echodata"
ek60_reversed_ping_time_test_data = [
    ek60_ooi_folder / "CE04OSPS-PC01B-05-ZPLSCB102_OOI-D20161106-T000000.raw",
    ek60_ooi_folder / "CE04OSPS-PC01B-05-ZPLSCB102_OOI-D20161107-T000000.raw",
]


@pytest.mark.parametrize(
    "files, sonar_model, xml_file, check_reversed_ping_time, concat_dims, concat_data_vars",
    [
        (
            azfp_test_data,
            "AZFP",
            azfp_xml_file,
            False,
            SONAR_MODELS["AZFP"]["concat_dims"],
            SONAR_MODELS["AZFP"]["concat_data_vars"],
        ),
        (
            ek60_test_data,
            "EK60",
            None,
            False,
            SONAR_MODELS["EK60"]["concat_dims"],
            SONAR_MODELS["EK60"]["concat_data_vars"],
        ),
        (
            ek60_reversed_ping_time_test_data,
            "EK60",
            None,
            True,
            SONAR_MODELS["EK60"]["concat_dims"],
            SONAR_MODELS["EK60"]["concat_data_vars"],
        ),
    ],
)
def test_combine_echodata(
    files: List[Path],
    sonar_model: "SonarModelsHint",
    xml_file: Optional[Path],
    check_reversed_ping_time: bool,
    concat_dims: Dict[str, Optional[Union[str, List[str]]]],
    concat_data_vars: Dict[str, str],
):
    eds = [echopype.open_raw(file, sonar_model, xml_file) for file in files]
    combined = echopype.combine_echodata(eds)  # type: ignore

    for group_name in combined.group_map:
        if group_name in ("top", "sonar", "provenance"):
            continue
        combined_group: xr.Dataset = getattr(combined, group_name)
        eds_groups = [
            getattr(ed, group_name) for ed in eds if getattr(ed, group_name) is not None
        ]

        def union_attrs(datasets: List[xr.Dataset]) -> List[xr.Dataset]:
            """
            Merges attrs from a list of datasets.
            Prioritizes keys from later datsets.
            """

            total_attrs = dict()
            for ds in datasets:
                total_attrs.update(ds.attrs)
            for ds in datasets:
                ds.attrs = total_attrs
            return datasets

        assert combined_group is None or xr.combine_nested(
            union_attrs(eds_groups),
            # foo,
            [concat_dims.get(group_name, concat_dims["default"])],
            data_vars=concat_data_vars.get(group_name, concat_data_vars["default"]),
            coords="minimal",
        ).drop_dims(
            ["concat_dim", "old_ping_time", "ping_time"], errors="ignore"
        ).identical(
            combined_group.drop_dims(["old_ping_time", "ping_time"], errors="ignore")
        )

        if (
            sonar_model == "EK60"
            and check_reversed_ping_time
            and combined_group is not None
            and "ping_time" in combined_group
        ):
            if "old_ping_time" in combined_group:
                assert exist_reversed_time(combined_group, "old_ping_time")
            assert not exist_reversed_time(combined_group, "ping_time")
