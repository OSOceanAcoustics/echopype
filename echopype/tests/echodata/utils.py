import os
import json
from pathlib import Path

import xarray as xr

from datatree import DataTree

import numpy as np

from echopype.convert.set_groups_base import SetGroupsBase
from echopype.echodata.echodata import EchoData
from echopype.echodata.convention import sonarnetcdf_1

class P2Z:
    _varattrs = sonarnetcdf_1.yaml_dict["variable_and_varattributes"]


class SetGroupsTest(SetGroupsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_beam(self) -> xr.Dataset:
        ds = xr.Dataset(
            attrs={"beam_mode": "vertical", "conversion_equation_t": "type_3"}
        )
        return ds

    def set_env(self) -> xr.Dataset:
        # TODO: add mock data
        ds = xr.Dataset()

        env_attr_dict = {
            "notes": "This is a mock env dataset, hence no data is found!"
        }
        ds = ds.assign_attrs(env_attr_dict)

        return ds

    def set_platform(self) -> xr.Dataset:
        # TODO: add mock data
        ds = xr.Dataset(
            attrs={
                "platform_code_ICES": 315,
                "platform_name": "My mock boat",
                "platform_type": "Research vessel",
            }
        )

        return ds

    def set_nmea(self) -> xr.Dataset:
        # TODO: add mock data
        ds = xr.Dataset(
            attrs={
                "description": "All Mock NMEA datagrams",
            }
        )

        return ds

    def set_sonar(self) -> xr.Dataset:
        # TODO: add mock data
        ds = xr.Dataset()

        # Assemble sonar group global attribute dictionary
        sonar_attr_dict = {
            "sonar_manufacturer": "Simrad",
            "sonar_model": self.sonar_model,
            # transducer (sonar) serial number is not stored in the EK60 raw data file,
            # so sonar_serial_number can't be populated from the raw datagrams
            "sonar_serial_number": "",
            "sonar_software_name": "",
            "sonar_software_version": "0.1.0",
            "sonar_type": "echosounder",
        }
        ds = ds.assign_attrs(sonar_attr_dict)

        return ds

    def set_vendor(self) -> xr.Dataset:
        # TODO: add mock data
        ds = xr.Dataset(attrs={"created_by": "Mock test"})
        return ds


def get_mock_echodata(
    sonar_model='TEST',
    file_chk='./test.raw',
    xml_chk=None,
):
    # Setup tree dictionary
    tree_dict = {}

    setgrouper = SetGroupsTest(
        parser_obj=None,
        input_file=file_chk,
        xml_path=xml_chk,
        output_path=None,
        sonar_model=sonar_model,
        params={"survey_name": "mock_survey"},
        parsed2zarr_obj=P2Z(),
    )
    tree_dict["/"] = setgrouper.set_toplevel(
        sonar_model, date_created=np.datetime64("1970-01-01")
    )
    tree_dict["Environment"] = setgrouper.set_env()
    tree_dict["Platform"] = setgrouper.set_platform()
    tree_dict["Platform/NMEA"] = setgrouper.set_nmea()
    tree_dict["Provenance"] = setgrouper.set_provenance()
    tree_dict["Sonar"] = None
    tree_dict["Sonar/Beam_group1"] = setgrouper.set_beam()
    tree_dict["Sonar"] = setgrouper.set_sonar()
    tree_dict["Vendor_specific"] = setgrouper.set_vendor()

    tree = DataTree.from_dict(tree_dict, name="root")
    echodata = EchoData(
        source_file=file_chk, xml_path=xml_chk, sonar_model=sonar_model
    )
    echodata._set_tree(tree)
    echodata._load_tree()
    return echodata


def check_consolidated(echodata: EchoData, zmeta_path: Path) -> None:
    """
    Checks for the presence of `.zgroup`
    for every group in echodata within the `.zmetadata`
    file.

    Parameters
    ----------
    echodata : EchoData
        The echodata object to be checked.
    zmeta_path : pathlib.Path
        The path to the .zmetadata for the zarr file.
    """
    # Check that every group is in
    # the zmetadata if consolidated
    expected_zgroups = [
        os.path.join(p, '.zgroup') if p != 'Top-level' else '.zgroup'
        for p in echodata.group_paths
    ]

    with open(zmeta_path) as f:
        meta_json = json.load(f)

    file_groups = [
        k
        for k in meta_json['metadata'].keys()
        if k.endswith('.zgroup')
    ]

    for g in expected_zgroups:
        assert g in file_groups, f"{g} not Found!"
