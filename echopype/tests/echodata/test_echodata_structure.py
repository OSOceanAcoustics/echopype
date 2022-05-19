from typing import Any, Dict, Optional
from datatree import open_datatree
import xarray as xr
import pytest

# TODO: change the below imports to absolute imports
from ...echodata.echodata import EchoData, XARRAY_ENGINE_MAP
from ...echodata.api import open_converted


def _tree_from_file(converted_raw_path: str,
                    ed_storage_options: Optional[Dict[str, Any]] = {},
                    open_kwargs: Dict[str, Any] = {}):
    """
    Checks that converted_raw_path exists, sanitizes the path,
    obtains the path's suffix, and lastly opens the file
    as a datatree.

    Parameters
    ----------
    converted_raw_path : str
        path to converted data file
    ed_storage_options : dict
        options for cloud storage used by EchoData
    open_kwargs : dict
        optional keyword arguments to be passed
        into xr.open_dataset

    Returns
    -------
    A Datatree object representing the converted data file.
    """

    # the purpose of this class is so I can use
    # functions in EchoData as if they were static
    # TODO: There is a better way to do this if
    #  we change functions in EchoData to static methods
    class temp_class(object):
        storage_options = ed_storage_options

    EchoData._check_path(temp_class, converted_raw_path)
    converted_raw_path = EchoData._sanitize_path(temp_class,
                                                 converted_raw_path)
    suffix = EchoData._check_suffix(temp_class,
                                    converted_raw_path)

    tree = open_datatree(
        converted_raw_path,
        engine=XARRAY_ENGINE_MAP[suffix],
        **open_kwargs,
    )

    return tree


def _check_and_drop_var(ed, tree, grp_path, var):

    ed_var = ed[grp_path][var]
    tree_var = tree[grp_path].ds[var]

    # make sure that the dimensions and attributes
    # are the same for the variable
    print(f"--> {ed_var.dims == tree_var.dims}")
    print(f"--> {ed_var.attrs == tree_var.attrs}")

    # make sure that the data types are correct too
    print(f"--> {isinstance(ed_var.values, type(tree_var.values))}")

    # drop variables so we can check that datasets are identical
    ed[grp_path] = ed[grp_path].drop(var)
    tree[grp_path].ds = tree[grp_path].ds.drop(var)


def compare_ed_against_tree(ed, tree):

    for grp_path in ed.group_paths:

        print(f"grp_path = {grp_path}")
        if grp_path == "Top-level":
            print(f"--> {tree.ds.identical(ed[grp_path])}")
        else:
            print(f"--> {tree[grp_path].ds.identical(ed[grp_path])}")


def test_v05x_v06x_conversion_structure():
    """
    Tests that version 0.5.x echopype files
    have been correctly converted to the
    0.6.x structure.
    """

    ek60_path = "./"
    ek80_path = "./"
    azfp_path = "./"

    converted_raw_paths_v06x = [ek60_path + "ek60-Summer2017-D20170615-T190214-ep-v06x.nc",
                                ek80_path + "ek80-Summer2018--D20180905-T033113-ep-v06x.nc",
                                azfp_path + "azfp-17082117_01A_17041823_XML-ep-v06x.nc"]

    converted_raw_paths_v05x = [ek60_path + "ek60-Summer2017-D20170615-T190214-ep-v05x.nc",
                                ek80_path + "ek80-Summer2018--D20180905-T033113-ep-v05x.nc",
                                azfp_path + "azfp-17082117_01A_17041823_XML-ep-v05x.nc"]

    for path_v05x, path_v06x in zip(converted_raw_paths_v05x, converted_raw_paths_v06x):

        ed_v05x = open_converted(path_v05x)
        tree_v06x = _tree_from_file(converted_raw_path=path_v06x)

        # TODO: drop conversion_software_version and conversion_time
        #  attribute in Provenance group, make sure name exists and val is a string

        print(f"sensor = {ed_v05x['Top-level'].attrs['keywords']}")

        # ignore direct comparison of the variables Sonar.sonar_serial_number,
        # Platform.drop_keel_offset_is_manual, and Platform.water_level_draft_is_manual
        # for EK80, this data is not present in v0.5.x
        if ed_v05x["Top-level"].attrs["keywords"] == "EK80":

            # dictionary of variables to drop where the group path is the
            # key and the variables are the value
            vars_to_drop = {"Sonar": ["sonar_serial_number"],
                            "Platform": ["drop_keel_offset_is_manual",
                                         "water_level_draft_is_manual"],
                            "Environment": ["sound_velocity_profile",
                                            "sound_velocity_profile_depth",
                                            "sound_velocity_source",
                                            "transducer_name",
                                            "transducer_sound_speed"]}

            # check and drop variables that cannot be directly compared
            # because their values are not the same
            for key, val in vars_to_drop.items():
                for var in val:
                    _check_and_drop_var(ed_v05x, tree_v06x, key, var)
                    print(" ")

            # sort the beam groups for EK80 according to channel (necessary for comparison)
            ed_v05x['Sonar/Beam_group1'] = ed_v05x['Sonar/Beam_group1'].sortby("channel")
            ed_v05x['Sonar/Beam_group2'] = ed_v05x['Sonar/Beam_group2'].sortby("channel")

            # sort the Platform group by channel for EK80 (necessary for comparison)
            tree_v06x['Platform'].ds = tree_v06x['Platform'].ds.sortby('channel')
            ed_v05x['Platform'] = ed_v05x['Platform'].sortby('channel')

        compare_ed_against_tree(ed_v05x, tree_v06x)

        if ed_v05x["Top-level"].attrs["keywords"] == "EK80":

            # sort the channels in tree_v06x and ed_v05x since they are not consistent
            tree_v06x['Platform'].ds = tree_v06x['Platform'].ds.sortby('channel')
            ed_v05x['Platform'] = ed_v05x['Platform'].sortby('channel')

            for vars in ed_v05x["Platform"].variables:

                print(f"variable = {str(vars)}")
                print(f"identical = {ed_v05x['Platform'][str(vars)].identical(tree_v06x['Platform'][str(vars)])}")

        print("")

    # TODO: do a comparison of a combined file, one of the groups creates an attribute key


def test_echodata_structure():
    """
    Makes sure that all raw files opened
    create the expected EchoData structure.
    """

    # compare_ed_against_tree(ed_v05x, tree_v06x)

    pytest.xfail("Full testing of the EchoData Structure has not been implemented yet.")