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

        print(ed_v05x["Top-level"].attrs["keywords"])

        # ignore direct comparison of the variable Sonar.sonar_serial_number for
        # EK80, this data is not present in v0.5.x
        if ed_v05x["Top-level"].attrs["keywords"] == "EK80":

            # make sure that at least the coordinates are identical
            v05x_var_coords = xr.Dataset(ed_v05x['Sonar'].sonar_serial_number.coords)
            v06x_var_coords = xr.Dataset(tree_v06x['Sonar'].ds.sonar_serial_number.coords)
            print(f"--> {v05x_var_coords.identical(v06x_var_coords)}")

            # drop sonar_serial_number
            ed_v05x['Sonar'] = ed_v05x['Sonar'].drop("sonar_serial_number")
            tree_v06x['Sonar'].ds = tree_v06x['Sonar'].ds.drop("sonar_serial_number")

        compare_ed_against_tree(ed_v05x, tree_v06x)
        print("")


def test_echodata_structure():
    """
    Makes sure that all raw files opened
    create the expected EchoData structure.
    """

    # compare_ed_against_tree(ed_v05x, tree_v06x)

    pytest.xfail("Full testing of the EchoData Structure has not been implemented yet.")