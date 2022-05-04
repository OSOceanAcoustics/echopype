from typing import Any, Dict, Optional
from datatree import open_datatree
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


def test_v05x_v06x_conversion_structure():
    """
    Tests that version 0.5.x echopype files
    have been correctly converted to the
    0.6.x structure.
    """

    converted_raw_path_v06x = './ek60-Summer2017-D20170615-T190214-ep-v06x.nc'
    tree_v06x = _tree_from_file(converted_raw_path=converted_raw_path_v06x)

    converted_raw_path_v05x = './ek60-Summer2017-D20170615-T190214-ep-v05x.nc'
    ed_v05x = open_converted(converted_raw_path_v05x)

    for grp_path in ed_v05x.group_paths:

        print(f"grp_path = {grp_path}")
        if grp_path == "Top-level":

            print(f"--> {tree_v06x.ds.identical(ed_v05x[grp_path])}")
        else:
            print(f"--> {tree_v06x[grp_path].ds.identical(ed_v05x[grp_path])}")


def test_echodata_structure():
    """
    Makes sure that all raw files opened
    create the expected EchoData structure.
    """

    pytest.xfail("Full testing of the EchoData Structure has not been implemented yet.")