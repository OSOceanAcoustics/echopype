"""testing.py

Helper module for testing related things.
"""
import json
import os
from pathlib import Path

import fsspec
import xarray as xr

from echopype.echodata.echodata import EchoData

__all__ = [
    "_check_consolidated",
    "_check_output_files",
    "_create_path_str",
    "_check_and_drop_var",
    "_check_and_drop_attr",
    "_compare_ed_against_tree",
]

HERE = Path(__file__).parent.absolute()
TEST_DATA_FOLDER = HERE / "test_data"


def _check_consolidated(echodata: EchoData, zmeta_path: Path) -> None:
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
        os.path.join(p, ".zgroup") if p != "Top-level" else ".zgroup" for p in echodata.group_paths
    ]

    with open(zmeta_path) as f:
        meta_json = json.load(f)

    file_groups = [k for k in meta_json["metadata"].keys() if k.endswith(".zgroup")]

    for g in expected_zgroups:
        assert g in file_groups, f"{g} not Found!"


def _check_file_group(data_file, engine, groups):
    for g in groups:
        ds = xr.open_dataset(data_file, engine=engine, group=g)

        assert isinstance(ds, xr.Dataset) is True


def _check_output_files(engine, output_files, storage_options):
    groups = [
        "Provenance",
        "Environment",
        "Sonar/Beam_group1",
        "Sonar",
        "Vendor_specific",
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


def _check_and_drop_var(ed, tree, grp_path, var):
    """
    This function performs minimal checks of
    a variable contained both in an EchoData object
    and a Datatree. It ensures that the dimensions,
    attributes, and data types are the same. Once
    the checks have passed, it then drops these
    variables from both the EchoData object and the
    Datatree.

    Parameters
    ----------
    ed : EchoData
        EchoData object that contains the variable
        to check and drop.
    tree : Datatree
        Datatree object that contains the variable
        to check and drop.
    grp_path : str
        The path to the group that the variable is in.
    var : str
        The variable to be checked and dropped.

    Notes
    -----
    The Datatree object is created from an EchoData
    object written to a netcdf file.
    """

    ed_var = ed[grp_path][var]
    tree_var = tree[grp_path].ds[var]

    # make sure that the dimensions and attributes
    # are the same for the variable
    assert ed_var.dims == tree_var.dims
    assert ed_var.attrs == tree_var.attrs

    # make sure that the data types are correct too
    assert isinstance(ed_var.values, type(tree_var.values))

    # drop variables so we can check that datasets are identical
    ed[grp_path] = ed[grp_path].drop(var)
    tree[grp_path].ds = tree[grp_path].ds.drop(var)


def _check_and_drop_attr(ed, tree, grp_path, attr, typ):
    """
    This function performs minimal checks of
    an attribute contained both in an EchoData object
    and a Datatree group. This function only works for
    a group's attribute, it cannot work on variable
    attributes. It ensures that the attribute exists
    and that it has the expected data type. Once
    the checks have passed, it then drops the
    attribute from both the EchoData object and the
    Datatree.

    Parameters
    ----------
    ed : EchoData
        EchoData object that contains the attribute
        to check and drop.
    tree : Datatree
        Datatree object that contains the attribute
        to check and drop.
    grp_path : str
        The path to the group that the attribute  is in.
    attr : str
        The attribute to be checked and dropped.
    typ : type
        The expected data type of the attribute.

    Notes
    -----
    The Datatree object is created from an EchoData
    object written to a netcdf file.
    """

    # make sure that the attribute exists
    assert attr in ed[grp_path].attrs.keys()
    assert attr in tree[grp_path].ds.attrs.keys()

    # make sure that the value of the attribute is the right type
    assert isinstance(ed[grp_path].attrs[attr], typ)
    assert isinstance(tree[grp_path].ds.attrs[attr], typ)

    # drop the attribute so we can directly compare datasets
    del ed[grp_path].attrs[attr]
    del tree[grp_path].ds.attrs[attr]


def _compare_ed_against_tree(ed, tree):
    """
    This function compares the Datasets
    of ed against tree and makes sure they
    are identical.

    Parameters
    ----------
    ed : EchoData
        EchoData object
    tree : Datatree
        Datatree object

    Notes
    -----
    The Datatree object is created from an EchoData
    object written to a netcdf file.
    """

    for grp_path in ed.group_paths:
        if grp_path == "Top-level":
            assert tree.ds.identical(ed[grp_path])
        else:
            assert tree[grp_path].ds.identical(ed[grp_path])
