"""
Note 2023-08-30:
None of the test in this module is actually run (they are xfailed),
and especially since we have removed automatic version conversion in open_converted in #1143.
However, since we kept the previous data format conversion mechanisms
(under echodata/sensor_ep_version_mapping) intact in case we are able to
reinstate such capability in the future, we also keep these tests here.

See https://github.com/OSOceanAcoustics/echopype/pull/1143 for discussions.
"""


from typing import Any, Dict, Optional
from datatree import open_datatree
import pytest
from echopype.echodata.echodata import EchoData, XARRAY_ENGINE_MAP
from echopype.echodata.api import open_converted


@pytest.fixture
def azfp_path(test_path):
    return test_path['AZFP']


@pytest.fixture
def ek60_path(test_path):
    return test_path['EK60']


@pytest.fixture
def ek80_path(test_path):
    return test_path['EK80']


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
    ed[grp_path] = ed[grp_path].drop_vars(var)
    tree[grp_path].ds = tree[grp_path].ds.drop_vars(var)


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


def compare_ed_against_tree(ed, tree):
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


def _get_conversion_file_lists(azfp_path, ek60_path, ek80_path):

    converted_raw_paths_v06x = [ek60_path / "ek60-Summer2017-D20170615-T190214-ep-v06x.nc",
                                ek60_path / "ek60-combined-ep-v06x.nc",
                                ek80_path / "ek80-Summer2018--D20180905-T033113-ep-v06x.nc",
                                ek80_path / "ek80-2018115-D20181213-T094600-ep-v06x.nc",
                                ek80_path / "ek80-2019118-group2survey-D20191214-T081342-ep-v06x.nc",
                                ek80_path / "ek80-Green2-Survey2-FM-short-slow-D20191004-T211557-ep-v06x.nc",
                                azfp_path / "azfp-17082117_01A_17041823_XML-ep-v06x.nc"]

    converted_raw_paths_v05x = [ek60_path / "ek60-Summer2017-D20170615-T190214-ep-v05x.nc",
                                ek60_path / "ek60-combined-ep-v05x.nc",
                                ek80_path / "ek80-Summer2018--D20180905-T033113-ep-v05x.nc",
                                ek80_path / "ek80-2018115-D20181213-T094600-ep-v05x.nc",
                                ek80_path / "ek80-2019118-group2survey-D20191214-T081342-ep-v05x.nc",
                                ek80_path / "ek80-Green2-Survey2-FM-short-slow-D20191004-T211557-ep-v05x.nc",
                                azfp_path / "azfp-17082117_01A_17041823_XML-ep-v05x.nc"]

    return converted_raw_paths_v06x, converted_raw_paths_v05x


def test_v05x_v06x_conversion_structure(azfp_path, ek60_path, ek80_path):
    """
    Tests that version 0.5.x echopype files
    have been correctly converted to the
    0.6.x structure.
    """

    pytest.xfail("PR #881 has caused these tests to fail for EK80 sonar models. While we "
                 "revise this test structure, these tests will be skipped. Please see issue "
                 "https://github.com/OSOceanAcoustics/echopype/issues/884 for more information.")

    converted_raw_paths_v06x, converted_raw_paths_v05x = \
        _get_conversion_file_lists(azfp_path, ek60_path, ek80_path)

    for path_v05x, path_v06x in zip(converted_raw_paths_v05x, converted_raw_paths_v06x):

        ed_v05x = open_converted(path_v05x)
        tree_v06x = _tree_from_file(converted_raw_path=path_v06x)

        # dictionary of attributes to drop (from the group only) where
        # the group path is the key and the value is a list of tuples
        # of the form (attr, type of attr expected)
        attrs_to_drop = {
            "Provenance": [("conversion_software_version", str),
                           ("conversion_time", str)]
        }

        # check and drop attributes that cannot be directly compared
        # because their values are not the same
        for key, val in attrs_to_drop.items():
            for var in val:
                _check_and_drop_attr(ed_v05x, tree_v06x, key, var[0], var[1])

        _check_and_drop_var(ed_v05x, tree_v06x, "Provenance", "source_filenames")

        # The following if block is for the case where we have a combined file
        # TODO: look into this after v0.6.0 release
        if "echodata_filename" in ed_v05x["Provenance"]:
            prov_comb_names = ["echodata_filename", "top_attrs", "environment_attrs",
                               "platform_attrs", "nmea_attrs", "provenance_attrs",
                               "sonar_attrs", "beam_attrs", "vendor_attrs",
                               "top_attr_key", "environment_attr_key",
                               "platform_attr_key", "nmea_attr_key", "provenance_attr_key",
                               "sonar_attr_key", "beam_attr_key", "vendor_attr_key"]

            for name in prov_comb_names:
                _check_and_drop_var(ed_v05x, tree_v06x, "Provenance", name)

            ed_v05x["Provenance"] = ed_v05x["Provenance"].drop_vars("src_filenames")

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
                                            "transducer_sound_speed"]
                            }

            # check and drop variables that cannot be directly compared
            # because their values are not the same
            for key, val in vars_to_drop.items():
                for var in val:
                    _check_and_drop_var(ed_v05x, tree_v06x, key, var)

            # sort the beam groups for EK80 according to channel (necessary for comparison)
            ed_v05x['Sonar/Beam_group1'] = ed_v05x['Sonar/Beam_group1'].sortby("channel")

            if 'Sonar/Beam_group2' in ed_v05x.group_paths:
                ed_v05x['Sonar/Beam_group2'] = ed_v05x['Sonar/Beam_group2'].sortby("channel")

            # sort the Platform group by channel for EK80 (necessary for comparison)
            tree_v06x['Platform'].ds = tree_v06x['Platform'].ds.sortby('channel')
            ed_v05x['Platform'] = ed_v05x['Platform'].sortby('channel')

            # remove all attributes from Vendor_specific (data is missing sometimes)
            tree_v06x["Vendor_specific"].ds.attrs = {"blank": 'None'}
            ed_v05x["Vendor_specific"].attrs = {"blank": 'None'}

        compare_ed_against_tree(ed_v05x, tree_v06x)


def test_echodata_structure(azfp_path, ek60_path, ek80_path):
    """
    Makes sure that all raw files opened
    create the expected EchoData structure.
    """

    # TODO: create this test once dev is in its final form.
    # check and remove conversion time from attributes
    # _check_and_drop_attr(ed_v05x, tree_v06x, "Provenance", "conversion_time", str)
    # compare_ed_against_tree(ed_v05x, tree_v06x)

    pytest.xfail("Full testing of the EchoData Structure has not been implemented yet.")
