from datetime import datetime
from textwrap import dedent
from pathlib import Path
import tempfile

import numpy as np
import pytest
import xarray as xr

import echopype
from echopype.utils.coding import DEFAULT_ENCODINGS
from echopype.echodata import EchoData

from echopype.echodata.combine import (
    _create_channel_selection_dict,
    _check_channel_consistency,
    _merge_attributes
)


@pytest.fixture
def ek60_diff_range_sample_test_data(test_path):
    files = [
        ("ncei-wcsd", "SH1701", "TEST-D20170114-T202932.raw"),
        ("ncei-wcsd", "SH1701", "TEST-D20170114-T203337.raw"),
        ("ncei-wcsd", "SH1701", "TEST-D20170114-T203853.raw"),
    ]
    return [test_path["EK60"].joinpath(*f) for f in files]


@pytest.fixture(scope="module")
def ek60_test_data(test_path):
    files = [
        ("ncei-wcsd", "Summer2017-D20170620-T011027.raw"),
        ("ncei-wcsd", "Summer2017-D20170620-T014302.raw"),
        ("ncei-wcsd", "Summer2017-D20170620-T021537.raw"),
    ]
    return [test_path["EK60"].joinpath(*f) for f in files]

@pytest.fixture(scope="module")
def ek60_multi_test_data(test_path):
    files = [
        ("ncei-wcsd", "Summer2017-D20170620-T011027.raw"),
        ("ncei-wcsd", "Summer2017-D20170620-T014302.raw"),
        ("ncei-wcsd", "Summer2017-D20170620-T021537.raw"),
        ("ncei-wcsd", "Summer2017-D20170620-T024811.raw")
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
def ek80_broadband_same_range_sample_test_data(test_path):
    files = [
        ("ncei-wcsd", "SH1707", "Reduced_D20170826-T205615.raw"),
        ("ncei-wcsd", "SH1707", "Reduced_D20170826-T205659.raw"),
        ("ncei-wcsd", "SH1707", "Reduced_D20170826-T205742.raw"),
    ]
    return [test_path["EK80"].joinpath(*f) for f in files]


@pytest.fixture
def ek80_narrowband_diff_range_sample_test_data(test_path):
    files = [
        ("ncei-wcsd", "SH2106", "EK80", "Reduced_Hake-D20210701-T130426.raw"),
        ("ncei-wcsd", "SH2106", "EK80", "Reduced_Hake-D20210701-T131325.raw"),
        ("ncei-wcsd", "SH2106", "EK80", "Reduced_Hake-D20210701-T131621.raw"),
    ]
    return [test_path["EK80"].joinpath(*f) for f in files]


@pytest.fixture
def azfp_test_data(test_path):

    # TODO: in the future we should replace these files with another set of 
    #  similarly small set of files, for example the files from the location below:
    #  "https://rawdata.oceanobservatories.org/files/CE01ISSM/R00015/instrmts/dcl37/ZPLSC_sn55076/DATA/202109/*"
    #  This is because we have lost track of where the current files came from,
    #  since the filenames does not contain the site identifier.
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
            "files": "ek60_test_data"
        },
        {
            "sonar_model": "EK60",
            "xml_file": None,
            "files": "ek60_diff_range_sample_test_data"
        },
        {
            "sonar_model": "AZFP",
            "xml_file": "azfp_test_xml",
            "files": "azfp_test_data"
        },
        {
            "sonar_model": "EK80",
            "xml_file": None,
            "files": "ek80_broadband_same_range_sample_test_data"
        },
        {
            "sonar_model": "EK80",
            "xml_file": None,
            "files": "ek80_narrowband_diff_range_sample_test_data"
        }
    ],
    ids=["ek60", "ek60_diff_range_sample", "azfp",
         "ek80_bb_same_range_sample", "ek80_nb_diff_range_sample"]
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
        request.node.callspec.id
    )


def test_combine_echodata(raw_datasets):
    (
        files,
        sonar_model,
        xml_file,
        param_id,
    ) = raw_datasets

    eds = [echopype.open_raw(file, sonar_model, xml_file) for file in files]

    append_dims = {"filenames", "time1", "time2", "time3", "nmea_time", "ping_time"}

    combined = echopype.combine_echodata(eds)

    # Test Provenance conversion and combination attributes
    for attr_token in ["software_name", "software_version", "time"]:
        assert f"conversion_{attr_token}" in combined['Provenance'].attrs
        assert f"combination_{attr_token}" in combined['Provenance'].attrs

    def attr_time_to_dt(time_str):
        return datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S%z')
    assert (
            attr_time_to_dt(combined['Provenance'].attrs['conversion_time']) <=
            attr_time_to_dt(combined['Provenance'].attrs['combination_time'])
    )

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
            grp_drop_dims = list(set(combined_group.sizes).intersection(set(all_drop_dims)))
            concat_dims = list(set(combined_group.sizes).intersection(append_dims))

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
                test_ds = test_ds.assign_coords(
                    filenames=np.arange(test_ds.sizes["filenames"], dtype=int)
                )

            # correctly modify Provenance attributes, so we can do a direct compare
            if group_name == "Provenance":
                del test_ds.attrs["conversion_time"]
                del combined_group.attrs["conversion_time"]

        if group_name != "Provenance":
            # TODO: Skip for Provenance group for now, need to figure out how to test this properly
            if (combined_group is not None) and (test_ds is not None):
                assert test_ds.identical(combined_group.drop_dims(grp_drop_dims))


def _check_prov_ds(prov_ds, eds):
    """Checks the Provenance dataset against source_filenames variable
    and global attributes in the original echodata object"""
    for i in range(prov_ds.sizes["echodata_filename"]):
        ed_ds = eds[i]
        one_ds = prov_ds.isel(echodata_filename=i, filenames=i)
        for key, value in one_ds.data_vars.items():
            if key == "source_filenames":
                ed_group = "Provenance"
                assert np.array_equal(
                    ed_ds[ed_group][key].isel(filenames=0).values, value.values
                )
            else:
                ed_group = value.attrs.get("echodata_group")
                expected_val = ed_ds[ed_group].attrs[key]
                if not isinstance(expected_val, str):
                    expected_val = str(expected_val)
                assert str(value.values) == expected_val


@pytest.mark.parametrize("test_param", [
        "single",
        "multi",
        "combined"
    ]
)
def test_combine_echodata_combined_append(ek60_multi_test_data, test_param, sonar_model="EK60"):
        """
        Integration test for combine_echodata with the following cases:
        - a single combined echodata object and a single echodata object
        - a single combined echodata object and 2 single echodata objects
        - a single combined echodata object and another combined single echodata object
        """
        eds = [
            echopype.open_raw(raw_file=file, sonar_model=sonar_model)
            for file in ek60_multi_test_data
        ]
        # create temporary directory for zarr store
        temp_zarr_dir = tempfile.TemporaryDirectory()
        first_zarr = (
            temp_zarr_dir.name
            + f"/combined_echodata.zarr"
        )
        second_zarr = (
            temp_zarr_dir.name
            + f"/combined_echodata2.zarr"
        )
        # First combined file
        combined_ed = echopype.combine_echodata(eds[:2])
        combined_ed.to_zarr(first_zarr, overwrite=True)
        
        def _check_prov_ds_and_dims(sel_comb_ed, n_val_expected):
            prov_ds = sel_comb_ed["Provenance"]
            for _, n_val in prov_ds.sizes.items():
                assert n_val == n_val_expected
            _check_prov_ds(prov_ds, eds)

        # Checks for Provenance group
        # Both dims of filenames and echodata filename should be 2
        expected_n_vals = 2
        _check_prov_ds_and_dims(combined_ed, expected_n_vals)

        # Second combined file
        combined_ed_other = echopype.combine_echodata(eds[2:])
        combined_ed_other.to_zarr(second_zarr, overwrite=True)

        combined_ed = echopype.open_converted(first_zarr)
        combined_ed_other = echopype.open_converted(second_zarr)

        # Set expected values for Provenance
        if test_param == "single":
            data_inputs = [combined_ed, eds[2]]
            expected_n_vals = 3
        elif test_param == "multi":
            data_inputs = [combined_ed, eds[2], eds[3]]
            expected_n_vals = 4
        else:
            data_inputs = [combined_ed, combined_ed_other]
            expected_n_vals = 4

        combined_ed2 = echopype.combine_echodata(data_inputs)

        # Verify that combined objects are all EchoData objects
        assert isinstance(combined_ed, EchoData)
        assert isinstance(combined_ed_other, EchoData)
        assert isinstance(combined_ed2, EchoData)

        # Ensure that they're from the same file source
        group_path = "Provenance"
        for i in range(4):
            ds_i = eds[i][group_path]
            select_comb_ds = combined_ed[group_path] if i < 2 else combined_ed2[group_path]
            if i < 3 or (i == 3 and test_param != "single"):
                assert ds_i.source_filenames[0].values == select_comb_ds.source_filenames[i].values

        # Check beam_group1. Should be exactly same xr dataset
        group_path = "Sonar/Beam_group1"
        for i in range(4):
            ds_i = eds[i][group_path]
            select_comb_ds = combined_ed[group_path] if i < 2 else combined_ed2[group_path]
            if i < 3 or (i == 3 and test_param != "single"):
                filt_ds_i = select_comb_ds.sel(ping_time=ds_i.ping_time)
                assert filt_ds_i.identical(ds_i) is True

        filt_combined = combined_ed2[group_path].sel(ping_time=combined_ed[group_path].ping_time)
        assert filt_combined.identical(combined_ed[group_path]) is True
        
        # Checks for Provenance group
        # Both dims of filenames and echodata filename should be expected_n_vals
        _check_prov_ds_and_dims(combined_ed2, expected_n_vals)


def test_combine_echodata_channel_selection():
    """
    This test ensures that the ``channel_selection`` input
    of ``combine_echodata`` is producing the correct output
    for all sonar models except AD2CP.
    """

    # TODO: Once a mock EchoData structure can be easily formed,
    #  we should implement this test.

    pytest.skip("This test will not be implemented until after a mock EchoData object can be created.")


def test_attr_storage(ek60_test_data):
    # check storage of attributes before combination in provenance group
    eds = [echopype.open_raw(file, "EK60") for file in ek60_test_data]

    combined = echopype.combine_echodata(eds)

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


def test_combined_encodings(ek60_test_data):
    eds = [echopype.open_raw(file, "EK60") for file in ek60_test_data]

    combined = echopype.combine_echodata(eds)

    encodings_to_drop = {'chunks', 'preferred_chunks', 'compressor', 'filters'}

    group_checks = []
    for _, value in combined.group_map.items():
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

    if len(group_checks) > 0:
        all_messages = ['Encoding mismatch found!'] + group_checks
        message_text = '\n'.join(all_messages)
        raise AssertionError(message_text)


def test_combined_echodata_repr(ek60_test_data):
    eds = [echopype.open_raw(file, "EK60") for file in ek60_test_data]

    combined = echopype.combine_echodata(eds)

    expected_repr = dedent(
        f"""\
        <EchoData: standardized raw data from Internal Memory>
        Top-level: contains metadata about the SONAR-netCDF4 file format.
        ├── Environment: contains information relevant to acoustic propagation through water.
        ├── Platform: contains information about the platform on which the sonar is installed.
        │   └── NMEA: contains information specific to the NMEA protocol.
        ├── Provenance: contains metadata about how the SONAR-netCDF4 version of the data were obtained.
        ├── Sonar: contains sonar system metadata and sonar beam groups.
        │   └── Beam_group1: contains backscatter power (uncalibrated) and other beam or channel-specific data, including split-beam angle data when they exist.
        └── Vendor_specific: contains vendor-specific information about the sonar and the data."""
    )

    assert isinstance(repr(combined), str) is True

    actual = "\n".join(x.rstrip() for x in repr(combined).split("\n"))
    assert actual == expected_repr


@pytest.mark.parametrize(
    ("all_chan_list", "channel_selection"),
    [
        (
            [['a', 'b', 'c'], ['a', 'b', 'c']],
            None
        ),
        pytest.param(
            [['a', 'b', 'c'], ['a', 'b']],
            None,
            marks=pytest.mark.xfail(strict=True,
                                    reason="This test should not pass because the channels are not consistent")
        ),
        (
            [['a', 'b', 'c'], ['a', 'b', 'c']],
            ['a', 'b', 'c']
        ),
        (
            [['a', 'b', 'c'], ['a', 'b', 'c']],
            ['a', 'b']
        ),
        (
            [['a', 'b', 'c'], ['a', 'b']],
            ['a', 'b']
        ),
        pytest.param(
            [['a', 'c'], ['a', 'b', 'c']],
            ['a', 'b'],
            marks=pytest.mark.xfail(strict=True,
                                    reason="This test should not pass because we are selecting "
                                           "channels that do not occur in each Dataset")
        ),
    ],
    ids=["chan_sel_none_pass", "chan_sel_none_fail",
         "chan_sel_same_as_given_chans", "chan_sel_subset_of_given_chans",
         "chan_sel_subset_of_given_chans_uneven", "chan_sel_diff_from_some_given_chans"]
)
def test_check_channel_consistency(all_chan_list, channel_selection):
    """
    Ensures that the channel consistency check for combine works
    as expected using mock data.
    """

    _check_channel_consistency(all_chan_list, "test_group", channel_selection)


# create duplicated dictionaries used within pytest parameterize
has_chan_dim_1_beam = {'Top-level': False, 'Environment': False, 'Platform': True,
                       'Platform/NMEA': False, 'Provenance': False, 'Sonar': True,
                       'Sonar/Beam_group1': True, 'Vendor_specific': True}

has_chan_dim_2_beam = {'Top-level': False, 'Environment': False, 'Platform': True,
                       'Platform/NMEA': False, 'Provenance': False, 'Sonar': True,
                       'Sonar/Beam_group1': True, 'Sonar/Beam_group2': True, 'Vendor_specific': True}

expected_1_beam_none = {'Top-level': None, 'Environment': None, 'Platform': None,
                        'Platform/NMEA': None, 'Provenance': None, 'Sonar': None,
                        'Sonar/Beam_group1': None, 'Vendor_specific': None}

expected_1_beam_a_b_sel = {'Top-level': None, 'Environment': None, 'Platform': ['a', 'b'],
                           'Platform/NMEA': None, 'Provenance': None, 'Sonar': ['a', 'b'],
                           'Sonar/Beam_group1': ['a', 'b'], 'Vendor_specific': ['a', 'b']}


@pytest.mark.parametrize(
    ("sonar_model", "has_chan_dim", "user_channel_selection", "expected_dict"),
    [
        (
            ["EK60", "ES70", "AZFP"],
            has_chan_dim_1_beam,
            [None],
            expected_1_beam_none
        ),
        (
            ["EK80", "ES80", "EA640"],
            has_chan_dim_1_beam,
            [None],
            expected_1_beam_none
        ),
        (
            ["EK80", "ES80", "EA640"],
            has_chan_dim_2_beam,
            [None],
            {'Top-level': None, 'Environment': None, 'Platform': None, 'Platform/NMEA': None,
             'Provenance': None, 'Sonar': None, 'Sonar/Beam_group1': None,
             'Sonar/Beam_group2': None, 'Vendor_specific': None}
        ),
        (
            ["EK60", "ES70", "AZFP"],
            has_chan_dim_1_beam,
            [['a', 'b'], {'Sonar/Beam_group1': ['a', 'b']}],
            expected_1_beam_a_b_sel
        ),
        (
            ["EK80", "ES80", "EA640"],
            has_chan_dim_1_beam,
            [['a', 'b'], {'Sonar/Beam_group1': ['a', 'b']}],
            expected_1_beam_a_b_sel
        ),
        (
            ["EK80", "ES80", "EA640"],
            has_chan_dim_2_beam,
            [['a', 'b']],
            {'Top-level': None, 'Environment': None, 'Platform': ['a', 'b'], 'Platform/NMEA': None,
             'Provenance': None, 'Sonar': ['a', 'b'], 'Sonar/Beam_group1': ['a', 'b'],
             'Sonar/Beam_group2': ['a', 'b'], 'Vendor_specific': ['a', 'b']}
        ),
        (
            ["EK80", "ES80", "EA640"],
            has_chan_dim_2_beam,
            [{'Sonar/Beam_group1': ['a', 'b'], 'Sonar/Beam_group2': ['c', 'd']}],
            {'Top-level': None, 'Environment': None, 'Platform': ['a', 'b', 'c', 'd'], 'Platform/NMEA': None,
             'Provenance': None, 'Sonar': ['a', 'b', 'c', 'd'], 'Sonar/Beam_group1': ['a', 'b'],
             'Sonar/Beam_group2': ['c', 'd'], 'Vendor_specific': ['a', 'b', 'c', 'd']}
        ),
        (
            ["EK80", "ES80", "EA640"],
            has_chan_dim_2_beam,
            [{'Sonar/Beam_group1': ['a', 'b'], 'Sonar/Beam_group2': ['b', 'c', 'd']}],
            {'Top-level': None, 'Environment': None, 'Platform': ['a', 'b', 'c', 'd'], 'Platform/NMEA': None,
             'Provenance': None, 'Sonar': ['a', 'b', 'c', 'd'], 'Sonar/Beam_group1': ['a', 'b'],
             'Sonar/Beam_group2': ['b', 'c', 'd'], 'Vendor_specific': ['a', 'b', 'c', 'd']}
        ),
    ],
    ids=["EK60_no_sel", "EK80_no_sel_1_beam", "EK80_no_sel_2_beam", "EK60_chan_sel",
         "EK80_chan_sel_1_beam", "EK80_list_chan_sel_2_beam", "EK80_dict_chan_sel_2_beam_diff_beam_group_chans",
         "EK80_dict_chan_sel_2_beam_overlap_beam_group_chans"]
)
def test_create_channel_selection_dict(sonar_model, has_chan_dim,
                                       user_channel_selection, expected_dict):
    """
    Ensures that ``create_channel_selction_dict`` is constructing the correct output
    for the sonar models ``EK60, EK80, AZFP`` and varying inputs for the input
    ``user_channel_selection``.

    Notes
    -----
    The input ``has_chan_dim`` is unchanged except for the case where we are considering
    an EK80 sonar model with two beam groups.
    """

    for model in sonar_model:
        for usr_sel_chan in user_channel_selection:

            channel_selection_dict = _create_channel_selection_dict(model, has_chan_dim, usr_sel_chan)
            assert channel_selection_dict == expected_dict

@pytest.mark.parametrize(
    ["attributes", "expected"],
    [
        ([{"key1": ""}, {"key1": "test2"}, {"key1": "test1"}], {"key1": "test2"}),
        (
            [{"key1": "test1"}, {"key1": ""}, {"key1": "test2"}, {"key2": ""}],
            {"key1": "test1", "key2": ""},
        ),
        (
            [
                {"key1": ""},
                {"key2": "test1", "key1": "test2"},
                {"key2": "test3"},
            ],
            {"key2": "test1", "key1": "test2"},
        ),
    ],
)
def test__merge_attributes(attributes, expected):
    merged = _merge_attributes(attributes)

    assert merged == expected
