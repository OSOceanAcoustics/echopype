from textwrap import dedent
from pathlib import Path

import numpy as np
import xarray as xr

import echopype
from echopype.utils.coding import DEFAULT_ENCODINGS
import os.path

import tempfile
from dask.distributed import Client

from echopype.echodata.combine import _create_channel_selection_dict, _check_echodata_channels, \
    _check_channel_consistency


def test_combine_echodata(raw_datasets):
    (
        files,
        sonar_model,
        xml_file,
        param_id,
    ) = raw_datasets

    eds = [echopype.open_raw(file, sonar_model, xml_file) for file in files]

    append_dims = {"filenames", "time1", "time2", "time3", "ping_time"}

    # create temporary directory for zarr store
    temp_zarr_dir = tempfile.TemporaryDirectory()
    zarr_file_name = os.path.join(temp_zarr_dir.name, "combined_echodatas.zarr")

    # create dask client
    client = Client()

    combined = echopype.combine_echodata(eds, zarr_file_name, client=client)

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
            grp_drop_dims = list(set(combined_group.dims).intersection(set(all_drop_dims)))
            concat_dims = list(set(combined_group.dims).intersection(append_dims))

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
                test_ds.filenames.values[:] = np.arange(len(test_ds.filenames), dtype=int)

            # correctly modify Provenance attributes, so we can do a direct compare
            if group_name == "Provenance":
                del test_ds.attrs["conversion_time"]
                del combined_group.attrs["conversion_time"]

        if (combined_group is not None) and (test_ds is not None):
            assert test_ds.identical(combined_group.drop_dims(grp_drop_dims))

    temp_zarr_dir.cleanup()

    # close client
    client.close()


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

    # create temporary directory for zarr store
    temp_zarr_dir = tempfile.TemporaryDirectory()
    zarr_file_name = os.path.join(temp_zarr_dir.name, "combined_echodatas.zarr")

    # create dask client
    client = Client()

    combined = echopype.combine_echodata(eds, zarr_file_name, client=client)

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

    temp_zarr_dir.cleanup()

    # close client
    client.close()


def test_combined_encodings(ek60_test_data):
    eds = [echopype.open_raw(file, "EK60") for file in ek60_test_data]

    # create temporary directory for zarr store
    temp_zarr_dir = tempfile.TemporaryDirectory()
    zarr_file_name = os.path.join(temp_zarr_dir.name, "combined_echodatas.zarr")

    # create dask client
    client = Client()

    combined = echopype.combine_echodata(eds, zarr_file_name, client=client)

    encodings_to_drop = {'chunks', 'preferred_chunks', 'compressor', 'filters'}

    group_checks = []
    for group, value in combined.group_map.items():
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

    temp_zarr_dir.cleanup()

    # close client
    client.close()

    if len(group_checks) > 0:
        all_messages = ['Encoding mismatch found!'] + group_checks
        message_text = '\n'.join(all_messages)
        raise AssertionError(message_text)


def test_combined_echodata_repr(ek60_test_data):
    eds = [echopype.open_raw(file, "EK60") for file in ek60_test_data]

    # create temporary directory for zarr store
    temp_zarr_dir = tempfile.TemporaryDirectory()
    zarr_file_name = os.path.join(temp_zarr_dir.name, "combined_echodatas.zarr")

    # create dask client
    client = Client()

    combined = echopype.combine_echodata(eds, zarr_file_name, client=client)

    expected_repr = dedent(
        f"""\
        <EchoData: standardized raw data from {zarr_file_name}>
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

    temp_zarr_dir.cleanup()

    # close client
    client.close()


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
