from typing import Any, Dict, Optional
from datatree import open_datatree

import echopype
from echopype.calibrate.env_params import EnvParams
from echopype.echodata.echodata import EchoData, XARRAY_ENGINE_MAP
from echopype import open_converted

import pytest
import xarray as xr
import numpy as np

from echopype.testing import (
    _compare_ed_against_tree,
    _check_and_drop_attr,
    _check_and_drop_var,
)


def test_compute_range(compute_range_samples):
    (
        filepath,
        sonar_model,
        azfp_xml_path,
        azfp_cal_type,
        ek_waveform_mode,
        ek_encode_mode,
    ) = compute_range_samples
    ed = echopype.open_raw(filepath, sonar_model, azfp_xml_path)
    rng = np.random.default_rng(0)
    stationary_env_params = EnvParams(
        xr.Dataset(
            data_vars={
                "pressure": ("time3", np.arange(50)),
                "salinity": ("time3", np.arange(50)),
                "temperature": ("time3", np.arange(50)),
            },
            coords={
                "time3": np.arange(
                    "2017-06-20T01:00",
                    "2017-06-20T01:25",
                    np.timedelta64(30, "s"),
                    dtype="datetime64[ns]",
                )
            },
        ),
        data_kind="stationary",
    )
    if "time3" in ed["Platform"] and sonar_model != "AD2CP":
        ed.compute_range(
            stationary_env_params, azfp_cal_type, ek_waveform_mode
        )
    else:
        try:
            ed.compute_range(
                stationary_env_params,
                ek_waveform_mode="CW",
                azfp_cal_type="Sv",
            )
        except ValueError:
            pass
        else:
            raise AssertionError

    mobile_env_params = EnvParams(
        xr.Dataset(
            data_vars={
                "pressure": ("time", np.arange(100)),
                "salinity": ("time", np.arange(100)),
                "temperature": ("time", np.arange(100)),
            },
            coords={
                "latitude": ("time", rng.random(size=100) + 44),
                "longitude": ("time", rng.random(size=100) - 125),
            },
        ),
        data_kind="mobile",
    )
    if (
        "latitude" in ed["Platform"]
        and "longitude" in ed["Platform"]
        and sonar_model != "AD2CP"
        and not np.isnan(ed["Platform"]["time1"]).all()
    ):
        ed.compute_range(mobile_env_params, azfp_cal_type, ek_waveform_mode)
    else:
        try:
            ed.compute_range(
                mobile_env_params, ek_waveform_mode="CW", azfp_cal_type="Sv"
            )
        except ValueError:
            pass
        else:
            raise AssertionError

    env_params = {"sound_speed": 343}
    if sonar_model == "AD2CP":
        try:
            ed.compute_range(
                env_params, ek_waveform_mode="CW", azfp_cal_type="Sv"
            )
        except ValueError:
            pass  # AD2CP is not currently supported in ed.compute_range
        else:
            raise AssertionError
    else:
        echo_range = ed.compute_range(
            env_params,
            azfp_cal_type,
            ek_waveform_mode,
        )
        assert isinstance(echo_range, xr.DataArray)


def test_nan_range_entries(range_check_files):
    sonar_model, ek_file = range_check_files
    echodata = echopype.open_raw(ek_file, sonar_model=sonar_model)
    if sonar_model == "EK80":
        ds_Sv = echopype.calibrate.compute_Sv(
            echodata, waveform_mode='BB', encode_mode='complex'
        )
        range_output = echodata.compute_range(
            env_params=[], ek_waveform_mode='BB'
        )
        nan_locs_backscatter_r = (
            ~echodata["Sonar/Beam_group1"]
            .backscatter_r.isel(beam=0)
            .drop("beam")
            .isnull()
        )
    else:
        ds_Sv = echopype.calibrate.compute_Sv(echodata)
        range_output = echodata.compute_range(env_params=[])
        nan_locs_backscatter_r = (
            ~echodata["Sonar/Beam_group1"]
            .backscatter_r.isel(beam=0)
            .drop("beam")
            .isnull()
        )

    nan_locs_Sv_range = ~ds_Sv.echo_range.isnull()
    nan_locs_range = ~range_output.isnull()
    assert xr.Dataset.equals(nan_locs_backscatter_r, nan_locs_range)
    assert xr.Dataset.equals(nan_locs_backscatter_r, nan_locs_Sv_range)


@pytest.mark.parametrize(
    [
        "ext_type",
        "sonar_model",
        "updated",
        "path_model",
        "raw_path",
        "platform_data",
    ],
    [
        (
            "external-trajectory",
            "EK80",
            ("pitch", "roll", "longitude", "latitude"),
            "EK80",
            (
                "saildrone",
                "SD2019_WCS_v05-Phase0-D20190617-T125959-0.raw",
            ),
            (
                "saildrone",
                "saildrone-gen_5-fisheries-acoustics-code-sprint-sd1039-20190617T130000-20190618T125959-1_hz-v1.1595357449818.nc",  # noqa
            ),
        ),
        (
            "fixed-location",
            "EK60",
            ("longitude", "latitude"),
            "EK60",
            ("ooi", "CE02SHBP-MJ01C-07-ZPLSCB101_OOI-D20191201-T000000.raw"),
            (-100.0, -50.0),
        ),
    ],
)
def test_update_platform(
    ext_type,
    sonar_model,
    updated,
    path_model,
    raw_path,
    platform_data,
    test_path,
):
    raw_file = test_path[path_model] / raw_path[0] / raw_path[1]
    ed = echopype.open_raw(raw_file, sonar_model=sonar_model)

    for variable in updated:
        assert np.isnan(ed["Platform"][variable].values).all()

    if ext_type == "external-trajectory":
        extra_platform_data_file_name = platform_data[1]
        extra_platform_data = xr.open_dataset(
            test_path[path_model]
            / platform_data[0]
            / extra_platform_data_file_name
        )
    elif ext_type == "fixed-location":
        extra_platform_data_file_name = None
        extra_platform_data = xr.Dataset(
            {
                "longitude": (["time"], np.array([float(platform_data[0])])),
                "latitude": (["time"], np.array([float(platform_data[1])])),
            },
            coords={
                "time": (
                    ["time"],
                    np.array([ed['Sonar/Beam_group1'].ping_time.values.min()]),
                )
            },
        )

    ed.update_platform(
        extra_platform_data,
        extra_platform_data_file_name=extra_platform_data_file_name,
    )

    for variable in updated:
        assert not np.isnan(ed["Platform"][variable].values).all()

    # times have max interval of 2s
    # check times are > min(ed["Sonar/Beam_group1"]["ping_time"]) - 2s
    assert (
        ed["Platform"]["time1"]
        > ed["Sonar/Beam_group1"]["ping_time"].min() - np.timedelta64(2, "s")
    ).all()
    # check there is only 1 time < min(ed["Sonar/Beam_group1"]["ping_time"])
    assert (
        np.count_nonzero(
            ed["Platform"]["time1"]
            < ed["Sonar/Beam_group1"]["ping_time"].min()
        )
        <= 1
    )
    # check times are < max(ed["Sonar/Beam_group1"]["ping_time"]) + 2s
    assert (
        ed["Platform"]["time1"]
        < ed["Sonar/Beam_group1"]["ping_time"].max() + np.timedelta64(2, "s")
    ).all()
    # check there is only 1 time > max(ed["Sonar/Beam_group1"]["ping_time"])
    assert (
        np.count_nonzero(
            ed["Platform"]["time1"]
            > ed["Sonar/Beam_group1"]["ping_time"].max()
        )
        <= 1
    )


def _tree_from_file(
    converted_raw_path: str,
    ed_storage_options: Optional[Dict[str, Any]] = {},
    open_kwargs: Dict[str, Any] = {},
):
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
    converted_raw_path = EchoData._sanitize_path(
        temp_class, converted_raw_path
    )
    suffix = EchoData._check_suffix(temp_class, converted_raw_path)

    tree = open_datatree(
        converted_raw_path,
        engine=XARRAY_ENGINE_MAP[suffix],
        **open_kwargs,
    )

    return tree


def _get_conversion_file_lists(azfp_path, ek60_path, ek80_path):

    converted_raw_paths_v06x = [
        ek60_path / "ek60-Summer2017-D20170615-T190214-ep-v06x.nc",
        ek60_path / "ek60-combined-ep-v06x.nc",
        ek80_path / "ek80-Summer2018--D20180905-T033113-ep-v06x.nc",
        ek80_path / "ek80-2018115-D20181213-T094600-ep-v06x.nc",
        ek80_path / "ek80-2019118-group2survey-D20191214-T081342-ep-v06x.nc",
        ek80_path
        / "ek80-Green2-Survey2-FM-short-slow-D20191004-T211557-ep-v06x.nc",
        azfp_path / "azfp-17082117_01A_17041823_XML-ep-v06x.nc",
    ]

    converted_raw_paths_v05x = [
        ek60_path / "ek60-Summer2017-D20170615-T190214-ep-v05x.nc",
        ek60_path / "ek60-combined-ep-v05x.nc",
        ek80_path / "ek80-Summer2018--D20180905-T033113-ep-v05x.nc",
        ek80_path / "ek80-2018115-D20181213-T094600-ep-v05x.nc",
        ek80_path / "ek80-2019118-group2survey-D20191214-T081342-ep-v05x.nc",
        ek80_path
        / "ek80-Green2-Survey2-FM-short-slow-D20191004-T211557-ep-v05x.nc",
        azfp_path / "azfp-17082117_01A_17041823_XML-ep-v05x.nc",
    ]

    return converted_raw_paths_v06x, converted_raw_paths_v05x


def test_v05x_v06x_conversion_structure(azfp_path, ek60_path, ek80_path):
    """
    Tests that version 0.5.x echopype files
    have been correctly converted to the
    0.6.x structure.
    """

    pytest.xfail(
        "PR #881 has caused these tests to fail for EK80 sonar models. While we "
        "revise this test structure, these tests will be skipped. Please see issue "
        "https://github.com/OSOceanAcoustics/echopype/issues/884 for more information."
    )

    (
        converted_raw_paths_v06x,
        converted_raw_paths_v05x,
    ) = _get_conversion_file_lists(azfp_path, ek60_path, ek80_path)

    for path_v05x, path_v06x in zip(
        converted_raw_paths_v05x, converted_raw_paths_v06x
    ):

        ed_v05x = open_converted(path_v05x)
        tree_v06x = _tree_from_file(converted_raw_path=path_v06x)

        # dictionary of attributes to drop (from the group only) where
        # the group path is the key and the value is a list of tuples
        # of the form (attr, type of attr expected)
        attrs_to_drop = {
            "Provenance": [
                ("conversion_software_version", str),
                ("conversion_time", str),
            ]
        }

        # check and drop attributes that cannot be directly compared
        # because their values are not the same
        for key, val in attrs_to_drop.items():
            for var in val:
                _check_and_drop_attr(ed_v05x, tree_v06x, key, var[0], var[1])

        _check_and_drop_var(
            ed_v05x, tree_v06x, "Provenance", "source_filenames"
        )

        # The following if block is for the case where we have a combined file
        # TODO: look into this after v0.6.0 release
        if "echodata_filename" in ed_v05x["Provenance"]:
            prov_comb_names = [
                "echodata_filename",
                "top_attrs",
                "environment_attrs",
                "platform_attrs",
                "nmea_attrs",
                "provenance_attrs",
                "sonar_attrs",
                "beam_attrs",
                "vendor_attrs",
                "top_attr_key",
                "environment_attr_key",
                "platform_attr_key",
                "nmea_attr_key",
                "provenance_attr_key",
                "sonar_attr_key",
                "beam_attr_key",
                "vendor_attr_key",
            ]

            for name in prov_comb_names:
                _check_and_drop_var(ed_v05x, tree_v06x, "Provenance", name)

            ed_v05x["Provenance"] = ed_v05x["Provenance"].drop("src_filenames")

        # ignore direct comparison of the variables Sonar.sonar_serial_number,
        # Platform.drop_keel_offset_is_manual, and Platform.water_level_draft_is_manual
        # for EK80, this data is not present in v0.5.x
        if ed_v05x["Top-level"].attrs["keywords"] == "EK80":

            # dictionary of variables to drop where the group path is the
            # key and the variables are the value
            vars_to_drop = {
                "Sonar": ["sonar_serial_number"],
                "Platform": [
                    "drop_keel_offset_is_manual",
                    "water_level_draft_is_manual",
                ],
                "Environment": [
                    "sound_velocity_profile",
                    "sound_velocity_profile_depth",
                    "sound_velocity_source",
                    "transducer_name",
                    "transducer_sound_speed",
                ],
            }

            # check and drop variables that cannot be directly compared
            # because their values are not the same
            for key, val in vars_to_drop.items():
                for var in val:
                    _check_and_drop_var(ed_v05x, tree_v06x, key, var)

            # sort the beam groups for EK80 according to channel (necessary for comparison)
            ed_v05x['Sonar/Beam_group1'] = ed_v05x['Sonar/Beam_group1'].sortby(
                "channel"
            )

            if 'Sonar/Beam_group2' in ed_v05x.group_paths:
                ed_v05x['Sonar/Beam_group2'] = ed_v05x[
                    'Sonar/Beam_group2'
                ].sortby("channel")

            # sort the Platform group by channel for EK80 (necessary for comparison)
            tree_v06x['Platform'].ds = tree_v06x['Platform'].ds.sortby(
                'channel'
            )
            ed_v05x['Platform'] = ed_v05x['Platform'].sortby('channel')

            # remove all attributes from Vendor_specific (data is missing sometimes)
            tree_v06x["Vendor_specific"].ds.attrs = {"blank": 'None'}
            ed_v05x["Vendor_specific"].attrs = {"blank": 'None'}

        _compare_ed_against_tree(ed_v05x, tree_v06x)


def test_echodata_structure(azfp_path, ek60_path, ek80_path):
    """
    Makes sure that all raw files opened
    create the expected EchoData structure.
    """

    # TODO: create this test once dev is in its final form.
    # check and remove conversion time from attributes
    # _check_and_drop_attr(ed_v05x, tree_v06x, "Provenance", "conversion_time", str)
    # compare_ed_against_tree(ed_v05x, tree_v06x)

    pytest.xfail(
        "Full testing of the EchoData Structure has not been implemented yet."
    )
