import math
import os
import dask
import pathlib
import tempfile

import pytest

import numpy as np
import pandas as pd
import xarray as xr
import scipy.io as io
import echopype as ep
from typing import List

from echopype.consolidate.ek_depth_utils import (
    ek_use_platform_vertical_offsets, ek_use_platform_angles, ek_use_beam_angles
)

"""
For future reference:

For ``test_add_splitbeam_angle`` the test data is in the following locations:
- the EK60 raw file is in `test_data/ek60/DY1801_EK60-D20180211-T164025.raw` and the
associated echoview split-beam data is in `test_data/ek60/splitbeam`.
- the EK80 raw file is in `test_data/ek80_bb_with_calibration/2018115-D20181213-T094600.raw` and
the associated echoview split-beam data is in `test_data/ek80_bb_with_calibration/splitbeam`
"""


@pytest.fixture(
    params=[
        (
            ("EK60", "DY1002_EK60-D20100318-T023008_rep_freq.raw"),
            "EK60",
            None,
            {},
        ),
        (
            ("EK80_NEW", "D20211004-T233354.raw"),
            "EK80",
            None,
            {'waveform_mode': 'CW', 'encode_mode': 'power'},
        ),
        (
            ("AZFP", "17082117.01A"),
            "AZFP",
            ("AZFP", "17041823.XML"),
            {},
        ),
    ],
    ids=[
        "ek60_dup_freq",
        "ek80_cw_power",
        "azfp",
    ],
)
def test_data_samples(request, test_path):
    (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    ) = request.param
    path_model, *paths = filepath
    filepath = test_path[path_model].joinpath(*paths)

    if azfp_xml_path is not None:
        path_model, *paths = azfp_xml_path
        azfp_xml_path = test_path[path_model].joinpath(*paths)

    return (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    )


def _check_swap(ds, ds_swap):
    assert "channel" in ds.dims
    assert "frequency_nominal" not in ds.dims
    assert "frequency_nominal" in ds_swap.dims
    assert "channel" not in ds_swap.dims


def test_swap_dims_channel_frequency(test_data_samples):
    """
    Test swapping dimension/coordinate from channel to frequency_nominal.
    """
    (
        filepath,
        sonar_model,
        azfp_xml_path,
        range_kwargs,
    ) = test_data_samples
    ed = ep.open_raw(filepath, sonar_model, azfp_xml_path)
    if ed.sonar_model.lower() == 'azfp':
        avg_temperature = ed['Environment']['temperature'].values.mean()
        env_params = {
            'temperature': avg_temperature,
            'salinity': 27.9,
            'pressure': 59,
        }
        range_kwargs['env_params'] = env_params
        if 'azfp_cal_type' in range_kwargs:
            range_kwargs.pop('azfp_cal_type')

    dup_freq_valueerror = (
        "Duplicated transducer nominal frequencies exist in the file. "
        "Operation is not valid."
    )

    Sv = ep.calibrate.compute_Sv(ed, **range_kwargs)
    try:
        Sv_swapped = ep.consolidate.swap_dims_channel_frequency(Sv)
        _check_swap(Sv, Sv_swapped)
    except Exception as e:
        assert isinstance(e, ValueError) is True
        assert str(e) == dup_freq_valueerror

    MVBS = ep.commongrid.compute_MVBS(Sv)
    try:
        MVBS_swapped = ep.consolidate.swap_dims_channel_frequency(MVBS)
        _check_swap(Sv, MVBS_swapped)
    except Exception as e:
        assert isinstance(e, ValueError) is True
        assert str(e) == dup_freq_valueerror


def _build_ds_Sv(channel, range_sample, ping_time, sample_interval):
    return xr.Dataset(
        data_vars={
            "Sv": (
                ("channel", "range_sample", "ping_time"),
                np.random.random((len(channel), range_sample.size, ping_time.size)),
            ),
            "echo_range": (
                ("channel", "range_sample", "ping_time"),
                (
                    np.swapaxes(np.tile(range_sample, (len(channel), ping_time.size, 1)), 1, 2)
                    * sample_interval
                ),
            ),
        },
        coords={
            "channel": channel,
            "range_sample": range_sample,
            "ping_time": ping_time,
        },
    )


@pytest.mark.unit
def test_ek_use_platform_vertical_offsets_output():
    """
    Test `use_platform_vertical_offsets` outputs.
    """
    ping_time_da = xr.DataArray(pd.date_range(start="2024-07-04", periods=5, freq="4h"), dims=("ping_time"))
    time2_da = xr.DataArray(pd.date_range(start="2024-07-04", periods=4, freq="5h"), dims=("time2"))
    platform_ds = xr.Dataset(
        {
            "water_level": xr.DataArray(
                [1.5, 0.5, 0.0, 1.0],
                dims=("time2")
            ),
            "vertical_offset": xr.DataArray(
                [1.0, 0.0, 0.0, 1.0],
                dims=("time2")
            ),
            "transducer_offset_z": xr.DataArray(
                [3.0, 1.5, 0.0, 11.15],
                dims=("time2")
            ),
        },
        coords={"time2": time2_da}
    )
    transducer_depth = ep.consolidate.ek_depth_utils.ek_use_platform_vertical_offsets(
        platform_ds,
        ping_time_da
    )

    # Check transducer depth values
    assert np.allclose(transducer_depth.values, np.array([0.5, 1.0, 0.0, 0.0, 9.15]))


@pytest.mark.unit
def test_ek_use_platform_angles_output():
    """
    Test `use_platform_angles` outputs for 2 times when the boat is completely sideways
    and 1 time when the boat has no sideways component (completely straight on z-axis).
    """
    # Create a Dataset with ping-time-specific pitch and roll arc degrees
    # (with possible range -90 deg to 90 deg).
    # During the 1st and 2nd time2, the platform is completely sideways so the echo range scaling
    # should be 0 (i.e zeros out the entire depth).
    # During the 3rd time2, the platform is completely vertical so the echo range scaling should
    # be 1 (i.e no change).
    # During the 4th time2, the platform is tilted by 45 deg so the echo range scaling should
    # be 1/sqrt(2).
    ping_time_da = xr.DataArray(pd.date_range(start="2024-07-04", periods=5, freq="4h"), dims=("ping_time"))
    time2_da = xr.DataArray(pd.date_range(start="2024-07-04", periods=4, freq="5h"), dims=("time2"))
    platform_ds = xr.Dataset(
        {
            "pitch": xr.DataArray(
                [-90, 0, 0, -45],
                dims=("time2")
            ),
            "roll": xr.DataArray(
                [0, 90, 0, 0],
                dims=("time2")
            ),
        },
        coords={"time2": time2_da}
    )
    echo_range_scaling = ep.consolidate.ek_depth_utils.ek_use_platform_angles(platform_ds, ping_time_da)

    # The two 1.0s here are from the interpolation
    assert np.allclose(echo_range_scaling.values, np.array([0.0, 0.0, 1.0, 1.0, 1/np.sqrt(2)]))


@pytest.mark.unit
def test_ek_use_beam_angles_output():
    """
    Test `use_beam_angle` outputs for 2 sideways looking beams and 1 vertical looking beam.
    """
    # Create a Dataset with channel-specific beam direction vectors
    # In channels 1 and 2, the transducers are not pointing vertically at all so the echo
    # range scaling should be 0 (i.e zeros out the entire depth).
    # In channel 3, the transducer is completely vertical so the echo range scaling should
    # be 1 (i.e no change).
    # In channel 4, the transducer is tilted to the x direction by 30 deg, so the
    # echo range scaling should be sqrt(3)/2.
    channel_da = xr.DataArray(["chan1", "chan2", "chan3", "chan4"], dims=("channel"))
    beam_ds = xr.Dataset(
        {
            "beam_direction_x": xr.DataArray([1, 0, 0, 1/2], dims=("channel")),
            "beam_direction_y": xr.DataArray([0, 1, 0, 0], dims=("channel")),
            "beam_direction_z": xr.DataArray([0, 0, 1, np.sqrt(3)/2], dims=("channel")),
        },
        coords={"channel": channel_da}
    )
    echo_range_scaling = ep.consolidate.ek_depth_utils.ek_use_beam_angles(beam_ds)
    assert np.allclose(echo_range_scaling.values, np.array([0.0, 0.0, 1.0, np.sqrt(3)/2]))


@pytest.mark.integration
@pytest.mark.parametrize("file, sonar_model, compute_Sv_kwargs", [
    (
        "echopype/test_data/ek60/NBP_B050N-D20180118-T090228.raw",
        "EK60",
        {}
    ),
    (
        "echopype/test_data/ek60/ncei-wcsd/Summer2017-D20170620-T021537.raw",
        "EK60",
        {}
    ),
    (
        "echopype/test_data/ek80/ncei-wcsd/SH1707/Reduced_D20170826-T205615.raw",
        "EK80",
        {"waveform_mode":"BB", "encode_mode":"complex"}
    ),
    (
        "echopype/test_data/ek80/ncei-wcsd/SH2106/EK80/Reduced_Hake-D20210701-T131621.raw",
        "EK80",
        {"waveform_mode":"CW", "encode_mode":"power"}
    )
])
def test_ek_depth_utils_dims(file, sonar_model, compute_Sv_kwargs):
    """
    Tests `ek_use_platform_vertical_offsets`, `ek_use_platform_angles`, and
    `ek_use_beam_angles` for correct dimensions.
    """
    # Open EK Raw file and Compute Sv
    ed = ep.open_raw(file, sonar_model=sonar_model)
    ds_Sv = ep.calibrate.compute_Sv(ed, **compute_Sv_kwargs)

    # Check dimensions for using EK platform vertical offsets to compute
    # `transducer_depth`.
    transducer_depth = ek_use_platform_vertical_offsets(
        platform_ds=ed["Platform"], ping_time_da=ds_Sv["ping_time"]
    )
    assert transducer_depth.dims == ('channel', 'ping_time')
    assert transducer_depth["channel"].equals(ds_Sv["channel"])
    assert transducer_depth["ping_time"].equals(ds_Sv["ping_time"])

    # Check dimensions for using EK platform angles to compute
    # `platform_echo_range_scaling`.
    platform_echo_range_scaling = ek_use_platform_angles(platform_ds=ed["Platform"], ping_time_da=ds_Sv["ping_time"])
    assert platform_echo_range_scaling.dims == ('ping_time',)
    assert platform_echo_range_scaling["ping_time"].equals(ds_Sv["ping_time"])

    # Check dimensions for using EK beam angles to compute `beam_echo_range_scaling`.
    beam_echo_range_scaling = ek_use_beam_angles(beam_ds=ed["Sonar/Beam_group1"])
    assert beam_echo_range_scaling.dims == ('channel',)
    assert beam_echo_range_scaling["channel"].equals(ds_Sv["channel"])


@pytest.mark.integration
def test_ek_depth_utils_group_variable_NaNs_logger_warnings(caplog):
    """
    Tests `ek_use_platform_vertical_offsets`, `ek_use_platform_angles`, and
    `ek_use_beam_angles` for correct logger warnings when NaNs exist in group
    variables.
    """
    # Open EK Raw file and Compute Sv
    ed = ep.open_raw(
        "echopype/test_data/ek80/ncei-wcsd/SH2106/EK80/Reduced_Hake-D20210701-T131621.raw",
        sonar_model="EK80"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed, **{"waveform_mode":"CW", "encode_mode":"power"})

    # Set first index of group variables to NaN
    ed["Platform"]["water_level"] = np.nan # Is a scalar
    ed["Platform"]["vertical_offset"].values[0] = np.nan
    ed["Platform"]["transducer_offset_z"].values[0] = np.nan
    ed["Platform"]["pitch"].values[0] = np.nan
    ed["Platform"]["roll"].values[0] = np.nan
    ed["Sonar/Beam_group1"]["beam_direction_x"].values[0] = np.nan
    ed["Sonar/Beam_group1"]["beam_direction_y"].values[0] = np.nan
    ed["Sonar/Beam_group1"]["beam_direction_z"].values[0] = np.nan

    # Turn on logger verbosity
    ep.utils.log.verbose(override=False)

    # Run EK depth util functions:
    ek_use_platform_vertical_offsets(platform_ds=ed["Platform"], ping_time_da=ds_Sv["ping_time"])
    ek_use_platform_angles(platform_ds=ed["Platform"], ping_time_da=ds_Sv["ping_time"])
    ek_use_beam_angles(beam_ds=ed["Sonar/Beam_group1"])

    # Set (group, variable) name pairs
    group_variable_name_pairs = [
        ["Platform", "water_level"],
        ["Platform", "vertical_offset"],
        ["Platform", "transducer_offset_z"],
        ["Platform", "pitch"],
        ["Platform", "roll"],
        ["Sonar/Beam_group1", "beam_direction_x"],
        ["Sonar/Beam_group1", "beam_direction_y"],
        ["Sonar/Beam_group1", "beam_direction_z"],
    ]

    # Check if the expected warnings are logged
    for group_name, variable_name in group_variable_name_pairs:
        expected_warning = (
            f"The Echodata `{group_name}` group `{variable_name}` variable array contains "
            "NaNs. This will result in NaNs in the final `depth` array. Consider filling the "
            "NaNs and calling `.add_depth(...)` again."
        )
        assert any(expected_warning in record.message for record in caplog.records)

    # Turn off logger verbosity
    ep.utils.log.verbose(override=True)


@pytest.mark.integration
def test_add_depth_tilt_depth_use_arg_logger_warnings(caplog):
    """
    Tests warnings when `tilt` and `depth_offset` are being passed in when other
    `use_*` arguments are passed in as `True`.
    """
    # Open EK Raw file and Compute Sv
    ed = ep.open_raw(
        "echopype/test_data/ek80/ncei-wcsd/SH2106/EK80/Reduced_Hake-D20210701-T131621.raw",
        sonar_model="EK80"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed, **{"waveform_mode":"CW", "encode_mode":"power"})

    # Turn on logger verbosity
    ep.utils.log.verbose(override=False)

    # Run `add_depth` with `tilt`, `depth_offset` as Non-NaN, using beam group angles,
    # and platform vertical offset values
    ep.consolidate.add_depth(
        ds_Sv,
        ed,
        depth_offset=9.15,
        tilt=0.1,
        use_platform_vertical_offsets=True,
        use_beam_angles=True,
    )

    # Check if the expected warnings are logged
    depth_offset_warning = (
        "When `depth_offset` is specified, platform vertical offset "
        "variables will not be used."
    )
    tilt_warning = (
        "When `tilt` is specified, beam/platform angle variables will not be used."
    )
    for warning in [depth_offset_warning, tilt_warning]:
        assert any(warning in record.message for record in caplog.records)

    # Turn off logger verbosity
    ep.utils.log.verbose(override=True)


@pytest.mark.integration
def test_add_depth_without_echodata():
    """
    Test `add_depth` without using Echodata Platform or Beam groups.
    """
    # Build test Sv dataset
    channel = ["channel_0", "channel_1", "channel_2"]
    range_sample = np.arange(100)
    ping_time = pd.date_range(start="2022-08-10T10:00:00", end="2022-08-10T12:00:00", periods=121)
    sample_interval = 0.01
    ds_Sv = _build_ds_Sv(channel, range_sample, ping_time, sample_interval)

    # User input `depth_offset`
    water_level = 10
    ds_Sv_depth = ep.consolidate.add_depth(ds_Sv, depth_offset=water_level)
    assert ds_Sv_depth["depth"].equals(ds_Sv["echo_range"] + water_level)

    # User input `depth_offset` and `tilt`
    tilt = 15
    ds_Sv_depth = ep.consolidate.add_depth(ds_Sv, depth_offset=water_level, tilt=tilt)
    assert ds_Sv_depth["depth"].equals(ds_Sv["echo_range"] * np.cos(tilt / 180 * np.pi) + water_level)

    # Inverted echosounder with `depth_offset` and `tilt`
    ds_Sv_depth = ep.consolidate.add_depth(ds_Sv, depth_offset=water_level, tilt=tilt, downward=False)
    assert ds_Sv_depth["depth"].equals(-1 * ds_Sv["echo_range"] * np.cos(tilt / 180 * np.pi) + water_level)

    # Check history attribute
    history_attribute = ds_Sv_depth["depth"].attrs["history"]
    history_attribute_without_time = history_attribute[33:]
    assert history_attribute_without_time == ". depth` calculated using: Sv `echo_range`."


@pytest.mark.integration
def test_add_depth_errors():
    """Check if all `add_depth` errors are raised appropriately."""
    # Open EK80 Raw file and Compute Sv
    ed = ep.open_raw(
        "echopype/test_data/ek80/ncei-wcsd/SH2106/EK80/Reduced_Hake-D20210701-T131621.raw",
        sonar_model="EK80"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed, **{"waveform_mode":"CW", "encode_mode":"power"})

    # Test that all three errors are called:
    with pytest.raises(ValueError, match=(
        "If any of `use_platform_vertical_offsets`, "
        + "`use_platform_angles` "
        + "or `use_beam_angles` is `True`, "
        + "then `echodata` cannot be `None`."
    )):
        ep.consolidate.add_depth(ds_Sv, None, use_platform_angles=True)
    with pytest.raises(NotImplementedError, match=(
        "Computing depth with both platform and beam angles is not implemented yet."
    )):
        ep.consolidate.add_depth(ds_Sv, ed, use_platform_angles=True, use_beam_angles=True)
    with pytest.raises(NotImplementedError, match=(
        "`use_platform/beam_...` not implemented yet for `AZFP`."
    )):
        ed["Sonar"].attrs["sonar_model"] = "AZFP"
        ep.consolidate.add_depth(ds_Sv, ed, use_platform_angles=True)


@pytest.mark.integration
@pytest.mark.parametrize("file, sonar_model, compute_Sv_kwargs", [
    (
        "echopype/test_data/ek60/NBP_B050N-D20180118-T090228.raw",
        "EK60",
        {}
    ),
    (
        "echopype/test_data/ek60/ncei-wcsd/Summer2017-D20170620-T021537.raw",
        "EK60",
        {}
    ),
    (
        "echopype/test_data/ek80/ncei-wcsd/SH1707/Reduced_D20170826-T205615.raw",
        "EK80",
        {"waveform_mode":"BB", "encode_mode":"complex"}
    ),
    (
        "echopype/test_data/ek80/ncei-wcsd/SH2106/EK80/Reduced_Hake-D20210701-T131621.raw",
        "EK80",
        {"waveform_mode":"CW", "encode_mode":"power"}
    )
])
def test_add_depth_EK_with_platform_vertical_offsets(file, sonar_model, compute_Sv_kwargs):
    """Test `depth` values when using EK Platform vertical offset values to compute it."""
    # Open EK Raw file and Compute Sv
    ed = ep.open_raw(file, sonar_model=sonar_model)
    ds_Sv = ep.calibrate.compute_Sv(ed, **compute_Sv_kwargs)

    # Subset ds_Sv to include only first 5 `range_sample` coordinates
    # since the test takes too long to iterate through every value
    ds_Sv = ds_Sv.isel(range_sample=slice(0,5))

    # Replace any Platform Vertical Offset NaN values with 0
    ed["Platform"]["water_level"] = ed["Platform"]["water_level"].fillna(0)
    ed["Platform"]["vertical_offset"] = ed["Platform"]["vertical_offset"].fillna(0)
    ed["Platform"]["transducer_offset_z"] = ed["Platform"]["transducer_offset_z"].fillna(0)

    # Compute `depth` using platform vertical offset values
    ds_Sv_with_depth = ep.consolidate.add_depth(ds_Sv, ed, use_platform_vertical_offsets=True)

    # Check history attribute
    history_attribute = ds_Sv_with_depth["depth"].attrs["history"]
    history_attribute_without_time = history_attribute[33:]
    assert history_attribute_without_time == (
        ". depth` calculated using: Sv `echo_range`, Echodata `Platform` Vertical Offsets."
    )

    # Compute transducer depth
    transducer_depth = ek_use_platform_vertical_offsets(ed["Platform"], ds_Sv["ping_time"])
    
    # Check if depth value is equal to corresponding `echo_range` value + transducer depth value
    assert np.allclose(
        ds_Sv_with_depth["depth"].data,
        (ds_Sv["echo_range"] + transducer_depth).data,
        rtol=1e-10,
        atol=1e-10
    )


@pytest.mark.integration
@pytest.mark.parametrize("file, sonar_model, compute_Sv_kwargs", [
    (
        "echopype/test_data/ek60/NBP_B050N-D20180118-T090228.raw",
        "EK60",
        {}
    ),
    (
        "echopype/test_data/ek60/ncei-wcsd/Summer2017-D20170620-T021537.raw",
        "EK60",
        {}
    ),
    (
        "echopype/test_data/ek80/ncei-wcsd/SH1707/Reduced_D20170826-T205615.raw",
        "EK80",
        {"waveform_mode":"BB", "encode_mode":"complex"}
    ),
    (
        "echopype/test_data/ek80/ncei-wcsd/SH2106/EK80/Reduced_Hake-D20210701-T131621.raw",
        "EK80",
        {"waveform_mode":"CW", "encode_mode":"power"}
    )
])
def test_add_depth_EK_with_platform_angles(file, sonar_model, compute_Sv_kwargs):
    """Test `depth` values when using EK Platform angles to compute it."""
    # Open EK Raw file and Compute Sv
    ed = ep.open_raw(file, sonar_model=sonar_model)
    ds_Sv = ep.calibrate.compute_Sv(ed, **compute_Sv_kwargs)

    # Replace any Beam Angle NaN values with 0
    ed["Platform"]["pitch"] = ed["Platform"]["pitch"].fillna(0)
    ed["Platform"]["roll"] = ed["Platform"]["roll"].fillna(0)

    # Compute `depth` using platform angle values
    ds_Sv_with_depth = ep.consolidate.add_depth(ds_Sv, ed, use_platform_angles=True)

    # Check history attribute
    history_attribute = ds_Sv_with_depth["depth"].attrs["history"]
    history_attribute_without_time = history_attribute[33:]
    assert history_attribute_without_time == (
        ". depth` calculated using: Sv `echo_range`, Echodata `Platform` Angles."
    )

    # Compute transducer depth
    echo_range_scaling = ek_use_platform_angles(ed["Platform"], ds_Sv["ping_time"])

    # Check if depth is equal to echo range scaling value * echo range
    assert np.allclose(
        ds_Sv_with_depth["depth"].data,
        (echo_range_scaling * ds_Sv["echo_range"]).transpose("channel", "ping_time", "range_sample").data, 
        equal_nan=True
    )


@pytest.mark.integration
@pytest.mark.parametrize("file, sonar_model, compute_Sv_kwargs", [
    (
        "echopype/test_data/ek60/NBP_B050N-D20180118-T090228.raw",
        "EK60",
        {}
    ),
    (
        "echopype/test_data/ek60/ncei-wcsd/Summer2017-D20170620-T021537.raw",
        "EK60",
        {}
    ),
    (
        "echopype/test_data/ek80/ncei-wcsd/SH1707/Reduced_D20170826-T205615.raw",
        "EK80",
        {"waveform_mode":"BB", "encode_mode":"complex"}
    ),
    (
        "echopype/test_data/ek80/ncei-wcsd/SH2106/EK80/Reduced_Hake-D20210701-T131621.raw",
        "EK80",
        {"waveform_mode":"CW", "encode_mode":"power"}
    )
])
def test_add_depth_EK_with_beam_angles(file, sonar_model, compute_Sv_kwargs):
    """Test `depth` values when using EK Beam angles to compute it."""
    # Open EK Raw file and Compute Sv
    ed = ep.open_raw(file, sonar_model=sonar_model)
    ds_Sv = ep.calibrate.compute_Sv(ed, **compute_Sv_kwargs)

    # Replace Beam Angle NaN values
    ed["Sonar/Beam_group1"]["beam_direction_x"] = ed["Sonar/Beam_group1"]["beam_direction_x"].fillna(0)
    ed["Sonar/Beam_group1"]["beam_direction_y"] = ed["Sonar/Beam_group1"]["beam_direction_y"].fillna(0)
    ed["Sonar/Beam_group1"]["beam_direction_z"] = ed["Sonar/Beam_group1"]["beam_direction_z"].fillna(1)

    # Compute `depth` using beam angle values
    ds_Sv_with_depth = ep.consolidate.add_depth(ds_Sv, ed, use_beam_angles=True)

    # Check history attribute
    history_attribute = ds_Sv_with_depth["depth"].attrs["history"]
    history_attribute_without_time = history_attribute[33:]
    assert history_attribute_without_time == (
        ". depth` calculated using: Sv `echo_range`, Echodata `Beam_group1` Angles."
    )

    # Compute echo range scaling values
    echo_range_scaling = ek_use_beam_angles(ed["Sonar/Beam_group1"])

    # Check if depth is equal to echo range scaling value * echo range
    assert np.allclose(
        ds_Sv_with_depth["depth"].data,
        (echo_range_scaling * ds_Sv["echo_range"]).transpose("channel", "ping_time", "range_sample").data, 
        equal_nan=True
    )


@pytest.mark.integration
@pytest.mark.parametrize("file, sonar_model, compute_Sv_kwargs, expected_beam_group_name", [
    (
        "echopype/test_data/ek80/Summer2018--D20180905-T033113.raw",
        "EK80",
        {"waveform_mode":"BB", "encode_mode":"complex"},
        "Beam_group1"
    ),
    (
        "echopype/test_data/ek80/Summer2018--D20180905-T033113.raw",
        "EK80",
        {"waveform_mode":"CW", "encode_mode":"power"},
        "Beam_group2"
    )
])
def test_add_depth_EK_with_beam_angles_with_different_beam_groups(
    file, sonar_model, compute_Sv_kwargs, expected_beam_group_name
):
    """
    Test `depth` channel when using EK Beam angles from two separate calibrated
    Sv datasets (that are from the same raw file) using two differing pairs of
    calibration key word arguments. The two tests should correspond to different
    beam groups i.e. beam group 1 and beam group 2.
    """
    # Open EK Raw file and Compute Sv
    ed = ep.open_raw(file, sonar_model=sonar_model)
    ds_Sv = ep.calibrate.compute_Sv(ed, **compute_Sv_kwargs)

    # Compute `depth` using beam angle values
    ds_Sv = ep.consolidate.add_depth(ds_Sv, ed, use_beam_angles=True)

    # Check history attribute
    history_attribute = ds_Sv["depth"].attrs["history"]
    history_attribute_without_time = history_attribute[33:]
    assert history_attribute_without_time == (
        f". depth` calculated using: Sv `echo_range`, Echodata `{expected_beam_group_name}` Angles."
    )


def _create_array_list_from_echoview_mats(paths_to_echoview_mat: List[pathlib.Path]) -> List[np.ndarray]:
    """
    Opens each mat file in ``paths_to_echoview_mat``, selects the first ``ping_time``,
    and then stores the array in a list.

    Parameters
    ----------
    paths_to_echoview_mat: list of pathlib.Path
        A list of paths corresponding to mat files, where each mat file contains the
        echoview generated angle alongship and athwartship data for a channel

    Returns
    -------
    list of np.ndarray
        A list of numpy arrays generated by choosing the appropriate data from the mat files.
        This list will have the same length as ``paths_to_echoview_mat``
    """

    list_of_mat_arrays = []
    for mat_file in paths_to_echoview_mat:

        # open mat file and grab appropriate data
        list_of_mat_arrays.append(io.loadmat(file_name=mat_file)["P0"]["Data_values"][0][0])

    return list_of_mat_arrays


@pytest.mark.integration
@pytest.mark.parametrize(
    ["location_type", "sonar_model", "path_model", "raw_and_xml_paths", "lat_lon_name_dict", "extras"],
    [
        (
            "empty-location",
            "EK60",
            "EK60",
            ("ooi/CE02SHBP-MJ01C-07-ZPLSCB101_OOI-D20191201-T000000.raw", None),
            {"lat_name": "latitude", "lon_name": "longitude"},
            None,
        ),
        (
            "with-track-location",
            "EK60",
            "EK60",
            ("Winter2017-D20170115-T150122.raw", None),
            {"lat_name": "latitude", "lon_name": "longitude"},
            None,
        ),
        (
            "fixed-location",
            "AZFP",
            "AZFP",
            ("17082117.01A", "17041823.XML"),
            {"lat_name": "latitude", "lon_name": "longitude"},
            {'longitude': -60.0, 'latitude': 45.0, 'salinity': 27.9, 'pressure': 59},
        ),
    ],
)
def test_add_location(
        location_type,
        sonar_model,
        path_model,
        raw_and_xml_paths,
        lat_lon_name_dict,
        extras,
        test_path
):
    # Prepare the Sv dataset
    raw_path = test_path[path_model] / raw_and_xml_paths[0]
    if raw_and_xml_paths[1]:
        xml_path = test_path[path_model] / raw_and_xml_paths[1]
    else:
        xml_path = None

    ed = ep.open_raw(raw_path, xml_path=xml_path, sonar_model=sonar_model)
    if location_type == "fixed-location":
        point_ds = xr.Dataset(
            {
                lat_lon_name_dict["lat_name"]: (["time"], np.array([float(extras['latitude'])])),
                lat_lon_name_dict["lon_name"]: (["time"], np.array([float(extras['longitude'])])),
            },
            coords={
                "time": (["time"], np.array([ed["Sonar/Beam_group1"]["ping_time"].values.min()]))
            },
        )
        ed.update_platform(
            point_ds,
            variable_mappings={
                lat_lon_name_dict["lat_name"]: lat_lon_name_dict["lat_name"],
                lat_lon_name_dict["lon_name"]: lat_lon_name_dict["lon_name"]
            }
        )

    env_params = None
    # AZFP data require external salinity and pressure
    if sonar_model == "AZFP":
        env_params = {
            "temperature": ed["Environment"]["temperature"].values.mean(),
            "salinity": extras["salinity"],
            "pressure": extras["pressure"],
        }

    ds = ep.calibrate.compute_Sv(echodata=ed, env_params=env_params)

    # add_location tests
    if location_type == "empty-location":
        with pytest.raises(Exception) as exc:
            ep.consolidate.add_location(ds=ds, echodata=ed)
        assert exc.type is ValueError
        assert "Coordinate variables not present or all nan" in str(exc.value)
    else:
        def _tests(ds_test, location_type, nmea_sentence=None):
            # lat,lon & time1 existence
            assert "latitude" in ds_test
            assert "longitude" in ds_test
            assert "time1" not in ds_test

            # lat & lon have a single dimension: 'ping_time'
            assert len(ds_test["latitude"].dims) == 1 and ds_test["latitude"].dims[0] == "ping_time" # noqa
            assert len(ds_test["longitude"].dims) == 1 and ds_test["longitude"].dims[0] == "ping_time" # noqa

            # Check interpolated or broadcast values
            if location_type == "with-track-location":
                for ed_position, ds_position in [
                    (lat_lon_name_dict["lat_name"], "latitude"),
                    (lat_lon_name_dict["lon_name"], "longitude")
                ]:
                    position_var = ed["Platform"][ed_position]
                    if nmea_sentence:
                        position_var = position_var[ed["Platform"]["sentence_type"] == nmea_sentence]
                    position_interp = position_var.interp(time1=ds_test["ping_time"])
                    # interpolated values are identical
                    assert np.allclose(ds_test[ds_position].values, position_interp.values, equal_nan=True) # noqa
            elif location_type == "fixed-location":
                for position in ["latitude", "longitude"]:
                    position_uniq = set(ds_test[position].values)
                    # contains a single repeated value equal to the value passed to update_platform
                    assert (
                            len(position_uniq) == 1 and
                            math.isclose(list(position_uniq)[0], extras[position])
                    )

        ds_all = ep.consolidate.add_location(ds=ds, echodata=ed)
        _tests(ds_all, location_type)

        # the test for nmea_sentence="GGA" is limited to the with-track-location case
        if location_type == "with-track-location" and sonar_model.startswith("EK"):
            ds_sel = ep.consolidate.add_location(ds=ds, echodata=ed, nmea_sentence="GGA")
            _tests(ds_sel, location_type, nmea_sentence="GGA")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("raw_path, sonar_model, datagram_type, time_dim_name, compute_Sv_kwargs"),
    [
        (
            "echopype/test_data/ek80/D20170912-T234910.raw",
            "EK80",
            "NMEA",
            "time1",
            {
                "waveform_mode": "BB",
                "encode_mode": "complex"
            }
        ),
        (
            "echopype/test_data/ek80/RL2407_ADCP-D20240709-T150437.raw",
            "EK80",
            "MRU1",
            "time3",
            {
                "waveform_mode": "CW",
                "encode_mode": "complex"
            }
        ),
        (
            "echopype/test_data/ek80/idx_bot/Hake-D20230711-T181910.raw",
            "EK80",
            "IDX",
            "time4",
            {
                "waveform_mode": "CW",
                "encode_mode": "power"
            }
        ),
    ],
)
def test_add_location_time_duplicates_value_error(
    raw_path, sonar_model, datagram_type, time_dim_name, compute_Sv_kwargs,
):   
    """Tests for duplicate time value error in ``add_location``.""" 
    # Open raw and compute the Sv dataset
    if not datagram_type == "IDX":
        ed = ep.open_raw(raw_path, sonar_model=sonar_model)
    else:
        ed = ep.open_raw(raw_path, include_idx=True, sonar_model=sonar_model)
    ds = ep.calibrate.compute_Sv(
        echodata=ed,
        **compute_Sv_kwargs,
    )

    # Add duplicates to `time_dim`
    ed["Platform"][time_dim_name].data[0] = ed["Platform"][time_dim_name].data[1]
    
    # Check if the expected error is logged
    with pytest.raises(ValueError) as exc_info:
        # Run add location with duplicated time
        ep.consolidate.add_location(ds=ds, echodata=ed, datagram_type=datagram_type)

    # Check if the specific error message is in the logs
    assert (
        f'The ``echodata["Platform"]["{time_dim_name}"]`` array contains duplicate values. '
        "Downstream interpolation on the position variables requires unique time values."
    ) == str(exc_info.value)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("raw_path, sonar_model, datagram_type, time_dim_name, compute_Sv_kwargs"),
    [
        (
            "echopype/test_data/ek80/D20170912-T234910.raw",
            "EK80",
            "NMEA",
            "time1",
            {
                "waveform_mode": "BB",
                "encode_mode": "complex"
            }
        ),
        (
            "echopype/test_data/ek80/RL2407_ADCP-D20240709-T150437.raw",
            "EK80",
            "MRU1",
            "time3",
            {
                "waveform_mode": "CW",
                "encode_mode": "complex"
            }
        ),
        (
            "echopype/test_data/ek80/idx_bot/Hake-D20230711-T181910.raw",
            "EK80",
            "IDX",
            "time4",
            {
                "waveform_mode": "CW",
                "encode_mode": "power"
            }
        ),
    ],
)
def test_add_location_lat_lon_0_NaN_warnings(
    raw_path, sonar_model, datagram_type, time_dim_name, compute_Sv_kwargs, caplog
):
    """Tests for lat lon 0 and NaN value warnings in ``add_warning``."""
    # Open raw and compute the Sv dataset
    if not datagram_type == "IDX":
        ed = ep.open_raw(raw_path, sonar_model=sonar_model)
    else:
        ed = ep.open_raw(raw_path, include_idx=True, sonar_model=sonar_model)
    ds = ep.calibrate.compute_Sv(
        echodata=ed,
        **compute_Sv_kwargs,
    )
    
    # Add NaN to latitude and 0 to longitude
    if datagram_type in ["MRU1", "IDX"]:
        ed["Platform"][f"latitude_{datagram_type.lower()}"][0] = np.nan
        ed["Platform"][f"longitude_{datagram_type.lower()}"][0] = 0
    else:
        ed["Platform"]["latitude"][0] = np.nan
        ed["Platform"]["longitude"][0] = 0

    # Turn on logger verbosity
    ep.utils.log.verbose(override=False)

    # Run add location with 0 and NaN lat/lon values
    ep.consolidate.add_location(ds=ds, echodata=ed, datagram_type=datagram_type)
    
    # Check if the expected warnings are logged
    interp_msg = (
        "Interpolation may be negatively impacted, "
        "consider handling these values before calling ``add_location``."
    )
    expected_warnings = [
        f"Latitude and/or longitude arrays contain NaNs. {interp_msg}",
        f"Latitude and/or longitude arrays contain zeros. {interp_msg}"
    ]
    for warning in expected_warnings:
        assert any(warning in record.message for record in caplog.records)
    
    # Turn off logger verbosity
    ep.utils.log.verbose(override=True)


@pytest.mark.parametrize(
    ("sonar_model", "test_path_key", "raw_file_name", "paths_to_echoview_mat",
     "waveform_mode", "encode_mode", "pulse_compression", "to_disk"),
    [
        # ek60_CW_power
        (
            "EK60", "EK60", "DY1801_EK60-D20180211-T164025.raw",
            [
                'splitbeam/DY1801_EK60-D20180211-T164025_angles_T1.mat',
                'splitbeam/DY1801_EK60-D20180211-T164025_angles_T2.mat',
                'splitbeam/DY1801_EK60-D20180211-T164025_angles_T3.mat',
                'splitbeam/DY1801_EK60-D20180211-T164025_angles_T4.mat',
                'splitbeam/DY1801_EK60-D20180211-T164025_angles_T5.mat'
            ],
            "CW", "power", False, False
        ),
        # ek60_CW_power_Sv_path
        (
            "EK60", "EK60", "DY1801_EK60-D20180211-T164025.raw",
            [
                'splitbeam/DY1801_EK60-D20180211-T164025_angles_T1.mat',
                'splitbeam/DY1801_EK60-D20180211-T164025_angles_T2.mat',
                'splitbeam/DY1801_EK60-D20180211-T164025_angles_T3.mat',
                'splitbeam/DY1801_EK60-D20180211-T164025_angles_T4.mat',
                'splitbeam/DY1801_EK60-D20180211-T164025_angles_T5.mat'
            ],
            "CW", "power", False, True
        ),
        # ek80_CW_complex
        (
            "EK80", "EK80_CAL", "2018115-D20181213-T094600.raw",
            [
                'splitbeam/2018115-D20181213-T094600_angles_T1.mat',
                'splitbeam/2018115-D20181213-T094600_angles_T4.mat',
                'splitbeam/2018115-D20181213-T094600_angles_T6.mat',
                'splitbeam/2018115-D20181213-T094600_angles_T5.mat'
            ],
            "CW", "complex", False, False
        ),
        # ek80_BB_complex_no_pc
        (
            "EK80", "EK80_CAL", "2018115-D20181213-T094600.raw",
            [
                'splitbeam/2018115-D20181213-T094600_angles_T3_nopc.mat',
                'splitbeam/2018115-D20181213-T094600_angles_T2_nopc.mat',
            ],
            "BB", "complex", False, False,
        ),
        # ek80_CW_power
        (
            "EK80", "EK80", "Summer2018--D20180905-T033113.raw",
            [
                'splitbeam/Summer2018--D20180905-T033113_angles_T2.mat',
                'splitbeam/Summer2018--D20180905-T033113_angles_T1.mat',
            ],
            "CW", "power", False, False,
        ),
    ],
    ids=[
        "ek60_CW_power",
        "ek60_CW_power_Sv_path",
        "ek80_CW_complex",
        "ek80_BB_complex_no_pc",
        "ek80_CW_power",
    ],
)
def test_add_splitbeam_angle(sonar_model, test_path_key, raw_file_name, test_path,
                             paths_to_echoview_mat, waveform_mode, encode_mode,
                             pulse_compression, to_disk):

    # obtain the EchoData object with the data needed for the calculation
    ed = ep.open_raw(test_path[test_path_key] / raw_file_name, sonar_model=sonar_model)

    # compute Sv as it is required for the split-beam angle calculation
    ds_Sv = ep.calibrate.compute_Sv(ed, waveform_mode=waveform_mode, encode_mode=encode_mode)

    # initialize temporary directory object
    temp_dir = None

    # allows us to test for the case when source_Sv is a path
    if to_disk:

        # create temporary directory for mask_file
        temp_dir = tempfile.TemporaryDirectory()

        # write DataArray to temporary directory
        zarr_path = os.path.join(temp_dir.name, "Sv_data.zarr")
        ds_Sv.to_zarr(zarr_path)

        # assign input to a path
        ds_Sv = zarr_path

    # add the split-beam angles to Sv dataset
    ds_Sv = ep.consolidate.add_splitbeam_angle(source_Sv=ds_Sv, echodata=ed,
                                               waveform_mode=waveform_mode,
                                               encode_mode=encode_mode,
                                               pulse_compression=pulse_compression,
                                               to_disk=to_disk)

    if to_disk:
        assert isinstance(ds_Sv["angle_alongship"].data, dask.array.core.Array)
        assert isinstance(ds_Sv["angle_athwartship"].data, dask.array.core.Array)

    # obtain corresponding echoview output
    full_echoview_path = [test_path[test_path_key] / path for path in paths_to_echoview_mat]
    echoview_arr_list = _create_array_list_from_echoview_mats(full_echoview_path)

    # compare echoview output against computed output for all channels
    for chan_ind in range(len(echoview_arr_list)):

        # grabs the appropriate ds data to compare against
        reduced_angle_alongship = ds_Sv.isel(channel=chan_ind, ping_time=0).angle_alongship.dropna("range_sample")
        reduced_angle_athwartship = ds_Sv.isel(channel=chan_ind, ping_time=0).angle_athwartship.dropna("range_sample")

        # TODO: make "start" below a parameter in the input so that this is not ad-hoc but something known
        # for some files the echoview data is shifted by one index, here we account for that
        if reduced_angle_alongship.shape == (echoview_arr_list[chan_ind].shape[1], ):
            start = 0
        else:
            start = 1

        # note for the checks below:
        #   - angles from CW power data are similar down to 1e-7
        #   - angles computed from complex samples deviates a lot more

        # check the computed angle_alongship values against the echoview output
        assert np.allclose(reduced_angle_alongship.values[start:],
                           echoview_arr_list[chan_ind][0, :], rtol=1e-1, atol=1e-2)

        # check the computed angle_alongship values against the echoview output
        assert np.allclose(reduced_angle_athwartship.values[start:],
                           echoview_arr_list[chan_ind][1, :], rtol=1e-1, atol=1e-2)

    if temp_dir:
        # remove the temporary directory, if it was created
        temp_dir.cleanup()


def test_add_splitbeam_angle_BB_pc(test_path):

    # obtain the EchoData object with the data needed for the calculation
    ed = ep.open_raw(test_path["EK80_CAL"] / "2018115-D20181213-T094600.raw", sonar_model="EK80")

    # compute Sv as it is required for the split-beam angle calculation
    ds_Sv = ep.calibrate.compute_Sv(ed, waveform_mode="BB", encode_mode="complex")

    # add the split-beam angles to Sv dataset
    ds_Sv = ep.consolidate.add_splitbeam_angle(
        source_Sv=ds_Sv, echodata=ed,
        waveform_mode="BB", encode_mode="complex", pulse_compression=True,
        to_disk=False
    )

    # Load pyecholab pickle
    import pickle
    with open(test_path["EK80_EXT"] / "pyecholab/pyel_BB_splitbeam.pickle", 'rb') as handle:
        pyel_BB_p_data = pickle.load(handle)

    # Compare 70kHz channel
    chan_sel = "WBT 714590-15 ES70-7C"

    # Compare cal params
    # dict mappgin:  {pyecholab : echopype}
    cal_params_dict = {
        "angle_sensitivity_alongship": "angle_sensitivity_alongship",
        "angle_sensitivity_athwartship": "angle_sensitivity_athwartship",
        "beam_width_alongship": "beamwidth_alongship",
        "beam_width_athwartship": "beamwidth_athwartship",
    }
    for p_pyel, p_ep in cal_params_dict.items():
        assert np.allclose(pyel_BB_p_data["cal_parms"][p_pyel],
                           ds_Sv[p_ep].sel(channel=chan_sel).values)

    # alongship angle
    pyel_vals = pyel_BB_p_data["alongship_physical"]
    ep_vals = ds_Sv["angle_alongship"].sel(channel=chan_sel).values
    assert pyel_vals.shape == ep_vals.shape
    assert np.allclose(pyel_vals, ep_vals, atol=1e-5)

    # athwartship angle
    pyel_vals = pyel_BB_p_data["athwartship_physical"]
    ep_vals = ds_Sv["angle_athwartship"].sel(channel=chan_sel).values
    assert pyel_vals.shape == ep_vals.shape
    assert np.allclose(pyel_vals, ep_vals, atol=1e-6)


# TODO: need a test for power/angle data, with mock EchoData object
# containing some channels with single-beam data and some channels with split-beam data
