import pytest
import xarray as xr
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

from echopype.utils.align import align_to_ping_time

import echopype as ep
from echopype.consolidate.ek_depth_utils import (
    ek_use_platform_vertical_offsets, ek_use_platform_angles, ek_use_beam_angles
)


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


@pytest.mark.integration
def test_add_depth_with_external_glider_depth_and_tilt_array():
    """
    Test add_depth with external glider depth offset and tilt array data.
    """
    # Open RAW
    ed = ep.open_raw(
        raw_file="echopype/test_data/azfp/rutgers_glider_external_nc/18011107.01A",
        xml_path="echopype/test_data/azfp/rutgers_glider_external_nc/18011107.XML",
        sonar_model="azfp"
    )

    # Open external glider dataset
    glider_ds = xr.open_dataset(
        "echopype/test_data/azfp/rutgers_glider_external_nc/ru32-20180109T0531-profile-sci-delayed-subset.nc",
        engine="netcdf4"
    )

    # Grab external environment parameters
    env_params_means = {}
    for env_var in ["temperature", "salinity", "pressure"]:
        env_params_means[env_var] = float(glider_ds[env_var].mean().values)

    # Compute Sv with external environment parameters
    ds_Sv = ep.calibrate.compute_Sv(ed, env_params=env_params_means)

    # Grab pitch and roll from platform group
    pitch = np.rad2deg(glider_ds["m_pitch"])
    roll = np.rad2deg(glider_ds["m_roll"])

    # Compute tilt in degrees from pitch roll rotations
    yaw = np.zeros_like(pitch.values)
    yaw_pitch_roll_euler_angles_stack = np.column_stack([yaw, pitch.values, roll.values])
    yaw_rot_pitch_roll = R.from_euler("ZYX", yaw_pitch_roll_euler_angles_stack, degrees=True)
    glider_tilt = yaw_rot_pitch_roll.as_matrix()[:, -1, -1]
    glider_tilt = xr.DataArray(
        glider_tilt, dims="time", coords={"time": glider_ds["time"]}
    )
    glider_tilt = np.rad2deg(glider_tilt)

    # Add auxiliary depth and tilt
    ds_Sv_with_depth = ep.consolidate.add_depth(
        ds_Sv,
        depth_offset=glider_ds["depth"].dropna("time"),
        tilt=glider_tilt.dropna("time")
    )
    depth_da = ds_Sv_with_depth["depth"]

    # Align glider depth and glider tilt to ping time
    glider_depth_aligned = align_to_ping_time(
        glider_ds["depth"].dropna("time"), "time", ds_Sv["ping_time"],
    )
    glider_tilt_aligned = align_to_ping_time(
        glider_tilt.dropna("time"), "time", ds_Sv["ping_time"],
    )

    # Compute expected depth
    expected_depth_da = (
        glider_depth_aligned + (
            ds_Sv["echo_range"] * np.cos(np.deg2rad(glider_tilt_aligned))
        )
    )

    # Check that the two depth arrays are equal
    assert np.allclose(expected_depth_da, depth_da, equal_nan=True)


@pytest.mark.unit
def test_multi_dim_depth_offset_and_tilt_array_error():
    """
    Test that the correct `ValueError`s are raised when a multi-dimensional
    array is passed into `add_depth` for the `depth_offset` and `tilt`
    arguments.
    """
    # Open EK Raw file and Compute Sv
    ed = ep.open_raw(
        "echopype/test_data/ek80/ncei-wcsd/SH2106/EK80/Reduced_Hake-D20210701-T131621.raw",
        sonar_model="EK80"
    )
    ds_Sv = ep.calibrate.compute_Sv(ed, **{"waveform_mode":"CW", "encode_mode":"power"})

    # Multi-dimensional mock array
    multi_dim_da = xr.DataArray(
        np.random.randn(5, 3),
        dims=("x", "y"),
        coords={"x": range(5), "y": range(3)}
    )

    # Test `add_depth` with multi-dim `depth_offset`
    with pytest.raises(
        ValueError,
        match=(
            "If depth_offset is passed in as an xr.DataArray, "
            "it must contain a single dimension."
        )
    ):
        ep.consolidate.add_depth(ds_Sv, depth_offset=multi_dim_da)

    # Test `add_depth` with multi-dim `tilt`
    with pytest.raises(
        ValueError,
        match=(
            "If tilt is passed in as an xr.DataArray, "
            "it must contain a single dimension."
        )
    ):
        ep.consolidate.add_depth(ds_Sv, tilt=multi_dim_da)
