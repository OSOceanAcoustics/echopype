import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation as R


def ek_use_platform_vertical_offsets(
    platform_ds: xr.Dataset,
    ping_time_da: xr.DataArray,
):
    """
    Use `water_level`, `vertical_offset` and `transducer_offset_z` from the EK Platform group
    to compute transducer depth.
    """
    # Grab vertical offset platform variables
    water_level = platform_ds["water_level"]
    vertical_offset = platform_ds["vertical_offset"]
    transducer_offset_z = platform_ds["transducer_offset_z"]

    # Compute z translation for transducer position vs water level
    transducer_depth = transducer_offset_z - (water_level + vertical_offset)

    # Interpolate `transducer_depth`'s `time2` dimension to `ping_time`:
    transducer_depth = transducer_depth.interp(
        {"time2": ping_time_da}, method="nearest", kwargs={"fill_value": "extrapolate"}
    ).drop_vars("time2")

    return transducer_depth


def ek_use_platform_angles(
    platform_ds: xr.Dataset,
    ping_time_da: xr.DataArray,
):
    """
    Use `pitch` and `roll` from the EK Platform group to compute echo range rotational values.
    """
    # Grab pitch and roll from platform group
    pitch = platform_ds["pitch"]
    roll = platform_ds["roll"]

    # Compute echo range scaling from pitch roll rotations
    yaw = np.zeros_like(pitch.values)
    yaw_pitch_roll_euler_angles_stack = np.column_stack([yaw, pitch.values, roll.values])
    yaw_rot_pitch_roll = R.from_euler("ZYX", yaw_pitch_roll_euler_angles_stack, degrees=True)
    echo_range_scaling = yaw_rot_pitch_roll.as_matrix()[:, -1, -1]
    echo_range_scaling = xr.DataArray(
        echo_range_scaling, dims="ping_time", coords={"ping_time": ping_time_da}
    )

    return echo_range_scaling


def ek_use_beam_angles(
    beam_ds: xr.Dataset,
    channel_da: xr.DataArray,
):
    """
    Use `beam_direction_x`, `beam_direction_y`, and `beam_direction_z` from the EK Beam group to
    compute echo range rotational values.
    """
    # Grab beam angles from beam group
    beam_direction_x = beam_ds["beam_direction_x"]
    beam_direction_y = beam_ds["beam_direction_y"]
    beam_direction_z = beam_ds["beam_direction_z"]

    # Compute echo range scaling from pitch roll rotations
    beam_dir_rotmatrix_stack = [
        [
            np.array([0, 0, beam_direction_x[c]]),
            np.array([0, 0, beam_direction_y[c]]),
            np.array([0, 0, beam_direction_z[c]]),
        ]
        for c in range(len(beam_direction_x))
    ]
    rot_beam_direction = R.from_matrix(beam_dir_rotmatrix_stack)
    echo_range_scaling = rot_beam_direction.as_matrix()[:, -1, -1]
    echo_range_scaling = xr.DataArray(
        echo_range_scaling, dims="channel", coords={"channel": channel_da}
    )

    return echo_range_scaling
