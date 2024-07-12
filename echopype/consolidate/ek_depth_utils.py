import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation as R

from ..utils.log import _init_logger

logger = _init_logger(__name__)


def _check_and_log_nans(
    echodata_group: xr.Dataset, group_name: str, variable_names: list[str]
) -> None:
    """
    Checks for NaNs in Echodata group variables and raises logger warning.
    """
    # Iterate through group variable names
    for variable_name in variable_names:
        # Extract group and check if it contains any NaNs
        group_var = echodata_group[variable_name]
        # Log warning if the group variable contains any NaNs
        if np.any(np.isnan(group_var.values)):
            logger.warning(
                f"The Echodata `{group_name}` group `{variable_name}` variable array contains "
                "NaNs. This will result in NaNs in the final `depth` array. Consider filling the "
                "NaNs and calling `.add_depth(...)` again."
            )


def _var_time2_to_ping_time(var_with_time2, ping_time_da):
    """
    If `time2` does not differ from `var`, we rename `time2` to 'ping_time',
    else interpolate `transducer_depth`'s `time2` dimension to `ping_time`.

    Useful for handling EK60 and EK80 platform data:

    EK80 will have platform variables with time2 dimension that does not match
    Beam group ping time values, while EK60 will have time2 dimension that
    matches Beam group ping time values.
    """
    if not ping_time_da.equals(var_with_time2["time2"].rename({"time2": "ping_time"})):
        var_with_ping_time = var_with_time2.interp(
            {"time2": ping_time_da},
            method="nearest",
            # More details for `fill_value` and `extrapolate` can be found here:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html # noqa
            kwargs={"fill_value": "extrapolate"},
        ).drop_vars("time2")
    else:
        var_with_ping_time = var_with_time2.rename({"time2": "ping_time"})

    return var_with_ping_time


def ek_use_platform_vertical_offsets(
    platform_ds: xr.Dataset,
    ping_time_da: xr.DataArray,
) -> xr.DataArray:
    """
    Use `water_level`, `vertical_offset` and `transducer_offset_z` from the EK Platform group
    to compute transducer depth.
    """
    # Check and log NaNs if they exist in the Platform group variables
    _check_and_log_nans(
        platform_ds, "Platform", ["water_level", "vertical_offset", "transducer_offset_z"]
    )

    # Grab vertical offset platform variables
    water_level = platform_ds["water_level"]
    vertical_offset = platform_ds["vertical_offset"]
    transducer_offset_z = platform_ds["transducer_offset_z"]

    # Compute z translation for transducer position vs water level
    transducer_depth = transducer_offset_z - (water_level + vertical_offset)

    return _var_time2_to_ping_time(transducer_depth, ping_time_da)


def ek_use_platform_angles(platform_ds: xr.Dataset, ping_time_da: xr.DataArray) -> xr.DataArray:
    """
    Use `pitch` and `roll` from the EK Platform group to compute echo range rotational values.
    """
    # Check and log NaNs if they exist in the Platform group variables
    _check_and_log_nans(platform_ds, "Platform", ["pitch", "roll"])

    # Grab pitch and roll from platform group
    pitch = platform_ds["pitch"]
    roll = platform_ds["roll"]

    # Compute echo range scaling from pitch roll rotations
    yaw = np.zeros_like(pitch.values)
    yaw_pitch_roll_euler_angles_stack = np.column_stack([yaw, pitch.values, roll.values])
    yaw_rot_pitch_roll = R.from_euler("ZYX", yaw_pitch_roll_euler_angles_stack, degrees=True)
    echo_range_scaling = yaw_rot_pitch_roll.as_matrix()[:, -1, -1]
    echo_range_scaling = xr.DataArray(
        echo_range_scaling, dims="time2", coords={"time2": platform_ds["time2"]}
    )

    return _var_time2_to_ping_time(echo_range_scaling, ping_time_da)


def ek_use_beam_angles(
    beam_ds: xr.Dataset,
) -> xr.DataArray:
    """
    Use `beam_direction_x`, `beam_direction_y`, and `beam_direction_z` from the EK Beam group to
    compute echo range rotational values.
    """
    # Check and log NaNs if they exist in the Beam group variables
    _check_and_log_nans(
        beam_ds, "Sonar/Beam_group1", ["beam_direction_x", "beam_direction_y", "beam_direction_z"]
    )

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
        echo_range_scaling, dims="channel", coords={"channel": beam_ds["channel"]}
    )

    return echo_range_scaling
