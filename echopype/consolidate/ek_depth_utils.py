import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation as R

from ..utils.align import align_to_ping_time
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

    return align_to_ping_time(transducer_depth, "time2", ping_time_da)


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

    return align_to_ping_time(echo_range_scaling, "time2", ping_time_da)


def ek_use_beam_angles(beam_ds: xr.Dataset) -> xr.DataArray:
    """
    Compute echo range scaling from beam_direction components. For each channel, we expect that
    the beam direction vector is normalized. If not, then we normalize the z direction and set
    that as the echo range scaling.
    Additionally, if a nonzero vector is not normalized, a warning is issued and it is normalized.
    If a channel-wise beam direction vector is zero, a warning is issued and the returned z value
    is set to NaN.
    """
    # Check and log NaNs if they exist in the Beam direction variables
    _check_and_log_nans(
        beam_ds, "Sonar/Beam_group1", ["beam_direction_x", "beam_direction_y", "beam_direction_z"]
    )

    beam_direction_x = beam_ds["beam_direction_x"]
    beam_direction_y = beam_ds["beam_direction_y"]
    beam_direction_z = beam_ds["beam_direction_z"]

    # Calculate the norm for each channel
    norm = np.sqrt(beam_direction_x**2 + beam_direction_y**2 + beam_direction_z**2)

    # Warn if any nonzero vector is not normalized
    tolerance = 1e-8
    if ((norm > tolerance) & (np.abs(norm - 1) > tolerance)).any():
        logger.warning(
            "Beam direction vector was not normalized; applying normalization. "
            "By definition, it should have been normalized."
        )

    # Warn if any channel has a (nearly) zero vector
    if (norm < tolerance).any():
        logger.warning("Some beam direction vectors are zero. Outputting NaN for those channels.")

    # For channels with near-zero norm, we return NaN. Otherwise, we return the normalized
    # z component.
    return xr.where(norm < tolerance, np.nan, beam_direction_z / norm)
