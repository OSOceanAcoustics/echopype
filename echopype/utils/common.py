import numpy as np
import xarray as xr


def swap_dims_channel_frequency(ds: xr.Dataset) -> xr.Dataset:
    """
    Use frequency_nominal in place of channel to be dataset dimension and coorindate.

    This is useful because the nominal transducer frequencies are commonly used to
    refer to data collected from a specific transducer.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset for which the dimension will be swapped

    Returns
    -------
    The input dataset with the dimension swapped

    Note
    ----
    This operation is only possible when there are no duplicated frequencies present in the file.
    """
    # Only possible if no duplicated frequencies
    if np.unique(ds["frequency_nominal"]).size == ds["frequency_nominal"].size:
        return (
            ds.set_coords("frequency_nominal")
            .swap_dims({"channel": "frequency_nominal"})
            .reset_coords("channel")
        )
    else:
        raise ValueError(
            "Duplicated transducer nominal frequencies exist in the file. "
            "Operation is not valid."
        )


def add_depth(
    ds: xr.Dataset,
    vertical: bool = True,
    downward: bool = True,
    tilt: float = None,
    water_level: float = None,
) -> xr.Dataset:
    """
    Create a depth data variable based on data in Sv dataset.

    The depth is generated based on whether the transducers are mounted vertically
    or with a polar angle to vertical, and whether the transducers were pointed
    up or down.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset for which the dimension will be swapped.
        Must contain `echo_range`.
    vertical : bool
        Whether or not the transducers are mounted vertically.
        Default to True.
    downward : bool
        Whether or not the transducers point downward.
        Default to True.
    tilt : float
        Tilt angle (degree) if transducers are not mounted vertically (when `vertical=False`).
        Required if `vertical=False` and `tilt` does not exist in the dataset.
    water_level : float
        User-defined `water_level` to replace the water_level stored in the dataset.
        Optional if `water_level` exists in the dataset.
        Required if `water_level` does not exist in the dataset.

    Returns
    -------
    The input dataset with a `depth` variable added
    """
    # Water level has to come from somewhere
    if water_level is None:
        if "water_level" not in ds:
            water_level = ds["water_level"]
        else:
            raise ValueError(
                "water_level not found in dataset and needs to be supplied by the user"
            )

    # If not vertical needs to have tilt
    if not vertical:
        if tilt is None:
            if "tilt" in ds:
                tilt = ds["tilt"]
            else:
                raise ValueError(
                    "tilt not found in dataset and needs to be supplied by the user. "
                    "Required when vertical=False"
                )
    else:
        tilt = 0

    # Multiplication factor depending on if transducers are pointing downward
    if downward:
        mult = 1  # no flip
    else:
        mult = -1  # flip upside down (closer to transducer is deeper)

    # Compute depth
    ds["depth"] = mult * ds["echo_range"] * np.cos(tilt) + water_level

    return ds
