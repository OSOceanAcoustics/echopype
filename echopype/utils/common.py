from ast import Num
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
    ds: xr.Dataset, vertical: bool, downward: bool = True, tilt: Num = None, water_level: Num = None
) -> xr.Dataset:
    """
    Create a depth data variable based on data in Sv dataset.

    The depth is generated based on whether the transducers are mounted vertically
    or with a polar angle to vertical, and whether the transducers were pointed
    up or down.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset for which the dimension will be swapped,
        must contain `echo_range`
    vertical : bool
        Whether or not the transducers are mounted vertically
    downward : bool
        Whether or not the transducers point downward
        Default to True
    tilt : num
        Tilt angle (degree) if transducers are not mounted vertically (when `vertical=False`)
        Required if `vertical=False` and `tilt` does not exist in the dataset
    water_level : num
        User-defined `water_level` to replace the water_level stored in the dataset
        Optional if `water_level` exists in the dataset
        Required if `water_level` does not exist in the dataset

    Returns
    -------
    The input dataset with a `depth` variable added
    """
    # Water level has to come from somewhere
    if "water_level" not in ds and water_level is None:
        raise ValueError("water_level not found in dataset and needs to be supplied by the user")

    # If not vertical needs to have tilt
    if not vertical:
        if tilt is None:
            if "tilt" in ds:
                tilt = ds["tilt"]
            else:
                raise ValueError("tilt not found in dataset and needs to be supplied by the user when vertical=False")
    else:
        tilt = 0

    # Multiplication factor dependeing on if transducers are pointing downward
    if downward:
        mult = -1
    else:
        mult = 1

    # Compute depth
    ds["depth"] = mult * ds["sonar_range"] * np.cos(tilt) + ds["water_level"]

    return ds
