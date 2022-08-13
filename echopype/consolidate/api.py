import datetime
from typing import Optional

import numpy as np
import xarray as xr

from ..echodata import EchoData


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
    depth_offset: float = 0,
    tilt: float = 0,
    downward: bool = True,
) -> xr.Dataset:
    """
    Create a depth data variable based on data in Sv dataset.

    The depth is generated based on whether the transducers are mounted vertically
    or with a polar angle to vertical, and whether the transducers were pointed
    up or down.

    Parameters
    ----------
    ds : xr.Dataset
        Source Sv dataset to which a depth variable will be added.
        Must contain `echo_range`.
    depth_offset : float
        Offset along the vertical (depth) dimension to account for actual transducer
        position in water, since `echo_range` is counted from transducer surface.
        Default is 0.
    tilt : float
        Transducer tilt angle [degree].
        Default is 0 (transducer mounted vertically).
    downward : bool
        Whether or not the transducers point downward.
        Default to True.

    Returns
    -------
    The input dataset with a `depth` variable added

    Notes
    -----
    Currently this function only scalar inputs of depth_offset and tilt angle.
    In future expansion we plan to add the following options:

    * Allow inputs as xr.DataArray for time-varying variations of these variables
    * Use data stored in the EchoData object or raw-converted file from which the Sv is derived,
      specifically `water_level`, `vertical_offtset` and `tilt` in the `Platform` group.
    """
    # TODO: add options to use water_depth, vertical_offset, tilt stored in EchoData
    # # Water level has to come from somewhere
    # if depth_offset is None:
    #     if "water_level" in ds:
    #         depth_offset = ds["water_level"]
    #     else:
    #         raise ValueError(
    #             "water_level not found in dataset and needs to be supplied by the user"
    #         )

    # # If not vertical needs to have tilt
    # if not vertical:
    #     if tilt is None:
    #         if "tilt" in ds:
    #             tilt = ds["tilt"]
    #         else:
    #             raise ValueError(
    #                 "tilt not found in dataset and needs to be supplied by the user. "
    #                 "Required when vertical=False"
    #             )
    # else:
    #     tilt = 0

    # Multiplication factor depending on if transducers are pointing downward
    mult = 1 if downward else -1

    # Compute depth
    ds["depth"] = mult * ds["echo_range"] * np.cos(tilt / 180 * np.pi) + depth_offset
    ds["depth"].attrs = {"long_name": "Depth", "standard_name": "depth"}

    return ds


def add_location(ds: xr.Dataset, echodata: EchoData = None, nmea_sentence: Optional[str] = None):
    """
    Add geographical location (latitude/longitude) to the Sv dataset.

    This function interpolates the location from the Platform group in the original data file
    based on the time when the latitude/longitude data are recorded and the time the acoustic
    data are recorded (`ping_time`).

    Parameters
    ----------
    ds : xr.Dataset
        An Sv or MVBS dataset for which the geographical locations will be added to
    echodata
        An `EchoData` object holding the raw data
    nmea_sentence
        NMEA sentence to select a subset of location data (optional)

    Returns
    -------
    The input dataset with the the location data added
    """

    def sel_interp(var):
        # NMEA sentence selection
        if nmea_sentence:
            coord_var = echodata["Platform"][var][
                echodata["Platform"]["sentence_type"] == nmea_sentence
            ]
        else:
            coord_var = echodata["Platform"][var]

        if len(coord_var) == 1:
            # Propagate single, fixed-location coordinate
            return xr.DataArray(
                data=coord_var.values[0] * np.ones(len(ds["ping_time"]), dtype=np.float64),
                dims=["ping_time"],
                attrs=coord_var.attrs,
            )
        else:
            # Interpolation. time1 is always associated with location data
            return coord_var.interp(time1=ds["ping_time"])

    if "longitude" not in echodata["Platform"] or echodata["Platform"]["longitude"].isnull().all():
        raise ValueError("Coordinate variables not present or all nan")

    interp_ds = ds.copy()
    interp_ds["latitude"] = sel_interp("latitude")
    interp_ds["longitude"] = sel_interp("longitude")
    # Most attributes are attached automatically via interpolation
    # here we add the history
    history = (
        f"{datetime.datetime.utcnow()} +00:00. "
        "Interpolated or propagated from Platform latitude/longitude."  # noqa
    )
    interp_ds["latitude"] = interp_ds["latitude"].assign_attrs({"history": history})
    interp_ds["longitude"] = interp_ds["longitude"].assign_attrs({"history": history})

    return interp_ds.drop_vars("time1")
