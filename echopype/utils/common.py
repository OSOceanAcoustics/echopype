import datetime
from typing import Optional

import numpy as np
import xarray as xr

from ..echodata import EchoData


def swap_dims_channel_frequency(ds: xr.Dataset) -> xr.Dataset:
    """
    Use frequency_nominal in place of channel as dataset dimension and coordinate.

    This is useful because the nominal transducer frequencies are commonly used to
    refer to data collected from a specific transducer.

    Parameters
    ----------
    ds : xr.Dataset
        An Sv or MVBS dataset for which the dimension will be swapped

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
            x_var = echodata["Platform"][var][
                echodata["Platform"]["sentence_type"] == nmea_sentence
            ]
        else:
            x_var = echodata["Platform"][var]
        # Interpolation
        return x_var.interp(time1=ds["ping_time"])  # time1 is always associated with location data

    # Most attritbues are attached automatically via interpolation
    # here we add the history
    ds_copy = ds.copy()
    ds_copy["latitude"] = sel_interp("latitude")
    ds_copy["longitude"] = sel_interp("longitude")
    history = (
        f"{datetime.datetime.utcnow()} +00:00. " "Interpolated from Platform latitude/longitude."
    )
    ds_copy["latitude"] = ds_copy["latitude"].assign_attrs({"history": history})
    ds_copy["longitude"] = ds_copy["longitude"].assign_attrs({"history": history})

    return ds_copy.drop("time1")
