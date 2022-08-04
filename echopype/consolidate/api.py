import datetime
from typing import Optional

import numpy as np
import xarray as xr

from ..echodata import EchoData


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
                attrs=coord_var.attrs
            )
        else:
            # Interpolation. time1 is always associated with location data
            return coord_var.interp(time1=ds["ping_time"])

    if ("longitude" not in echodata["Platform"]
            or echodata["Platform"]["longitude"].isnull().all()):
        raise ValueError("Coordinate variables not present or all nan")

    interp_ds = ds.copy()
    interp_ds["latitude"] = sel_interp("latitude")
    interp_ds["longitude"] = sel_interp("longitude")
    # Most attributes are attached automatically via interpolation
    # here we add the history
    history = (
        f"{datetime.datetime.utcnow()} +00:00. " "Interpolated or propagated from Platform latitude/longitude."  # noqa
    )
    interp_ds["latitude"] = interp_ds["latitude"].assign_attrs({"history": history})
    interp_ds["longitude"] = interp_ds["longitude"].assign_attrs({"history": history})

    return interp_ds.drop_vars("time1")
