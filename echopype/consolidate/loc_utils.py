import numpy as np
import xarray as xr

from ..echodata import EchoData
from ..utils.log import _init_logger

logger = _init_logger(__name__)


def check_loc_vars_present_all_NaN(
    echodata: EchoData, lat_name: str, lon_name: str, datagram_type: str
):
    """
    Checks if any of the location variables are missing or are all NaN. Recommends using another
    datagram_type if the corresponding lat/lon variables are not missing and are not all NaN.
    """
    # Grab variables from EchoData object
    lat_var = echodata["Platform"][lat_name]
    lon_var = echodata["Platform"][lon_name]

    # Check if loc vars missing or all NaN
    any_var_missing = (lat_var is None) or (lon_var is None)
    any_var_all_nan = lat_var.isnull().all() or lon_var.isnull().all()

    if any_var_missing or any_var_all_nan:
        # Initialize error message
        error_message = "Coordinate variables not present or all nan."
        if echodata.sonar_model.startswith("EK"):
            # Initialize good datagram type list
            good_datagram_types = []

            # Iterate through potential datagram type sets
            potential_datagram_type_set = set(["NMEA", "MRU1", "IDX"])
            potential_datagram_type_set.discard(datagram_type)
            for potential_datagram_type in potential_datagram_type_set:
                # Grab variables from EchoData object
                if potential_datagram_type != "NMEA":
                    potential_lat_var = echodata["Platform"].get(
                        f"latitude_{potential_datagram_type.lower()}"
                    )
                    potential_lon_var = echodata["Platform"].get(
                        f"latitude_{potential_datagram_type.lower()}"
                    )
                elif potential_datagram_type == "NMEA":
                    potential_lat_var = echodata["Platform"]["latitude"]
                    potential_lon_var = echodata["Platform"]["longitude"]

                # Check if all potential variables are present and not all NaN
                all_present_and_not_all_nan = (
                    potential_lat_var is not None
                    and potential_lon_var is not None
                    and not (potential_lat_var.isnull().all() or potential_lon_var.isnull().all())
                )

                if all_present_and_not_all_nan:
                    good_datagram_types.append(potential_datagram_type)

            if len(good_datagram_types) > 0:
                # Append to error message
                error_message = (
                    error_message
                    + f" Consider setting datagram_type to any of {good_datagram_types}."
                )

        # Raise error
        raise ValueError(error_message)


def check_loc_vars_any_NaN_0(echodata: EchoData, lat_name: str, lon_name: str, datagram_type: str):
    """
    Checks if the location variables contain any NaN or 0. Recommends using another datagram_type
    if the corresponding lat/lon variables do not contain any NaN or 0.
    """
    # Grab variables from EchoData object
    lat_var = echodata["Platform"][lat_name]
    lon_var = echodata["Platform"][lon_name]

    # Check if loc vars contains any 0s or NaNs
    contains_zero_lat_lon = (lat_var.values == 0).any() or (lon_var.values == 0).any()
    contains_nan_lat_lon = np.isnan(lat_var.values).any() or np.isnan(lon_var.values).any()

    if contains_zero_lat_lon or contains_nan_lat_lon:
        # Initialize shared warning message
        shared_warning_message = (
            "Interpolation may be negatively impacted, "
            "consider handling these values before calling ``add_location``."
        )
        if echodata.sonar_model.startswith("EK"):
            # Initialize good datagram type list
            good_datagram_types = []

            # Iterate through potential datagram type sets
            if echodata.sonar_model.startswith("EK80"):
                potential_datagram_type_set = set(["NMEA", "MRU1", "IDX"])
            else:
                potential_datagram_type_set = set(["NMEA", "IDX"])
            potential_datagram_type_set.discard(datagram_type)
            for potential_datagram_type in potential_datagram_type_set:
                # Grab variables from EchoData object
                if potential_datagram_type != "NMEA":
                    potential_lat_var = echodata["Platform"].get(
                        f"latitude_{potential_datagram_type.lower()}"
                    )
                    potential_lon_var = echodata["Platform"].get(
                        f"latitude_{potential_datagram_type.lower()}"
                    )
                else:
                    potential_lat_var = echodata["Platform"]["latitude"]
                    potential_lon_var = echodata["Platform"]["longitude"]

                # Check if potential variables are present and don't contain all 0s or NaNs
                if potential_lat_var is not None and potential_lon_var is not None:
                    potential_contains_zero_lat_lon = (potential_lat_var.values == 0).any() or (
                        potential_lon_var.values == 0
                    ).any()
                    potential_contains_nan_lat_lon = (
                        np.isnan(potential_lat_var.values).any()
                        or np.isnan(potential_lon_var.values).any()
                    )

                    if not (potential_contains_zero_lat_lon or potential_contains_nan_lat_lon):
                        good_datagram_types.append(potential_datagram_type)

            if len(good_datagram_types) > 0:
                # Append to shared warning message
                shared_warning_message = (
                    shared_warning_message
                    + f" Consider setting datagram_type to any of {good_datagram_types}."
                )

        if contains_zero_lat_lon:
            logger.warning(f"Echodata Platform arrays contain zeros. {shared_warning_message}")
        if contains_nan_lat_lon:
            logger.warning(f"Echodata Platform arrays contain NaNs. {shared_warning_message}")


def check_loc_time_dim_duplicates(echodata: EchoData, time_dim_name: str) -> None:
    """Check if there are duplicates in time_dim_name"""
    if len(np.unique(echodata["Platform"][time_dim_name].data)) != len(
        echodata["Platform"][time_dim_name].data
    ):
        raise ValueError(
            f'The ``echodata["Platform"]["{time_dim_name}"]`` array contains duplicate values. '
            "Downstream interpolation on the position variables requires unique time values."
        )


def sel_interp(
    ds: xr.Dataset,
    echodata: EchoData,
    datagram_type: str,
    loc_name: str,
    time_dim_name: str,
    nmea_sentence: str,
) -> xr.DataArray:
    """
    Selection and interpolation for a location variable.

    The selection logic is as follows, with 4 possible scenarios:

    1) If datagram_type == NMEA is used and NMEA sentence is NaN, then do nothing.
    2) If datagram_type == NMEA is used and NMEA sentence is not NaN, then do the selection.
    3) If datagram_type != NMEA and NMEA sentence is NaN, then do nothing.
    4) If datagram_type != NMEA and NMEA sentence is not NaN, then raise ValueError since NMEA
    sentence selection can only be used on location variables from the NMEA datagram.

    After selection logic, the location variable is then interpolated time-wise to match
    that of the input dataset's time dimension.
    """
    # NMEA sentence selection if datagram_type is None (NMEA corresponds to None)
    if nmea_sentence and datagram_type is None:
        position_var = echodata["Platform"][loc_name][
            echodata["Platform"]["sentence_type"] == nmea_sentence
        ]
    elif nmea_sentence and datagram_type is not None:
        raise ValueError(
            "If datagram_type is not `None`, then `nmea_sentence` cannot be specified."
        )
    else:
        position_var = echodata["Platform"][loc_name]

    if len(position_var) == 1:
        # Propagate single, fixed-location coordinate
        return xr.DataArray(
            data=position_var.values[0] * np.ones(len(ds["ping_time"]), dtype=np.float64),
            dims=["ping_time"],
            attrs=position_var.attrs,
        )
    else:
        # Values may be nan if there are ping_time values outside the time_dim_name range
        return position_var.interp(**{time_dim_name: ds["ping_time"]})
