from typing import Union

import numpy as np
import xarray as xr

from ..echodata import EchoData
from ..utils.log import _init_logger

logger = _init_logger(__name__)


def compute_invalid_check(lat_var: xr.DataArray, lon_var: xr.DataArray, validity_check: str):
    """Helper function to check if loc vars are invalid in 4 separate ways."""
    if validity_check == "missing":
        return (lat_var is None) or (lon_var is None)
    elif lat_var is not None and lon_var is not None and validity_check == "all_nan":
        return lat_var.isnull().all() or lon_var.isnull().all()
    elif lat_var is not None and lon_var is not None and validity_check == "some_nan":
        return np.isnan(lat_var.values).any() or np.isnan(lon_var.values).any()
    elif lat_var is not None and lon_var is not None and validity_check == "some_zero":
        return (lat_var.values == 0).any() or (lon_var.values == 0).any()
    else:
        return True


def check_loc_vars_validity(
    echodata: EchoData,
    lat_name: str,
    lon_name: str,
    datagram_type: str,
    validity_check: str,
):
    """
    Checks if any of the location variables are valid. Recommends using another
    datagram_type if the corresponding lat/lon variables are valid.
    """
    # Grab variables from EchoData object
    lat_var = echodata["Platform"].get(lat_name)
    lon_var = echodata["Platform"].get(lon_name)

    # Check if loc vars are invalid
    invalid_bool = compute_invalid_check(lat_var, lon_var, validity_check)

    if invalid_bool:
        # Initialize output message
        if validity_check == "missing":
            output_message = "Coordinate variables not present."
        elif validity_check == "all_nan":
            output_message = "Coordinate variables are all NaN."
        elif validity_check == "some_nan":
            output_message = (
                "Coordinate variables contain NaN(s). "
                "Interpolation may be negatively impacted, "
                "consider handling these values before calling ``add_location``."
            )
        elif validity_check == "some_zero":
            output_message = (
                "Coordinate variables contain zero(s). "
                "Interpolation may be negatively impacted, "
                "consider handling these values before calling ``add_location``."
            )

        if echodata.sonar_model.startswith("EK"):
            # Initialize good datagram type list
            good_datagram_types = []

            # Iterate through potential datagram type sets
            potential_datagram_type_set = set([None, "MRU1", "IDX"])
            potential_datagram_type_set.discard(datagram_type)
            for potential_datagram_type in potential_datagram_type_set:
                # Grab variables from EchoData object
                if potential_datagram_type is not None:
                    potential_lat_var = echodata["Platform"].get(
                        f"latitude_{potential_datagram_type.lower()}"
                    )
                    potential_lon_var = echodata["Platform"].get(
                        f"latitude_{potential_datagram_type.lower()}"
                    )
                elif potential_datagram_type is None:
                    potential_lat_var = echodata["Platform"].get("latitude")
                    potential_lon_var = echodata["Platform"].get("longitude")

                # Check if loc vars are invalid
                if validity_check in ["missing", "all_nan"]:
                    potential_invalid_bool = compute_invalid_check(
                        potential_lat_var, potential_lon_var, "missing"
                    ) or compute_invalid_check(potential_lat_var, potential_lon_var, "all_nan")
                elif validity_check in ["some_nan", "some_zero"]:
                    potential_invalid_bool = compute_invalid_check(
                        potential_lat_var, potential_lon_var, "some_nan"
                    ) or compute_invalid_check(potential_lat_var, potential_lon_var, "some_zero")
                # Append to good datagram types
                if not potential_invalid_bool:
                    good_datagram_types.append(potential_datagram_type)

            if len(good_datagram_types) > 0:
                # Append to output message
                output_message = (
                    output_message
                    + f" Consider setting datagram_type to any of {good_datagram_types}."
                )

        # Raise error / warning depending on validity_check value
        if validity_check in ["missing", "all_nan"]:
            raise ValueError(output_message)
        elif validity_check in ["some_nan", "some_zero"]:
            logger.warning(output_message)


def check_loc_time_dim_duplicates(echodata: EchoData, time_dim_name: str) -> None:
    """Check if there are duplicates in time_dim_name"""
    if len(np.unique(echodata["Platform"][time_dim_name].data)) != len(
        echodata["Platform"][time_dim_name].data
    ):
        raise ValueError(
            f'The ``echodata["Platform"]["{time_dim_name}"]`` array contains duplicate values. '
            "Downstream interpolation on the position variables requires unique time values."
        )


def sel_nmea(
    echodata: EchoData,
    loc_name: str,
    nmea_sentence: Union[str, None] = None,
    datagram_type: Union[str, None] = None,
) -> xr.DataArray:
    """
    Select location subset for a location variable based on NMEA sentence.

    The selection logic is as follows, with 4 possible scenarios:
    Note here datagram_type = None is equivalent to datagram_type = NMEA

    1) If datagram_type is None and nmea_sentence is None, then do nothing.
    2) If datagram_type is None and nmea_sentence is not None, then do the selection.
    3) If datagram_type is not None and nmea_sentence is None, then do nothing.
    4) If datagram_type is not None and nmea_sentence is not None, then raise ValueError since NMEA
       sentence selection can only be used on location variables from the NMEA datagram.
    """
    # NMEA sentence selection if datagram_type is None (NMEA corresponds to None)
    if nmea_sentence and datagram_type is None:
        return echodata["Platform"][loc_name][
            echodata["Platform"]["sentence_type"] == nmea_sentence
        ]
    elif nmea_sentence and datagram_type is not None:
        raise ValueError(
            "If datagram_type is not `None`, then `nmea_sentence` cannot be specified."
        )
    else:
        return echodata["Platform"][loc_name]
