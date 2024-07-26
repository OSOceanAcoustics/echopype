import datetime
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import xarray as xr

from ..echodata import EchoData
from ..utils import uwa
from ..utils.align import align_to_ping_time
from .cal_params import param2da

ENV_PARAMS = (
    "sound_speed",
    "sound_absorption",
    "temperature",
    "salinity",
    "pressure",
    "pH",
    "formula_sound_speed",
    "formula_absorption",
)


def harmonize_env_param_time(
    p: Union[int, float, xr.DataArray],
    ping_time: Optional[Union[xr.DataArray, datetime.datetime]] = None,
):
    """
    Harmonize time coordinate between Beam_groupX data and env_params to make sure
    the timestamps are broadcast correctly in calibration and range calculations.

    Regardless of the source, if `p` is an xr.DataArray, the time coordinate name
    needs to be `time1` to be consistent with the time coordinate in EchoData["Environment"].
    If `time1` is of length=1, the dimension `time1` is dropped.
    Otherwise, `p` is interpolated to `ping_time`.
    If `p` is not an xr.DataArray it is returned directly.

    Parameters
    ----------
    p
        The environment parameter for timestamp check/correction
    ping_time
        Beam_groupX ping_time to interpolate env_params timestamps to.
        Only used if p.time1 has length >1

    Returns
    -------
    Environment parameter with correctly broadcasted timestamps
    """
    if isinstance(p, xr.DataArray):
        if "time1" not in p.coords:
            return p

        # If there's only 1 time1 value:
        if p["time1"].size == 1:
            return p.squeeze(dim="time1").drop_vars("time1")

        # If after dropping NaN along time1 dimension there's only 1 time1 value:
        if p.dropna(dim="time1").size == 1:
            return p.dropna(dim="time1").squeeze(dim="time1").drop_vars("time1")

        if ping_time is None:
            raise ValueError(
                f"ping_time needs to be provided for comparison or interpolating {p.name}"
            )

        # Align array to ping time
        return align_to_ping_time(
            p.dropna(dim="time1", how="all"), "time1", ping_time, method="linear"
        )
    return p


def sanitize_user_env_dict(
    user_dict: Dict[str, Union[int, float, List, xr.DataArray]],
    channel: Union[List, xr.DataArray],
) -> Dict[str, Union[int, float, xr.DataArray]]:
    """
    Creates a blueprint for ``env_params`` dictionary and
    check the format/organize user-provided parameters.

    This function is very similar to ``sanitize_user_cal_dict`` but much simpler,
    without the interpolation routines needed for calibration parameters.

    Parameters
    ----------
    user_dict : dict
        A dictionary containing user input calibration parameters
        as {parameter name: parameter value}.
        Parameter value has to be a scalar (int or float), a list or an ``xr.DataArray``.
        If parameter value is an ``xr.DataArray``, it has to have a'channel' as a coordinate.

    channel : list or xr.DataArray
        A list of channels to be calibrated.
        For EK80 data, this list has to corresponds with the subset of channels
        selected based on waveform_mode and encode_mode

    Returns
    -------
    dict
        A dictionary containing sanitized user-provided environmental parameters.

    Notes
    -----
        The user-provided 'sound_absorption' parameter has to be a list or an xr.DataArray,
        because this parameter is frequency-dependen.
    """
    # Make channel a sorted list
    if not isinstance(channel, (list, xr.DataArray)):
        raise ValueError("'channel' has to be a list or an xr.DataArray")

    if isinstance(channel, xr.DataArray):
        channel_sorted = sorted(channel.values)
    else:
        channel_sorted = sorted(channel)

    # Screen parameters: only retain those defined in ENV_PARAMS
    #  -- transform params in list to xr.DataArray
    #  -- directly pass through params that are scalar or str
    #  -- check channel coordinate if params are xr.DataArray and pass it through
    out_dict = dict.fromkeys(ENV_PARAMS)
    for p_name, p_val in user_dict.items():
        if p_name in out_dict:
            # Param "sound_absorption" has to be an xr.DataArray or a list because it is freq-dep
            if p_name == "sound_absorption" and not isinstance(p_val, (xr.DataArray, list)):
                raise ValueError(
                    "The 'sound_absorption' parameter has to be a list or an xr.DataArray, "
                    "with 'channel' as an coordinate."
                )

            # If p_val an xr.DataArray, check existence and coordinates
            if isinstance(p_val, xr.DataArray):
                # if 'channel' is a coordinate, it has to match that of the data
                if "channel" in p_val.coords:
                    if not (sorted(p_val.coords["channel"].values) == channel_sorted):
                        raise ValueError(
                            f"The 'channel' coordinate of {p_name} has to match "
                            "that of the data to be calibrated"
                        )
                else:
                    raise ValueError(f"{p_name} has to have 'channel' as a coordinate")
                out_dict[p_name] = p_val

            # If p_val a scalar or str, do nothing
            elif isinstance(p_val, (int, float, str)):
                out_dict[p_name] = p_val

            # If p_val a list, make it xr.DataArray
            elif isinstance(p_val, list):
                # check for list dimension happens within param2da()
                out_dict[p_name] = param2da(p_val, channel)

            # p_val has to be one of int, float, xr.DataArray
            else:
                raise ValueError(f"{p_name} has to be a scalar, list, or an xr.DataArray")

    return out_dict


def get_env_params_AZFP(echodata: EchoData, user_dict: Optional[dict] = None):
    """Get env params using user inputs or values from data file.

    Parameters
    ----------
    echodata : EchoData
        an echodata object containing the env params to be pulled from
    user_dict : dict
        user input dict containing env params

    Returns
    -------
    dict
        A dictionary containing the environmental parameters.
    """
    # AZFP only has 1 beam group
    beam = echodata["Sonar/Beam_group1"]

    # Use sanitized user dict as blueprint
    # out_dict contains only and all of the allowable cal params
    out_dict = sanitize_user_env_dict(user_dict=user_dict, channel=beam["channel"])
    out_dict.pop("pH")  # AZFP formulae do not use pH

    # For AZFP, salinity and pressure always come from user input
    if ("salinity" not in out_dict) or ("pressure" not in out_dict):
        raise ReferenceError("Please supply both salinity and pressure in env_params.")

    # Needs to fill in temperature first before sound speed and absorption can be calculated
    if out_dict["temperature"] is None:
        out_dict["temperature"] = echodata["Environment"]["temperature"]

    # Set sound speed and absorption formula source if not in user_dict
    if out_dict["formula_sound_speed"] is None:
        out_dict["formula_sound_speed"] = "AZFP"
    if out_dict["formula_absorption"] is None:
        out_dict["formula_absorption"] = "AZFP"

    # Only fill in params that are None
    for p, v in out_dict.items():
        if v is None:
            if p == "sound_speed":
                out_dict[p] = uwa.calc_sound_speed(
                    temperature=out_dict["temperature"],
                    salinity=out_dict["salinity"],
                    pressure=out_dict["pressure"],
                    formula_source=out_dict["formula_sound_speed"],
                )
            elif p == "sound_absorption":
                out_dict[p] = uwa.calc_absorption(
                    frequency=beam["frequency_nominal"],
                    temperature=out_dict["temperature"],
                    salinity=out_dict["salinity"],
                    pressure=out_dict["pressure"],
                    formula_source=out_dict["formula_absorption"],
                )

    # Harmonize time coordinate between Beam_groupX (ping_time) and env_params (time1)
    # Note for AZFP data is always in Sonar/Beam_group1
    for p in out_dict.keys():
        out_dict[p] = harmonize_env_param_time(out_dict[p], ping_time=beam["ping_time"])

    return out_dict


def get_env_params_EK(
    sonar_type: Literal["EK60", "EK80"],
    beam: xr.Dataset,
    env: xr.Dataset,
    user_dict: Optional[Dict] = None,
    freq: xr.DataArray = None,
) -> Dict:
    """
    Get env params using user inputs or values from data file.

    Parameters
    ----------
    sonar_type : str
        Type of sonar, one of "EK60" or "EK80"
    beam : xr.Dataset
        A subset of Sonar/Beam_groupX that contains only the channels specified for calibration
    env : xr.Dataset
        A subset of Environment group that contains only the channels specified for calibration
    user_dict : dict
        User input dict containing env params
    freq : xr.DataArray
        Center frequency for the selected channels.
        Required for EK80 calibration.
        If provided for EK60 calibration,
        it will be overwritten by the values in ``beam['frequency_nominal']``

    Returns
    -------
    A dictionary containing the environmental parameters.

    Notes
    -----
    EK60 file by default contains only sound speed and absorption.
    In cases when temperature, salinity, and pressure values are supplied
    by the user simultaneously, the sound speed and absorption are re-calculated.

    EK80 file by default contains sound speed, temperature, depth, salinity, and acidity,
    therefore absorption is always calculated unless it is supplied by the user.
    In cases when temperature, salinity, and pressure values are supplied
    by the user simultaneously, both the sound speed and absorption are re-calculated.

    """
    if sonar_type not in ["EK60", "EK80"]:
        raise ValueError("'sonar_type' has to be 'EK60' or 'EK80'")

    # EK80 calibration requires freq, which is the channel center frequency
    if sonar_type == "EK80":
        if freq is None:
            raise ValueError("'freq' is required for calibrating EK80-style data.")
    else:  # EK60
        freq = beam["frequency_nominal"]  # overwriting input if exists

    # Use sanitized user dict as blueprint
    # out_dict contains only and all of the allowable cal params
    out_dict = sanitize_user_env_dict(user_dict=user_dict, channel=beam["channel"])

    # Check absorption and sound speed formula
    if out_dict["formula_absorption"] not in [None, "AM", "FG"]:
        raise ValueError("'formula_absorption' has to be None, 'FG' or 'AM' for EK echosounders.")
    if out_dict["formula_sound_speed"] not in (None, "Mackenzie"):
        raise ValueError("'formula_absorption' has to be None or 'Mackenzie' for EK echosounders.")

    # Calculation sound speed and absorption requires at least T, S, P
    # tsp_all_exist controls wherher to calculate sound speed and absorption
    tspa_all_exist = np.all(
        [out_dict[p] is not None for p in ["temperature", "salinity", "pressure", "pH"]]
    )

    # If EK80, get env parameters from data if not provided in user dict
    # All T, S, P, pH are needed because we always have to compute sound absorption for EK80 data
    if not tspa_all_exist and sonar_type == "EK80":
        for p_user, p_data in zip(
            ["temperature", "salinity", "pressure", "pH"],  # name in defined env params
            ["temperature", "salinity", "depth", "acidity"],  # name in EK80 data
        ):
            out_dict[p_user] = user_dict.get(p_user, env[p_data])

    # Sound speed
    if out_dict["sound_speed"] is None:
        if not tspa_all_exist:
            # sounds speed always exist in EK60 and EK80 data
            out_dict["sound_speed"] = env["sound_speed_indicative"]
            out_dict.pop("formula_sound_speed")
        else:
            # default to Mackenzie sound speed formula if not in user dict
            if out_dict["formula_sound_speed"] is None:
                out_dict["formula_sound_speed"] = "Mackenzie"

            out_dict["sound_speed"] = uwa.calc_sound_speed(
                temperature=out_dict["temperature"],
                salinity=out_dict["salinity"],
                pressure=out_dict["pressure"],
                formula_source=out_dict["formula_sound_speed"],
            )
    else:
        out_dict.pop("formula_sound_speed")  # remove this since no calculation

    # Sound absorption
    if out_dict["sound_absorption"] is None:
        if not tspa_all_exist and sonar_type != "EK80":  # this should not happen for EK80
            # absorption always exist in EK60 data
            out_dict["sound_absorption"] = env["absorption_indicative"]
            out_dict.pop("formula_absorption")
        else:
            # default to FG absorption if not in user dict
            if out_dict["formula_absorption"] is None:
                out_dict["formula_absorption"] = "FG"

            out_dict["sound_absorption"] = uwa.calc_absorption(
                frequency=freq,
                temperature=out_dict["temperature"],
                salinity=out_dict["salinity"],
                pressure=out_dict["pressure"],
                pH=out_dict["pH"],
                sound_speed=out_dict["sound_speed"],
                formula_source=out_dict["formula_absorption"],
            )
    else:
        out_dict.pop("formula_absorption")  # remove this since no calculation

    # Remove params if calculation for both sound speed and absorption didn't happen
    if not ("formula_sound_speed" in out_dict or "formula_absorption" in out_dict):
        [out_dict.pop(p) for p in ["temperature", "salinity", "pressure", "pH"]]

    # Harmonize time coordinate between Beam_groupX (ping_time) and env_params (time1)
    # Note for EK60 data is always in Sonar/Beam_group1
    for p in out_dict.keys():
        out_dict[p] = harmonize_env_param_time(out_dict[p], ping_time=beam["ping_time"])

    return out_dict
