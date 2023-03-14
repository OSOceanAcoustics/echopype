import datetime
from typing import Dict, Optional, Union, List

import numpy as np
import xarray as xr

from ..echodata import EchoData
from .cal_params import param2da
from ..utils import uwa


ENV_PARAMS = (
    "sound_speed",
    "sound_absorption",
    "temperature",
    "salinity",
    "pressure",
    "pH",
    "formula_source_sound_speed",
    "formula_source_absorption",
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
        else:
            # If there's only 1 time1 value,
            # or if after dropping NaN there's only 1 time1 value
            if p["time1"].size == 1 or p.dropna(dim="time1").size == 1:
                return p.dropna(dim="time1").squeeze(dim="time1").drop("time1")

            # Direct assignment if all timestamps are identical (EK60 data)
            elif np.all(p["time1"].values == ping_time.values):
                return p.rename({"time1": "ping_time"})

            elif ping_time is None:
                raise ValueError(f"ping_time needs to be provided for interpolating {p.name}")

            else:
                return p.dropna(dim="time1").interp(time1=ping_time)
    else:
        return p


def sanitize_user_env_dict(
    user_dict: Dict[str, Union[int, float, xr.DataArray]],
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
        Parameter value has to be a scalar (int or float) or an ``xr.DataArray``.
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
    # TODO: check allowable formula sources for sound speed and absorption


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
            # Param "sound_absorption" has to be an xr.DataArray because it is freq-dependent
            if p_name == "sound_absorption" and not (p_val, xr.DataArray):
                raise ValueError("The 'sound_absorption' parameter has to be an xr.DataArray!")

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
    if out_dict["formula_source_sound_speed"] is None:
        out_dict["formula_source_sound_speed"] = "AZFP"
    if out_dict["formula_source_absorption"] is None:
        out_dict["formula_source_absorption"] = "AZFP"

    # Only fill in params that are None
    for p, v in out_dict.items():
        if v is None:
            if p == "sound_speed":
                out_dict[p] = uwa.calc_sound_speed(
                    temperature=out_dict["temperature"],
                    salinity=out_dict["salinity"],
                    pressure=out_dict["pressure"],
                    formula_source=out_dict["formula_source_sound_speed"],
                )
            elif p == "sound_absorption":
                out_dict[p] = uwa.calc_absorption(
                    frequency=beam["frequency_nominal"],
                    temperature=out_dict["temperature"],
                    salinity=out_dict["salinity"],
                    pressure=out_dict["pressure"],
                    formula_source=out_dict["formula_source_absorption"],
                )

    # Harmonize time coordinate between Beam_groupX (ping_time) and env_params (time1)
    # Note for AZFP data is always in Sonar/Beam_group1
    for p in out_dict.keys():
        out_dict[p] = harmonize_env_param_time(
            out_dict[p], ping_time=beam["ping_time"]
        )

    return out_dict


def get_env_params_EK60(echodata: EchoData, user_dict: Optional[Dict] = None) -> Dict:
    """Get env params using user inputs or values from data file.

    EK60 file by default contains only sound speed and absorption.
    In cases when temperature, salinity, and pressure values are supplied
    by the user simultaneously, the sound speed and absorption are re-calculated.

    Parameters
    ----------
    echodata : EchoData
        an echodata object containing the env params to be pulled from
    user_dict : dict
        user input dict containing env params

    Returns
    -------
    A dictionary containing the calibration parameters.
    """
    # EK60 only has 1 beam group
    beam = echodata["Sonar/Beam_group1"]

    # Use sanitized user dict as blueprint
    # out_dict contains only and all of the allowable cal params
    out_dict = sanitize_user_env_dict(user_dict=user_dict, channel=beam["channel"])

    # Re-calculate environment parameters if user supply all env variables
    tsp_all_exist = np.all([out_dict[p] is not None for p in ["temperature", "salinity", "pressure"]])

    # Remove temperature, salinity, pressure so that it is not included in the final env param set
    if not tsp_all_exist:
        [out_dict.pop(p) for p in ["temperature", "salinity", "pressure", "pH"]]

    # Set sound speed and absorption formula source if not in out_dict
    if out_dict["formula_source_sound_speed"] is None:
        out_dict["formula_source_sound_speed"] = "Mackenzie"
    if out_dict["formula_source_absorption"] is None:
        out_dict["formula_source_absorption"] = "FG"

    # Only fill in params that are None
    for p, v in out_dict.items():
        if v is None:
            if p == "sound_speed":
                if tsp_all_exist:
                    out_dict[p] = uwa.calc_sound_speed(
                        temperature=out_dict["temperature"],
                        salinity=out_dict["salinity"],
                        pressure=out_dict["pressure"],
                        formula_source=out_dict["formula_source_sound_speed"]
                    )
                else:
                    out_dict[p] = echodata["Environment"]["sound_speed_indicative"]
            elif p == "sound_absorption":
                if tsp_all_exist:
                    # Fill in pH if not in out_dict
                    if out_dict["pH"] is None:
                        out_dict["pH"] = 8.1  # required to compute absorption

                    out_dict[p] = uwa.calc_absorption(
                        frequency=beam["frequency_nominal"],
                        temperature=out_dict["temperature"],
                        salinity=out_dict["salinity"],
                        pressure=out_dict["pressure"],
                        pH=out_dict["pH"],
                        formula_source=out_dict["formula_source_absorption"],
                    )
                else:
                    out_dict[p] = echodata["Environment"]["absorption_indicative"]

    # Harmonize time coordinate between Beam_groupX (ping_time) and env_params (time1)
    # Note for EK60 data is always in Sonar/Beam_group1
    for p in out_dict.keys():
        out_dict[p] = harmonize_env_param_time(
            out_dict[p], ping_time=echodata["Sonar/Beam_group1"]["ping_time"]
        )

    return out_dict


def get_env_params_EK80(
    echodata: EchoData,
    freq: xr.DataArray,
    user_env_dict: Optional[Dict] = None,
    ed_group: str = None,
) -> Dict:
    """Get env params using user inputs or values from data file.

    EK80 file by default contains sound speed, temperature, depth, salinity, and acidity,
    therefore absorption is always calculated unless it is supplied by the user.
    In cases when temperature, salinity, and pressure values are supplied
    by the user simultaneously, both the sound speed and absorption are re-calculated.

    Parameters
    ----------
    echodata : EchoData
        an echodata object containing the env params to be pulled from
    freq : xr.DataArray
        center frequency for the selected channels
    user_env_dict : dict
        user input dict containing env params
    ed_group : str
        the right Sonar/Beam_groupX given waveform and encode mode

    Returns
    -------
    A dictionary containing the environmental parameters.
    """

    # Initiate input/output dict
    out_dict = {}
    if user_env_dict is None:
        user_env_dict = {}

    # Re-calculate environment parameters if user supply all env variables
    tsp_all_exist = np.all([p in user_env_dict for p in ["temperature", "salinity", "pressure"]])

    if tsp_all_exist:
        out_dict["sound_speed"] = uwa.calc_sound_speed(
            temperature=user_env_dict["temperature"],
            salinity=user_env_dict["salinity"],
            pressure=user_env_dict["pressure"],
        )
        out_dict["sound_absorption"] = uwa.calc_absorption(
            frequency=freq,
            temperature=user_env_dict["temperature"],
            salinity=user_env_dict["salinity"],
            pressure=user_env_dict["pressure"],
        )
    # Otherwise
    #  get temperature, salinity, and pressure from raw data file
    #  get sound speed from user inputs or raw data file
    #  get absorption from user inputs or computing from env params stored in raw data file
    else:
        # pressure is encoded as "depth" in EK80  # TODO: change depth to pressure in EK80 file?
        for p_user, p_data in zip(
            ["temperature", "salinity", "pressure", "pH", "sound_speed"],
            ["temperature", "salinity", "depth", "acidity", "sound_speed_indicative"],
        ):
            out_dict[p_user] = user_env_dict.get(p_user, echodata["Environment"][p_data])

        out_dict["sound_absorption"] = user_env_dict.get(
            "sound_absorption",
            uwa.calc_absorption(
                frequency=freq,
                temperature=out_dict["temperature"],
                salinity=out_dict["salinity"],
                pressure=out_dict["pressure"],
                sound_speed=out_dict["sound_speed"],
                pH=out_dict["pH"],
                formula_source=(
                    user_env_dict["formula_source"] if "formula_source" in user_env_dict else "FG"
                ),
            ),
        )

    # Harmonize time coordinate between Beam_groupX (ping_time) and env_params (time1)
    for p in out_dict.keys():
        if isinstance(out_dict[p], xr.DataArray):
            if "channel" in out_dict[p].coords:
                out_dict[p] = out_dict[p].sel(channel=freq["channel"])  # only ch subset needed
        out_dict[p] = harmonize_env_param_time(
            out_dict[p], ping_time=echodata[ed_group]["ping_time"]
        )

    return out_dict
