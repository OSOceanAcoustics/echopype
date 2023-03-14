import datetime
from typing import Dict, Optional, Union

import numpy as np
import xarray as xr

from ..echodata import EchoData
from ..utils import uwa

# TODO: create default dict with empty values but specific keys for out_dict
# like in cal_params, right now it is initiated as {}


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


def get_env_params_AZFP(echodata: EchoData, user_env_dict: Optional[dict] = None):
    """Get env params using user inputs or values from data file.

    Parameters
    ----------
    echodata : EchoData
        an echodata object containing the env params to be pulled from
    user_env_dict : dict
        user input dict containing env params

    Returns
    -------
    A dictionary containing the calibration parameters.
    """
    out_dict = {}

    # Temperature comes from either user input or data file
    out_dict["temperature"] = user_env_dict.get(
        "temperature", echodata["Environment"]["temperature"]
    )

    # Salinity and pressure always come from user input
    if ("salinity" not in user_env_dict) or ("pressure" not in user_env_dict):
        raise ReferenceError("Please supply both salinity and pressure in env_params.")
    else:
        out_dict["salinity"] = user_env_dict["salinity"]
        out_dict["pressure"] = user_env_dict["pressure"]

    # Always calculate sound speed and absorption
    out_dict["sound_speed"] = uwa.calc_sound_speed(
        temperature=out_dict["temperature"],
        salinity=out_dict["salinity"],
        pressure=out_dict["pressure"],
        formula_source="AZFP",
    )
    out_dict["sound_absorption"] = uwa.calc_absorption(
        frequency=echodata["Sonar/Beam_group1"]["frequency_nominal"],
        temperature=out_dict["temperature"],
        salinity=out_dict["salinity"],
        pressure=out_dict["pressure"],
        formula_source="AZFP",
    )

    # Harmonize time coordinate between Beam_groupX (ping_time) and env_params (time1)
    # Note for AZFP data is always in Sonar/Beam_group1
    for p in out_dict.keys():
        out_dict[p] = harmonize_env_param_time(
            out_dict[p], ping_time=echodata["Sonar/Beam_group1"]["ping_time"]
        )

    return out_dict


def get_env_params_EK60(echodata: EchoData, user_env_dict: Optional[Dict] = None) -> Dict:
    """Get env params using user inputs or values from data file.

    EK60 file by default contains only sound speed and absorption.
    In cases when temperature, salinity, and pressure values are supplied
    by the user simultaneously, the sound speed and absorption are re-calculated.

    Parameters
    ----------
    echodata : EchoData
        an echodata object containing the env params to be pulled from
    user_env_dict : dict
        user input dict containing env params

    Returns
    -------
    A dictionary containing the calibration parameters.
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
            frequency=echodata["Sonar/Beam_group1"]["frequency_nominal"],
            temperature=user_env_dict["temperature"],
            salinity=user_env_dict["salinity"],
            pressure=user_env_dict["pressure"],
        )
    # Otherwise get sound speed and absorption from user inputs or raw data file
    else:
        p_map_dict = {
            "sound_speed": "sound_speed_indicative",
            "sound_absorption": "absorption_indicative",
        }
        for p_out, p_data in p_map_dict.items():
            out_dict[p_out] = user_env_dict.get(p_out, echodata["Environment"][p_data])

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
