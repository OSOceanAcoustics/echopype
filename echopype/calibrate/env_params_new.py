from typing import Optional
import numpy as np

from ..echodata import EchoData
from ..echodata.simrad import _check_mode_input_with_data_EK80, _check_mode_input_without_data
from ..utils import uwa


def get_env_params_AZFP(echodata: EchoData, user_env_dict: Optional[dict] = None):
    """Get env params using user inputs or values from data file.

    Parameters
    ----------
    user_env_dict : dict
    """
    out_dict = {}

    # Temperature comes from either user input or data file
    out_dict["temperature"] = (
        user_env_dict["temperature"]
        if "temperature" in user_env_dict
        else echodata["Environment"]["temperature"]
    )

    # Salinity and pressure always come from user input
    if ("salinity" not in user_env_dict) or ("pressure" not in user_env_dict):
        raise ReferenceError("Please supply both salinity and pressure in env_params.")
    else:
        out_dict["salinity"] = user_env_dict["salinity"]
        out_dict["pressure"] = user_env_dict["pressure"]

    # Always calculate sound speed and absorption
    out_dict["sound_speed"] = uwa.calc_sound_speed(
        temperature=user_env_dict["temperature"],
        salinity=user_env_dict["salinity"],
        pressure=user_env_dict["pressure"],
        formula_source="AZFP",
    )
    out_dict["sound_absorption"] = uwa.calc_absorption(
        frequency=echodata["Sonar/Beam_group1"]["frequency_nominal"],
        temperature=user_env_dict["temperature"],
        salinity=user_env_dict["salinity"],
        pressure=user_env_dict["pressure"],
        formula_source="AZFP",
    )

    return out_dict


def get_env_params_EK60(echodata: EchoData, user_env_dict: Optional[dict] = None):
    """Get env params using user inputs or values from data file.

    EK60 file by default contains only sound speed and absorption.
    In cases when temperature, salinity, and pressure values are supplied
    by the user simultaneously, the sound speed and absorption are re-calculated.

    Parameters
    ----------
    user_env_dict : dict
    """
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
        out_dict["sound_speed"] = (
            user_env_dict["sound_speed"]
            if "sound_speed" in user_env_dict
            else echodata["Environment"]["sound_speed_indicative"]
        )
        out_dict["sound_absorption"] = (
            user_env_dict["sound_absorption"]
            if "sound_absorption" in user_env_dict
            else echodata["Environment"]["absorption_indicative"]
        )
    
    return out_dict


def get_env_params_EK80(
    echodata: EchoData, env_params: Optional[dict] = None,
    waveform_mode: Optional[str] = None, encode_mode: Optional[str] = None
):
    """Get env params using user inputs or values from data file.

    EK80 file by default contains sound speed, temperature, depth, salinity, and acidity,
    therefore absorption is always calculated unless it is supplied by the user.
    In cases when temperature, salinity, and pressure values are supplied
    by the user simultaneously, both the sound speed and absorption are re-calculated.

    Parameters
    ----------
    env_params : dict

    waveform_mode : {"CW", "BB"}
        Type of transmit waveform.

        - `"CW"` for narrowband transmission,
            returned echoes recorded either as complex or power/angle samples
        - (default) `"BB"` for broadband transmission,
            returned echoes recorded as complex samples

    encode_mode : {"complex", "power"}
        Type of encoded return echo data.

        - (default) `"complex"` for complex samples
        - `"power"` for power/angle samples, only allowed when
            the echosounder is configured for narrowband transmission
    """
    # Verify input
    _check_mode_input_without_data(waveform_mode=waveform_mode, encode_mode=encode_mode)

    # Retrieve the correct beam group
    power_ed_group, complex_ed_group = _check_mode_input_with_data_EK80(
        echodata, waveform_mode, encode_mode
    )
    if encode_mode == "complex":
        beam = echodata[complex_ed_group]
    else:
        beam = echodata[power_ed_group]

    # Use center frequency if in BB mode, else use nominal channel frequency
    if waveform_mode == "BB":
        freq = (beam["frequency_start"] + beam["frequency_end"]) / 2
    else:
        freq = beam["frequency_nominal"]

    # Re-calculate environment parameters if user supply all env variables
    tsp_all_exist = np.all([p in env_params for p in ["temperature", "salinity", "pressure"]])

    if tsp_all_exist:
        env_params["sound_speed"] = uwa.calc_sound_speed(
            temperature=env_params["temperature"],
            salinity=env_params["salinity"],
            pressure=env_params["pressure"],
        )
        env_params["sound_absorption"] = uwa.calc_absorption(
            frequency=freq,
            temperature=env_params["temperature"],
            salinity=env_params["salinity"],
            pressure=env_params["pressure"],
        )
    # Otherwise
    #  get temperature, salinity, and pressure from raw data file
    #  get sound speed from user inputs or raw data file
    #  get absorption from user inputs or computing from env params stored in raw data file
    else:
        # pressure is encoded as "depth" in EK80  # TODO: change depth to pressure in EK80 file?
        for p1, p2 in zip(
            ["temperature", "salinity", "pressure"],
            ["temperature", "salinity", "depth"],
        ):
            env_params[p1] = (
                env_params[p1]
                if p1 in env_params
                else echodata["Environment"][p2]
            )
        env_params["sound_speed"] = (
            env_params["sound_speed"]
            if "sound_speed" in env_params
            else echodata["Environment"]["sound_speed_indicative"]
        )
        env_params["sound_absorption"] = (
            env_params["sound_absorption"]
            if "sound_absorption" in env_params
            else uwa.calc_absorption(
                frequency=freq,
                temperature=env_params["temperature"],
                salinity=env_params["salinity"],
                pressure=env_params["pressure"],
            )
        )


def get_env_params(
    sonar_model: str, echodata: EchoData, env_params: Optional[dict] = None, **kwarg
):
    if sonar_model == "AZFP":
        return get_env_params_AZFP(echodata=echodata, user_env_dict=env_params)
    elif sonar_model in ["EK60", "ES70"]:
        return get_env_params_EK60(echodata=echodata, user_env_dict=env_params)
    elif sonar_model in ["EK80", "ES80", "EA640"]:
        return get_env_params_EK80(echodata=echodata, env_params=env_params, **kwarg)
