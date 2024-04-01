"""
Contains functions that are specific to Simrad echo sounders
"""

from typing import Optional, Tuple

import numpy as np

from .echodata import EchoData


def check_input_args_combination(
    waveform_mode: str, encode_mode: str, pulse_compression: bool = None
) -> None:
    """
    Checks that the ``waveform_mode`` and ``encode_mode`` have
    the correct values and that the combination of input arguments are valid, without
    considering the actual data.

    Parameters
    ----------
    waveform_mode: str
        Type of transmit waveform
    encode_mode: str
        Type of encoded return echo data
    pulse_compression: bool
        States whether pulse compression should be used
    """

    if waveform_mode not in ["CW", "BB"]:
        raise ValueError("The input waveform_mode must be either 'CW' or 'BB'!")

    if encode_mode not in ["complex", "power"]:
        raise ValueError("The input encode_mode must be either 'complex' or 'power'!")

    # BB has complex data only, but CW can have complex or power data
    if (waveform_mode == "BB") and (encode_mode == "power"):
        raise ValueError(
            "Data from broadband ('BB') transmission must be recorded as complex samples"
        )

    # make sure that we have BB and complex inputs, if pulse compression is selected
    if pulse_compression is not None:
        if pulse_compression and ((waveform_mode != "BB") or (encode_mode != "complex")):
            raise RuntimeError(
                "Pulse compression can only be used with "
                "waveform_mode='BB' and encode_mode='complex'"
            )


def _retrieve_correct_beam_group_EK60(
    echodata: EchoData, waveform_mode: str, encode_mode: str
) -> Optional[str]:
    """
    Ensures that the provided ``waveform_mode`` and ``encode_mode`` are consistent
    with the EK60-like data supplied by ``echodata``. Additionally, select the
    appropriate beam group corresponding to this input.

    Parameters
    ----------
    echodata: EchoData
        An ``EchoData`` object holding the data
    waveform_mode : {"CW", "BB"}
        Type of transmit waveform
    encode_mode : {"complex", "power"}
        Type of encoded return echo data

    Returns
    -------
    power_ed_group: str, optional
        The ``EchoData`` beam group path containing the power data
    """

    # initialize power EchoData group value
    power_ed_group = None

    # EK60-like sensors must have 'power' and 'CW' modes only
    if waveform_mode != "CW":
        raise RuntimeError("Incorrect waveform_mode input provided!")
    if encode_mode != "power":
        raise RuntimeError("Incorrect encode_mode input provided!")

    # ensure that no complex data exists (this should never be triggered)
    if "backscatter_i" in echodata["Sonar/Beam_group1"].variables:
        raise RuntimeError(
            "Provided echodata object does not correspond to an EK60-like "
            "sensor, but is labeled as data from an EK60-like sensor!"
        )
    else:
        power_ed_group = "Sonar/Beam_group1"

    return power_ed_group


def _retrieve_correct_beam_group_EK80(
    echodata: EchoData, waveform_mode: str, encode_mode: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Ensures that the provided ``waveform_mode`` and ``encode_mode`` are consistent
    with the EK80-like data supplied by ``echodata``. Additionally, select the
    appropriate beam group corresponding to this input.

    Parameters
    ----------
    echodata: EchoData
        An ``EchoData`` object holding the data
    waveform_mode : {"CW", "BB"}
        Type of transmit waveform
    encode_mode : {"complex", "power"}
        Type of encoded return echo data

    Returns
    -------
    power_ed_group: str, optional
        The ``EchoData`` beam group path containing the power data
    complex_ed_group: str, optional
        The ``EchoData`` beam group path containing the complex data
    """

    # initialize power and complex EchoData group values
    power_ed_group = None
    complex_ed_group = None

    transmit_type = echodata["Sonar/Beam_group1"]["transmit_type"]
    # assume transmit_type identical for all pings in a channel
    # TODO: change when allowing within-channel CW-BB switch
    first_ping_transmit_type = transmit_type.isel(ping_time=0)
    if waveform_mode == "BB":
        # check BB waveform_mode, BB must always have complex data, can have 2 beam groups
        # when echodata contains CW power and BB complex samples
        if np.all(first_ping_transmit_type == "CW"):
            raise ValueError("waveform_mode='BB', but complex data does not exist!")
        elif echodata["Sonar/Beam_group2"] is not None:
            power_ed_group = "Sonar/Beam_group2"
            complex_ed_group = "Sonar/Beam_group1"
        else:
            complex_ed_group = "Sonar/Beam_group1"
    else:
        # CW can have complex or power data, so we just need to make sure that
        # 1) complex samples always exist in Sonar/Beam_group1
        # 2) power samples are in Sonar/Beam_group1 if only one beam group exists
        # 3) power samples are in Sonar/Beam_group2 if two beam groups exist

        # Raise error if waveform_mode="CW" but CW data does not exist (not a single ping is CW)
        # TODO: change when allowing within-channel CW-BB switch
        if encode_mode == "complex" and np.all(first_ping_transmit_type != "CW"):
            raise ValueError("waveform_mode='CW', but all data are broadband (BB)!")

        if echodata["Sonar/Beam_group2"] is None:
            if encode_mode == "power":
                # power samples must be in Sonar/Beam_group1 (thus no complex samples)
                if "backscatter_i" in echodata["Sonar/Beam_group1"].variables:
                    raise RuntimeError("Data provided does not correspond to encode_mode='power'!")
                else:
                    power_ed_group = "Sonar/Beam_group1"
            elif encode_mode == "complex":
                # complex samples must be in Sonar/Beam_group1
                if "backscatter_i" not in echodata["Sonar/Beam_group1"].variables:
                    raise RuntimeError(
                        "Data provided does not correspond to encode_mode='complex'!"
                    )
                else:
                    complex_ed_group = "Sonar/Beam_group1"

        else:
            # complex should be in Sonar/Beam_group1 and power in Sonar/Beam_group2
            # the RuntimeErrors below should never be triggered
            if "backscatter_i" not in echodata["Sonar/Beam_group1"].variables:
                raise RuntimeError(
                    "Complex data does not exist in Sonar/Beam_group1, "
                    "input echodata object must have been incorrectly constructed!"
                )
            elif "backscatter_r" not in echodata["Sonar/Beam_group2"].variables:
                raise RuntimeError(
                    "Power data does not exist in Sonar/Beam_group2, "
                    "input echodata object must have been incorrectly constructed!"
                )
            else:
                complex_ed_group = "Sonar/Beam_group1"
                power_ed_group = "Sonar/Beam_group2"

    return power_ed_group, complex_ed_group


def retrieve_correct_beam_group(echodata: EchoData, waveform_mode: str, encode_mode: str) -> str:
    """
    A function to make sure that the user has provided the correct
    ``waveform_mode`` and ``encode_mode`` inputs based off of the
    supplied ``echodata`` object. Additionally, determine the
    ``EchoData`` beam group corresponding to ``encode_mode``.

    Parameters
    ----------
    echodata: EchoData
        An ``EchoData`` object holding the data corresponding to the
        waveform and encode modes
    waveform_mode : {"CW", "BB"}
        Type of transmit waveform
    encode_mode : {"complex", "power"}
        Type of encoded return echo data
    pulse_compression: bool
        States whether pulse compression should be used

    Returns
    -------
    str
        The ``EchoData`` beam group path corresponding to the ``encode_mode`` input
    """

    if echodata.sonar_model in ["EK60", "ES70"]:
        # initialize complex_data_location (needed only for EK60)
        complex_ed_group = None

        # check modes against data for EK60 and get power EchoData group
        power_ed_group = _retrieve_correct_beam_group_EK60(echodata, waveform_mode, encode_mode)

    elif echodata.sonar_model in ["EK80", "ES80", "EA640"]:
        # check modes against data for EK80 and get power/complex EchoData groups
        power_ed_group, complex_ed_group = _retrieve_correct_beam_group_EK80(
            echodata, waveform_mode, encode_mode
        )

    else:
        # raise error for unknown or unaccounted for sonar model
        raise RuntimeError("EchoData was produced by a non-Simrad or unknown Simrad echo sounder!")

    if encode_mode == "complex":
        return complex_ed_group
    else:
        return power_ed_group
