"""
Contains functions that are specific to Simrad echo sounders
"""

from typing import Optional, Tuple

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
    power_group: str, optional
        The ``EchoData`` beam group path containing the power data
    """

    # initialize power EchoData group value
    power_group = None

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
        power_group = "Sonar/Beam_group1"

    return power_group


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
    power_group: str, optional
        The ``EchoData`` beam group path containing the power data
    complex_group: str, optional
        The ``EchoData`` beam group path containing the complex data
    """
    if "waveform_encode_descr" not in echodata["Sonar"]:
        raise ValueError(
            "Echodata missing `waveform_encode_descr`. Reconvert using latest Echopype version."
        )

    # Get the waveform_encode descriptions indexed by beam group.
    # The keys are beam group index, and values are encode and
    # waveform descriptions.
    descr = echodata["Sonar"]["waveform_encode_descr"]

    power_group = None
    complex_FM_group = None
    complex_CW_group = None

    # Iterate over beam groups:
    for beam_idx, desc in zip(descr.coords["beam_group_index"].values, descr.values):
        if encode_mode == "power" and "power" in desc:
            power_group = f"Sonar/Beam_group{beam_idx}"
        elif encode_mode == "complex":
            if "FM" in desc and waveform_mode == "BB":
                complex_FM_group = f"Sonar/Beam_group{beam_idx}"
            elif "CW" in desc and waveform_mode == "CW":
                complex_CW_group = f"Sonar/Beam_group{beam_idx}"

    return power_group, complex_FM_group, complex_CW_group


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
        # check modes against data for EK60 and get power EchoData group
        power_group = _retrieve_correct_beam_group_EK60(echodata, waveform_mode, encode_mode)

    elif echodata.sonar_model in ["EK80", "ES80", "EA640"]:
        # check modes against data for EK80 and get power/complex EchoData groups
        power_group, complex_FM_group, complex_CW_group = _retrieve_correct_beam_group_EK80(
            echodata, waveform_mode, encode_mode
        )

    else:
        # raise error for unknown or unaccounted for sonar model
        raise RuntimeError("EchoData was produced by a non-Simrad or unknown Simrad echo sounder!")

    if encode_mode == "complex":
        if waveform_mode == "CW":
            return complex_CW_group
        else:
            return complex_FM_group
    else:
        return power_group
