"""
Contains functions that are specific to Simrad echo sounders
"""

from typing import Optional, Tuple

import numpy as np

from ..calibrate.utils import check_input_args_combination
from ..core import SONAR_MODELS
from .echodata import EchoData

__all__ = ["check_input_args_combination", "retrieve_correct_beam_group"]


def _retrieve_correct_beam_group_EK60(
    echodata: EchoData, waveform_mode: str, encode_mode: str
) -> Optional[str]:
    """
    Ensures that the provided ``waveform_mode`` and ``encode_mode`` are consistent
    with EK60-like data supplied by ``echodata``. Additionally, select the
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

    return "Sonar/Beam_group1"


def _retrieve_correct_beam_group_EK80(
    echodata: EchoData, waveform_mode: str, encode_mode: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Ensures that the provided ``waveform_mode`` and ``encode_mode`` are consistent
    with EK80-like data supplied by ``echodata``. Additionally, select the
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
            "Echodata missing `waveform_encode_descr`. "
            "Reconvert using the latest Echopype version."
        )

    # Get the waveform_encode descriptions indexed by beam group.
    # The keys are beam group index, and values are encode and
    # waveform descriptions.
    descr = echodata["Sonar"]["waveform_encode_descr"]

    if encode_mode == "power":
        match_str = "power"
    elif encode_mode == "complex":
        if waveform_mode == "CW":
            match_str = "complex_CW"
        else:
            match_str = "complex_FM"
    idx_match = np.argwhere((descr == match_str).values).squeeze()
    if idx_match.size == 0:
        raise RuntimeError(
            f"No beam group with the specified encode_mode {encode_mode} "
            f"and waveform_mode {waveform_mode} found in the provided echodata!"
        )

    return f"Sonar/Beam_group{idx_match + 1}"  # Beam_groupX is 1-based, index is 0-based


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

    # TODO: can simplify the checks here since
    #       1) checks under _retrieve_correct_beam_group_EK60 are redundant, and
    #       2) only power data would exist for EK60-like data
    #          and we have echodata["Sonar"]["waveform_encode_descr"] now
    model_family = SONAR_MODELS[echodata.sonar_model]["family"]
    if model_family == "Ex60":
        # check modes against data for EK60 and get power EchoData group
        return _retrieve_correct_beam_group_EK60(echodata, waveform_mode, encode_mode)

    elif model_family == "Ex80":
        return _retrieve_correct_beam_group_EK80(echodata, waveform_mode, encode_mode)
    else:
        # raise error for unknown or unaccounted for sonar model
        raise RuntimeError("EchoData was produced by a non-Simrad or unknown Simrad echo sounder!")
