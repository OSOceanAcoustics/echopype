"""
Functions to unpack Simrad EK60 .raw and save to .nc.
"""

from __future__ import absolute_import, division, print_function
from .set_groups_ek60 import SetGroupsEK60
from .set_groups_azfp import SetGroupsAZFP


def SetGroups(file_path, echo_type, compress=True):
    """Wrapper function to set groups in converted files.

    Parameters
    ----------
    file_path : str
        Path to .nc file to be generated
    echo_type: str
        Type of echosounder from which data were generated
    compress: bool
        Whether or not to compress the backscatter data

    Returns
    -------
        Returns a specialized SetGroups object depending on
        the type of echosounder
    """

    # Returns specific SetGroup object
    if echo_type == "EK60":
        return SetGroupsEK60(file_path, compress)
    elif echo_type == "AZFP":
        return SetGroupsAZFP(file_path, compress)
    else:
        raise ValueError("Unsupported file type")
