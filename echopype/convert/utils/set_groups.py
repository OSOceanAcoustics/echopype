"""
Functions to unpack Simrad EK60 .raw and save to .nc.
"""

from __future__ import absolute_import, division, print_function
from .set_groups_ek60 import SetGroupsEK60
from .set_groups_ek80 import SetGroupsEK80
from .set_groups_azfp import SetGroupsAZFP


class SetGroups:
    def __new__(cls, file_path, echo_type):
        """Wrapper class to use for setting groups in .nc files.

        Parameters
        ----------
        file_path : str
            Path to .nc file to be generated
        echo_type: str
            Type of echosounder from which data were generated

        Returns
        -------
            Returns a specialized SetGroups object depending on
            the type of echosounder
        """

        # Returns specific EchoData object
        if echo_type == "EK60":
            return SetGroupsEK60(file_path)
        elif echo_type == "EK80":
            return SetGroupsEK80(file_path)
        elif echo_type == "AZFP":
            return SetGroupsAZFP(file_path)
        else:
            raise ValueError("Unsupported file type")
