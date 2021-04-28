"""
UI class for converting raw data from different echosounders to netcdf or zarr.
"""
from echopype.echodata.echodata import EchoData
import os
import warnings
from datetime import datetime as dt
from pathlib import Path

import fsspec
from fsspec.implementations.local import LocalFileSystem

from ..convert import convert_combine as combine_fcn
from ..utils import io
from .parse_azfp import ParseAZFP
from .parse_ek60 import ParseEK60
from .parse_ek80 import ParseEK80
from .set_groups_azfp import SetGroupsAZFP
from .set_groups_ek60 import SetGroupsEK60
from .set_groups_ek80 import SetGroupsEK80
from .api import open_raw


warnings.simplefilter("always", DeprecationWarning)

MODELS = {
    "AZFP": {
        "ext": ".01A",
        "xml": True,
        "parser": ParseAZFP,
        "set_groups": SetGroupsAZFP,
    },
    "EK60": {
        "ext": ".raw",
        "xml": False,
        "parser": ParseEK60,
        "set_groups": SetGroupsEK60,
    },
    "EK80": {
        "ext": ".raw",
        "xml": False,
        "parser": ParseEK80,
        "set_groups": SetGroupsEK80,
    },
    "EA640": {
        "ext": ".raw",
        "xml": False,
        "parser": ParseEK80,
        "set_groups": SetGroupsEK80,
    },
}

NMEA_SENTENCE_DEFAULT = ["GGA", "GLL", "RMC"]

CONVERT_PARAMS = [
    "survey_name",
    "platform_name",
    "platform_code_ICES",
    "platform_type",
    "water_level",
    "nmea_gps_sentence",
]


# TODO: Used for backwards compatibility. Delete in future versions
def ConvertEK80(_filename=""):
    warnings.warn(
        "`ConvertEK80` is deprecated, use `echopype.open_raw` instead.",
        DeprecationWarning,
        2,
    )
    return Convert(file=_filename, model="EK80")


class Convert:
    """Object for converting data from manufacturer-specific formats to a standardized format."""

    _instance = None

    def __new__(cls, file=None, xml_path=None, model=None, storage_options=None):
        warnings.warn(
            "Calling `echopype.Convert` is deprecated, "
            "use `echopype.open_raw(raw_file, sonar_model, ...)` instead.",
            DeprecationWarning,
            2,
        )
        if not isinstance(cls._instance, cls):
            cls._instance = open_raw(
                raw_file=file,
                sonar_model=model,
                xml_path=xml_path,
                storage_options=storage_options,
            )
        return cls._instance
