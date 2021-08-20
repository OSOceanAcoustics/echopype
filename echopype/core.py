import os
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, Union

from fsspec.mapping import FSMap
from typing_extensions import Literal

from .convert.parse_ad2cp import ParseAd2cp
from .convert.parse_azfp import ParseAZFP
from .convert.parse_ek60 import ParseEK60
from .convert.parse_ek80 import ParseEK80
from .convert.set_groups_ad2cp import SetGroupsAd2cp
from .convert.set_groups_azfp import SetGroupsAZFP
from .convert.set_groups_ek60 import SetGroupsEK60
from .convert.set_groups_ek80 import SetGroupsEK80

if TYPE_CHECKING:
    # Please keep SonarModelsHint updated with the keys of the SONAR_MODELS dict
    SonarModelsHint = Literal["AZFP", "EK60", "EK80", "EA640", "AD2CP"]
    PathHint = Union[str, os.PathLike, FSMap]
    FileFormatHint = Literal[".nc", ".zarr"]
    EngineHint = Literal["netcdf4", "zarr"]


def validate_azfp_ext(test_ext: str):
    if not re.fullmatch(r"\.\d{2}[a-zA-Z]", test_ext):
        raise ValueError(
            'Expecting a file in the form ".XXY" '
            f"where XX is a number and Y is a letter but got {test_ext}"
        )


def validate_ext(ext: str) -> Callable[[str], None]:
    def inner(test_ext: str):
        if ext.casefold() != test_ext.casefold():
            raise ValueError(f"Expecting a {ext} file but got {test_ext}")

    return inner


SONAR_MODELS: Dict["SonarModelsHint", Dict[str, Any]] = {
    "AZFP": {
        "validate_ext": validate_azfp_ext,
        "xml": True,
        "parser": ParseAZFP,
        "set_groups": SetGroupsAZFP,
        "concat_dims": {
            "platform": None,
            "nmea": "location_time",
            "vendor": ["ping_time", "frequency"],
            "default": "ping_time",
        },
        "concat_data_vars": {
            "platform": "all",
            "default": "minimal",
        },
    },
    "EK60": {
        "validate_ext": validate_ext(".raw"),
        "xml": False,
        "parser": ParseEK60,
        "set_groups": SetGroupsEK60,
        "concat_dims": {
            "platform": ["location_time", "ping_time"],
            "nmea": "location_time",
            "vendor": None,
            "default": "ping_time",
        },
        "concat_data_vars": {
            "default": "minimal",
        },
    },
    "EK80": {
        "validate_ext": validate_ext(".raw"),
        "xml": False,
        "parser": ParseEK80,
        "set_groups": SetGroupsEK80,
        "concat_dims": {
            "platform": ["location_time", "mru_time"],
            "nmea": "location_time",
            "vendor": None,
            "default": "ping_time",
        },
        "concat_data_vars": {
            "default": "minimal",
        },
    },
    "EA640": {
        "validate_ext": validate_ext(".raw"),
        "xml": False,
        "parser": ParseEK80,
        "set_groups": SetGroupsEK80,
        "concat_dims": {
            "platform": ["location_time", "mru_time"],
            "nmea": "location_time",
            "vendor": None,
            "default": "ping_time",
        },
        "concat_data_vars": {
            "default": "minimal",
        },
    },
    "AD2CP": {
        "validate_ext": validate_ext(".ad2cp"),
        "xml": False,
        "parser": ParseAd2cp,
        "set_groups": SetGroupsAd2cp,
        "concat_dims": {
            "platform": "ping_time",
            "nmea": "location_time",
            "vendor": None,
            "default": "ping_time",
        },
        "concat_data_vars": {
            "default": "minimal",
        },
    },
}
