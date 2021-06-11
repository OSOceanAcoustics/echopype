from typing import Literal

# from .convert import (
#     ParseAZFP,
#     SetGroupsAZFP,
#     ParseEK60,
#     SetGroupsEK60,
#     ParseEK80,
#     SetGroupsEK80,
#     ParseAd2cp,
#     SetGroupsAd2cp,
# )

# avoid circular imports
from .convert.parse_ad2cp import ParseAd2cp
from .convert.parse_azfp import ParseAZFP
from .convert.parse_ek60 import ParseEK60
from .convert.parse_ek80 import ParseEK80
from .convert.set_groups_ad2cp import SetGroupsAd2cp
from .convert.set_groups_azfp import SetGroupsAZFP
from .convert.set_groups_ek60 import SetGroupsEK60
from .convert.set_groups_ek80 import SetGroupsEK80

SONAR_MODELS = {
    "AZFP": {
        "ext": ".01A",
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
        "ext": ".raw",
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
        "ext": ".raw",
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
        "ext": ".raw",
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
        "ext": ".ad2cp",
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
# Please update keep this updated with the keys of the SONAR_MODELS dict
SONAR_MODELS_TYPE_HINT = Literal["AZFP", "EK60", "EK80", "EA640", "AD2CP"]
