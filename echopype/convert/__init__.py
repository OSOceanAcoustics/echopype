"""
Include code to unpack manufacturer-specific data files into an interoperable netCDF format.

The current version supports:

- Simrad EK60 echosounder ``.raw`` data
- Simrad EK80 echosounder ``.raw`` data
- ASL Environmental Sciences AZFP echosounder ``.01A`` data
"""
from .convert import Convert, ConvertEK80       # TODO remove ConvertEK80 in later version
from .parse_ek60 import ParseEK60
from .parse_ek80 import ParseEK80
from .parse_azfp import ParseAZFP
from .parse_ad2cp import ParseAd2cp
from .parse_base import ParseBase
from .set_groups_azfp import SetGroupsAZFP
from .set_groups_ek60 import SetGroupsEK60
from .set_groups_ek80 import SetGroupsEK80
from .set_groups_ad2cp import SetGroupsAd2cp