"""
Include code to unpack manufacturer-specific data files into an interoperable netCDF format.

The current version supports:

- Simrad EK60 echosounder ``.raw`` data
- Simrad EK80 echosounder ``.raw`` data
- ASL Environmental Sciences AZFP echosounder ``.01A`` data
"""
from .convert_ui import Convert
from .ek60 import ParseEK60
from .ek80 import ParseEK80
from .azfp import ParseAZFP
from .set_groups_azfp import SetGroupsAZFP
from .set_groups_ek60 import SetGroupsEK60
from .set_groups_ek80 import SetGroupsEK80
