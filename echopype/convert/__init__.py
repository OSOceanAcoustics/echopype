"""
Include code to unpack manufacturer-specific data files into an interoperable netCDF format.

The current version supports:

- Simrad EK60 echosounder ``.raw`` data
- ASL Environmental Sciences AZFP echosounder ``.01A`` data
"""
from .convert import Convert
from .ek60 import ConvertEK60
from .azfp import ConvertAZFP