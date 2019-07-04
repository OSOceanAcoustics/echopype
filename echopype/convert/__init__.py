"""
Include code to unpack manufacturer-specific data files into an interoperable netCDF format.
"""

from .ek60 import ConvertEK60
from .azfp import ConvertAZFP
from .set_nc_groups import SetGroups
