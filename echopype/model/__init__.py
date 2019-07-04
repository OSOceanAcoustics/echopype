"""
Include methods to manipulate echo data that is already converted to netCDF.
Currently the operations include:

- calibration
- noise removal
- calculating mean volume backscattering strength (MVBS)

The current version supports EK60 `.raw` data.
"""

from .ek60 import EchoDataEK60
from .azfp import EchoDataAZFP
