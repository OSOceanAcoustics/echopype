"""
Base echo data model.

Include methods to do the following:
- calibration
- noise removal
- calculate MVBS

The current version supports EK60 .raw data.
"""

from .ek60 import EchoData
