"""
EchoData is an object that handles interfacing raw converted data.
It is used for calibration and other processing.
"""

from . import convention
from .echodata import EchoData

__all__ = ["EchoData", "convention"]
