"""
EchoData is an object that handles interfacing raw converted data.
It is used for calibration and other processing.
"""
from .echodata import EchoData
from . import convention

__all__ = ["EchoData", "convention"]
