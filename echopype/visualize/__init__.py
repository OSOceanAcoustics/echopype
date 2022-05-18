"""
Visualization module to quickly plot raw, Sv, and MVBS dataset.

**NOTE: To use this subpackage. `Matplotlib` and `cmocean` package must be installed.**
"""
from .api import create_echogram
from . import cm

__all__ = ["create_echogram", "cm"]
