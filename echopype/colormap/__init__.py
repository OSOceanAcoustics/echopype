"""
Colormaps for plotting echograms.

To use this subpackage the `Matplotlib` and `cmocean` packages must be installed.

Importing this package adds echogram-specific colormaps to Matplotlib's existing colormaps. These
always start with `ep` and come in pairs (the colormap and a reversed version with a name
ending in `_r`). The list of colormaps added can be found with this code snippet:

>>> import echopype.colormap
>>> from matplotlib import colormaps
>>> cmaps = [name for name in colormaps if name.startswith('ep')]

"""

from . import cm

__all__ = ["cm"]
