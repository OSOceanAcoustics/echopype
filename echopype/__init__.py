from __future__ import absolute_import, division, print_function

from _echopype_version import version as __version__  # noqa

from . import calibrate, clean, commongrid, consolidate, mask, utils
from .convert.api import open_raw
from .echodata.api import open_converted
from .echodata.combine import combine_echodata
from .utils.io import init_ep_dir
from .utils.log import verbose

# Turn off verbosity for echopype
verbose(override=True)

init_ep_dir()

__all__ = [
    "calibrate",
    "clean",
    "combine_echodata",
    "commongrid",
    "consolidate",
    "mask",
    "metrics",
    "open_converted",
    "open_raw",
    "utils",
    "verbose",
]
