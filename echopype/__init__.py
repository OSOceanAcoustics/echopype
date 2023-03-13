from __future__ import absolute_import, division, print_function

from _echopype_version import version as __version__  # noqa

from . import calibrate, consolidate, mask, preprocess, utils
from .convert.api import open_raw
from .echodata.api import open_converted
from .echodata.combine import combine_echodata
from .utils.io import init_ep_dir
from .utils.log import verbose

# Turn off verbosity for echopype
verbose(override=True)

init_ep_dir()

__all__ = [
    "open_raw",
    "open_converted",
    "combine_echodata",
    "calibrate",
    "consolidate",
    "mask",
    "preprocess",
    "utils",
    "verbose",
]
