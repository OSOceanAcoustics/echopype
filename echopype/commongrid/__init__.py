from ..utils.misc import is_package_installed
from .api import compute_MVBS, compute_MVBS_index_binning, compute_NASC

__all__ = [
    "compute_MVBS",
    "compute_NASC",
    "compute_MVBS_index_binning",
]

# Optional dependency, only import
# if scitools-iris is installed
if is_package_installed("iris"):
    from .regrid import regrid_Sv  # noqa

    __all__.append("regrid_Sv")
