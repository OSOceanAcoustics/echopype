import xarray as xr

from ...utils.prov import add_processing_level
from .detect_basic import DetectBasic
from .detect_blackwell import DetectBlackwell
from .detect_manual import DetectManual

METHODS = {"basic": DetectBasic, "manual": DetectManual, "blackwell": DetectBlackwell}


@add_processing_level(
    "L*B"
)  # verify with WuJung if ok , because same levels as clean/background_correction??
def compute_bottom(ds: xr.Dataset, method: str = "basic", **kwargs) -> xr.Dataset:
    """
    Detect seafloor using a selected method.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing Sv and echo_range.
    method : str
        Detection method ("dumb" supported).

    Returns
    -------
    xr.Dataset
        Bottom detection result.
    """
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")
    detector = METHODS[method](ds, **kwargs)
    return detector.compute()
