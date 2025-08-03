import numpy as np
import xarray as xr

from echopype.mask.seafloor_detection.utils import (
    _check_inputs,
    _validate_threshold,
)


def detect_basic(
    ds: xr.Dataset,
    var_name: str,
    channel: str,
    threshold: float = -50.0,
    offset_m: float = 0.5,
    surface_skip: int = 200,
) -> xr.DataArray:
    """
    Basic seafloor detection algorithm returning 1D bottom line (depth).
    """

    Sv_sel, depth_sel = _check_inputs(ds, var_name, channel)
    tmin, tmax = _validate_threshold(threshold)

    depth_ref = depth_sel.isel(ping_time=0)
    Sv_sliced = Sv_sel.isel(range_sample=slice(surface_skip, None))
    cond = (Sv_sliced > tmin) & (Sv_sliced < tmax)

    # Find index of first match along range_sample
    idx = cond.argmax(dim="range_sample") + surface_skip  # add back skipped samples

    # Map index to depth using depth_ref
    bottom_depth = xr.apply_ufunc(
        lambda i: depth_ref[int(i)] if np.isfinite(i) else np.nan,
        idx.astype(float),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ) - float(
        offset_m
    )  # we remove the offset

    return bottom_depth.rename("bottom_depth").assign_attrs(
        {
            "detector": "basic",
            "threshold_min": float(tmin),
            "threshold_max": float(tmax),
            "offset_m": float(offset_m),
            "surface_skip": int(surface_skip),
            "channel": str(channel),
        }
    )
