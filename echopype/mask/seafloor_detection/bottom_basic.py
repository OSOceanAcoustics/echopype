import numpy as np
import xarray as xr

from echopype.mask.seafloor_detection.utils import (
    _check_inputs,
    _validate_threshold,
)


def bottom_basic(
    ds: xr.Dataset,
    var_name: str,
    channel: str,
    threshold: float = -50.0,
    offset_m: float = 0.5,
    bin_skip_from_surface: int = 200,
) -> xr.DataArray:
    """
    Simple threshold-based seafloor detection returning a 1-D bottom line (depth).

    Summary
    -------
    For the selected `channel`, the algorithm skips the top `bin_skip_from_surface`
    range bins, then finds (per ping) the first range sample where `Sv` is within a
    user-defined dB interval. That depth (looked up from `depth`) minus `offset_m`
    is returned as the bottom.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
          • `var_name` (Sv in dB), typically with dims
            (`channel`, `ping_time`, `range_sample`);
          • a vertical coordinate (e.g., `depth`) aligned with `range_sample`.
    var_name : str
        Name of the Sv variable to use (e.g., `"Sv"`).
    channel : str
        Channel identifier to process (must match an entry in `ds['channel']`).
    threshold : float or tuple(float, float), default -50.0
        Sv threshold(s) in dB. If a single float is given, it is treated as the
        lower bound and the upper bound is set to 10 dB above the lower
        bound. If a 2-tuple `(tmin, tmax)` is provided, both the lower
        and upper bounds are used directly.
    offset_m : float, default 0.5
        Meters subtracted from the detected crossing to place the bottom slightly
        above the echo maximum.
    bin_skip_from_surface : int, default 200
        Number of shallow range bins to ignore before searching (index units,
        not meters).

    Returns
    -------
    xr.DataArray
        1-D bottom depth per `ping_time` (no `channel` dimension) with attributes:
        `detector='basic'`, `threshold_min`, `threshold_max`, `offset_m`,
        `bin_skip_from_surface`, and `channel`.

    Notes
    -----
    * Depth lookup uses `depth.isel(ping_time=0)` as the reference vector.
    * If no sample meets the threshold in a ping, `argmax` on a False-only
      mask returns 0 (i.e., the first searched bin after the skipped region).
    """

    Sv_sel, depth_sel = _check_inputs(ds, var_name, channel)
    tmin, tmax = _validate_threshold(threshold)

    depth_ref = depth_sel.isel(ping_time=0)
    Sv_sliced = Sv_sel.isel(range_sample=slice(bin_skip_from_surface, None))
    cond = (Sv_sliced > tmin) & (Sv_sliced < tmax)

    # Find index of first match along range_sample
    idx = cond.argmax(dim="range_sample") + bin_skip_from_surface  # add back skipped samples

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

    # Return 1D DataArray with attributes
    return xr.DataArray(
        bottom_depth.data,
        dims=["ping_time"],
        coords={"ping_time": ds["ping_time"]},
        name="bottom_depth",
        attrs={
            "detector": "basic",
            "threshold_min": float(tmin),
            "threshold_max": float(tmax),
            "offset_m": float(offset_m),
            "bin_skip_from_surface": int(bin_skip_from_surface),
            "channel": str(channel),
        },
    )
