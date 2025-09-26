import xarray as xr


def _check_inputs(ds: xr.Dataset, var_name: str, channel: str, required_vars: list[str] = None):
    """Validate dataset and select the reference channel for bottom detection."""
    if var_name not in ds:
        raise KeyError(f"{var_name!r} not found in dataset")
    if "depth" not in ds:
        raise KeyError("'depth' variable not found in dataset")
    if "channel" not in ds.coords:
        raise ValueError("Dataset must have 'channel' coordinate")

    required_vars = required_vars or []
    for var in required_vars:
        if var not in ds:
            raise KeyError(f"Required variable {var!r} not found in dataset")

    Sv_all = ds[var_name]
    depth_all = ds["depth"]
    Sv_sel = Sv_all.sel(channel=channel)
    depth_sel = depth_all.sel(channel=channel)

    # Ensure uniform depth grid
    depth_ref = depth_sel.isel(ping_time=0)
    is_uniform = (abs(depth_sel - depth_ref).max(dim="range_sample") < 1e-16).all()
    if not bool(is_uniform):
        raise ValueError("Depth grid varies across ping_time for the selected channel.")

    return Sv_sel, depth_sel


def _validate_threshold(threshold):
    """Ensure threshold is a valid tuple (tmin, tmax)."""
    if isinstance(threshold, (int, float)):
        tmin, tmax = float(threshold), float(threshold) + 10.0
    else:
        tmin, tmax = map(float, threshold)
        if tmax <= tmin:
            raise ValueError("threshold upper bound must be > lower bound")
    return tmin, tmax


def _parse_blackwell_thresholds(threshold):
    """
    Parse threshold for Blackwell detection.

    Returns
    -------
    tuple : (tSv, ttheta, tphi)
        Thresholds for Sv (dB), angle_major, angle_minor.
    """
    if isinstance(threshold, (list, tuple)):
        if len(threshold) == 3:
            tSv, ttheta, tphi = threshold
        elif len(threshold) == 2:
            tSv, ttheta, tphi = threshold[0], 702, 282
        else:
            raise ValueError("`threshold` must have 1, 2, or 3 values")
    elif isinstance(threshold, (int, float)):
        tSv, ttheta, tphi = threshold, 702, 282
    else:
        raise TypeError("`threshold` must be float or tuple/list of 1–3 floats")

    return float(tSv), float(ttheta), float(tphi)
