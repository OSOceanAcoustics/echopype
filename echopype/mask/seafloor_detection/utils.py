import xarray as xr


def _check_inputs(ds: xr.Dataset, var_name: str, channel: str):
    """Validate dataset and select the reference channel for bottom detection."""
    if var_name not in ds:
        raise KeyError(f"{var_name!r} not found in dataset")
    if "depth" not in ds:
        raise KeyError("'depth' variable not found in dataset")
    if "channel" not in ds.coords:
        raise ValueError("Dataset must have 'channel' coordinate")

    Sv_all = ds[var_name]
    depth_all = ds["depth"]
    Sv_sel = Sv_all.sel(channel=channel)
    depth_sel = depth_all.sel(channel=channel)

    # Ensure uniform depth grid across pings
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
