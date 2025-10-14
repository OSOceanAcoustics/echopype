import numpy as np
import xarray as xr
from scipy.ndimage import binary_dilation

from echopype.utils.compute import _lin2log, _log2lin


def _matecho_core_numpy(
    Sv,
    r,
    bottom_depth=None,
    start_depth=220,
    window_meter=450,
    window_ping=100,
    percentile=25,
    delta_db=12,
    extend_ping=0,
    min_window=20,
):
    """
    Matecho-style transient detector (NumPy core operating on columns).

    Parameters
    ----------
    Sv : np.ndarray, shape (range, ping) or (ping, range)
        Sv in dB. Internally converted to (range, ping).
    r : array-like, shape (range,)
        Vertical coordinate (meters). Must match Sv’s range axis length.
    bottom_depth : array-like or None, shape (ping,), optional
        Bottom depth per ping (meters). If None/NaN, set to r[-1] per ping.
    start_depth : float
        Top of vertical detection window (m).
    window_meter : float
        Vertical window thickness (m).
    window_ping : int
        Temporal window width (pings) centered on each ping for local stats.
    percentile : float
        Reference percentile (dB) computed from the local window.
    delta_db : float
        Excess over percentile (dB) required to flag a ping.
    extend_ping : int
        Horizontal dilation (pings) for flagged columns.
    min_window : float
        Minimum usable window height (m). Skip if smaller.

    Returns
    -------
    mask_bad : np.ndarray of bool, shape (range, ping)
        True = BAD (transient noise) — columns for flagged pings.
    aux_2d : np.ndarray of bool, shape (range, ping)
        Reserved for diagnostics (currently all False).
    """

    n_ping = Sv.shape[1]
    r = np.asarray(r)
    depth_mask = (r >= start_depth) & (r <= start_depth + window_meter)

    # Sv must be (range, ping)
    if Sv.shape[0] != len(r):
        if Sv.shape[1] == len(r):
            Sv = Sv.T
        else:
            raise ValueError(f"Mismatch Sv {Sv.shape} vs range {len(r)}")

    # bottom_depth
    if bottom_depth is None:
        bottom_depth = np.full(n_ping, r[-1], dtype=float)
    else:
        bottom_depth = np.array(bottom_depth, dtype=float, copy=True)
        bottom_depth[np.isnan(bottom_depth)] = r[-1]

    pings_bad = np.zeros(n_ping, dtype=bool)

    for j in range(n_ping):
        j0 = max(0, j - window_ping // 2)
        j1 = min(n_ping, j + window_ping // 2)
        local_bottom = np.min(bottom_depth[j0:j1])

        refined_mask = depth_mask & (r < local_bottom)
        if not np.any(refined_mask):
            continue

        H = (r[1] - r[0]) * np.sum(refined_mask)
        if H < min_window:
            continue

        sv_window = Sv[refined_mask, j0:j1]
        sv_ping = Sv[refined_mask, j]

        sv_window_flat = sv_window[~np.isnan(sv_window)]
        if sv_window_flat.size == 0:
            continue

        pctl_val = np.percentile(sv_window_flat, percentile)  # could compute in linear
        sv_ping_lin = _log2lin(sv_ping)
        sv_ping_mean_db = _lin2log(np.nanmean(sv_ping_lin))

        if sv_ping_mean_db > pctl_val + delta_db:
            pings_bad[j] = True

    if extend_ping > 0:
        structure = np.ones(2 * extend_ping + 1, dtype=bool)
        pings_bad = binary_dilation(pings_bad, structure=structure)

    mask_bad = np.zeros_like(Sv, dtype=bool)  # (range, ping)
    mask_bad[:, pings_bad] = True
    aux_2d = np.zeros_like(mask_bad, dtype=bool)
    return mask_bad, aux_2d  # True = BAD


def transient_noise_matecho(
    ds: xr.Dataset,
    var_name: str = "Sv",
    range_var: str = "depth",
    time_var: str = "ping_time",
    bottom_var: str | None = None,
    start_depth: float = 220,
    window_meter: float = 450,
    window_ping: int = 100,
    percentile: float = 25,
    delta_db: float = 12,
    extend_ping: int = 0,
    min_window: float = 20,
) -> xr.DataArray:
    """
    Matecho-style transient-noise mask that masks the entire water column for noisy pings.

    Overview
    --------
    Flags entire pings as transient when, within a deep window, the ping’s
    mean Sv (computed in linear units, then converted back to dB) exceeds a
    local reference percentile by `delta_db`.

    1) Depth window: Use a vertical slice from `start_depth` to
       `start_depth + window_meter`, limited by a local bottom (if provided;
       otherwise r[-1]). Skip if usable height < `min_window`.
    2) Local reference: For ping j, form a temporal neighborhood
       [j - window_ping/2, j + window_ping/2] and compute the chosen `percentile`
       (in dB) over that neighborhood within the deep window.
    3) Mean Sv within the depth window (`ping_mean_db`): 
    Compute the mean Sv (in the linear domain and converted back to dB).
    4) Decision: If `ping_mean_db > percentile + delta_db`, mark ping j as BAD.
       Optionally dilate flagged pings horizontally by `extend_ping`
       (binary dilation).

    This function prepares the vertical coordinate (and optional bottom), then
    calls a NumPy core via `xarray.apply_ufunc`, vectorized across leading dims
    (e.g., `channel`).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing `var_name` (Sv in dB) and `range_var`.
    var_name : str, default "Sv"
        Name of the Sv DataArray. Must include dims (..., "ping_time", "range_sample").
    range_var : str, default "depth"
        Vertical coordinate. Can be 1-D or 2-D; reduced to a 1-D vector per leading dim.
    time_var : str, default "ping_time"
        Time/ping dimension name.
    bottom_var : str | None, default None
        Name of a 1-D bottom-depth-per-ping variable (meters). If None or NaN, the
        maximum range is used. **Currently not plumbed through**; future work.
    start_depth : float, default 220
        Top of the vertical detection window (m).
    window_meter : float, default 450
        Vertical window thickness (m).
    window_ping : int, default 100
        Temporal window width (pings) for the local reference statistics.
    percentile : float, default 25
        Reference percentile (in dB space) of the local window.
    delta_db : float, default 12
        Threshold (dB) added to the percentile for flagging.
    extend_ping : int, default 0
        Optional horizontal dilation in pings applied to flagged columns.
    min_window : float, default 20
        Minimum usable vertical window height (m); skip if smaller.

    Returns
    -------
    xr.DataArray (bool)
        Boolean mask aligned to `ds[var_name]` with the **same dims and order**,
        where **True = VALID (keep)** and **False = transient noise**.
        Name: "matecho_mask_valid". Dtype: bool.

    Examples to use with dispatcher
    --------
    >>> mask_matecho = mask_transient_noise_dispatch(
        ds=ds_Sv,
        method="matecho",
        params={
            "var_name": "Sv",
            "range_var": "depth",
            "time_var": "ping_time",
            "bottom_var": None,
            "start_depth": 700,
            "window_meter": 300,
            "window_ping": 50,
            "percentile": 25,
            "delta_db": 8,
            "extend_ping": 0,
            "min_window": 5,
        },
    )
    """

    if var_name not in ds:
        raise ValueError(f"{var_name!r} not found.")
    if range_var not in ds:
        raise ValueError(f"{range_var!r} not found.")
    if time_var not in ds[var_name].dims:
        raise ValueError(f"{time_var!r} must be a dim of {var_name!r}.")

    Sv_da = ds[var_name]

    rng_dim = None
    for cand in (range_var, "range_sample"):
        if cand in Sv_da.dims:
            rng_dim = cand
            break
    if rng_dim is None:
        raise ValueError(f"No range dim in {var_name!r} dims {Sv_da.dims}")

    # per-channel depth vector (channel, range_sample) -> drop time
    r_1d = ds[range_var].isel({time_var: 0})  # keeps 'channel'

    # 1D time vector
    t_1d = ds[var_name][time_var]  # ('ping_time',)
    if t_1d.ndim != 1:
        t_1d = t_1d.squeeze()

    # if provided, use ds[bottom_var] (1D per ping); else NaN → core falls back to r[-1]
    if bottom_var is not None and bottom_var in ds:
        bottom_1d = ds[bottom_var].transpose(..., time_var).astype(float)
    else:
        bottom_1d = xr.DataArray(np.full(ds[var_name][time_var].shape, np.nan), dims=(time_var,))

    # ensure Sv’s last two dims are (range, time) for the core (keep leading dims like channel)
    Sv_core = Sv_da.transpose(..., rng_dim, time_var)

    mask_bad, _ = xr.apply_ufunc(
        _matecho_core_numpy,
        Sv_core,
        r_1d,
        bottom_1d,
        input_core_dims=[
            [rng_dim, time_var],
            [rng_dim],
            [time_var],
        ],
        output_core_dims=[
            [rng_dim, time_var],
            [rng_dim, time_var],
        ],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool, bool],
        kwargs=dict(
            start_depth=start_depth,
            window_meter=window_meter,
            window_ping=window_ping,
            percentile=percentile,
            delta_db=delta_db,
            extend_ping=extend_ping,
            min_window=min_window,
        ),
    )

    mask_valid = (
        (~mask_bad)
        .astype(bool)
        .rename("matecho_mask_valid")
        .assign_attrs({"meaning": "True = VALID (False = transient noise)"})
    )

    # Return with SAME dims/order as input Sv (e.g., ('channel','ping_time','range_sample'))
    return mask_valid.transpose(*Sv_da.dims)
