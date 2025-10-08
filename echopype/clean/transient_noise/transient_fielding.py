import numpy as np
import xarray as xr

from echopype.utils.compute import _lin2log, _log2lin


def _fielding_core_numpy(
    Sv_pr,
    r,
    r0,
    r1,
    n,
    thr,
    roff,
    jumps=5,
    maxts=-35,
    start=0,
):
    """
    Core Fielding detector (NumPy).

    Parameters
    ----------
    Sv_pr : array-like, shape (ping, range) or (range, ping)
        Sv in dB. Internally converted to shape (range, ping).
    r : array-like, shape (range,)
        Vertical coordinate in meters (monotonic). Used to compute steps/indices.
    r0, r1 : float
        Vertical window bounds (m). If invalid/outside data, returns all-False mask.
    n : int
        Half-width of temporal neighborhood (pings).
    thr : tuple[float, float]
        Thresholds (dB) for decision stages.
    roff : float
        Stop depth (m) for upward propagation.
    jumps : float
        Vertical step (m) when moving the window upward.
    maxts : float
        Max 75th percentile (dB) to treat a ping as “quiet”.
    start : int
        Number of initial pings flagged uncomputable in aux mask.

    Returns
    -------
    mask_bad_full : np.ndarray of bool, shape (ping, range)
        True = BAD (transient noise).
    mask_aux_full : np.ndarray of bool, shape (ping, range)
        True = uncomputable ping (e.g., insufficient context). Not used by the wrapper.
    """
    # transpose to (range, ping) to match your original code
    Sv = np.asarray(Sv_pr).T  # (range, ping)
    r = np.asarray(r)

    if r0 > r1:
        # same behavior as original: nothing masked if window invalid
        mask = np.zeros_like(Sv, dtype=bool)
        mask_ = np.zeros_like(Sv, dtype=bool)
        return mask.T, mask_.T

    if (r0 > r[-1]) or (r1 < r[0]):
        mask = np.zeros_like(Sv, dtype=bool)
        mask_ = np.zeros_like(Sv, dtype=bool)
        return mask.T, mask_.T

    up = np.argmin(abs(r - r0))
    lw = np.argmin(abs(r - r1))
    rmin = np.argmin(abs(r - roff))

    dr = float(np.nanmedian(np.diff(r)))
    sf = max(1, int(round(jumps / dr)))

    mask = np.zeros_like(Sv, dtype=bool)  # True = BAD
    mask_ = np.zeros_like(Sv, dtype=bool)  # True = "uncomputable" ping

    n_pings = Sv.shape[1]
    for j in range(start, n_pings):
        if (j - n < 0) or (j + n > n_pings - 1) or np.all(np.isnan(Sv[up:lw, j])):
            mask_[:, j] = True
        else:
            pingmedian = _lin2log(np.nanmedian(_log2lin(Sv[up:lw, j])))
            pingp75 = _lin2log(np.nanpercentile(_log2lin(Sv[up:lw, j]), 75))
            blockmedian = _lin2log(np.nanmedian(_log2lin(Sv[up:lw, j - n : j + n])))

            if (pingp75 < maxts) and ((pingmedian - blockmedian) > thr[0]):
                r0_, r1_ = up - sf, up
                while r0_ > rmin:
                    pingmedian = _lin2log(np.nanmedian(_log2lin(Sv[r0_:r1_, j])))
                    blockmedian = _lin2log(np.nanmedian(_log2lin(Sv[r0_:r1_, j - n : j + n])))
                    r0_, r1_ = r0_ - sf, r1_ - sf
                    if (pingmedian - blockmedian) < thr[1]:
                        break
                mask[r0_:, j] = True

    # restore to (ping, range) for xarray core-dims ("ping_time","range_sample")
    # also: pad the first `start` pings with False (good) / True (aux) so shape == input
    if start > 0:
        pad_bad = np.zeros((start, Sv.shape[0]), dtype=bool)  # False = keep
        pad_aux = np.ones((start, Sv.shape[0]), dtype=bool)  # True = uncomputable
        mask_bad_full = np.vstack([pad_bad, mask.T[:, : Sv.shape[1] - start]])
        mask_aux_full = np.vstack([pad_aux, mask_.T[:, : Sv.shape[1] - start]])
    else:
        mask_bad_full = mask.T
        mask_aux_full = mask_.T

    return mask_bad_full, mask_aux_full


def transient_noise_fielding(
    ds_Sv: xr.Dataset,
    var_name: str = "Sv",
    range_var: str = "depth",
    r0: float = 900,
    r1: float = 1000,
    n: int = 30,
    thr=(3, 1),
    roff: float = 20,
    jumps: float = 5,
    maxts: float = -35,
    start: int = 0,
) -> xr.DataArray:
    """
    Transient noise detector modified from the "fielding"
    function in `mask_transient.py`, originally written by
    Alejandro ARIZA for the Echopy library (C) 2020.

    Overview
    -------------------
    This algorithm identifies deep transient noise in echosounder
    data by comparing the echo level of each ping with its local
    temporal neighbourhood in a deep water window.
    It operates in linear Sv space and uses a two-stage decision:

    1. Deep window test – In a specified depth interval (e.g., 900–1000 m),
    compute the ping median and the median over neighbouring pings.
    If the ping’s deep-window 75th percentile is below maxts (i.e.,
    the window is not broadly high), and the ping median exceeds the
    neighborhood median by more than thr[0], mark the ping as potentially
    transient.

    2. Upward propagation – Move the vertical window upward in fixed steps
    (e.g., 5 m). Continue masking shallower ranges until the difference
    between the ping and block medians drops below the second threshold
    (thr[1]). This limits the mask to the part of the column affected
    by the transient.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        Dataset containing `var_name` (Sv, in dB) and `range_var`.
    var_name : str, default "Sv"
        Name of the Sv DataArray. Must have dims (..., "ping_time", "range_sample").
    range_var : str, default "depth"
        Name of the vertical coordinate. Can be 1-D over ("range_sample") or 2-D over
        ("ping_time","range_sample"); the wrapper reduces it to a 1-D vector per
        leading dimension (e.g., channel) by selecting the first ping.
    r0, r1 : float
        Upper/lower bounds of the vertical window (meters). If the window is invalid
        or outside the data range, nothing is masked.
    n : int
        Half-width of the temporal neighborhood (pings) used to compute the block median.
    thr : tuple[float, float], default (3, 1)
        Thresholds (dB) used in the two-stage decision.
    roff : float
        Minimum depth (m) to which the masking can propagate upward (stop depth).
    jumps : float, default 5
        Vertical step (m) used when iteratively moving the window upward.
    maxts : float, default -35
        Maximum allowable 75th-percentile Sv (dB) to consider a ping “quiet enough.”
    start : int, default 0
        Number of initial pings to mark as uncomputable in the auxiliary mask (internal).

    Returns
    -------
    xr.DataArray (bool)
        Boolean mask aligned to `ds_Sv[var_name]` with the **same dims and order**,
        where **True = VALID (keep)** and **False = transient noise**.
        Name: "fielding_mask_valid". Dtype: bool.

    Examples, to be used with dispatcher
    --------
    >>> mask_fielding = mask_transient_noise_dispatch(
        ds=ds_Sv,
        method="fielding",
        params={
            "var_name": "Sv",
            "range_var": "depth",
            "r0": 900,
            "r1": 1000,
            "n": 10,
            "thr": (3, 1),
            "roff": 20,
            "jumps": 5,
            "maxts": -35,
            "start": 0,
        },
    )
    """

    if var_name not in ds_Sv:
        raise ValueError(f"{var_name!r} not found in Dataset.")
    if range_var not in ds_Sv:
        raise ValueError(f"{range_var!r} not found in Dataset.")

    Sv_da = ds_Sv[var_name]

    # 1D range vector along 'range_sample'
    r_da = ds_Sv[range_var]
    if {"ping_time", "range_sample"}.issubset(r_da.dims):
        r_1d = r_da.isel(ping_time=0)
    elif (r_da.ndim == 1) and ("range_sample" in r_da.dims):
        r_1d = r_da
    else:
        raise ValueError(f"Cannot infer 1D '{range_var}' from dims {r_da.dims}.")

    mask_bad, mask_aux = xr.apply_ufunc(
        _fielding_core_numpy,
        Sv_da,
        r_1d,
        input_core_dims=[["ping_time", "range_sample"], ["range_sample"]],
        output_core_dims=[["ping_time", "range_sample"], ["ping_time", "range_sample"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool, bool],
        kwargs=dict(r0=r0, r1=r1, n=n, thr=thr, roff=roff, jumps=jumps, maxts=maxts, start=start),
    )

    # Flip to True = VALID for apply_mask
    mask_valid = (
        (~mask_bad)
        .astype(bool)
        .rename("fielding_mask_valid")
        .assign_attrs({"meaning": "True = VALID (False = transient noise)"})
    )
    # Ensure dims order matches `Sv_da`
    return mask_valid.transpose(*Sv_da.dims)
