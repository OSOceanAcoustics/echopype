import numpy as np
import scipy.ndimage as ndi
import xarray as xr


def shoal_weill(
    ds: xr.Dataset,
    var_name: str,
    channel: str | None = None,
    thr: float = -70.0,
    maxvgap: int = 5,
    maxhgap: int = 0,
    minvlen: int = 0,
    minhlen: int = 0,
) -> xr.DataArray:
    """
    Transient noise detector modified from the "weill"
    function in `mask_shoals.py`, originally written by
    Alejandro ARIZA for the Echopy library (C) 2020.

    Detects and masks shoals following the algorithm described in:

        "Weill et al. (1993): MOVIES-B — an acoustic detection description
        software . Application to shoal species' classification".

    Contiguous regions of Sv above a given threshold are grouped
    as a single shoal, following the contiguity rules of Weill et al. (1993):s

    - Vertical contiguity: Gaps along the ping are tolerated
    up to roughly half the pulse length.

    - Horizontal contiguity: Features in consecutive pings are
    considered part of the same shoal if at least one sample
    occurs at the same range depth.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing `var_name` with Sv in dB.
    var_name : str
        Name of the Sv variable in `ds`.
    channel : str | None
        If a "channel" dimension exists, select this channel.
    thr : float
        Threshold in dB (keep values <= thr).
    maxvgap : int
        Max vertical gap (in range samples) to fill.
    maxhgap : int
        Max horizontal gap (in pings) to fill.
    minvlen : int
        Minimum vertical length (in range samples) to keep a feature.
    minhlen : int
        Minimum horizontal length (in pings) to keep a feature.

    Returns
    -------
    xr.DataArray
        Boolean mask with dims ("ping_time", "range_sample") where True = detected.
    """
    if var_name not in ds:
        raise ValueError(f"Variable '{var_name}' not found in dataset")

    var = ds[var_name]

    # If multi-channel, select one
    if "channel" in var.dims:
        if channel is None:
            raise ValueError("Please specify 'channel' for multi-channel data.")
        var = var.sel(channel=channel)

    # Ensure we have the two core dims
    if not {"ping_time", "range_sample"}.issubset(set(var.dims)):
        raise ValueError(
            f"'{var_name}' must have dims including 'ping_time' and 'range_sample', "
            f"got {tuple(var.dims)}"
        )

    # Arrange as (range, ping) for processing, similar to echopy
    Sv = var.transpose("range_sample", "ping_time").values

    # --- 1) Thresholding: keep (mask=True) where Sv <= thr
    mask = np.ma.masked_greater(Sv, thr).mask
    if np.isscalar(mask):
        mask = np.zeros_like(Sv, dtype=bool)

    n_range, n_ping = mask.shape

    # --- 2) Fill short vertical gaps per ping
    # Work on each column (over range axis)
    for jdx in range(n_ping):
        col = mask[:, jdx]
        # Label False regions (gaps) within the True mask
        labelled = ndi.label(~col)[0]
        # If all False or all True, nothing to do
        if (labelled == 0).all() or (labelled == 1).all():
            continue
        for lab in range(1, labelled.max() + 1):
            gap = labelled == lab
            gap_size = int(gap.sum())
            if gap_size <= maxvgap:
                idx = np.where(gap)[0]
                # Do not fill if the gap touches top/bottom boundary
                if 0 in idx or (n_range - 1) in idx:
                    continue
                mask[idx, jdx] = True

    # --- 3) Fill short horizontal gaps per depth
    # Work on each row (over ping axis)
    for idx in range(n_range):
        row = mask[idx, :]
        labelled = ndi.label(~row)[0]
        if (labelled == 0).all() or (labelled == 1).all():
            continue
        for lab in range(1, labelled.max() + 1):
            gap = labelled == lab
            gap_size = int(gap.sum())
            if gap_size <= maxhgap:
                jdxs = np.where(gap)[0]
                # Do not fill if the gap touches left/right boundary
                if 0 in jdxs or (n_ping - 1) in jdxs:
                    continue
                mask[idx, jdxs] = True

    # --- 4) Remove features smaller than (minvlen, minhlen)
    # Label True regions and filter by size in (range, ping) coordinates
    features = ndi.label(mask)[0]
    if features.max() > 0:
        for lab in range(1, features.max() + 1):
            feat = features == lab
            if not feat.any():
                continue
            ii, jj = np.where(feat)
            vlen = int(ii.max() - ii.min() + 1)  # vertical length in samples
            hlen = int(jj.max() - jj.min() + 1)  # horizontal length in pings
            if (vlen < minvlen) or (hlen < minhlen):
                mask[ii, jj] = False

    # Return as (ping_time, range_sample) to match echopype convention
    out = xr.DataArray(
        mask.T.astype(bool),
        dims=("ping_time", "range_sample"),
        coords={"ping_time": ds["ping_time"], "range_sample": ds["range_sample"]},
        name="shoal_mask_weill",
        attrs={
            "description": f"Weill-style threshold+gap-fill mask on '{var_name}'",
            "threshold_dB": float(thr),
            "maxvgap": int(maxvgap),
            "maxhgap": int(maxhgap),
            "minvlen": int(minvlen),
            "minhlen": int(minhlen),
            **({"channel": str(channel)} if channel is not None else {}),
        },
    )
    return out
