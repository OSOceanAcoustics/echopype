import numpy as np
import pandas as pd
import scipy.ndimage as ndima
import xarray as xr


def shoal_echoview(
    ds: xr.Dataset,
    var_name: str,
    channel: str | None,
    idim: np.ndarray,
    jdim: np.ndarray,
    thr: float = -70.0,
    mincan: tuple[float, float] = (3.0, 10.0),
    maxlink: tuple[float, float] = (3.0, 15.0),
    minsho: tuple[float, float] = (3.0, 15.0),
) -> xr.DataArray:
    """
    Shoal detector modified from the "echoview" function
    in `mask_shoals.py`, originally written by
    Alejandro ARIZA for the Echopy library © 2020.

    Overview
    ---------
    Perform shoal detection on a Sv matrix using Echoview-like algorithm.

    1) Threshold (candidates): mark samples where Sv > `thr`.

    2) Minimum candidate size: remove connected components whose
        height < `mincan[0]` (in idim units) or width < `mincan[1]` (in jdim units).

    3) Linking: for each remaining component, search a surrounding box expanded
        by `maxlink` (height, width) and link neighbouring components by assigning
        them the same label.

    4) Minimum shoal size: after linking, remove shoals whose height < `minsho[0]`
        or width < `minsho[1]`.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing `var_name` (Sv in dB).
    var_name : str
        Name of the Sv variable in `ds`.
    channel : str | None
        If a "channel" dimension exists, select this channel.
    idim : np.ndarray
        Vertical axis coordinates (length = n_rows + 1).
    jdim : np.ndarray
        Horizontal axis coordinates (length = n_cols + 1).
    thr : float
        Threshold in dB for initial detection (detect values > `thr`).
    mincan : tuple[float, float]
        Minimum candidate size (height, width).
    maxlink : tuple[float, float]
        Maximum linking distance (height, width).
    minsho : tuple[float, float]
        Minimum shoal size (height, width) after linking.

    Returns
    -------
    xr.DataArray
        Boolean mask of detected shoals (dims: "ping_time", "range_sample"; True = detected).
    """

    # Validate variable
    if var_name not in ds:
        raise ValueError(f"Variable '{var_name}' not found in dataset")

    var = ds[var_name]

    if "channel" in var.dims:
        if channel is None:
            raise ValueError("Please specify channel for multi-channel data")
        var = var.sel(channel=channel)

    if np.isnan(idim).any() or np.isnan(jdim).any():
        raise ValueError("idim and jdim must not contain NaN")

    Sv = var.transpose("range_sample", "ping_time").values  # (range, ping)

    # 1. Thresholding
    mask = np.ma.masked_greater(Sv, thr).mask
    if isinstance(mask, np.bool_):  # scalar
        mask = np.zeros_like(Sv, dtype=bool)

    # 2. Remove candidates below mincan size
    candidateslabeled = ndima.label(mask, np.ones((3, 3)))[0]
    candidateslabels = pd.factorize(candidateslabeled[candidateslabeled != 0])[1]
    for cl in candidateslabels:
        candidate = candidateslabeled == cl
        idx = np.where(candidate)[0]
        jdx = np.where(candidate)[1]
        height = idim[max(idx + 1)] - idim[min(idx)]
        width = jdim[max(jdx + 1)] - jdim[min(jdx)]
        if (height < mincan[0]) or (width < mincan[1]):
            mask[idx, jdx] = False

    # 3. Linking neighbours
    linked = np.zeros(mask.shape, dtype=int)
    shoalslabeled = ndima.label(mask, np.ones((3, 3)))[0]
    shoalslabels = pd.factorize(shoalslabeled[shoalslabeled != 0])[1]
    for fl in shoalslabels:
        shoal = shoalslabeled == fl
        i0, i1 = np.min(np.where(shoal)[0]), np.max(np.where(shoal)[0])
        j0, j1 = np.min(np.where(shoal)[1]), np.max(np.where(shoal)[1])
        i00 = np.nanargmin(abs(idim - (idim[i0] - (maxlink[0] + 1))))
        i11 = np.nanargmin(abs(idim - (idim[i1] + (maxlink[0] + 1)))) + 1
        j00 = np.nanargmin(abs(jdim - (jdim[j0] - (maxlink[1] + 1))))
        j11 = np.nanargmin(abs(jdim - (jdim[j1] + (maxlink[1] + 1)))) + 1

        around = np.zeros_like(mask, dtype=bool)
        around[i00:i11, j00:j11] = True
        neighbours = around & mask
        neighbourlabels = pd.factorize(shoalslabeled[neighbours])[1]
        neighbourlabels = neighbourlabels[neighbourlabels != 0]
        neighbours = np.isin(shoalslabeled, neighbourlabels)

        if (pd.factorize(linked[neighbours])[1] == 0).all():
            linked[neighbours] = np.max(linked) + 1
        else:
            formerlabels = pd.factorize(linked[neighbours])[1]
            minlabel = np.min(formerlabels[formerlabels != 0])
            linked[neighbours] = minlabel
            for fl in formerlabels[formerlabels != 0]:
                linked[linked == fl] = minlabel

    # 4. Remove linked shoals smaller than minsho
    linkedlabels = pd.factorize(linked[linked != 0])[1]
    for ll in linkedlabels:
        shoal = linked == ll
        idx, jdx = np.where(shoal)
        height = idim[max(idx + 1)] - idim[min(idx)]
        width = jdim[max(jdx + 1)] - jdim[min(jdx)]
        if (height < minsho[0]) or (width < minsho[1]):
            mask[idx, jdx] = False

    return xr.DataArray(
        mask.T.astype(bool),
        dims=["ping_time", "range_sample"],
        coords={"ping_time": ds["ping_time"], "range_sample": ds["range_sample"]},
        name="shoal_mask",
        attrs={"description": f"Shoal mask using Echoview algorithm on {var_name}"},
    )
