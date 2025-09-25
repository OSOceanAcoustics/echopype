import numpy as np
import scipy.ndimage as ndima
import xarray as xr
from scipy.signal import convolve2d

from echopype.mask.seafloor_detection.utils import _check_inputs, _parse_blackwell_thresholds
from echopype.utils.compute import _lin2log, _log2lin


def bottom_blackwell(
    ds: xr.Dataset,
    var_name: str,
    channel: str,
    threshold: float | list | tuple = -75,
    offset: float = 0.3,
    r0: float = 0,
    r1: float = 500,
    wtheta: int = 28,
    wphi: int = 52,
) -> xr.DataArray:
    """
    Seafloor detection from Sv + split-beam angles (Blackwell et al., 2019).

    Briefly: along-ship and athwart-ship angle fields are smoothed with square
    windows (``wtheta``, ``wphi``). Pixels with large angle activity are flagged,
    an Sv threshold is set from the median Sv within those pixels (or from the
    user-provided value), and connected Sv patches above that threshold are kept.
    The shallowest range of each kept patch per ping is taken as the bottom; an
    ``offset`` (m) is subtracted to place the line slightly above it.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
          • ``var_name`` (Sv in dB) with dims typically
            (``channel``, ``ping_time``, ``range_sample``),
          • ``angle_alongship`` and ``angle_athwartship`` with compatible dims,
          • a vertical coordinate (e.g., ``depth``) aligned with ``range_sample``.
    var_name : str
        Name of the Sv variable to use (e.g., ``"Sv"``).
    channel : str
        Channel identifier to process (must match an entry in ``ds['channel']``).
    threshold : float | list | tuple, default -75
        Either a single Sv threshold in dB (angle thresholds use defaults), or a
        3-tuple/list ``(tSv_dB, ttheta, tphi)`` where ``ttheta`` and ``tphi`` are
        post-smoothing angle activity thresholds (same units as the squared,
        smoothed angles in your pipeline).
    offset : float, default 0.3
        Meters subtracted from the detected range to place the bottom line slightly
        above the echo maximum.
    r0, r1 : float, default 0, 500
        Shallow and deep bounds (meters) of the search interval.
    wtheta, wphi : int, default 28, 52
        Side length (pixels) of the square smoothing windows for the along-ship
        and athwart-ship angle fields.

    Returns
    -------
    xr.DataArray
        1-D bottom depth per ``ping_time`` with attributes:
        ``detector='blackwell'``, ``threshold_Sv``, ``threshold_angle_major``,
        ``threshold_angle_minor``, ``offset_m``, and ``channel``.

    Notes
    -----
    Based on: Blackwell et al., 2019, ICES J. Mar. Sci., “An automated method for
    seabed detection using split-beam echosounders.”
    """

    # Validate input variables and structure
    Sv_sel, depth_sel = _check_inputs(
        ds,
        var_name=var_name,
        channel=channel,
        required_vars=["angle_alongship", "angle_athwartship"],
    )

    # parse thresholds
    tSv, ttheta, tphi = _parse_blackwell_thresholds(threshold)

    # Direct selection without using ch_sel
    theta = ds["angle_alongship"].sel(channel=channel)
    phi = ds["angle_athwartship"].sel(channel=channel)

    # to match with blackwell echopy format
    Sv_sel = Sv_sel.transpose("range_sample", "ping_time")
    theta = theta.transpose("range_sample", "ping_time")
    phi = phi.transpose("range_sample", "ping_time")

    ping_time = Sv_sel.coords["ping_time"]
    r = depth_sel.isel(ping_time=0).values

    # Define core detection range
    r0_idx = np.nanargmin(abs(r - r0))
    r1_idx = np.nanargmin(abs(r - r1)) + 1

    Svchunk = Sv_sel[r0_idx:r1_idx, :]
    thetachunk = theta[r0_idx:r1_idx, :]
    phichunk = phi[r0_idx:r1_idx, :]

    # Build angle masks
    ktheta = np.ones((wtheta, wtheta)) / wtheta**2
    kphi = np.ones((wphi, wphi)) / wphi**2

    # Angle masks
    thetamaskchunk = convolve2d(thetachunk, ktheta, "same", boundary="symm") ** 2 > ttheta
    phimaskchunk = convolve2d(phichunk, kphi, "same", boundary="symm") ** 2 > tphi
    anglemaskchunk = thetamaskchunk | phimaskchunk

    # Apply Blackwell algorithm
    if anglemaskchunk.any():

        Svmedian_anglemasked = float(
            _lin2log(np.nanmedian(_log2lin(Svchunk.values[anglemaskchunk])))
        )

        if np.isnan(Svmedian_anglemasked):
            Svmedian_anglemasked = np.inf
        if Svmedian_anglemasked < tSv:
            Svmedian_anglemasked = tSv

        Svmaskchunk = Svchunk > Svmedian_anglemasked

        # Connected components
        items = ndima.label(Svmaskchunk, ndima.generate_binary_structure(2, 2))[0]
        intercepted = list(set(items[anglemaskchunk]))
        if 0 in intercepted:
            intercepted.remove(0)

        # Combine intercepted items
        maskchunk = np.zeros(Svchunk.shape, dtype=bool)
        for i in intercepted:
            maskchunk |= items == i

        # Add padding
        above = np.zeros((r0_idx, maskchunk.shape[1]), dtype=bool)
        below = np.zeros((len(r) - r1_idx, maskchunk.shape[1]), dtype=bool)

        mask = np.r_[above, maskchunk, below]

    else:
        mask = np.zeros_like(Sv_sel, dtype=bool)

    # Bottom detection from mask - offset
    bottom_sample_idx = mask.argmax(axis=0)
    bottom_depth = r[bottom_sample_idx] - offset

    # Return 1D DataArray with attributes
    return xr.DataArray(
        bottom_depth,
        dims=["ping_time"],
        coords={"ping_time": ping_time},
        name="bottom_depth",
        attrs={
            "detector": "blackwell",
            "threshold_Sv": float(tSv),
            "threshold_angle_major": float(ttheta),
            "threshold_angle_minor": float(tphi),
            "offset_m": float(offset),
            "channel": channel,
        },
    )
