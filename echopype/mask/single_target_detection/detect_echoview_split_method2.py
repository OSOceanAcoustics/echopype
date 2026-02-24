from __future__ import annotations

import numpy as np
import xarray as xr

# ##
# Params validation
# ##

REQUIRED_PARAMS = {
    "ts_threshold_db",
    "pldl_db",
    "min_norm_pulse",
    "max_norm_pulse",
    "beam_comp_model",
    "max_beam_comp_db",
    "max_sd_minor_deg",
    "max_sd_major_deg",
}

OPTIONAL_PARAMS = {
    "dec_tir_samples",
    "bottom_offset_m",
    "exclude_above_m",
    "exclude_below_m",
    "allow_nans_inside_envelope",
}


def _validate_params(params: dict) -> dict:
    if params is None:
        raise ValueError("No parameters given.")
    unknown = set(params.keys()) - (REQUIRED_PARAMS | OPTIONAL_PARAMS)
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")
    missing = REQUIRED_PARAMS - set(params.keys())
    if missing:
        raise ValueError(f"Missing required parameters: {sorted(missing)}")
    if params["min_norm_pulse"] > params["max_norm_pulse"]:
        raise ValueError("min_norm_pulse must be <= max_norm_pulse")
    return params


def _validate_ds_m2(ds_m2: xr.Dataset) -> xr.Dataset:
    must = [
        "TS",
        "echo_range",
        "angle_alongship",
        "angle_athwartship",
        "sound_absorption",
        "sample_interval",
        "tau_effective",
    ]
    for v in must:
        if v not in ds_m2:
            raise ValueError(f"ds_m2 missing required variable: {v}")

    ts_mat = ds_m2["TS"]
    if ts_mat.dims != ("ping_time", "range_sample"):
        raise ValueError("Expected ds_m2['TS'] dims exactly ('ping_time','range_sample').")
    for v in ["echo_range", "angle_alongship", "angle_athwartship"]:
        if ds_m2[v].dims != ("ping_time", "range_sample"):
            raise ValueError(f"Expected ds_m2['{v}'] dims exactly ('ping_time','range_sample').")

    # Broadcast alpha to 2D if needed (for simple math)
    alpha = ds_m2["sound_absorption"]
    if alpha.ndim == 1:
        ds_m2 = ds_m2.assign(sound_absorption=alpha.broadcast_like(ts_mat))
    elif alpha.ndim == 0:
        ds_m2 = ds_m2.assign(sound_absorption=(xr.zeros_like(ts_mat) + alpha))
    elif alpha.ndim == 2:
        pass
    else:
        raise ValueError("sound_absorption must be scalar, 1D(ping_time), or 2D like TS.")

    return ds_m2


# ##
# Core algorithm equations
# ##


def _plike_from_ts(
    ts_db: xr.DataArray, r_m: xr.DataArray, alpha_db_m: xr.DataArray
) -> xr.DataArray:
    # Plike = TS - 40log10(r) - 2 alpha r
    r = xr.where(r_m > 0, r_m, np.nan)
    return ts_db - 40.0 * xr.apply_ufunc(np.log10, r) - 2.0 * alpha_db_m * r_m


def _local_max_first_plateau(plike_mat: xr.DataArray) -> xr.DataArray:
    prev = plike_mat.shift(range_sample=1)
    nxt = plike_mat.shift(range_sample=-1)
    peak = (plike_mat > prev) & (plike_mat >= nxt) & ~(plike_mat == prev)
    peak = peak & xr.apply_ufunc(np.isfinite, prev) & xr.apply_ufunc(np.isfinite, nxt)
    peak = peak & xr.apply_ufunc(np.isfinite, plike_mat)
    return peak


def _nech_p_samples(ds_m2: xr.Dataset) -> np.ndarray:
    # NechP (samples) = tau_effective / sample_interval
    tau = ds_m2["tau_effective"]
    dt = ds_m2["sample_interval"]

    # ---- tau -> 1D per ping_time ----
    if tau.ndim == 0:
        tau_vec = np.full(ds_m2.sizes["ping_time"], float(tau.values), dtype=float)
    elif tau.ndim == 1 and tau.dims == ("ping_time",):
        tau_vec = tau.values.astype(float)
    else:
        tau_vec = tau.isel(range_sample=0).values.astype(float)

    if dt.ndim == 0:
        dt_vec = np.full(ds_m2.sizes["ping_time"], float(dt.values), dtype=float)
    elif dt.ndim == 1 and dt.dims == ("ping_time",):
        dt_vec = dt.values.astype(float)
    else:
        dt_vec = dt.isel(range_sample=0).values.astype(float)

    sample_interval_sec = dt_vec

    return tau_vec / sample_interval_sec


def _envelope_bounds_1d(
    plike_row: np.ndarray, p: int, thr: float, allow_nans: bool
) -> tuple[int | None, int | None]:
    # expand left
    m = p
    while m > 0:
        v = plike_row[m - 1]
        if not np.isfinite(v):
            return (None, None) if not allow_nans else (m, p)
        if v >= thr:
            m -= 1
        else:
            break

    # expand right
    last = p
    while last < plike_row.size - 1:
        v = plike_row[last + 1]
        if not np.isfinite(v):
            return (None, None) if not allow_nans else (m, last)
        if v >= thr:
            last += 1
        else:
            break

    return m, last


# ##
# Beam compensation
# ##


def _beam_comp_db(ds_m2: xr.Dataset, params: dict) -> xr.DataArray:

    model = params["beam_comp_model"]

    if model == "none":
        return xr.zeros_like(ds_m2["TS"])

    elif model == "simrad_lobe":

        th_al = ds_m2["angle_alongship"]
        th_at = ds_m2["angle_athwartship"]

        bw_al = ds_m2["beamwidth_alongship"].broadcast_like(th_al)
        bw_at = ds_m2["beamwidth_athwartship"].broadcast_like(th_at)

        off_al = ds_m2["angle_offset_alongship"].broadcast_like(th_al)
        off_at = ds_m2["angle_offset_athwartship"].broadcast_like(th_at)

        x = 2 * (th_al - off_al) / bw_al
        y = 2 * (th_at - off_at) / bw_at

        beam_comp_db = 6.0206 * (x**2 + y**2 - 0.18 * x**2 * y**2)

        return beam_comp_db.broadcast_like(ds_m2["TS"])

    elif model == "provided":
        if "beam_comp_db" not in ds_m2:
            raise ValueError("beam_comp_db must exist if beam_comp_model='provided'")
        return ds_m2["beam_comp_db"].broadcast_like(ds_m2["TS"])

    else:
        raise ValueError(f"Unknown beam_comp_model: {model}")


# ##
# Phase I
# ##


def _phase1_simple(ds_m2: xr.Dataset, params: dict, beam_comp_db: xr.DataArray) -> xr.Dataset:
    """
    Returns a compact xr.Dataset with dim 'target' and vars:
      ping_index, range_sample, iinf, isup, pulse_len_samples, norm_pulse_len, plike_peak
    """
    plike_mat = _plike_from_ts(ds_m2["TS"], ds_m2["echo_range"], ds_m2["sound_absorption"])

    cand_mask = _local_max_first_plateau(plike_mat)
    cand_mask = cand_mask & (beam_comp_db <= float(params["max_beam_comp_db"]))

    # optional gates
    if params.get("dec_tir_samples") is not None:
        dec_tir = int(params["dec_tir_samples"])
        idx = xr.DataArray(
            np.arange(plike_mat.sizes["range_sample"]),
            dims=("range_sample",),
            coords={"range_sample": plike_mat["range_sample"]},
        )
        cand_mask = cand_mask & (idx >= dec_tir)

    if params.get("exclude_above_m") is not None:
        cand_mask = cand_mask & (ds_m2["echo_range"] >= float(params["exclude_above_m"]))
    if params.get("exclude_below_m") is not None:
        cand_mask = cand_mask & (ds_m2["echo_range"] <= float(params["exclude_below_m"]))

    if "bottom" in ds_m2 and params.get("bottom_offset_m") is not None:
        off = float(params["bottom_offset_m"])
        bottom2d = ds_m2["bottom"].broadcast_like(ds_m2["TS"])
        cand_mask = cand_mask & (ds_m2["echo_range"] <= (bottom2d - off))

    # numpy loop (envelopes)
    plike_np = plike_mat.values
    cand_np = cand_mask.values
    al_np = ds_m2["angle_alongship"].values
    ath_np = ds_m2["angle_athwartship"].values

    nech_p = _nech_p_samples(ds_m2)
    pldl_db = float(params["pldl_db"])
    min_norm_pulse = float(params["min_norm_pulse"])
    max_norm_pulse = float(params["max_norm_pulse"])
    max_sd_minor_deg = float(params["max_sd_minor_deg"])
    max_sd_major_deg = float(params["max_sd_major_deg"])
    allow_nans = bool(params.get("allow_nans_inside_envelope", False))

    (
        ping_index_list,
        range_sample_list,
        iinf_list,
        isup_list,
        pulse_len_samples_list,
        norm_pulse_len_list,
        plike_peak_list,
    ) = ([] for _ in range(7))

    for it in range(plike_np.shape[0]):
        peaks = np.where(cand_np[it])[0]
        if peaks.size == 0:
            continue

        nch = nech_p[it]
        if not np.isfinite(nch) or nch <= 0:
            continue

        plike_row = plike_np[it]
        ali = al_np[it]
        athi = ath_np[it]

        for p in peaks:
            plike_peak = plike_row[p]
            if not np.isfinite(plike_peak):
                continue

            iinf, isup = _envelope_bounds_1d(
                plike_row, int(p), plike_peak - pldl_db, allow_nans=allow_nans
            )
            if iinf is None:
                continue

            pulse_len_samples = isup - iinf + 1
            norm_pulse_len = pulse_len_samples / nch
            if norm_pulse_len < min_norm_pulse or norm_pulse_len > max_norm_pulse:
                continue

            seg_al = ali[iinf : isup + 1] * 180.0 / np.pi
            seg_ath = athi[iinf : isup + 1] * 180.0 / np.pi
            if np.nanstd(seg_ath) > max_sd_minor_deg:
                continue
            if np.nanstd(seg_al) > max_sd_major_deg:
                continue

            ping_index_list.append(it)
            range_sample_list.append(int(p))
            iinf_list.append(int(iinf))
            isup_list.append(int(isup))
            pulse_len_samples_list.append(int(pulse_len_samples))
            norm_pulse_len_list.append(float(norm_pulse_len))
            plike_peak_list.append(float(plike_peak))

    if len(range_sample_list) == 0:
        return xr.Dataset(coords={"target": np.arange(0, dtype=np.int64)})

    return xr.Dataset(
        data_vars=dict(
            ping_index=("target", np.array(ping_index_list, dtype=np.int64)),
            range_sample=("target", np.array(range_sample_list, dtype=np.int64)),
            iinf=("target", np.array(iinf_list, dtype=np.int64)),
            isup=("target", np.array(isup_list, dtype=np.int64)),
            pulse_len_samples=("target", np.array(pulse_len_samples_list, dtype=np.int64)),
            norm_pulse_len=("target", np.array(norm_pulse_len_list, dtype=np.float64)),
            plike_peak=("target", np.array(plike_peak_list, dtype=np.float64)),
        ),
        coords=dict(target=np.arange(len(range_sample_list), dtype=np.int64)),
    )


# ##
# Phase II (TS computation + threshold + overlap rejection)
# ##


def _compute_ts_at_peaks(
    feats: xr.Dataset, ds_m2: xr.Dataset, beam_comp_db: xr.DataArray
) -> xr.Dataset:
    it = feats["ping_index"].values.astype(np.int64)
    p = feats["range_sample"].values.astype(np.int64)

    r = ds_m2["echo_range"].values[it, p]
    a = ds_m2["sound_absorption"].values[it, p]
    plike_peak = feats["plike_peak"].values
    beam_comp_p = beam_comp_db.values[it, p]

    ts_uncomp = plike_peak + 40.0 * np.log10(r) + 2.0 * a * r
    ts_comp = ts_uncomp + beam_comp_p

    return feats.assign(
        target_range=("target", r.astype(float)),
        ts_uncomp=("target", ts_uncomp.astype(float)),
        ts_comp=("target", ts_comp.astype(float)),
    )


def _reject_overlaps_per_ping(feats: xr.Dataset) -> xr.Dataset:
    if feats.dims.get("target", 0) <= 1:
        return feats

    ping_idx = feats["ping_index"].values
    iinf = feats["iinf"].values
    isup = feats["isup"].values
    ts_comp = feats["ts_comp"].values

    keep = np.ones(feats.dims["target"], dtype=bool)

    for it in np.unique(ping_idx):
        ii = np.where(ping_idx == it)[0]
        if ii.size <= 1:
            continue

        order = ii[np.argsort(iinf[ii])]
        accepted = []

        for j in order:
            if not accepted:
                accepted.append(j)
                continue

            k = accepted[-1]
            if iinf[j] > isup[k]:
                accepted.append(j)
                continue

            # overlap: reject lower ts_comp
            if ts_comp[j] >= ts_comp[k]:
                keep[k] = False
                accepted[-1] = j
            else:
                keep[j] = False

    return feats.isel(target=keep)


def _phase2(
    ds_m2: xr.Dataset, params: dict, beam_comp_db: xr.DataArray, feats: xr.Dataset
) -> xr.Dataset:
    if feats.dims.get("target", 0) == 0:
        return feats

    feats = _compute_ts_at_peaks(feats, ds_m2, beam_comp_db)

    # EV Method2: threshold applied to compensated TS
    keep0 = feats["ts_comp"] >= float(params["ts_threshold_db"])
    feats = feats.isel(target=keep0)

    if feats.dims.get("target", 0) == 0:
        return feats

    feats = _reject_overlaps_per_ping(feats)
    return feats


# # ##
# Output packing
# ##


def _pack_targets(feats: xr.Dataset, ds_m2: xr.Dataset) -> xr.Dataset:
    if feats.sizes.get("target", 0) == 0:
        return xr.Dataset(coords={"target": np.arange(0, dtype=np.int64)})

    it = feats["ping_index"].values.astype(np.int64)
    p = feats["range_sample"].values.astype(np.int64)

    ping_time = ds_m2["ping_time"].values[it]
    angle_major_deg = ds_m2["angle_alongship"].values[it, p] * 180.0 / np.pi
    angle_minor_deg = ds_m2["angle_athwartship"].values[it, p] * 180.0 / np.pi

    # add constant per target for single-channel ds_m2
    fn = ds_m2["frequency_nominal"]
    if fn.ndim == 0:
        freq_val = float(fn.values)
    else:
        freq_val = float(fn.values[0])
    frequency_nominal = np.full(it.shape[0], freq_val, dtype=np.float64)

    return xr.Dataset(
        data_vars=dict(
            ping_time=("target", ping_time),
            range_sample=("target", p),
            frequency_nominal=("target", frequency_nominal),
            ping_index=("target", it),
            iinf=("target", feats["iinf"].values.astype(np.int64)),
            isup=("target", feats["isup"].values.astype(np.int64)),
            pulse_len_samples=("target", feats["pulse_len_samples"].values.astype(np.int64)),
            norm_pulse_len=("target", feats["norm_pulse_len"].values.astype(np.float64)),
            target_range=("target", feats["target_range"].values.astype(np.float64)),
            ts_uncomp=("target", feats["ts_uncomp"].values.astype(np.float64)),
            ts_comp=("target", feats["ts_comp"].values.astype(np.float64)),
            angle_major_deg=("target", angle_major_deg.astype(np.float64)),
            angle_minor_deg=("target", angle_minor_deg.astype(np.float64)),
        ),
        coords=dict(target=np.arange(feats.sizes["target"], dtype=np.int64)),
        attrs=dict(method="echoview_split_method2"),
    )


# ##
# Public API
# ##


def detect_echoview_split_method2(ds_m2: xr.Dataset, params: dict) -> xr.Dataset:
    """
    Echoview split-beam single-target detection (Method 2) port.

    Implements in python the EV Method 2 workflow on precomputed TS (dB) and split-beam angles:
    (i) compute Plike from TS + range + absorption, (ii) find local Plike maxima and build
    a -PLDL envelope, (iii) apply Phase I gates (beam-comp limit, echo-length / normalised
    pulse length, angle std-dev), then (iv) compute TS at peaks, apply the compensated TS
    threshold, and reject overlaps within each ping.

    Notes:
    - Input TS must be computed upstream.
    - Angles/beam geometry are converted from degrees to radians for the beam compensation model.

    Parameters
    ----------
    ds_m2 : xr.Dataset
        Single-channel dataset with 2D fields (ping_time, range_sample), at minimum:
        TS, echo_range, angle_alongship, angle_athwartship, sound_absorption,
        sample_interval, tau_effective, plus beam geometry terms if beam compensation is used.

    params : dict
        Required keys:
          - ts_threshold_db, pldl_db
          - min_norm_pulse, max_norm_pulse
          - beam_comp_model ('none'|'simrad_lobe'|'provided'), max_beam_comp_db
          - max_sd_minor_deg, max_sd_major_deg
        Optional keys:
          - dec_tir_samples, bottom_offset_m, exclude_above_m, exclude_below_m,
            allow_nans_inside_envelope

    Returns
    -------
    xr.Dataset
        Dataset with `target` detections including ts_comp/ts_uncomp, target_range,
        ping_time, range_sample, angles, and method metadata.
    """
    params = _validate_params(params)
    ds_m2 = _validate_ds_m2(ds_m2)

    # ADAPTATION: degrees -> radians
    deg2rad = np.pi / 180.0

    ds_m2 = ds_m2.copy()

    ds_m2["angle_alongship"] = ds_m2["angle_alongship"] * deg2rad
    ds_m2["angle_athwartship"] = ds_m2["angle_athwartship"] * deg2rad

    ds_m2["beamwidth_alongship"] = ds_m2["beamwidth_alongship"] * deg2rad
    ds_m2["beamwidth_athwartship"] = ds_m2["beamwidth_athwartship"] * deg2rad

    ds_m2["angle_offset_alongship"] = ds_m2["angle_offset_alongship"] * deg2rad
    ds_m2["angle_offset_athwartship"] = ds_m2["angle_offset_athwartship"] * deg2rad
    ###

    beam_comp_db = _beam_comp_db(ds_m2, params)

    feats = _phase1_simple(ds_m2, params, beam_comp_db)
    feats = _phase2(ds_m2, params, beam_comp_db, feats)

    out = _pack_targets(feats, ds_m2)
    return out
