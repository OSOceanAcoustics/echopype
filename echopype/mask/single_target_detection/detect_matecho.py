from __future__ import annotations

from typing import Any, Dict

import numpy as np
import xarray as xr


def _compute_core_matrices(sv_mat, al_mat, ath_mat, meta, params, derived):
    sv = np.asarray(sv_mat, dtype=np.float64)
    al = np.asarray(al_mat, dtype=np.float64)
    ath = np.asarray(ath_mat, dtype=np.float64)

    _, n_samp = sv.shape

    ind_range0 = int(derived["ind_range0"])
    delta = float(derived["delta"])
    sv2ts_f32 = np.float32(derived["sv2ts_f32"])
    alpha = float(meta["alpha"])

    # --- range vectors in float64 (Matecho parity) ---
    idx = np.arange(1, n_samp + 1, dtype=np.float64)
    ts_range64 = np.maximum(np.float64(delta), (idx - ind_range0) * np.float64(delta))
    log20 = 20.0 * np.log10(ts_range64)
    log40 = 40.0 * np.log10(ts_range64)

    # --- TSU: compute in float64 then cast to float32 (Matecho single) ---
    tsu = (sv + log20[None, :] + np.float64(sv2ts_f32)).astype(np.float32)

    # --- Plike: do the casts once ---
    ts_range32 = ts_range64.astype(np.float32)
    log40_32 = log40.astype(np.float32)
    two_alpha32 = np.float32(2.0 * alpha)

    plike = tsu - log40_32[None, :] - two_alpha32 * ts_range32[None, :]

    # --- beam compensation (float32) ---
    x = (2.0 * (al - float(meta["dec_al"])) / float(meta["ouv_al"])).astype(np.float32)
    y = (2.0 * (ath - float(meta["dec_at"])) / float(meta["ouv_at"])).astype(np.float32)
    ts = tsu + np.float32(6.0206) * (x * x + y * y - np.float32(0.18) * x * x * y * y)

    # --- bottom gate ---
    ir = (
        meta["bot"]
        - meta["TD_vec"]
        - meta["heave_vec"]
        - float(params.get("floor_filter_tolerance_m", 0.0))
    ) / delta
    ir = np.clip(ir, 1.0, float(n_samp))

    return {
        "tsu_mat": tsu,
        "ts_mat": ts,
        "plike_mat": plike,
        "ir": ir,
        "nbSamp": int(n_samp),
        "dec_tir": int(params.get("dec_tir", derived.get("dec_tir", 8))),
    }


def _matlab_round_pos(x: float) -> int:
    # MATLAB round for positive numbers: halves away from zero
    return int(np.floor(x + 0.5))


def _require(ds: xr.Dataset, name: str) -> xr.DataArray:
    if name not in ds:
        raise ValueError(f"Missing required variable: '{name}'")
    return ds[name]


def _require_dims(da: xr.DataArray, name: str, dims: tuple[str, ...]) -> None:
    got = tuple(da.dims)
    for d in dims:
        if d not in got:
            raise ValueError(f"Variable '{name}' must have dim '{d}'. Got dims={got}")


def validate_matecho_inputs_strict(ds: xr.Dataset, params: dict) -> tuple[str, xr.DataArray]:
    if params is None:
        raise ValueError("params is required.")
    if "channel" not in params or params["channel"] is None:
        raise ValueError("Missing required parameter: params['channel']")
    channel = str(params["channel"])

    if "channel" not in ds.coords:
        raise ValueError("Missing required coordinate: 'channel'")
    if "ping_time" not in ds.coords and "ping_time" not in ds.dims:
        raise ValueError("Missing required coordinate/dimension: 'ping_time'")
    if "range_sample" not in ds.coords and "range_sample" not in ds.dims:
        raise ValueError("Missing required coordinate/dimension: 'range_sample'")

    if channel not in ds["channel"].values:
        raise ValueError(f"Channel '{channel}' not found in ds['channel'].")

    required_3d = ["Sv", "angle_alongship", "angle_athwartship"]
    for v in required_3d:
        da = _require(ds, v)
        _require_dims(da, v, ("channel", "ping_time", "range_sample"))

    if "echo_range" in ds:
        axis_name = "echo_range"
    else:
        raise ValueError("Missing required variable: 'echo_range'")

    axis = _require(ds, axis_name)
    if "range_sample" not in axis.dims:
        raise ValueError(
            f"Variable '{axis_name}' must include dim 'range_sample'. Got dims={axis.dims}"
        )

    required_meta = [
        "sound_speed",
        "transmit_duration_nominal",
        "tau_effective",
        "sample_interval",
        "sound_absorption",
        "equivalent_beam_angle",
        "sa_correction",
        "beamwidth_alongship",
        "beamwidth_athwartship",
        "angle_offset_alongship",
        "angle_offset_athwartship",
        "angle_sensitivity_alongship",
        "angle_sensitivity_athwartship",
        "transducer_depth",
        "heave_compensation",
    ]
    for v in required_meta:
        da = _require(ds, v)
        if v in (
            "beamwidth_alongship",
            "beamwidth_athwartship",
            "angle_offset_alongship",
            "angle_offset_athwartship",
            "angle_sensitivity_alongship",
            "angle_sensitivity_athwartship",
        ):
            _require_dims(da, v, ("channel",))
        if v in (
            "sound_absorption",
            "transmit_duration_nominal",
            "transducer_depth",
            "heave_compensation",
        ):
            _require_dims(da, v, ("ping_time", "channel"))

    if params.get("seafloor") is None:
        raise ValueError("Need bottom information: provide params['seafloor'].")

    return channel, ds["Sv"]


def _extract_matrices(
    ds: xr.Dataset,
    Sv: xr.DataArray,
    channel: str,
) -> Dict[str, Any]:
    sv_mat = Sv.sel(channel=channel).transpose("ping_time", "range_sample").values
    al_mat = (
        ds["angle_alongship"].sel(channel=channel).transpose("ping_time", "range_sample").values
    )
    ath_mat = (
        ds["angle_athwartship"].sel(channel=channel).transpose("ping_time", "range_sample").values
    )

    r_da = ds["echo_range"] if "echo_range" in ds else ds["depth"]
    if "channel" in r_da.dims:
        r_da = r_da.sel(channel=channel)
    axis = np.asarray(r_da.median("ping_time", skipna=True).values, dtype=float).squeeze()

    last_idx = int(np.where(np.isfinite(axis))[0].max()) + 1
    sv_mat, al_mat, ath_mat = sv_mat[:, :last_idx], al_mat[:, :last_idx], ath_mat[:, :last_idx]
    axis = axis[:last_idx]

    delta = float(axis[1] - axis[0])
    depth1 = float(axis[0])
    depth2 = float(axis[1])

    # SHIFT (Matecho StartDepthSample alignment)
    if "start_depth_m" not in ds:
        raise ValueError("Matecho alignment requires ds['start_depth_m'], but it is missing.")
    sdm_da = ds["start_depth_m"]

    if "channel" in sdm_da.dims:
        sdm_da = sdm_da.sel(channel=channel)
    sdm_vec = np.asarray(sdm_da.values, dtype=float).squeeze()

    npings, nbSamp0 = sv_mat.shape

    start_samp_1b = np.floor(sdm_vec / delta).astype(int)
    start_samp_1b = np.clip(start_samp_1b, 1, nbSamp0)

    lens = nbSamp0 - (start_samp_1b - 1)
    nbSamp_new = int(np.nanmax(lens))

    sv_shift = np.full((npings, nbSamp_new), np.nan, dtype=np.float64)
    al_shift = np.full((npings, nbSamp_new), np.nan, dtype=np.float64)
    ath_shift = np.full((npings, nbSamp_new), np.nan, dtype=np.float64)

    for k in range(npings):
        s0 = int(start_samp_1b[k] - 1)
        L = int(lens[k])
        sv_shift[k, :L] = sv_mat[k, s0 : s0 + L]
        al_shift[k, :L] = al_mat[k, s0 : s0 + L]
        ath_shift[k, :L] = ath_mat[k, s0 : s0 + L]

    sv_mat, al_mat, ath_mat = sv_shift, al_shift, ath_shift

    return {
        "sv_mat": sv_mat,
        "al_mat": al_mat,
        "ath_mat": ath_mat,
        "delta": delta,
        "nbSamp": sv_mat.shape[1],
        "depth1": depth1,
        "depth2": depth2,
        "start_samp_1b": start_samp_1b,
        "start_depth_m": sdm_vec,
    }


def _extract_metadata(ds: xr.Dataset, params: dict) -> dict:
    channel = params["channel"]

    def _need(name: str) -> xr.DataArray:
        if name not in ds:
            raise ValueError(f"Missing required variable in ds: '{name}'")
        return ds[name]

    def _sel_channel(da: xr.DataArray) -> xr.DataArray:
        if "channel" in da.dims:
            return da.sel(channel=channel)
        return da

    def _scalar_from_da(da: xr.DataArray, name: str) -> float:
        da = _sel_channel(da)
        if "ping_time" in da.dims:
            da = da.median("ping_time", skipna=True)
        val = float(np.asarray(da.values).squeeze())
        if not np.isfinite(val):
            raise ValueError(f"'{name}' is present but not finite for channel='{channel}'.")
        return val

    def _ping_vector_from_da(da: xr.DataArray, name: str, npings: int) -> np.ndarray:
        da = _sel_channel(da)
        if "ping_time" not in da.dims:
            val = float(np.asarray(da.values).squeeze())
            return np.full((npings,), val, dtype=float)
        v = np.asarray(da.values, dtype=float).squeeze()
        if v.shape[0] != npings:
            raise ValueError(f"'{name}' ping_time length mismatch: {v.shape[0]} != {npings}")
        return v

    if "channel" not in ds.coords:
        raise ValueError("ds must have a 'channel' coordinate.")
    if channel not in ds["channel"].values:
        raise ValueError(f"Channel '{channel}' not found in ds['channel'].")

    if "ping_time" in ds.coords:
        npings = int(ds.sizes["ping_time"])
    else:
        raise ValueError("ds must have 'ping_time' coordinate/dimension.")

    c = _scalar_from_da(ds["sound_speed"], "sound_speed")

    if "transmit_duration_nominal" in ds:
        pd = _scalar_from_da(ds["transmit_duration_nominal"], "transmit_duration_nominal")
    else:
        raise ValueError("Need ds['transmit_duration_nominal'] (seconds).")

    if "sample_interval" in ds:
        dt_sec = _scalar_from_da(ds["sample_interval"], "sample_interval")
        dt_usec = dt_sec * 1e6
    else:
        raise ValueError("Need ds['sample_interval'].")

    alpha = _scalar_from_da(ds["sound_absorption"], "sound_absorption")
    psi = _scalar_from_da(ds["equivalent_beam_angle"], "equivalent_beam_angle")
    sa_cor = _scalar_from_da(ds["sa_correction"], "sa_correction")

    effpd = float(np.asarray(_sel_channel(ds["tau_effective"]).values).squeeze())

    # EK80 nominal Sa term: compute-only (like Matecho)
    sa_cor_ek80_nominal = 0.0

    if np.isfinite(effpd) and (effpd > 0.0) and (pd > 0.0):
        sa_cor_ek80_nominal = float(np.float32(5.0 * np.log10(effpd / pd)))
    else:
        sa_cor_ek80_nominal = 0.0

    ##

    ouv_al = _scalar_from_da(_need("beamwidth_alongship"), "beamwidth_alongship")
    ouv_at = _scalar_from_da(_need("beamwidth_athwartship"), "beamwidth_athwartship")

    dec_al = _scalar_from_da(_need("angle_offset_alongship"), "angle_offset_alongship")
    dec_at = _scalar_from_da(_need("angle_offset_athwartship"), "angle_offset_athwartship")
    al_sens = _scalar_from_da(_need("angle_sensitivity_alongship"), "angle_sensitivity_alongship")
    ath_sens = _scalar_from_da(
        _need("angle_sensitivity_athwartship"), "angle_sensitivity_athwartship"
    )

    if "transducer_depth" in ds:
        TD_vec = _ping_vector_from_da(ds["transducer_depth"], "transducer_depth", npings)
    else:
        raise ValueError("Need ds['transducer_depth'].")

    if "heave_compensation" in ds:
        CH_vec = _ping_vector_from_da(ds["heave_compensation"], "heave_compensation", npings)
    else:
        raise ValueError("Need ds['heave_compensation'].")

    # seafloor
    sf = params["seafloor"]

    if isinstance(sf, xr.DataArray):
        sf = sf.sel(ping_time=ds["ping_time"], method="nearest")
        bot = np.asarray(sf.values, dtype=float).squeeze()
    else:
        bot = np.asarray(sf, dtype=float).squeeze()

    if bot.shape[0] != npings:
        raise ValueError(f"seafloor length mismatch: {bot.shape[0]} != {npings}")

    return {
        "c": float(c),
        "pd": float(pd),
        "sample_interval_usec": float(dt_usec),
        "alpha": float(alpha),
        "psi": float(psi),
        "sa_cor": float(sa_cor),
        "sa_cor_ek80_nominal": float(sa_cor_ek80_nominal),
        "tau_effective": float(effpd),
        "ouv_al": float(ouv_al),
        "ouv_at": float(ouv_at),
        "dec_al": float(dec_al),
        "dec_at": float(dec_at),
        "al_sens": float(al_sens),
        "ath_sens": float(ath_sens),
        "TD_vec": TD_vec,
        "heave_vec": CH_vec,
        "bot": bot,
    }


def _compute_derived_for_parity(meta: dict, ex: dict, params: dict) -> dict:
    c = float(meta["c"])
    pd = float(meta["pd"])
    dt_usec = float(meta["sample_interval_usec"])
    dt_s = dt_usec * 1e-6

    effpd = float(meta.get("tau_effective", np.nan))
    sa_cor_nominal = float(meta.get("sa_cor_ek80_nominal", np.nan))
    psi = float(meta["psi"])
    sa_cor = float(meta["sa_cor"])

    ind_range0 = 1 if np.isfinite(effpd) else 3
    nech_p = pd / dt_s

    sv2ts_raw64 = 10.0 * np.log10(c * pd / 2.0) + psi + 2.0 * sa_cor + 2.0 * sa_cor_nominal
    sv2ts_f32 = float(np.float32(sv2ts_raw64))

    if "min_echo_length_pulse" not in params or params["min_echo_length_pulse"] is None:
        raise ValueError("Missing required parameter: params['min_echo_length_pulse']")
    if "max_echo_length_pulse" not in params or params["max_echo_length_pulse"] is None:
        raise ValueError("Missing required parameter: params['max_echo_length_pulse']")

    min_echo_len = float(params["min_echo_length_pulse"])
    max_echo_len = float(params["max_echo_length_pulse"])

    n_min = _matlab_round_pos(nech_p * min_echo_len)
    n_max = _matlab_round_pos(nech_p * max_echo_len)

    n_min = max(n_min, 1)
    n_max = max(n_max, 1)

    return {
        "timeSampleInterval": float(dt_s),
        "tau_effective": float(effpd),
        "ind_range0": int(ind_range0),
        "nech_p": float(nech_p),
        "dec_tir": int(params.get("dec_tir", 8)),
        "depth1": float(ex.get("depth1", np.nan)),
        "depth2": float(ex.get("depth2", np.nan)),
        "delta": float(ex["delta"]),
        "n_depth": int(ex["nbSamp"]),
        "sv2ts_f32": float(sv2ts_f32),
        "sv2ts_raw64": float(sv2ts_raw64),
        "n_min": int(n_min),
        "n_max": int(n_max),
    }


# conditions


def _cond_1to5(core: dict, params: dict):
    ts_mat = core["ts_mat"]
    tsu_mat = core["tsu_mat"]
    plike_mat = core["plike_mat"]
    ir = core["ir"]
    nbSamp = core["nbSamp"]
    dec_tir = core["dec_tir"]

    # MATLAB: never evaluate 1st and last sample
    cols0 = np.arange(1, nbSamp - 1)  # Python 0-based
    cols1 = cols0 + 1  # MATLAB 1-based

    ts = ts_mat[:, 1:-1]
    tsu = tsu_mat[:, 1:-1]
    pmid = plike_mat[:, 1:-1]
    pleft = plike_mat[:, 0:-2]
    pright = plike_mat[:, 2:]

    # Cond-1 : TS threshold
    # ts_mat(:,2:end-1) > MinThreshold
    cond1 = ts > float(params["min_threshold_db"])
    # Cond-2 : angle compression
    # (ts − tsu) <= 2 * MaxAngleOneWayCompression
    cond2 = (ts - tsu) <= 2.0 * float(params["max_angle_oneway_compression_db"])
    # Cond-3 : local max of Plike
    cond3 = pmid >= np.maximum(pleft, pright)
    # Cond-4 : above bottom gate
    # col < ir(ping)
    cond4 = cols1[None, :] < ir[:, None]
    # Cond-5 : index > dec_tir
    cond5 = cols1[None, :] > dec_tir
    mask = cond1 & cond2 & cond3 & cond4 & cond5

    # Pixel indices of valid detections
    iping, icol0 = np.where(mask)
    icol = icol0 + 1

    return iping, icol


# --- Condition 6
def _cond_6_phase(
    i1: np.ndarray,
    ind: np.ndarray,
    al_mat: np.ndarray,
    ath_mat: np.ndarray,
    core: dict,
    meta: dict,
    params: dict,
):
    """
    Returns:
      ind2: indices into (i1, ind) that pass Cond-6
      std_ph_al: per-candidate std (same length as i1/ind)
      std_ph_ath: per-candidate std (same length as i1/ind)
    """
    nbSamp = int(core["nbSamp"])
    max_phase_deviation_deg = float(params["max_phase_deviation_deg"])

    # MATLAB scaling: * 128/pi * sens
    al_sens = float(meta["al_sens"])
    ath_sens = float(meta["ath_sens"])
    scale_al = (128.0 / np.pi) * al_sens
    scale_ath = (128.0 / np.pi) * ath_sens

    necho = ind.size
    std_ph_al = np.full(necho, np.nan, dtype=float)
    std_ph_ath = np.full(necho, np.nan, dtype=float)

    for k in range(necho):
        p = int(i1[k])
        j = int(ind[k])

        lo = j - 2
        hi = j + 2
        if hi >= nbSamp:
            hi = nbSamp - 1
        if lo < 0:
            lo = 0

        a = al_mat[p, lo : hi + 1] * scale_al
        b = ath_mat[p, lo : hi + 1] * scale_ath

        std_ph_al[k] = np.nanstd(a, ddof=1)
        std_ph_ath[k] = np.nanstd(b, ddof=1)

    ind2 = np.where(
        (std_ph_al <= max_phase_deviation_deg) & (std_ph_ath <= max_phase_deviation_deg)
    )[0]

    return ind2, std_ph_al, std_ph_ath


def _cond_7_echo_length(
    i1: np.ndarray,
    ind: np.ndarray,
    ind2: np.ndarray,
    core: dict,
    derived: dict,
    params: dict,
) -> dict:
    """
    Matecho Cond-7 (CW): echo length gate based on Plike -6 dB bounds.

    Uses echopype port parameter names:
      - params["min_echo_length_pulse"], params["max_echo_length_pulse"]
    """

    nbSamp = int(core["nbSamp"])
    plike_mat = core["plike_mat"]
    ir = core["ir"]

    n_min = int(derived["n_min"])
    n_max = int(derived["n_max"])

    # survivors after Cond-6
    iip = i1[ind2].astype(np.int64)
    iid0 = ind[ind2].astype(np.int64)

    necho = iid0.size
    iinf = np.empty(necho, dtype=np.int64)
    isup = np.empty(necho, dtype=np.int64)

    for k in range(necho):
        ip = int(iip[k])
        iid_1b = int(iid0[k] + 1)

        # MATLAB: ir_floor = floor(ir(ip))
        ir_floor = int(np.floor(ir[ip]))
        ir_floor = max(ir_floor, 1)
        ir_floor = min(ir_floor, nbSamp)

        peak = float(plike_mat[ip, iid_1b - 1])
        thr = peak - 6.0

        # forward: iid .. ir_floor
        fwd = plike_mat[ip, (iid_1b - 1) : ir_floor] < thr
        if np.any(fwd):
            umin0 = int(np.where(fwd)[0].min())
            # FIX: MATLAB isup = iid - 2 + min(u)
            # Here umin0 = min(u) - 1  => isup = iid_1b - 1 + umin0
            isup[k] = iid_1b - 1 + umin0
        else:
            isup[k] = iid_1b

        # backward: 1 .. iid
        bwd = plike_mat[ip, :iid_1b] < thr
        if np.any(bwd):
            umax0 = int(np.where(bwd)[0].max())
            # MATLAB: iinf = 1 + max(u) where max(u)=umax0+1
            iinf[k] = umax0 + 2
        else:
            iinf[k] = iid_1b

    L6dB = (isup - iinf + 1).astype(np.int64)
    ind4 = np.where((L6dB >= n_min) & (L6dB <= n_max))[0]

    return {
        "iip": iip,
        "iid0": iid0,
        "iinf_1b": iinf,
        "isup_1b": isup,
        "L6dB": L6dB,
        "ind4": ind4,
    }


# cond 8


def _cond_8_refine_and_spacing_matecho(
    c7: dict,
    core: dict,
    meta: dict,
    derived: dict,
    params: dict,
) -> dict:
    """
    Cond-8 (Matecho CW): refine detections, enforce within-ping spacing, and apply a depth gate.

    Steps (per ping)
    - Refine range index `ir_fin` as a power-weighted centroid of samples within the -6 dB bounds
    (`w = 10**(Plike/10)`).
    - Correct TS/TSU to `ir_fin` using Matecho’s range/absorption adjustment.
    - Keep strongest targets separated by at least `min_echo_space_pulse * nech_p` samples.
    - Convert to depth `D = (ir_fin - ind_range0) * delta + TD + heave` and keep `D` within
    [`min_echo_depth_m`, `max_echo_depth_m`].

    Returns: kept indices, refined `ir_fin` (1-based), refined TS/TSU,
    ranges/depths, and spacing in samples.
    """
    ind4 = np.asarray(c7.get("ind4", []), dtype=int)
    if ind4.size == 0:
        return {
            "keep_ind8": np.array([], dtype=int),
            "ir_fin_1b": np.array([], dtype=float),
            "ts_fin": np.array([], dtype=np.float32),
            "tsu_fin": np.array([], dtype=np.float32),
            "targetRange_m": np.array([], dtype=float),
            "d_m": np.array([], dtype=float),
            "min_space_samp": float(params.get("min_echo_space_pulse", 1.0))
            * float(derived["nech_p"]),
        }

    # Cond-6 survivors (0-based) + Cond-7 bounds (1-based)
    iip = np.asarray(c7["iip"], dtype=int)
    iid0 = np.asarray(c7["iid0"], dtype=int)
    iinf_1b = np.asarray(c7["iinf_1b"], dtype=int)
    isup_1b = np.asarray(c7["isup_1b"], dtype=int)

    # Subset that passed Cond-7
    ip0 = iip[ind4]
    ii0 = iid0[ind4]

    plike_mat = core["plike_mat"]
    ts_mat = core["ts_mat"]
    tsu_mat = core["tsu_mat"]

    nech_p = float(derived["nech_p"])
    ind_range0 = float(derived["ind_range0"])
    delta = float(derived["delta"])
    alpha = float(meta["alpha"])

    min_space_samp = float(params.get("min_echo_space_pulse", 1.0)) * nech_p

    TD_vec = np.asarray(meta["TD_vec"], dtype=float)
    heave_vec = np.asarray(meta["heave_vec"], dtype=float)

    min_d = float(params.get("min_echo_depth_m", 0.0))
    max_d = float(params.get("max_echo_depth_m", np.inf))

    keep_global, ir_fin_keep, ts_fin_keep, tsu_fin_keep, tr_keep, D_keep = [], [], [], [], [], []

    for ip in np.unique(ip0):
        idx_p = np.where(ip0 == ip)[0]
        if idx_p.size == 0:
            continue

        ir_fin = np.empty(idx_p.size, dtype=np.float64)
        ts_fin = np.empty(idx_p.size, dtype=np.float64)
        tsu_fin = np.empty(idx_p.size, dtype=np.float64)

        for k_local, j in enumerate(idx_p):
            j_surv = int(ind4[j])

            lo_1b = int(iinf_1b[j_surv])
            hi_1b = int(isup_1b[j_surv])

            il6dB_1b = np.arange(lo_1b, hi_1b + 1, dtype=np.float64)
            il6dB_0b = il6dB_1b.astype(int) - 1

            P = plike_mat[int(ip), il6dB_0b].astype(np.float64)
            w = 10.0 ** (P / 10.0)

            den = np.nansum(w)
            ir_fin[k_local] = (np.nansum(il6dB_1b * w) / den) if den != 0.0 else np.nan

            i_ini_1b = float(ii0[j] + 1)
            base_ts = float(ts_mat[int(ip), int(ii0[j])])
            base_tsu = float(tsu_mat[int(ip), int(ii0[j])])

            corr = 40.0 * np.log10(
                (ir_fin[k_local] - ind_range0) / (i_ini_1b - ind_range0)
            ) + 2.0 * alpha * delta * (ir_fin[k_local] - i_ini_1b)
            ts_fin[k_local] = base_ts + corr
            tsu_fin[k_local] = base_tsu + corr

        # Spacing: keep strongest TS_fin targets with min separation in ir_fin
        if idx_p.size > 1:
            order = np.argsort(ts_fin)[::-1]
            kept = [int(order[0])]
            for cand in order[1:]:
                cand = int(cand)
                sep = np.nanmin(np.abs(ir_fin[cand] - ir_fin[np.array(kept, dtype=int)]))
                if sep >= min_space_samp:
                    kept.append(cand)
        else:
            kept = [0]

        ir_keep = ir_fin[np.array(kept, dtype=int)]
        ts_keep = ts_fin[np.array(kept, dtype=int)]
        tsu_keep = tsu_fin[np.array(kept, dtype=int)]

        targetRange_m = (ir_keep - ind_range0) * delta
        d_m = targetRange_m + float(TD_vec[int(ip)]) + float(heave_vec[int(ip)])

        ok = np.where((d_m >= min_d) & (d_m <= max_d))[0]

        kept_surv = ind4[idx_p[np.array(kept, dtype=int)]][ok]

        keep_global.extend(kept_surv.tolist())
        ir_fin_keep.extend(ir_keep[ok].tolist())
        ts_fin_keep.extend(ts_keep[ok].tolist())
        tsu_fin_keep.extend(tsu_keep[ok].tolist())
        tr_keep.extend(targetRange_m[ok].tolist())
        D_keep.extend(d_m[ok].tolist())

    return {
        "keep_ind8": np.asarray(keep_global, dtype=int),
        "ir_fin_1b": np.asarray(ir_fin_keep, dtype=float),
        "ts_fin": np.asarray(ts_fin_keep, dtype=np.float32),
        "tsu_fin": np.asarray(tsu_fin_keep, dtype=np.float32),
        "targetRange_m": np.asarray(tr_keep, dtype=float),
        "d_m": np.asarray(D_keep, dtype=float),
        "min_space_samp": float(min_space_samp),
    }


# return steps


def _matecho_out_to_dataset(out: dict, *, channel: str) -> xr.Dataset:
    n = int(out.get("nb_valid_targets", 0))
    target = np.arange(n, dtype=int)

    if n == 0:
        empty_target = np.arange(0, dtype=int)

        return xr.Dataset(
            data_vars=dict(
                nb_valid_targets=((), 0),
                ping_time=("target", np.asarray([], dtype="datetime64[ns]")),
                range_sample=("target", np.asarray([], dtype=np.int64)),
                frequency_nominal=("target", np.asarray([], dtype=np.float64)),
            ),
            coords=dict(
                target=empty_target,
                channel=channel,
            ),
            attrs=dict(method="matecho", channel=str(channel), nb_valid_targets=0),
        )

    def _arr(key, dtype=float):
        a = np.asarray(out[key])
        if dtype is not None:
            a = a.astype(dtype)
        if a.size != n:
            raise ValueError(f"Field {key!r} has length {a.size}, expected {n}")
        return a

    ds_out = xr.Dataset(
        data_vars=dict(
            nb_valid_targets=((), n),
            # --- snake_case outputs ---
            ts_comp=("target", _arr("ts_comp", float), {"units": "dB"}),
            ts_uncomp=("target", _arr("ts_uncomp", float), {"units": "dB"}),
            target_range=("target", _arr("target_range", float), {"units": "m"}),
            target_range_disp=("target", _arr("target_range_disp", float), {"units": "m"}),
            target_range_min=("target", _arr("target_range_min", float), {"units": "m"}),
            target_range_max=("target", _arr("target_range_max", float), {"units": "m"}),
            idx_r=("target", _arr("idx_r", int)),
            idx_target_lin=("target", _arr("idx_target_lin", float)),
            angle_minor_axis=("target", _arr("angle_minor_axis", float), {"units": "rad"}),
            angle_major_axis=("target", _arr("angle_major_axis", float), {"units": "rad"}),
            stddev_angles_minor_axis=("target", _arr("stddev_angles_minor_axis", float)),
            stddev_angles_major_axis=("target", _arr("stddev_angles_major_axis", float)),
            heave=("target", _arr("heave", float), {"units": "m"}),
            roll=("target", _arr("roll", float), {"units": "rad"}),
            pitch=("target", _arr("pitch", float), {"units": "rad"}),
            heading=("target", _arr("heading", float), {"units": "rad"}),
            dist=("target", _arr("dist", float), {"units": "m"}),
            ping_time=("target", np.asarray(out["ping_time"], dtype="datetime64[ns]")),
            range_sample=("target", _arr("range_sample", int)),
            frequency_nominal=("target", _arr("frequency_nominal", float)),
        ),
        coords=dict(
            target=target,
            ping_number=("target", _arr("ping_number", int)),
            channel=channel,
        ),
        attrs=dict(method="matecho", channel=str(channel), nb_valid_targets=n),
    )
    return ds_out


def _build_outputs_from_refined(
    ds: xr.Dataset,
    params: dict,
    ex: dict,
    meta: dict,
    c7: dict,
    c8: dict,
) -> dict:
    keep = np.asarray(c8["keep_ind8"], dtype=int)
    n = keep.size
    if n == 0:
        return {"nb_valid_targets": 0}

    iip = np.asarray(c7["iip"], dtype=int)
    iid0 = np.asarray(c7["iid0"], dtype=int)

    ping0 = iip[keep]
    ii0 = iid0[keep]
    ping_number = ping0 + 1

    ts_comp = np.asarray(c8["ts_fin"], dtype=np.float32)
    ts_uncomp = np.asarray(c8["tsu_fin"], dtype=np.float32)

    target_range_m = np.asarray(c8["d_m"], dtype=float)
    idx_target_lin = np.asarray(c8["ir_fin_1b"], dtype=float)
    ping_time = np.asarray(ds["ping_time"].values[ping0], dtype="datetime64[ns]")

    range_sample = np.asarray(np.rint(idx_target_lin - 1.0), dtype=int)

    # API field: frequency_nominal per target
    if "frequency_nominal" in ds:
        fn = ds["frequency_nominal"]
        if "channel" in fn.dims:
            fn_val = float(fn.sel(channel=params["channel"]).values)
        else:
            fn_val = float(np.asarray(fn.values).squeeze())
        frequency_nominal = np.full(n, fn_val, dtype=float)
    else:
        frequency_nominal = np.full(n, np.nan, dtype=float)

    angle_minor_axis = np.asarray(ex["al_mat"][ping0, ii0], dtype=float)
    angle_major_axis = np.asarray(ex["ath_mat"][ping0, ii0], dtype=float)

    heave_m = np.asarray(meta["heave_vec"][ping0], dtype=float)

    out = {
        "nb_valid_targets": int(n),
        "ts_comp": ts_comp.astype(float).tolist(),
        "ts_uncomp": ts_uncomp.astype(float).tolist(),
        "target_range": target_range_m.tolist(),
        "target_range_disp": target_range_m.tolist(),
        "target_range_min": [float(params.get("min_echo_depth_m", np.nan))] * n,
        "target_range_max": [float(params.get("max_echo_depth_m", np.nan))] * n,
        "ping_number": ping_number.astype(int).tolist(),
        "ping_time": ping_time,
        "idx_r": (ii0 + 1).astype(int).tolist(),
        "idx_target_lin": idx_target_lin.tolist(),
        "angle_minor_axis": angle_minor_axis.tolist(),
        "angle_major_axis": angle_major_axis.tolist(),
        "stddev_angles_minor_axis": [np.nan] * n,
        "stddev_angles_major_axis": [np.nan] * n,
        "heave": heave_m.tolist(),
        "roll": [np.nan] * n,
        "pitch": [np.nan] * n,
        "heading": [np.nan] * n,
        "dist": [np.nan] * n,
        "range_sample": range_sample.tolist(),
        "frequency_nominal": frequency_nominal.tolist(),
        "tsfreq_matrix": np.empty((0, 23), dtype=float),
    }
    return out


# main


def detect_matecho(
    ds: xr.Dataset,
    params: dict,
) -> xr.Dataset:
    """
    Matecho-style CW split-beam single-target detector, ported to echopype.

    Implements the Matecho Cond-1…Cond-8 pipeline on Sv (dB) and split-beam
    angles for a single channel. TSU/TS/Plike are computed internally (Sv to TS transform,
    range/absorption terms, and beam compensation), then candidates are filtered by
    threshold, angle compression, local Plike maxima, bottom gate, phase stability,
    -6 dB echo length, and a final refinement + spacing + depth gate.

    Returns an `xr.Dataset` with `target` detections (TS_comp/TS_uncomp, refined index,
    ping_time/number, range/depth, angles) and `nb_valid_targets`.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the following variables/coords:
          • `Sv` in dB with dims (`channel`, `ping_time`, `range_sample`)
          • `angle_alongship`, `angle_athwartship` with the same dims
          • `echo_range` aligned with `range_sample` (and optionally `ping_time`)
          • `start_depth_m` (per ping; used for Matecho SHIFT alignment)
          • Scalars / vectors required by Matecho physics:
              - `sound_speed`
              - `transmit_duration_nominal`
              - `sample_interval`
              - `sound_absorption`
              - `equivalent_beam_angle`
              - `sa_correction`
              - `tau_effective` (recommended; controls `ind_range0`)
              - EK80 nominal Sa term (`Sa_EK80_nominal` or equivalent if present)
              - split-beam geometry terms:
                `beamwidth_alongship`, `beamwidth_athwartship`,
                `angle_offset_alongship`, `angle_offset_athwartship`,
                `angle_sensitivity_alongship`, `angle_sensitivity_athwartship`
              - `transducer_depth`
              - `heave_compensation`

        Notes
        -----
        * This implementation assumes **split-beam** data (valid angle fields).
        * Units must be internally consistent (e.g., absorption in dB/m).

    params : dict
        Detection configuration. Required keys:
          • `channel` : str
              Channel identifier to process (must match `ds["channel"]` entries).
          • `seafloor` : array-like or xr.DataArray
              Bottom depth/range per ping_time (used by Cond-4 bottom gate).
          • `min_threshold_db` : float
              Cond-1 TS threshold.
          • `max_angle_oneway_compression_db` : float
              Cond-2 beam compression threshold (one-way, used as 2× in the test).
          • `max_phase_deviation_deg` : float
              Cond-6 phase stability threshold (degrees, after Matecho scaling).
          • `min_echo_length_pulse` : float
              Cond-7 minimum echo length in units of pulse length (× nech_p).
          • `max_echo_length_pulse` : float
              Cond-7 maximum echo length in units of pulse length (× nech_p).

        Optional keys:
          • `dec_tir` : int, default 8
              Cond-5 minimum range index (1-based in Matecho; implemented accordingly).
          • `floor_filter_tolerance_m` : float, default 0.0
              Extra margin in the bottom gate (subtracted before converting to samples).
          • `min_echo_space_pulse` : float, default 1.0
              Cond-8 minimum within-ping separation, expressed in pulse lengths.
          • `min_echo_depth_m` : float, default 0.0
              Minimum accepted displayed depth (after TD + CH).
          • `max_echo_depth_m` : float, default +inf
              Maximum accepted displayed depth (after TD + CH).
          • `DBG` : dict, optional
              Debug switches / payload (implementation-specific).

    Returns
    -------
    xr.Dataset
        Dataset with dimension `target` (length = number of valid detections),
        including (at least):
          • `TS_comp`, `TS_uncomp` (dB)
          • `idx_r` (coarse peak sample index, 1-based)
          • `idx_target_lin` (refined index, float, 1-based)
          • `Ping_number` (1-based)
          • `time` (datetime64)
          • angle fields at the detected position
          • `nb_valid_targets` scalar
        and attrs: `method="matecho"`, `channel=<channel>`, `nb_valid_targets=<n>`.

    """

    channel, Sv_da = validate_matecho_inputs_strict(ds, params)
    params = dict(params)
    params["channel"] = channel

    ex = _extract_matrices(ds, Sv_da, channel)
    meta = _extract_metadata(ds, params)

    # adaptations to match matecho algorithm

    # convert angles degrees to radians
    deg2rad = np.pi / 180.0
    ex["al_mat"] = ex["al_mat"] * deg2rad
    ex["ath_mat"] = ex["ath_mat"] * deg2rad
    meta["ouv_al"] *= deg2rad
    meta["ouv_at"] *= deg2rad
    # Echopype angles are already offset-corrected (Matecho not), so prevent double-subtraction
    meta["dec_al"] = 0.0
    meta["dec_at"] = 0.0
    ####

    derived = _compute_derived_for_parity(meta, ex, params)
    core = _compute_core_matrices(
        sv_mat=ex["sv_mat"],
        al_mat=ex["al_mat"],
        ath_mat=ex["ath_mat"],
        meta=meta,
        params=params,
        derived=derived,
    )

    # --- Cond 1–5: candidate pixels (joint logical mask) ---
    # ip_15 / ir_15 are aligned 0-based indices into (ping, sample).
    ip_15, ir_15 = _cond_1to5(core, params)

    # --- Cond 6: phase stability (±2 samples window) ---
    # ind6 indexes into (ip_15, ir_15); survivors are ip_15[ind6], ir_15[ind6].
    ind6, std_ph_al, std_ph_ath = _cond_6_phase(
        i1=ip_15,
        ind=ir_15,
        al_mat=ex["al_mat"],
        ath_mat=ex["ath_mat"],
        core=core,
        meta=meta,
        params=params,
    )

    # --- Cond 7: echo length (-6 dB width in Plike) ---
    # c7 bundles Cond-6 survivors + 6 dB bounds + pass subset (ind4).
    c7 = _cond_7_echo_length(
        i1=ip_15,
        ind=ir_15,
        ind2=ind6,
        core=core,
        derived=derived,
        params=params,
    )

    # --- Cond 8: refine (centroid), spacing, depth gate ---
    # Operates on c7["ind4"]; returns final kept targets.
    c8 = _cond_8_refine_and_spacing_matecho(
        c7=c7,
        core=core,
        meta=meta,
        derived=derived,
        params=params,
    )

    out = _build_outputs_from_refined(ds=ds, params=params, ex=ex, meta=meta, c7=c7, c8=c8)
    return _matecho_out_to_dataset(out, channel=channel)
