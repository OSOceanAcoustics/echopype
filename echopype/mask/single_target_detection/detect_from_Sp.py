from __future__ import annotations

import numpy as np
import xarray as xr

REQUIRED_PARAMS = {
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


def _validate_from_Sp_dataset(ds_sp: xr.Dataset) -> xr.Dataset:
    must = [
        "Sp",
        "echo_range",
        "angle_alongship",
        "angle_athwartship",
        "sound_absorption",
        "sample_interval",
        "tau_effective",
    ]

    for v in must:
        if v not in ds_sp:
            raise ValueError(f"ds_sp missing required variable: {v}")

    sp_mat = ds_sp["Sp"]

    if sp_mat.dims != ("ping_time", "range_sample"):
        raise ValueError("Expected ds_sp['Sp'] dims exactly ('ping_time', 'range_sample').")

    for v in ["echo_range", "angle_alongship", "angle_athwartship"]:
        if ds_sp[v].dims != ("ping_time", "range_sample"):
            raise ValueError(f"Expected ds_sp['{v}'] dims exactly ('ping_time', 'range_sample').")

    alpha = ds_sp["sound_absorption"]

    if alpha.ndim == 1:
        ds_sp = ds_sp.assign(sound_absorption=alpha.broadcast_like(sp_mat))
    elif alpha.ndim == 0:
        ds_sp = ds_sp.assign(sound_absorption=xr.zeros_like(sp_mat) + alpha)
    elif alpha.ndim == 2:
        pass
    else:
        raise ValueError("sound_absorption must be scalar, 1D(ping_time), or 2D like Sp.")

    return ds_sp


def _plike_from_sp(
    sp_db: xr.DataArray,
    r_m: xr.DataArray,
    alpha_db_m: xr.DataArray,
) -> xr.DataArray:
    r = xr.where(r_m > 0, r_m, np.nan)
    return sp_db - 40.0 * xr.apply_ufunc(np.log10, r) - 2.0 * alpha_db_m * r_m


def _local_max_first_plateau(plike_mat: xr.DataArray) -> xr.DataArray:
    prev = plike_mat.shift(range_sample=1)
    nxt = plike_mat.shift(range_sample=-1)

    peak = (plike_mat > prev) & (plike_mat >= nxt) & ~(plike_mat == prev)
    peak = peak & xr.apply_ufunc(np.isfinite, prev)
    peak = peak & xr.apply_ufunc(np.isfinite, nxt)
    peak = peak & xr.apply_ufunc(np.isfinite, plike_mat)

    return peak


def _nech_p_samples(ds_sp: xr.Dataset) -> np.ndarray:
    tau = ds_sp["tau_effective"]
    dt = ds_sp["sample_interval"]

    if tau.ndim == 0:
        tau_vec = np.full(ds_sp.sizes["ping_time"], float(tau.values), dtype=float)
    elif tau.ndim == 1 and tau.dims == ("ping_time",):
        tau_vec = tau.values.astype(float)
    else:
        tau_vec = tau.isel(range_sample=0).values.astype(float)

    if dt.ndim == 0:
        dt_vec = np.full(ds_sp.sizes["ping_time"], float(dt.values), dtype=float)
    elif dt.ndim == 1 and dt.dims == ("ping_time",):
        dt_vec = dt.values.astype(float)
    else:
        dt_vec = dt.isel(range_sample=0).values.astype(float)

    return tau_vec / dt_vec


def _envelope_bounds_1d(
    plike_row: np.ndarray,
    p: int,
    thr: float,
    allow_nans: bool,
) -> tuple[int | None, int | None]:
    m = p
    while m > 0:
        v = plike_row[m - 1]
        if not np.isfinite(v):
            return (None, None) if not allow_nans else (m, p)
        if v >= thr:
            m -= 1
        else:
            break

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


def _beam_comp_db(ds_sp: xr.Dataset, params: dict) -> xr.DataArray:
    model = params["beam_comp_model"]

    if model == "none":
        return xr.zeros_like(ds_sp["Sp"])

    if model == "provided":
        if "beam_comp_db" not in ds_sp:
            raise ValueError("beam_comp_db must exist if beam_comp_model='provided'")
        return ds_sp["beam_comp_db"].broadcast_like(ds_sp["Sp"])

    if model == "simrad_lobe":
        th_al = ds_sp["angle_alongship"]
        th_at = ds_sp["angle_athwartship"]

        bw_al = ds_sp["beamwidth_alongship"].broadcast_like(th_al)
        bw_at = ds_sp["beamwidth_athwartship"].broadcast_like(th_at)

        # Angles from add_splitbeam_angle() are already offset-corrected.
        x = 2.0 * th_al / bw_al
        y = 2.0 * th_at / bw_at

        beam_comp_db = 6.0206 * (x**2 + y**2 - 0.18 * x**2 * y**2)
        return beam_comp_db.broadcast_like(ds_sp["Sp"])

    raise ValueError(f"Unknown beam_comp_model: {model}")


def _phase1_simple(
    ds_sp: xr.Dataset,
    params: dict,
    beam_comp_db: xr.DataArray,
) -> xr.Dataset:
    # Echoview Method 2 detects peaks on power-like data obtained by
    # removing TVG/range and absorption from the TS operand.
    #
    # In echopype, Sp is already the uncompensated TS-like quantity,
    # so reconstruct the power-like signal directly from Sp.
    #
    # Beam compensation is NOT part of the detection signal; it is only
    # used as a peak-selection criterion and later for the final TS.
    plike_mat = _plike_from_sp(
        ds_sp["Sp"],
        ds_sp["echo_range"],
        ds_sp["sound_absorption"],
    )

    cand_mask = _local_max_first_plateau(plike_mat)
    cand_mask = cand_mask & (beam_comp_db <= float(params["max_beam_comp_db"]))

    if params.get("dec_tir_samples") is not None:
        dec_tir = int(params["dec_tir_samples"])
        idx = xr.DataArray(
            np.arange(plike_mat.sizes["range_sample"]),
            dims=("range_sample",),
            coords={"range_sample": plike_mat["range_sample"]},
        )
        cand_mask = cand_mask & (idx >= dec_tir)

    if params.get("exclude_above_m") is not None:
        cand_mask = cand_mask & (ds_sp["echo_range"] >= float(params["exclude_above_m"]))

    if params.get("exclude_below_m") is not None:
        cand_mask = cand_mask & (ds_sp["echo_range"] <= float(params["exclude_below_m"]))

    if "bottom" in ds_sp and params.get("bottom_offset_m") is not None:
        off = float(params["bottom_offset_m"])
        bottom2d = ds_sp["bottom"].broadcast_like(ds_sp["Sp"])
        cand_mask = cand_mask & (ds_sp["echo_range"] <= (bottom2d - off))

    plike_np = plike_mat.values
    cand_np = cand_mask.values
    al_np = ds_sp["angle_alongship"].values
    ath_np = ds_sp["angle_athwartship"].values
    range_np = ds_sp["echo_range"].values
    beam_comp_np = beam_comp_db.values

    nech_p = _nech_p_samples(ds_sp)

    pldl_db = float(params["pldl_db"])
    min_norm_pulse = float(params["min_norm_pulse"])
    max_norm_pulse = float(params["max_norm_pulse"])
    max_sd_minor_deg = float(params["max_sd_minor_deg"])
    max_sd_major_deg = float(params["max_sd_major_deg"])
    allow_nans = bool(params.get("allow_nans_inside_envelope", False))

    ping_index_list = []
    range_sample_list = []
    iinf_list = []
    isup_list = []
    pulse_len_samples_list = []
    norm_pulse_len_list = []
    plike_peak_list = []
    target_range_list = []
    beam_comp_db_list = []
    angle_minor_sd_deg_list = []
    angle_major_sd_deg_list = []

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
                plike_row,
                int(p),
                plike_peak - pldl_db,
                allow_nans=allow_nans,
            )
            if iinf is None:
                continue

            pulse_len_samples = isup - iinf + 1
            norm_pulse_len = pulse_len_samples / nch

            if norm_pulse_len < min_norm_pulse or norm_pulse_len > max_norm_pulse:
                continue

            seg_al = ali[iinf : isup + 1] * 180.0 / np.pi
            seg_ath = athi[iinf : isup + 1] * 180.0 / np.pi

            angle_minor_sd_deg = float(np.nanstd(seg_ath))
            angle_major_sd_deg = float(np.nanstd(seg_al))

            if angle_minor_sd_deg > max_sd_minor_deg:
                continue

            if angle_major_sd_deg > max_sd_major_deg:
                continue

            ping_index_list.append(it)
            range_sample_list.append(int(p))
            iinf_list.append(int(iinf))
            isup_list.append(int(isup))
            pulse_len_samples_list.append(int(pulse_len_samples))
            norm_pulse_len_list.append(float(norm_pulse_len))
            plike_peak_list.append(float(plike_peak))
            target_range_list.append(float(range_np[it, p]))
            beam_comp_db_list.append(float(beam_comp_np[it, p]))
            angle_minor_sd_deg_list.append(angle_minor_sd_deg)
            angle_major_sd_deg_list.append(angle_major_sd_deg)

    return xr.Dataset(
        data_vars=dict(
            ping_index=(
                "single_target",
                np.array(ping_index_list, dtype=np.int64),
            ),
            range_sample=(
                "single_target",
                np.array(range_sample_list, dtype=np.int64),
            ),
            iinf=(
                "single_target",
                np.array(iinf_list, dtype=np.int64),
            ),
            isup=(
                "single_target",
                np.array(isup_list, dtype=np.int64),
            ),
            pulse_len_samples=(
                "single_target",
                np.array(pulse_len_samples_list, dtype=np.int64),
            ),
            norm_pulse_len=(
                "single_target",
                np.array(norm_pulse_len_list, dtype=np.float64),
            ),
            plike_peak=(
                "single_target",
                np.array(plike_peak_list, dtype=np.float64),
            ),
            single_target_range=(
                "single_target",
                np.array(target_range_list, dtype=np.float64),
            ),
            beam_comp_db=(
                "single_target",
                np.array(beam_comp_db_list, dtype=np.float64),
            ),
            single_target_athwartship_angle_sd=(
                "single_target",
                np.array(angle_minor_sd_deg_list, dtype=np.float64),
            ),
            single_target_alongship_angle_sd=(
                "single_target",
                np.array(angle_major_sd_deg_list, dtype=np.float64),
            ),
        ),
        coords={
            "single_target": np.arange(
                len(range_sample_list),
                dtype=np.int64,
            )
        },
    )


def _reject_overlaps_per_ping(feats: xr.Dataset) -> xr.Dataset:
    if feats.sizes.get("single_target", 0) <= 1:
        return feats

    ping_idx = feats["ping_index"].values
    range_sample = feats["range_sample"].values
    iinf = feats["iinf"].values
    isup = feats["isup"].values
    plike_peak = feats["plike_peak"].values

    keep = np.ones(feats.sizes["single_target"], dtype=bool)

    for it in np.unique(ping_idx):
        ii = np.where(ping_idx == it)[0]
        if ii.size <= 1:
            continue

        # Echoview: screen pulses from low to high range/depth.
        # Use peak sample order, not envelope-start order.
        order = ii[np.argsort(range_sample[ii])]
        accepted = []

        for j in order:
            if not accepted:
                accepted.append(j)
                continue

            k = accepted[-1]

            if iinf[j] > isup[k]:
                accepted.append(j)
                continue

            # If pulses overlap, reject the lower-power / lower-TS one.
            if plike_peak[j] >= plike_peak[k]:
                keep[k] = False
                accepted[-1] = j
            else:
                keep[j] = False

    return feats.isel(single_target=keep)


def _pack_targets(feats: xr.Dataset, ds_sp: xr.Dataset) -> xr.Dataset:
    n_targets = feats.sizes.get("single_target", 0)
    channel_value = ds_sp["channel"].item()

    if n_targets == 0:
        return xr.Dataset(
            data_vars=dict(
                channel=(
                    "single_target",
                    np.array([], dtype=object),
                ),
                ping_time=(
                    "single_target",
                    np.array([], dtype=ds_sp["ping_time"].dtype),
                ),
                range_sample=(
                    "single_target",
                    np.array([], dtype=np.int64),
                ),
                frequency_nominal=(
                    "single_target",
                    np.array([], dtype=np.float64),
                ),
                ping_index=(
                    "single_target",
                    np.array([], dtype=np.int64),
                ),
                iinf=(
                    "single_target",
                    np.array([], dtype=np.int64),
                ),
                isup=(
                    "single_target",
                    np.array([], dtype=np.int64),
                ),
                pulse_len_samples=(
                    "single_target",
                    np.array([], dtype=np.int64),
                ),
                norm_pulse_len=(
                    "single_target",
                    np.array([], dtype=np.float64),
                ),
                single_target_range=(
                    "single_target",
                    np.array([], dtype=np.float64),
                ),
                single_target_alongship_angle=(
                    "single_target",
                    np.array([], dtype=np.float64),
                ),
                single_target_athwartship_angle=(
                    "single_target",
                    np.array([], dtype=np.float64),
                ),
                single_target_athwartship_angle_sd=(
                    "single_target",
                    np.array([], dtype=np.float64),
                ),
                single_target_alongship_angle_sd=(
                    "single_target",
                    np.array([], dtype=np.float64),
                ),
                beam_comp_db=(
                    "single_target",
                    np.array([], dtype=np.float64),
                ),
                plike_peak=(
                    "single_target",
                    np.array([], dtype=np.float64),
                ),
            ),
            coords={
                "single_target": np.arange(
                    n_targets,
                    dtype=np.int64,
                )
            },
            attrs=dict(method="from_Sp"),
        )

    it = feats["ping_index"].values.astype(np.int64)
    p = feats["range_sample"].values.astype(np.int64)

    ping_time = ds_sp["ping_time"].values[it]
    single_target_alongship_angle = ds_sp["angle_alongship"].values[it, p] * 180.0 / np.pi
    single_target_athwartship_angle = ds_sp["angle_athwartship"].values[it, p] * 180.0 / np.pi

    fn = ds_sp["frequency_nominal"]
    if fn.ndim == 0:
        freq_val = float(fn.values)
    else:
        freq_val = float(fn.values[0])

    frequency_nominal = np.full(n_targets, freq_val, dtype=np.float64)
    channel = np.full(
        n_targets,
        channel_value,
        dtype=object,
    )

    return xr.Dataset(
        data_vars=dict(
            channel=(
                "single_target",
                channel,
            ),
            ping_time=(
                "single_target",
                ping_time,
            ),
            range_sample=(
                "single_target",
                p,
            ),
            frequency_nominal=(
                "single_target",
                frequency_nominal,
            ),
            ping_index=(
                "single_target",
                it,
            ),
            iinf=(
                "single_target",
                feats["iinf"].values.astype(np.int64),
            ),
            isup=(
                "single_target",
                feats["isup"].values.astype(np.int64),
            ),
            pulse_len_samples=(
                "single_target",
                feats["pulse_len_samples"].values.astype(np.int64),
            ),
            norm_pulse_len=(
                "single_target",
                feats["norm_pulse_len"].values.astype(np.float64),
            ),
            single_target_range=(
                "single_target",
                feats["single_target_range"].values.astype(np.float64),
            ),
            single_target_alongship_angle=(
                "single_target",
                single_target_alongship_angle.astype(np.float64),
            ),
            single_target_athwartship_angle=(
                "single_target",
                single_target_athwartship_angle.astype(np.float64),
            ),
            single_target_athwartship_angle_sd=(
                "single_target",
                feats["single_target_athwartship_angle_sd"].values.astype(np.float64),
            ),
            single_target_alongship_angle_sd=(
                "single_target",
                feats["single_target_alongship_angle_sd"].values.astype(np.float64),
            ),
            beam_comp_db=(
                "single_target",
                feats["beam_comp_db"].values.astype(np.float64),
            ),
            plike_peak=(
                "single_target",
                feats["plike_peak"].values.astype(np.float64),
            ),
        ),
        coords={
            "single_target": np.arange(
                n_targets,
                dtype=np.int64,
            )
        },
        attrs=dict(method="from_Sp"),
    )


def detect_from_Sp(ds_sp: xr.Dataset, params: dict) -> xr.Dataset:
    """
    Detect single-target candidate locations from point scattering strength Sp.

    This follows the detection part of Echoview split-beam Method 2, but stops
    before target-strength calculation. TS and TS(f) should be computed later
    from the returned target locations.
    """
    params = _validate_params(params)
    ds_sp = _validate_from_Sp_dataset(ds_sp)

    # no need for copy?
    ds_sp = ds_sp.copy()

    deg2rad = np.pi / 180.0

    ds_sp["angle_alongship"] = ds_sp["angle_alongship"] * deg2rad
    ds_sp["angle_athwartship"] = ds_sp["angle_athwartship"] * deg2rad

    if params["beam_comp_model"] == "simrad_lobe":
        for v in [
            "beamwidth_alongship",
            "beamwidth_athwartship",
            "angle_offset_alongship",
            "angle_offset_athwartship",
        ]:
            if v not in ds_sp:
                raise ValueError(f"ds_sp missing required variable for beam compensation: {v}")
            ds_sp[v] = ds_sp[v] * deg2rad

    beam_comp_db = _beam_comp_db(ds_sp, params)

    feats = _phase1_simple(ds_sp, params, beam_comp_db)
    feats = _reject_overlaps_per_ping(feats)

    return _pack_targets(feats, ds_sp)
