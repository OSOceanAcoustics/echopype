import math

import numpy as np
import xarray as xr

DEFAULTS = {
    "SoundSpeed": 1500.0,
    "TS_threshold": -50.0,  # dB
    "PLDL": 6.0,  # dB below peak
    "MinNormPL": 0.7,
    "MaxNormPL": 1.5,
    "DataType": "CW",  # CW only for now
    "block_len": 1e7 / 3,  # ~3.33M cells per block
    "MaxBeamComp": 4.0,  # dB (like default in MATLAB)
    "MaxStdMinAxisAngle": 0.6,  # rad (≈ 34°)
    "MaxStdMajAxisAngle": 0.6,  # rad
}


def init_st_struct():
    """Return a dict mirroring the MATLAB single_targets_tot struct."""
    return {
        "TS_comp": [],
        "TS_uncomp": [],
        "Target_range": [],
        "Target_range_disp": [],
        "Target_range_min": [],
        "Target_range_max": [],
        "idx_r": [],
        "StandDev_Angles_Minor_Axis": [],
        "StandDev_Angles_Major_Axis": [],
        "Angle_minor_axis": [],
        "Angle_major_axis": [],
        "Ping_number": [],
        "Time": [],
        "nb_valid_targets": 0,
        "idx_target_lin": [],
        "pulse_env_before_lin": [],
        "pulse_env_after_lin": [],
        "PulseLength_Normalized_PLDL": [],
        "Transmitted_pulse_length": [],
        "Heave": [],
        "Roll": [],
        "Pitch": [],
        "Heading": [],
        "Dist": [],
    }


def _validate_params(p: dict):
    if p.get("DataType", "CW") not in ("CW", "FM"):
        raise ValueError("DataType must be 'CW' or 'FM'.")
    if not (-120.0 <= float(p["TS_threshold"]) <= -20.0):
        raise ValueError("TS_threshold must be in [-120, -20] dB.")
    if not (1.0 <= float(p["PLDL"]) <= 30.0):
        raise ValueError("PLDL must be in [1, 30] dB.")
    if not (0.0 <= float(p["MinNormPL"]) <= 10.0):
        raise ValueError("MinNormPL must be in [0, 10].")
    if not (0.0 <= float(p["MaxNormPL"]) <= 10.0):
        raise ValueError("MaxNormPL must be in [0, 10].")
    if float(p["block_len"]) <= 0:
        raise ValueError("block_len must be > 0.")
    if not (0.0 <= float(p["MaxBeamComp"]) <= 18.0):
        raise ValueError("MaxBeamComp must be in [0, 18] dB.")
    if not (0.0 <= float(p["MaxStdMinAxisAngle"]) <= 45.0):
        # keep same numeric limits as MATLAB; here we assume radians
        raise ValueError("MaxStdMinAxisAngle must be in [0, 45].")
    if not (0.0 <= float(p["MaxStdMajAxisAngle"]) <= 45.0):
        raise ValueError("MaxStdMajAxisAngle must be in [0, 45].")


def _pull_nav(ds, name_candidates, ping_time_ref):
    """Return a 1D float array aligned to ping_time_ref (or None)."""
    for nm in name_candidates:
        if nm in ds:
            da = ds[nm]
            for tdim in ("ping_time", "time", "time1", "time2"):
                if tdim in da.dims:
                    if tdim != "ping_time":
                        da = da.rename({tdim: "ping_time"})
                    da = da.reindex(
                        ping_time=ping_time_ref,
                        method="nearest",
                        tolerance=np.timedelta64(500, "ms"),
                    )
                    return np.asarray(da.values, dtype=float)
            if da.ndim == 1 and da.size == ping_time_ref.size:
                return np.asarray(da.values, dtype=float)
    return None


def simrad_beam_compensation(
    along_angles_rad: np.ndarray,
    athwart_angles_rad: np.ndarray,
    bw_along_rad: float,
    bw_athwart_rad: float,
) -> np.ndarray:
    """
    Simrad beam pattern compensation [dB], same polynomial as in ESP3/Matecho.

    Parameters
    ----------
    along_angles_rad : array
        Alongship split-beam angles in radians.
    athwart_angles_rad : array
        Athwartship split-beam angles in radians.
    bw_along_rad : float
        3 dB beamwidth alongship in radians.
    bw_athwart_rad : float
        3 dB beamwidth athwartship in radians.

    Returns
    -------
    comp : array
        Beam compensation in dB (one-way).
    """
    # same form as used in Matecho & ESP3:
    # x = 2 * (phi_along / BW_along)
    # y = 2 * (phi_athwart / BW_athwart)
    # comp = 6.0206 * (x^2 + y^2 - 0.18 * x^2 * y^2)
    x = 2.0 * (along_angles_rad / bw_along_rad)
    y = 2.0 * (athwart_angles_rad / bw_athwart_rad)
    comp = 6.0206 * (x**2 + y**2 - 0.18 * (x**2) * (y**2))
    return comp


def detect_esp3(ds: xr.Dataset, params: dict):
    """
    ESP3-style single-target detection (CW branch) translated to Python.

    Parameters
    ----------
    ds : xarray.Dataset
        Calibrated Sv dataset (channel, ping_time, range_sample).
        Must contain Sv (TS-like), depth, and optionally angles.

    params : dict
        Must include:
          - "channel": channel name
        May include:
          - "bottom_da" : bottom depth vs ping_time (DataArray)
          - TS_threshold, PLDL, MinNormPL, MaxNormPL, SoundSpeed,
            MaxBeamComp, MaxStdMinAxisAngle, MaxStdMajAxisAngle,
            block_len, DataType

    Returns
    -------
    out : dict
        ESP3-like structure with TS_comp, TS_uncomp, Target_range, etc.
    """
    if params is None:
        params = {}
    channel = params.get("channel")
    if channel is None:
        raise ValueError("params['channel'] is required.")
    bottom_da = params.get("bottom_da", None)

    # Merge defaults
    p = {
        **DEFAULTS,
        **{k: v for k, v in params.items() if k not in ("channel", "bottom_da")},
    }
    p["DataType"] = p.get("DataType", "CW")
    _validate_params(p)

    # ------------------------------------------------------------------
    # 1) Select data
    # ------------------------------------------------------------------
    Sv = ds["Sv"].sel(channel=channel).transpose("ping_time", "range_sample")
    Depth = ds["depth"].sel(channel=channel).transpose("ping_time", "range_sample")

    along = ds.get("angle_alongship")
    if along is not None:
        if "channel" in along.dims:
            along = along.sel(channel=channel)
        along = along.transpose("ping_time", "range_sample")

    athwt = ds.get("angle_athwartship")
    if athwt is not None:
        if "channel" in athwt.dims:
            athwt = athwt.sel(channel=channel)
        athwt = athwt.transpose("ping_time", "range_sample")

    nb_pings_tot = Sv.sizes["ping_time"]
    nb_samples_tot = Sv.sizes["range_sample"]
    idx_pings_tot = np.arange(nb_pings_tot, dtype=int)
    idx_r_tot = np.arange(nb_samples_tot, dtype=int)

    # Optional bottom cropping (like idx_r_tot>max(idx_bot) removal)
    if bottom_da is not None:
        bb_global = bottom_da
        if "channel" in bb_global.dims:
            bb_global = bb_global.sel(channel=channel)
        bb_global = bb_global.sel(ping_time=Sv["ping_time"])

        D_all = Depth.values  # (ping, range_sample)
        b_all = bb_global.values  # (ping,)
        idx_bot = np.empty(nb_pings_tot, dtype=int)
        for ip in range(nb_pings_tot):
            under = np.nonzero(np.isfinite(D_all[ip]) & (D_all[ip] >= b_all[ip]))[0]
            idx_bot[ip] = under[0] if under.size else (nb_samples_tot - 1)
        max_idx = int(np.nanmax(idx_bot))
        idx_r_tot = np.arange(max_idx + 1, dtype=int)

    # Region/bad-data mask (placeholder: none => all False)
    mask_inter_tot = np.zeros((idx_r_tot.size, idx_pings_tot.size), dtype=bool)

    # Block size
    cells_per_ping = max(1, idx_r_tot.size)
    block_size = int(min(math.ceil(float(p["block_len"]) / cells_per_ping), idx_pings_tot.size))
    num_ite = int(math.ceil(idx_pings_tot.size / max(block_size, 1)))

    # ------------------------------------------------------------------
    # 2) Nav / attitude
    # ------------------------------------------------------------------
    pt_ref = Sv["ping_time"]
    heading_arr = _pull_nav(ds, ["heading", "Heading"], pt_ref)
    pitch_arr = _pull_nav(ds, ["pitch", "Pitch"], pt_ref)
    roll_arr = _pull_nav(ds, ["roll", "Roll"], pt_ref)
    heave_arr = _pull_nav(ds, ["heave", "Heave", "vertical_offset"], pt_ref)
    dist_arr = _pull_nav(ds, ["dist", "Dist", "distance"], pt_ref)

    def _fallback(a, n):
        return a if a is not None else np.zeros(n, dtype=float)

    heading = _fallback(heading_arr, nb_pings_tot)
    pitch = _fallback(pitch_arr, nb_pings_tot)
    roll = _fallback(roll_arr, nb_pings_tot)
    heave = _fallback(heave_arr, nb_pings_tot)
    dist = _fallback(dist_arr, nb_pings_tot)

    times = Sv["ping_time"].values

    # ------------------------------------------------------------------
    # 3) Prepare constants (alpha, c, T, Np)
    # ------------------------------------------------------------------
    alpha = 0.0
    if "sound_absorption" in ds:
        sa = ds["sound_absorption"]
        try:
            alpha = (
                float(sa.sel(channel=channel).values.item())
                if "channel" in sa.dims
                else float(sa.values.item())
            )
        except Exception:
            alpha = 0.0

    c = float(p["SoundSpeed"])

    # Pulse length & number of samples
    if "Np" in p and p["Np"] is not None and int(p["Np"]) > 2:
        Np = int(p["Np"])
        T = float(p.get("pulse_length", 1e-3))
    else:
        dstep_da = Depth.diff("range_sample").median(skipna=True)
        dstep = (
            float(dstep_da.values.item())
            if getattr(dstep_da, "size", 1) == 1
            else float(dstep_da.values)
        )
        dt = (2.0 * dstep) / c if dstep > 0 else 1e-4
        T = float(p.get("pulse_length", 1e-3))
        Np = max(3, int(round(T / dt)))

    TS_thr = float(p["TS_threshold"])
    PLDL = float(p["PLDL"])
    min_len = max(1, int(Np * float(p["MinNormPL"])))
    max_len = max(1, int(math.ceil(Np * float(p["MaxNormPL"]))))
    MaxBeamComp = float(p["MaxBeamComp"])
    MaxStdMin = float(p["MaxStdMinAxisAngle"])
    MaxStdMaj = float(p["MaxStdMajAxisAngle"])

    # Beamwidths (taken from metadata if present; else fallback)
    bw_along_rad = None
    bw_athwart_rad = None
    if "beamwidth_alongship" in ds:
        bw_al = ds["beamwidth_alongship"].sel(channel=channel).values
        bw_al = float(np.asarray(bw_al).item())
        if abs(bw_al) > np.pi:  # assume deg
            bw_al = np.deg2rad(bw_al)
        bw_along_rad = bw_al
    if "beamwidth_athwartship" in ds:
        bw_at = ds["beamwidth_athwartship"].sel(channel=channel).values
        bw_at = float(np.asarray(bw_at).item())
        if abs(bw_at) > np.pi:
            bw_at = np.deg2rad(bw_at)
        bw_athwart_rad = bw_at

    # Fallback beamwidths if missing
    if bw_along_rad is None:
        bw_along_rad = np.deg2rad(7.0)
    if bw_athwart_rad is None:
        bw_athwart_rad = np.deg2rad(7.0)

    # ------------------------------------------------------------------
    # 4) Output struct (like single_targets_tot)
    # ------------------------------------------------------------------
    out = init_st_struct()

    # Bottom DA per-ping, sliced once
    bb = None
    if bottom_da is not None:
        bb = bottom_da
        if "channel" in bb.dims:
            bb = bb.sel(channel=channel)
        bb = bb.sel(ping_time=Sv["ping_time"])

    # ------------------------------------------------------------------
    # 5) Block loop
    # ------------------------------------------------------------------
    for ui in range(num_ite):
        start = ui * block_size
        stop = min((ui + 1) * block_size, idx_pings_tot.size)
        idx_pings = idx_pings_tot[start:stop]

        # Per-block rows (respect bottom)
        idx_r = idx_r_tot.copy()
        if bb is not None and idx_pings.size > 0:
            D_block = Depth.isel(ping_time=idx_pings).values
            b_block = bb.isel(ping_time=idx_pings).values
            idx_bot = np.empty(idx_pings.size, dtype=int)
            for k in range(idx_pings.size):
                under = np.nonzero(np.isfinite(D_block[k]) & (D_block[k] >= b_block[k]))[0]
                idx_bot[k] = under[0] if under.size else (nb_samples_tot - 1)
            max_idx = int(np.nanmax(idx_bot))
            idx_r = idx_r[idx_r <= max_idx]

        # Mask (regions)
        mask = np.zeros((idx_r.size, idx_pings.size), dtype=bool)
        if mask_inter_tot.size:
            r_pos = np.searchsorted(idx_r_tot, idx_r)
            p_pos = np.searchsorted(idx_pings_tot, idx_pings)
            mask |= mask_inter_tot[np.ix_(r_pos, p_pos)]

        # Extract submatrices (TS ~ Sv here)
        TS = (
            Sv.isel(ping_time=idx_pings, range_sample=idx_r)
            .transpose("range_sample", "ping_time")
            .values.copy()
        )
        DEP = (
            Depth.isel(ping_time=idx_pings, range_sample=idx_r)
            .transpose("range_sample", "ping_time")
            .values
        )

        nb_samples, nb_pings = TS.shape

        # Angles sub-block (in radians)
        along_block = None
        athwt_block = None
        if along is not None:
            along_block = (
                along.isel(ping_time=idx_pings, range_sample=idx_r)
                .transpose("range_sample", "ping_time")
                .data
            )
            along_block = np.deg2rad(along_block)
        if athwt is not None:
            athwt_block = (
                athwt.isel(ping_time=idx_pings, range_sample=idx_r)
                .transpose("range_sample", "ping_time")
                .data
            )
            athwt_block = np.deg2rad(athwt_block)

        # Under-bottom mask
        if bb is not None and idx_pings.size > 0:
            bcol = bb.isel(ping_time=idx_pings).values.reshape(1, -1)
            mask |= DEP >= bcol

        TS[mask] = -999.0
        if not np.any(TS > -999.0):
            continue

        # Remove trailing all-masked rows
        valid_rows = np.where((TS > -999.0).any(axis=1))[0]
        last_row = valid_rows.max()
        row_sel = slice(0, last_row + 1)
        TS = TS[row_sel, :]
        DEP = DEP[row_sel, :]
        if along_block is not None:
            along_block = along_block[row_sel, :]
        if athwt_block is not None:
            athwt_block = athwt_block[row_sel, :]
        idx_r = idx_r[row_sel]
        nb_samples, nb_pings = TS.shape

        # TVG matrix and Power (as in MATLAB)
        r_eff = DEP - c * T / 4.0
        TVG_mat = np.where(
            r_eff > 0.0,
            40.0 * np.log10(r_eff) + 2.0 * alpha * r_eff,
            np.nan,
        )
        Power = TS - TVG_mat

        # Peak detection on smoothed TS (or Power), like ESP3
        peak_calc = "TS"  # ESP3 uses 'TS' by default in your snippet
        if peak_calc.lower() == "power":
            base_mat = Power
        else:
            base_mat = TS

        # moving average over Np/2 samples in linear domain
        win = max(int(np.floor(Np / 2)), 1)
        kernel = np.ones(win, dtype=float) / float(win)

        base_lin = 10.0 ** (base_mat / 10.0)
        smooth_lin = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode="same"),
            axis=0,
            arr=base_lin,
        )
        peak_mat = 10.0 * np.log10(smooth_lin)
        peak_mat[TS <= -999.0] = -999.0

        # --- Find peaks with MinSeparation ≈ Np (along range axis) ---
        idx_peaks_bool = np.zeros_like(peak_mat, dtype=bool)
        for jp in range(nb_pings):
            z = peak_mat[:, jp]
            valid = z > -999.0
            if np.count_nonzero(valid) < 3:
                continue
            cand = np.where(valid[1:-1] & (z[1:-1] > z[:-2]) & (z[1:-1] >= z[2:]))[0] + 1

            # MinSeparation = Np
            last = -(10**9)
            keep_idx = []
            for k in cand:
                if k - last >= Np:
                    keep_idx.append(k)
                    last = k
            if keep_idx:
                idx_peaks_bool[np.array(keep_idx, dtype=int), jp] = True

        idx_peaks_bool[TS <= -999.0] = False

        # linear indices of peaks (like idx_peaks_lin)
        i_peaks_lin, j_peaks_lin = np.where(idx_peaks_bool)
        nb_peaks = len(i_peaks_lin)
        if nb_peaks == 0:
            continue

        # Level of local maxima minus PLDL
        peak_vals = peak_mat[idx_peaks_bool]  # 1D (same order as i_peaks_lin)
        pulse_level = peak_vals - PLDL

        # To mimic MATLAB's pulse-env scan, we will grow left/right until
        # we fall below pulse_level or reach max_len.
        pulse_env_before = np.zeros(nb_peaks, dtype=int)
        pulse_env_after = np.zeros(nb_peaks, dtype=int)

        for idx_peak in range(nb_peaks):
            ip = i_peaks_lin[idx_peak]
            jp = j_peaks_lin[idx_peak]
            thr = pulse_level[idx_peak]
            zcol = peak_mat[:, jp]

            # grow up (before)
            left = 0
            i = ip - 1
            while i >= 0 and left < max_len and np.isfinite(zcol[i]) and zcol[i] >= thr:
                left += 1
                i -= 1

            # grow down (after)
            right = 0
            i = ip + 1
            nP = zcol.size
            while i < nP and right < max_len and np.isfinite(zcol[i]) and zcol[i] >= thr:
                right += 1
                i += 1

            pulse_env_before[idx_peak] = left
            pulse_env_after[idx_peak] = right

        pulse_length_lin = pulse_env_before + pulse_env_after + 1
        # good pulses according to [MinNormPL, MaxNormPL]
        good_pulses = (pulse_length_lin >= min_len) & (pulse_length_lin <= max_len)

        # Filter peaks to targets
        i_targets = i_peaks_lin[good_pulses]
        j_targets = j_peaks_lin[good_pulses]
        pulse_before = pulse_env_before[good_pulses]
        pulse_after = pulse_env_after[good_pulses]
        pulse_len = pulse_length_lin[good_pulses]
        nb_targets = len(i_targets)
        if nb_targets == 0:
            continue

        # ------------------------------------------------------------------
        # For each target, compute:
        #  - samples_targets_power / range / angles
        #  - target_range (weighted by Power)
        #  - beam comp at peak (target_comp)
        #  - TS_uncomp = target_peak_power + TVG(target_range)
        #  - TS_comp   = TS_uncomp + target_comp
        #  - apply angle std filter and MaxBeamComp & TS_threshold
        # ------------------------------------------------------------------
        for it in range(nb_targets):
            ip = int(i_targets[it])
            jp = int(j_targets[it])
            left = int(pulse_before[it])
            right = int(pulse_after[it])

            i0 = max(ip - left, 0)
            i1 = min(ip + right, nb_samples - 1)
            seg_slice = slice(i0, i1 + 1)

            pow_seg = Power[seg_slice, jp]
            dep_seg = DEP[seg_slice, jp]

            # If all NaNs or -inf => skip
            if not np.isfinite(pow_seg).any():
                continue

            # angles segment
            if along_block is not None and athwt_block is not None:
                al_seg = along_block[seg_slice, jp]
                at_seg = athwt_block[seg_slice, jp]
            else:
                al_seg = np.full_like(pow_seg, np.nan, dtype=float)
                at_seg = np.full_like(pow_seg, np.nan, dtype=float)

            # std + mean of angles (for filtering & output)
            std_al = float(np.nanstd(al_seg))
            std_at = float(np.nanstd(at_seg))
            phi_al = float(np.nanmean(al_seg))
            phi_at = float(np.nanmean(at_seg))

            # Angle stability filter (CW only)
            if p["DataType"] == "CW":
                if std_al > MaxStdMin or std_at > MaxStdMaj:
                    continue

            # Power-weighted range (like target_range)
            w = 10.0 ** (pow_seg / 10.0)
            if not np.isfinite(w).any() or np.nansum(w) <= 0:
                continue
            target_range = float(np.nansum(w * dep_seg) / np.nansum(w) - c * T / 4.0)
            if target_range < 0:
                target_range = 0.0

            target_range_min = float(np.nanmin(dep_seg))
            target_range_max = float(np.nanmax(dep_seg))

            # Peak power within pulse window
            # (we take max of pow_seg and its index)
            idx_loc_peak = int(np.nanargmax(pow_seg))
            target_peak_power = float(pow_seg[idx_loc_peak])
            if not np.isfinite(target_peak_power):
                continue

            # Beam comp at peak sample
            if along_block is not None and athwt_block is not None:
                al_peak = al_seg[idx_loc_peak]
                at_peak = at_seg[idx_loc_peak]
                if np.isfinite(al_peak) and np.isfinite(at_peak):
                    target_comp = float(
                        simrad_beam_compensation(
                            np.array([al_peak]),
                            np.array([at_peak]),
                            bw_along_rad,
                            bw_athwart_rad,
                        )[0]
                    )
                else:
                    target_comp = 0.0
            else:
                target_comp = 0.0

            # TVG at target_range (like ESP3: TVG = 40 log10(R) + 2 alpha R)
            if target_range > 0:
                TVG = 40.0 * np.log10(target_range) + 2.0 * alpha * target_range
            else:
                TVG = np.nan

            if not np.isfinite(TVG):
                continue

            target_TS_uncomp = target_peak_power + TVG
            target_TS_comp = target_TS_uncomp + target_comp

            # Filters on TS_comp and beam comp
            if target_TS_comp <= TS_thr:
                continue
            if abs(target_comp) > MaxBeamComp:
                continue

            # Store detection
            ping_glob = int(idx_pings[jp])
            range_idx_glob = int(idx_r[ip])

            out["TS_comp"].append(target_TS_comp)
            out["TS_uncomp"].append(target_TS_uncomp)
            out["Target_range"].append(target_range)
            out["Target_range_disp"].append(target_range + c * T / 4.0)
            out["Target_range_min"].append(target_range_min)
            out["Target_range_max"].append(target_range_max)

            out["idx_r"].append(range_idx_glob)
            out["Ping_number"].append(ping_glob)
            out["Time"].append(times[ping_glob])
            out["idx_target_lin"].append(int(ping_glob * nb_samples_tot + range_idx_glob))

            out["pulse_env_before_lin"].append(int(left))
            out["pulse_env_after_lin"].append(int(right))
            out["PulseLength_Normalized_PLDL"].append(pulse_len[it] / float(Np))
            out["Transmitted_pulse_length"].append(int(pulse_len[it]))

            out["StandDev_Angles_Minor_Axis"].append(std_al)
            out["StandDev_Angles_Major_Axis"].append(std_at)
            out["Angle_minor_axis"].append(phi_al)
            out["Angle_major_axis"].append(phi_at)

            out["Heave"].append(float(heave[ping_glob]))
            out["Roll"].append(float(roll[ping_glob]))
            out["Pitch"].append(float(pitch[ping_glob]))
            out["Heading"].append(float(heading[ping_glob]))
            out["Dist"].append(float(dist[ping_glob]))

    out["nb_valid_targets"] = len(out["TS_comp"])
    return out
