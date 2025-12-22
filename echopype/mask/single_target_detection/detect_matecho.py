import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import xarray as xr

# ---
# Defaults
PARAM_DEFAULTS = {
    "SoundSpeed": 1500.0,
    "MinThreshold": -60.0,
    "MaxAngleOneWayCompression": 6.0,
    "MaxPhaseDeviation": 8.0,
    "MinEchoLength": 0.8,
    "MaxEchoLength": 1.8,
    "MinEchoSpace": 1.0,
    "MinEchoDepthM": 3.0,
    "MaxEchoDepthM": 38.0,
    "FloorFilterTolerance": 0.0,
    "tvg_start_sample": 3,  # (EK60=3, EK80=1)
    # Sv->TS
    "psi_two_way": 0.0,
    "Sa_correction": 0.0,
    "Sa_EK80_nominal": 0.0,
    # Required for CW range/TS conversion
    "pulse_length": None,  # seconds (pd in MATLAB)
    "sound_speed": None,  # m/s (c in MATLAB)
    "alpha": 0.0,  # dB/m (alpha(kf))
    # TD/CH
    "transducer_depth": None,  # TD (m), from surface
    "heave_compensation": 0.0,  # CH (m) per ping or scalar
    # (optional)
    "bottom_da": None,
}


# ---
# Output conversion
def _matecho_struct_to_dataset(out: dict, *, channel: str | None = None) -> xr.Dataset:
    n = int(out.get("nb_valid_targets", len(out.get("TS_comp", []))))
    target = np.arange(n, dtype=int)

    def _arr(key, dtype=float):
        vals = out.get(key, [])
        if n == 0:
            return np.asarray([], dtype=dtype)
        a = np.asarray(vals)
        if dtype is not None:
            a = a.astype(dtype)
        if a.size != n:
            raise ValueError(f"Output field '{key}' has length {a.size}, expected {n}.")
        return a

    tsfreq = np.asarray(out.get("TSfreq_matrix", np.empty((0, 23))), dtype=float)
    if tsfreq.ndim == 1:
        tsfreq = tsfreq.reshape(1, -1)
    if tsfreq.size == 0:
        tsfreq = np.empty((0, 23), dtype=float)
    if tsfreq.shape[0] != n:
        raise ValueError(
            f"TSfreq_matrix has {tsfreq.shape[0]} rows but expected {n} (nb_valid_targets)."
        )

    ds_out = xr.Dataset(
        data_vars=dict(
            nb_valid_targets=((), n),
            TS_comp=("target", _arr("TS_comp", float), {"units": "dB"}),
            TS_uncomp=("target", _arr("TS_uncomp", float), {"units": "dB"}),
            Target_range=("target", _arr("Target_range", float), {"units": "m"}),
            Target_range_disp=("target", _arr("Target_range_disp", float), {"units": "m"}),
            Target_range_min=("target", _arr("Target_range_min", float), {"units": "m"}),
            Target_range_max=("target", _arr("Target_range_max", float), {"units": "m"}),
            idx_r=("target", _arr("idx_r", int)),
            idx_target_lin=("target", _arr("idx_target_lin", int)),
            pulse_env_before_lin=("target", _arr("pulse_env_before_lin", float)),
            pulse_env_after_lin=("target", _arr("pulse_env_after_lin", float)),
            PulseLength_Normalized_PLDL=("target", _arr("PulseLength_Normalized_PLDL", float)),
            Transmitted_pulse_length=("target", _arr("Transmitted_pulse_length", float)),
            Angle_minor_axis=("target", _arr("Angle_minor_axis", float), {"units": "rad"}),
            Angle_major_axis=("target", _arr("Angle_major_axis", float), {"units": "rad"}),
            StandDev_Angles_Minor_Axis=(
                "target",
                _arr("StandDev_Angles_Minor_Axis", float),
                {"units": "phase_steps"},
            ),
            StandDev_Angles_Major_Axis=(
                "target",
                _arr("StandDev_Angles_Major_Axis", float),
                {"units": "phase_steps"},
            ),
            Heave=("target", _arr("Heave", float), {"units": "m"}),
            Roll=("target", _arr("Roll", float), {"units": "rad"}),
            Pitch=("target", _arr("Pitch", float), {"units": "rad"}),
            Heading=("target", _arr("Heading", float), {"units": "rad"}),
            Dist=("target", _arr("Dist", float), {"units": "m"}),
            TSfreq_matrix=(("target", "tsfreq_col"), tsfreq, {}),
        ),
        coords=dict(
            target=target,
            tsfreq_col=np.arange(tsfreq.shape[1], dtype=int),
            Ping_number=("target", _arr("Ping_number", int)),
            Time=("target", _arr("Time", np.dtype("datetime64[ns]"))),
        ),
        attrs=dict(
            method="matecho",
            channel=str(channel) if channel is not None else "",
            nb_valid_targets=n,
        ),
    )
    if channel is not None:
        ds_out = ds_out.assign_coords(channel=channel)
    return ds_out


def _empty_out(channel: str) -> xr.Dataset:
    out = {
        "nb_valid_targets": 0,
        "TS_comp": [],
        "TS_uncomp": [],
        "Target_range": [],
        "Target_range_disp": [],
        "Target_range_min": [],
        "Target_range_max": [],
        "idx_r": [],
        "idx_target_lin": [],
        "pulse_env_before_lin": [],
        "pulse_env_after_lin": [],
        "PulseLength_Normalized_PLDL": [],
        "Transmitted_pulse_length": [],
        "Angle_minor_axis": [],
        "Angle_major_axis": [],
        "StandDev_Angles_Minor_Axis": [],
        "StandDev_Angles_Major_Axis": [],
        "Heave": [],
        "Roll": [],
        "Pitch": [],
        "Heading": [],
        "Dist": [],
        "Ping_number": [],
        "Time": [],
        "TSfreq_matrix": np.empty((0, 23)),
    }
    return _matecho_struct_to_dataset(out, channel=channel)


# ---
# Helpers
def _make_dbg() -> dict:  # to print debug
    return {
        "enable": True,
        "print_header": True,
        "print_block": True,
        "print_cond_1to5": True,
        "print_cond6": True,
        "print_cond7": True,
        "print_per_ping": False,
        "print_examples": True,
        "n_examples": 10,
    }


def _nanmin(a):
    return float(np.nanmin(a)) if np.any(np.isfinite(a)) else np.nan


def _nanmax(a):
    return float(np.nanmax(a)) if np.any(np.isfinite(a)) else np.nan


def _nanmed(a):
    return float(np.nanmedian(a)) if np.any(np.isfinite(a)) else np.nan


def _merge_params(params: dict) -> dict:
    if params is None:
        raise ValueError("params is required.")
    return {**PARAM_DEFAULTS, **params}


def _validate_inputs(ds: xr.Dataset, Param: dict) -> Tuple[str, xr.DataArray]:
    channel = Param.get("channel")
    if channel is None:
        raise ValueError("params['channel'] is required.")

    var_name = Param.get("var_name", "Sv")
    if var_name not in ds:
        raise ValueError(f"var_name '{var_name}' not found in input dataset.")

    Sv = ds[var_name]
    required_dims = {"channel", "ping_time", "range_sample"}
    if not required_dims.issubset(set(Sv.dims)):
        raise ValueError(f"{var_name} must have dims {sorted(required_dims)}. Got: {Sv.dims}.")

    if channel not in Sv["channel"].values:
        raise ValueError(f"Channel '{channel}' not found in {var_name}.")

    return channel, Sv


def _extract_matrices(ds: xr.Dataset, Sv: xr.DataArray, channel: str) -> Dict[str, Any]:
    sv_mat = Sv.sel(channel=channel).transpose("ping_time", "range_sample").values
    npings, NbSamp = sv_mat.shape

    if "angle_alongship" in ds:
        al_mat = (
            ds["angle_alongship"].sel(channel=channel).transpose("ping_time", "range_sample").values
        )
    else:
        al_mat = np.full_like(sv_mat, np.nan, dtype=float)

    if "angle_athwartship" in ds:
        ath_mat = (
            ds["angle_athwartship"]
            .sel(channel=channel)
            .transpose("ping_time", "range_sample")
            .values
        )
    else:
        ath_mat = np.full_like(sv_mat, np.nan, dtype=float)

    if "depth" in ds:
        depth = ds["depth"].sel(channel=channel).median("ping_time", skipna=True).values
    else:
        depth = np.arange(NbSamp, dtype=float)

    if len(depth) < 2:
        raise ValueError("Need at least 2 samples to compute delta.")

    # trim NaN tail
    valid = np.isfinite(depth)
    if not np.any(valid):
        raise ValueError("Depth vector is all-NaN for this channel.")
    last = np.where(valid)[0][-1] + 1

    depth = depth[:last]
    sv_mat = sv_mat[:, :last]
    al_mat = al_mat[:, :last]
    ath_mat = ath_mat[:, :last]
    NbSamp = last

    return {
        "sv_mat": sv_mat,
        "al_mat": al_mat,
        "ath_mat": ath_mat,
        "depth": depth,
        "npings": npings,
        "NbSamp": NbSamp,
    }


def _compute_delta(depth: np.ndarray) -> float:
    dd = np.diff(depth)
    dd = dd[np.isfinite(dd)]
    if dd.size == 0:
        raise ValueError("Cannot compute delta from depth: all diffs are NaN.")
    return float(np.nanmedian(dd))


def _extract_metadata(
    ds: xr.Dataset, Param: dict, params: dict, channel: str, npings: int, DBG: dict
) -> Dict[str, Any]:
    # c, pd
    c = Param["sound_speed"]
    if c is None:
        if "sound_speed" in ds:
            c = float(ds["sound_speed"].values)
        else:
            c = float(PARAM_DEFAULTS["SoundSpeed"])

    pd = Param["pulse_length"]
    if pd is None:
        raise ValueError("params['pulse_length'] is required for CW (pd in MATLAB).")

    alpha = float(Param.get("alpha", 0.0))
    psi = float(Param.get("psi_two_way", 0.0))
    sa_cor = float(Param.get("Sa_correction", 0.0))
    SacorEK80Nominal = float(Param.get("Sa_EK80_nominal", 0.0))

    # beam meta
    ouv_al = (
        float(ds.get("beamwidth_alongship", xr.DataArray(np.nan)).sel(channel=channel).values)
        if "beamwidth_alongship" in ds
        else np.nan
    )
    ouv_at = (
        float(ds.get("beamwidth_athwartship", xr.DataArray(np.nan)).sel(channel=channel).values)
        if "beamwidth_athwartship" in ds
        else np.nan
    )
    dec_al = (
        float(ds.get("angle_offset_alongship", xr.DataArray(0.0)).sel(channel=channel).values)
        if "angle_offset_alongship" in ds
        else 0.0
    )
    dec_at = (
        float(ds.get("angle_offset_athwartship", xr.DataArray(0.0)).sel(channel=channel).values)
        if "angle_offset_athwartship" in ds
        else 0.0
    )
    al_sens = (
        float(ds.get("angle_sensitivity_alongship", xr.DataArray(1.0)).sel(channel=channel).values)
        if "angle_sensitivity_alongship" in ds
        else 1.0
    )
    ath_sens = (
        float(
            ds.get("angle_sensitivity_athwartship", xr.DataArray(1.0)).sel(channel=channel).values
        )
        if "angle_sensitivity_athwartship" in ds
        else 1.0
    )

    missing_meta = []
    if np.isnan(ouv_al):
        missing_meta.append("beamwidth_alongship")
    if np.isnan(ouv_at):
        missing_meta.append("beamwidth_athwartship")
    if missing_meta:
        warnings.warn(
            f"Missing beam metadata for channel '{channel}': {', '.join(missing_meta)}. ",
            UserWarning,
            stacklevel=2,
        )

    # TD/CH
    TD = Param.get("transducer_depth", None)
    if TD is None:
        if "transducer_depth" in ds:
            TD = float(ds["transducer_depth"].sel(channel=channel).values)
        else:
            TD = 0.0
    TD = float(TD)

    CH = Param.get("heave_compensation", 0.0)
    if np.ndim(CH) == 0:
        CH_vec = np.full((npings,), float(CH), dtype=float)
    else:
        CH_vec = np.asarray(CH, dtype=float)
        if CH_vec.size != npings:
            raise ValueError("heave_compensation length must match number of pings.")

    # bottom
    bottom_da = Param.get("bottom_da", None)
    if bottom_da is None:
        bot = np.full((npings,), np.inf, dtype=float)
    else:
        bot = bottom_da.sel(ping_time=ds["ping_time"]).values.astype(float)

    # timing
    timeSampleInterval_usec = params.get("timeSampleInterval_usec", None)
    if timeSampleInterval_usec is None:
        raise ValueError("params['timeSampleInterval_usec'] is required.")

    # angle units detection
    beamwidth_units = ""
    if "beamwidth_alongship" in ds and hasattr(ds["beamwidth_alongship"], "attrs"):
        beamwidth_units = str(ds["beamwidth_alongship"].attrs.get("units", "")).lower()
    angles_are_degrees = "deg" in beamwidth_units

    if DBG["enable"]:
        warnings.warn(
            f"beamwidth_alongship units='{beamwidth_units}' -> \
                angles_are_degrees = {angles_are_degrees}",
            category=UserWarning,
            stacklevel=2,
        )

    return {
        "c": float(c),
        "pd": float(pd),
        "alpha": float(alpha),
        "psi": float(psi),
        "sa_cor": float(sa_cor),
        "SacorEK80Nominal": float(SacorEK80Nominal),
        "ouv_al": float(ouv_al),
        "ouv_at": float(ouv_at),
        "dec_al": float(dec_al),
        "dec_at": float(dec_at),
        "al_sens": float(al_sens),
        "ath_sens": float(ath_sens),
        "TD": float(TD),
        "CH_vec": CH_vec,
        "bot": bot,
        "timeSampleInterval_usec": float(timeSampleInterval_usec),
        "angles_are_degrees": bool(angles_are_degrees),
    }


def _normalise_angles(al_mat, ath_mat, meta: dict, DBG: dict):
    if not meta["angles_are_degrees"]:
        return al_mat, ath_mat, meta

    DEG2RAD = np.pi / 180.0
    if DBG["enable"]:
        warnings.warn(
            "Converting angles, beamwidths, and offsets from degrees to radians.",
            category=UserWarning,
            stacklevel=2,
        )

    al_mat = al_mat * DEG2RAD
    ath_mat = ath_mat * DEG2RAD
    meta = dict(meta)
    meta["dec_al"] *= DEG2RAD
    meta["dec_at"] *= DEG2RAD
    meta["ouv_al"] *= DEG2RAD
    meta["ouv_at"] *= DEG2RAD
    return al_mat, ath_mat, meta


def _compute_pulse_derived(meta: dict, Param: dict, delta: float) -> Dict[str, float]:
    c = meta["c"]
    pd = meta["pd"]
    pulse_len_m = c * pd / 2.0

    n_min = int(np.round((pulse_len_m * float(Param["MinEchoLength"])) / delta))
    n_max = int(np.round((pulse_len_m * float(Param["MaxEchoLength"])) / delta))
    n_min = max(n_min, 1)
    n_max = max(n_max, n_min)

    min_space_m = float(Param["MinEchoSpace"]) * pulse_len_m
    return {
        "pulse_len_m": float(pulse_len_m),
        "n_min": int(n_min),
        "n_max": int(n_max),
        "min_space_m": float(min_space_m),
    }


# ---
# Helpers: core matrices (ts_range, tsu, Plike, ts)
def _compute_core_matrices(
    sv_mat: np.ndarray,
    al_mat: np.ndarray,
    ath_mat: np.ndarray,
    depth: np.ndarray,
    delta: float,
    meta: dict,
    Param: dict,
    channel: str,
) -> Dict[str, Any]:
    npings, NbSamp = sv_mat.shape

    ind_range0 = int(Param["tvg_start_sample"])
    dec_tir = 8

    c = meta["c"]
    pd = meta["pd"]
    alpha = meta["alpha"]
    psi = meta["psi"]
    sa_cor = meta["sa_cor"]
    SacorEK80Nominal = meta["SacorEK80Nominal"]

    # CW conversion constant
    sv2ts = 10.0 * np.log10(c * pd / 2.0) + psi + 2.0 * sa_cor + 2.0 * SacorEK80Nominal

    # NechP (dt-based)
    NechP = pd / (meta["timeSampleInterval_usec"] * 1e-6)

    # range vector in metres (one-way)
    jj = np.arange(1, NbSamp + 1, dtype=float)
    ts_range_1d = np.maximum(delta, (jj - ind_range0) * delta)
    ts_range = np.tile(ts_range_1d[None, :], (npings, 1))

    # bottom index constraint
    bot = meta["bot"]
    TD = meta["TD"]
    CH_vec = meta["CH_vec"]
    ir = (bot - TD - CH_vec - float(Param["FloorFilterTolerance"])) / delta
    ir = np.minimum(ir, NbSamp).astype(float)

    # tsu / Plike / ts
    tsu_mat = sv_mat + 20.0 * np.log10(ts_range) + sv2ts
    Plike_mat = tsu_mat - 40.0 * np.log10(ts_range) - 2.0 * alpha * ts_range

    ouv_al = meta["ouv_al"]
    ouv_at = meta["ouv_at"]
    dec_al = meta["dec_al"]
    dec_at = meta["dec_at"]

    x = 2.0 * (al_mat - dec_al) / ouv_al
    y = 2.0 * (ath_mat - dec_at) / ouv_at
    ts_mat = tsu_mat + 6.0206 * (x * x + y * y - 0.18 * (x * x) * (y * y))

    return {
        "npings": npings,
        "NbSamp": NbSamp,
        "ind_range0": ind_range0,
        "dec_tir": dec_tir,
        "NechP": float(NechP),
        "sv2ts": float(sv2ts),
        "ts_range": ts_range,
        "ir": ir,
        "tsu_mat": tsu_mat,
        "Plike_mat": Plike_mat,
        "ts_mat": ts_mat,
        "depth": depth,
        "channel": channel,
    }


# ---
# Helpers: Conditions 1–5
def _cond_1to5(core: dict, Param: dict, DBG: dict) -> Tuple[np.ndarray, np.ndarray]:
    NbSamp = core["NbSamp"]
    ts_mat = core["ts_mat"]
    tsu_mat = core["tsu_mat"]
    Plike_mat = core["Plike_mat"]
    ir = core["ir"]
    dec_tir = core["dec_tir"]

    cols = np.arange(1, NbSamp - 1, dtype=int)  # never evaluate edge samples

    cond1 = ts_mat[:, 1:-1] > float(Param["MinThreshold"])
    cond2 = (ts_mat[:, 1:-1] - tsu_mat[:, 1:-1]) <= 2.0 * float(Param["MaxAngleOneWayCompression"])

    left = Plike_mat[:, 0:-2]
    mid = Plike_mat[:, 1:-1]
    right = Plike_mat[:, 2:]
    cond3 = mid >= np.nanmax(np.stack([left, right]), axis=0)

    col_1based = cols + 1
    cond4 = col_1based[None, :] < ir[:, None]
    cond5 = col_1based[None, :] > dec_tir

    mask_1to5 = cond1 & cond2 & cond3 & cond4 & cond5
    i1, ind0 = np.where(mask_1to5)
    ind = ind0 + 1  # back to full-matrix python col index

    if DBG["enable"] and DBG["print_cond_1to5"]:
        warnings.warn(
            "[Cond 1–5] Counts (pixel-level):\n"
            f"  c1(ts > thr): {int(np.count_nonzero(cond1))}\n"
            f"  c2(comp <=): {int(np.count_nonzero(cond2))}\n"
            f"  c3(loc max): {int(np.count_nonzero(cond3))}\n"
            f"  c4(above bottom): {int(np.count_nonzero(cond4))}\n"
            f"  c5(idx > dec_tir): {int(np.count_nonzero(cond5))}\n"
            f"  ALL: {int(np.count_nonzero(mask_1to5))}\n"
            f"  detections after Cond 1–5: {ind.size}",
            category=UserWarning,
            stacklevel=2,
        )

    return i1, ind


# ---
# Helpers: Condition 6
def _cond_6_phase(
    i1: np.ndarray,
    ind: np.ndarray,
    al_mat: np.ndarray,
    ath_mat: np.ndarray,
    core: dict,
    meta: dict,
    Param: dict,
    DBG: dict,
) -> np.ndarray:
    NbSamp = core["NbSamp"]
    al_sens = meta["al_sens"]
    ath_sens = meta["ath_sens"]

    std_ph_al = np.full((ind.size,), np.nan, dtype=float)
    std_ph_ath = np.full((ind.size,), np.nan, dtype=float)

    for i in range(ind.size):
        r0 = max(ind[i] - 2, 0)
        r1 = min(ind[i] + 2, NbSamp - 1)
        sl = slice(r0, r1 + 1)
        std_ph_al[i] = np.nanstd(al_mat[i1[i], sl] * 128.0 / np.pi * al_sens)
        std_ph_ath[i] = np.nanstd(ath_mat[i1[i], sl] * 128.0 / np.pi * ath_sens)

    ind2 = np.where(
        (std_ph_al <= float(Param["MaxPhaseDeviation"]))
        & (std_ph_ath <= float(Param["MaxPhaseDeviation"]))
    )[0]

    if DBG["enable"] and DBG["print_cond6"]:
        warnings.warn(
            "[Cond 6] Phase deviation:\n"
            f"  std_ph_al: min={_nanmin(std_ph_al):.3f} "
            f"med={_nanmed(std_ph_al):.3f} "
            f"max={_nanmax(std_ph_al):.3f}\n"
            f"  std_ph_ath: min={_nanmin(std_ph_ath):.3f} "
            f"med={_nanmed(std_ph_ath):.3f} "
            f"max={_nanmax(std_ph_ath):.3f}\n"
            f"  Passing (<= MaxPhaseDeviation={float(Param['MaxPhaseDeviation']):.3f}): "
            f"{ind2.size} / {ind.size}",
            category=UserWarning,
            stacklevel=2,
        )

    return ind2


# ---
# Helpers: Condition 7
def _cond_7_echo_length(
    i1: np.ndarray,
    ind: np.ndarray,
    ind2: np.ndarray,
    core: dict,
    derived: dict,
    Param: dict,
    DBG: dict,
) -> Dict[str, Any]:
    NbSamp = core["NbSamp"]
    Plike_mat = core["Plike_mat"]
    ir = core["ir"]

    iid = ind[ind2]  # python col index (0-based)
    iip = i1[ind2]  # ping indices
    necho = iid.size

    isup = np.full((necho,), np.nan, dtype=float)
    iinf = np.full((necho,), np.nan, dtype=float)

    for i in range(necho):
        ip = iip[i]
        ii = iid[i]

        ir_floor = int(np.floor(ir[ip]))
        ir_floor = max(ir_floor, 1)
        ir_floor = min(ir_floor, NbSamp)

        thr = Plike_mat[ip, ii] - 6.0

        forward = Plike_mat[ip, ii:ir_floor] < thr
        if np.any(forward):
            umin = np.where(forward)[0].min()
            isup[i] = ii + umin + 1.0
        else:
            isup[i] = ii + 1.0

        backward = Plike_mat[ip, 0 : ii + 1] < thr
        if np.any(backward):
            umax = np.where(backward)[0].max()
            iinf[i] = umax + 2.0
        else:
            iinf[i] = ii + 1.0

    L6dB = isup - iinf + 1.0

    n_min = derived["n_min"]
    n_max = derived["n_max"]
    ind4 = np.where((L6dB >= n_min) & (L6dB <= n_max))[0]

    if DBG["enable"] and DBG["print_cond7"]:
        warnings.warn(
            "[Cond 7] Echo length (L6dB):\n"
            f"  L6dB: min={_nanmin(L6dB):.0f} "
            f"med={_nanmed(L6dB):.0f} "
            f"max={_nanmax(L6dB):.0f}\n"
            f"  Bounds depth-based [{n_min}..{n_max}] samples\n"
            f"  Passing depth-based: {ind4.size} / {necho}",
            category=UserWarning,
            stacklevel=2,
        )

    return {"iid": iid, "iip": iip, "iinf": iinf, "isup": isup, "L6dB": L6dB, "ind4": ind4}


# ---
# Helpers: Condition 8 (refinement + spacing + output build)
def _refine_ir_fin_1b(Plike_row: np.ndarray, iinf_1b: int, isup_1b: int) -> float:
    il6dB = np.arange(iinf_1b - 1, isup_1b, dtype=int)
    w = 10.0 ** (Plike_row[il6dB] / 10.0)
    return float(np.nansum((il6dB + 1) * w) / np.nansum(w))


def _cond_8_build_outputs(
    ds: xr.Dataset,
    params: dict,
    i1: np.ndarray,
    ind2: np.ndarray,
    c7: dict,
    core: dict,
    meta: dict,
    derived: dict,
    al_mat: np.ndarray,
    ath_mat: np.ndarray,
    depth: np.ndarray,
    Param: dict,
    DBG: dict,
    delta: float,
) -> Tuple[dict, List[List[float]]]:

    ts_mat = core["ts_mat"]
    tsu_mat = core["tsu_mat"]
    Plike_mat = core["Plike_mat"]
    ind_range0 = core["ind_range0"]

    Ping_number_full = np.arange(1, core["npings"] + 1, dtype=int)
    ping_time = ds["ping_time"].values

    # candidates per ping after Cond7
    iid = c7["iid"]
    iinf = c7["iinf"]
    isup = c7["isup"]
    ind4 = c7["ind4"]

    ip0 = i1[ind2[ind4]]
    ipu = np.unique(ip0)

    out = {
        k: []
        for k in [
            "TS_comp",
            "TS_uncomp",
            "Target_range",
            "Target_range_disp",
            "Target_range_min",
            "Target_range_max",
            "idx_r",
            "StandDev_Angles_Minor_Axis",
            "StandDev_Angles_Major_Axis",
            "Angle_minor_axis",
            "Angle_major_axis",
            "Ping_number",
            "Time",
            "idx_target_lin",
            "pulse_env_before_lin",
            "pulse_env_after_lin",
            "PulseLength_Normalized_PLDL",
            "Transmitted_pulse_length",
            "Heave",
            "Roll",
            "Pitch",
            "Heading",
            "Dist",
        ]
    }
    TSfreq_rows: List[List[float]] = []

    alpha = meta["alpha"]
    TD = meta["TD"]
    CH_vec = meta["CH_vec"]
    bot = meta["bot"]
    min_space_m = derived["min_space_m"]

    for ip in ipu:
        ip2 = np.where(ip0 == ip)[0]
        sel = ind4[ip2]

        # compute fine range ir_fin
        ir_fin = np.full((sel.size,), np.nan, dtype=float)
        for j, idx in enumerate(sel):
            ir_fin[j] = _refine_ir_fin_1b(Plike_mat[ip, :], int(iinf[idx]), int(isup[idx]))

        i_ini = iid[sel]
        i_ini_1b = i_ini + 1.0

        ts_fin = ts_mat[ip, i_ini] + (
            40.0 * np.log10((ir_fin - ind_range0) / (i_ini_1b - ind_range0))
            + 2.0 * alpha * delta * (ir_fin - i_ini_1b)
        )
        tsu_fin = tsu_mat[ip, i_ini] + (
            40.0 * np.log10((ir_fin - ind_range0) / (i_ini_1b - ind_range0))
            + 2.0 * alpha * delta * (ir_fin - i_ini_1b)
        )

        # convert to metres once
        range_fin_m = (ir_fin - ind_range0) * delta

        # spacing in metres
        if sel.size > 1:
            inds = np.argsort(-ts_fin)
            keep = [inds[0]]
            for k in inds[1:]:
                if np.nanmin(np.abs(range_fin_m[k] - range_fin_m[np.array(keep)])) >= min_space_m:
                    keep.append(k)
            ind5 = np.array(keep, dtype=int)
        else:
            ind5 = np.array([0], dtype=int)

        targetRange = range_fin_m[ind5]  # metres (one-way range from transducer)
        D = targetRange + TD + CH_vec[ip]  # metres (depth from surface)

        ok = (D >= float(Param["MinEchoDepthM"])) & (D <= float(Param["MaxEchoDepthM"]))
        if not np.any(ok):
            continue

        targetRange = targetRange[ok]
        D = D[ok]
        ind5_ok = ind5[ok]

        compensatedTS = ts_fin[ind5_ok]
        unCompensatedTS = tsu_fin[ind5_ok]

        samp_idx = i_ini[ind5_ok]
        AlongShipAngleRad = al_mat[ip, samp_idx]
        AthwartShipAngleRad = ath_mat[ip, samp_idx]

        for pidx in range(targetRange.size):
            positionWorldCoord = [np.nan, np.nan, np.nan]  # to add
            positionBeamCoord = [
                targetRange[pidx] * np.tan(AlongShipAngleRad[pidx]),
                targetRange[pidx] * np.tan(AthwartShipAngleRad[pidx]),
                targetRange[pidx],
            ]

            DepthIndex = int(np.argmin(np.abs(depth - D[pidx])) + 1)
            IndexPingClean = int(Ping_number_full[ip])
            DepthIndexFromTransducer = int(np.round(targetRange[pidx] / delta))

            kf = int(params.get("Frequency_index", 1))

            TSfreq_rows.append(
                [
                    float(Ping_number_full[ip]),
                    float(D[pidx]),
                    float(DepthIndex),
                    float(compensatedTS[pidx]),
                    float(unCompensatedTS[pidx]),
                    float(AlongShipAngleRad[pidx]),
                    float(AthwartShipAngleRad[pidx]),
                    0.0,
                    float(positionWorldCoord[0]),
                    float(positionWorldCoord[1]),
                    float(positionWorldCoord[2]),
                    float(positionBeamCoord[0]),
                    float(positionBeamCoord[1]),
                    float(positionBeamCoord[2]),
                    float(Param["MinEchoDepthM"]),
                    float(Param["MaxEchoDepthM"]),
                    float(bot[ip]) if np.isfinite(bot[ip]) else np.nan,
                    float(IndexPingClean),
                    float(DepthIndexFromTransducer),
                    float(targetRange.size),
                    1.0,
                    0.0,
                    float(kf),
                ]
            )

            out["TS_comp"].append(float(compensatedTS[pidx]))
            out["TS_uncomp"].append(float(unCompensatedTS[pidx]))

            out["Target_range"].append(float(D[pidx]))
            out["Target_range_disp"].append(float(D[pidx]))
            out["Target_range_min"].append(float(Param["MinEchoDepthM"]))
            out["Target_range_max"].append(float(Param["MaxEchoDepthM"]))

            out["idx_r"].append(int(samp_idx[pidx] + 1))
            out["idx_target_lin"].append(int(samp_idx[pidx] + 1))

            out["pulse_env_before_lin"].append(np.nan)
            out["pulse_env_after_lin"].append(np.nan)
            out["PulseLength_Normalized_PLDL"].append(np.nan)
            out["Transmitted_pulse_length"].append(float(meta["pd"]))

            out["Angle_minor_axis"].append(float(AlongShipAngleRad[pidx]))
            out["Angle_major_axis"].append(float(AthwartShipAngleRad[pidx]))

            out["StandDev_Angles_Minor_Axis"].append(float(np.nan))
            out["StandDev_Angles_Major_Axis"].append(float(np.nan))

            out["Heave"].append(float(CH_vec[ip]))
            out["Roll"].append(
                float(ds.get("roll", xr.DataArray(np.nan)).isel(ping_time=ip).values)
                if "roll" in ds
                else np.nan
            )
            out["Pitch"].append(
                float(ds.get("pitch", xr.DataArray(np.nan)).isel(ping_time=ip).values)
                if "pitch" in ds
                else np.nan
            )
            out["Heading"].append(
                float(ds.get("heading", xr.DataArray(np.nan)).isel(ping_time=ip).values)
                if "heading" in ds
                else np.nan
            )
            out["Dist"].append(np.nan)

            out["Ping_number"].append(int(Ping_number_full[ip]))
            out["Time"].append(ping_time[ip])

    return out, TSfreq_rows


# ---
# main
def detect_matecho(ds: xr.Dataset, params: dict) -> xr.Dataset:
    """
    Single-target detector (split-beam style) on CW Sv data.

    Overview
    --------
    Detect candidate single targets ping-by-ping using a peak-based workflow:

    1) Build core matrices
       - Convert Sv (dB) to TS-like quantities (uncompensated and compensated).
       - Compute one-way range (m) and apply bottom/near-surface constraints.

    2) Phase I: candidate peak selection (Conditions 1–5)
       - C1: compensated TS above `MinThreshold`
       - C2: beam compensation not exceeding `MaxAngleOneWayCompression`
       - C3: local maximum in power-like metric (Plike)
       - C4: sample index above bottom (optionally provided) with `FloorFilterTolerance`
       - C5: exclude early samples (near transducer) using a fixed guard index

       Note: edge samples are not evaluated (requires left/right neighbours).

    3) Condition 6: angle stability
       - Reject peaks whose along/athwart angle standard deviation within ±2 samples
         exceeds `MaxPhaseDeviation` (in phase-step units after sensitivity scaling).

    4) Condition 7: echo-length
       - Define pulse envelope at -6 dB relative to the peak in Plike.
       - Keep candidates whose envelope length is within
         [`MinEchoLength`, `MaxEchoLength`] expressed in pulse-length units.

    5) Phase II: refinement and deconfliction (Condition 8)
       - Refine target position using a weighted centroid within the -6 dB envelope.
       - Convert refined index to range (m) and enforce minimum spacing
         `MinEchoSpace` (pulse-length units).
       - Apply depth window [`MinEchoDepthM`, `MaxEchoDepthM`] using
         `transducer_depth` and `heave_compensation`. (to check further)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing Sv (dB) and optional angle/metadata variables.
    params : dict
        Detection parameters (see PARAM_DEFAULTS). Must include:
        - channel, pulse_length, timeSampleInterval_usec
        Optional: sound_speed, alpha, bottom_da, transducer_depth,
        heave_compensation, etc. (add all)

    Returns
    -------
    xr.Dataset
        Per-target outputs (TS_comp, TS_uncomp, angles, indices, time/ping number, etc.).
        Empty dataset is returned when no targets are detected.
    """

    Param = _merge_params(params)
    DBG = _make_dbg()

    channel, Sv = _validate_inputs(ds, Param)

    # extract variables
    ex = _extract_matrices(ds, Sv, channel)
    sv_mat = ex["sv_mat"]
    al_mat = ex["al_mat"]
    ath_mat = ex["ath_mat"]
    depth = ex["depth"]
    npings = ex["npings"]

    # extract Delta + metadata + angle normalisation
    delta = _compute_delta(depth)
    meta = _extract_metadata(ds, Param, params, channel, npings, DBG)
    al_mat, ath_mat, meta = _normalise_angles(al_mat, ath_mat, meta, DBG)

    # compute derived pulse quantities
    derived = _compute_pulse_derived(meta, Param, delta)

    # get core matrices (ts_range, tsu, Plike, ts) + constraints
    core = _compute_core_matrices(
        sv_mat=sv_mat,
        al_mat=al_mat,
        ath_mat=ath_mat,
        depth=depth,
        delta=delta,
        meta=meta,
        Param=Param,
        channel=channel,
    )

    # pass conditions
    i1, ind = _cond_1to5(core, Param, DBG)
    if ind.size == 0:
        return _empty_out(channel)

    ind2 = _cond_6_phase(i1, ind, al_mat, ath_mat, core, meta, Param, DBG)
    if ind2.size == 0:
        return _empty_out(channel)

    c7 = _cond_7_echo_length(i1, ind, ind2, core, derived, Param, DBG)
    if c7["ind4"].size == 0:
        return _empty_out(channel)

    out, TSfreq_rows = _cond_8_build_outputs(
        ds=ds,
        params=params,
        i1=i1,
        ind2=ind2,
        c7=c7,
        core=core,
        meta=meta,
        derived=derived,
        al_mat=al_mat,
        ath_mat=ath_mat,
        depth=depth,
        Param=Param,
        DBG=DBG,
        delta=delta,
    )

    out["nb_valid_targets"] = len(out["TS_comp"])
    out["TSfreq_matrix"] = (
        np.asarray(TSfreq_rows, dtype=float) if TSfreq_rows else np.empty((0, 23))
    )
    return _matecho_struct_to_dataset(out, channel=channel)
