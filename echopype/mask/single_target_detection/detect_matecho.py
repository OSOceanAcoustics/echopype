import warnings

import numpy as np
import xarray as xr

PARAM_DEFAULTS = {
    "SoundSpeed": 1500.0,
    "TS_threshold": -50.0,  # dB
    "MaxAngleOneWayCompression": 6.0,  # dB (one-way; test uses 2x)
    "MaxPhaseDeviation": 8.0,  # phase steps (128/pi units), Matecho GUI
    "MinEchoLength": 0.8,  # in pulse lengths
    "MaxEchoLength": 1.8,
    "MinEchoSpace": 1.0,  # in pulse lengths
    "MinEchoDepthM": 3.0,
    "MaxEchoDepthM": 38.0,
    "tvg_start_sample": 3,  # EK60=3, EK80=1
    "block_len": 1e7 / 3,
    # Beam pattern (fallbacks if no metadata):
    "beamwidth_along_3dB_rad": np.deg2rad(7.0),
    "beamwidth_athwart_3dB_rad": np.deg2rad(7.0),
    "steer_along_rad": 0.0,
    "steer_athwart_rad": 0.0,
    # Angle sensitivities (phase steps), overwritten from ds:
    "angle_sens_al": 1.0,
    "angle_sens_at": 1.0,
    # Sv->TS constant terms; keep 0 if unknown:
    "psi_two_way": 0.0,
    "Sa_correction": 0.0,
    "Sa_EK80_nominal": 0.0,
}


def _matecho_struct_to_dataset(out: dict, *, channel: str | None = None) -> xr.Dataset:
    """
    Convert struct-of-lists into an xr.Dataset with dim 'target'.
    """
    n = int(out.get("nb_valid_targets", len(out.get("TS_comp", []))))
    target = np.arange(n, dtype=int)

    def _arr(key, dtype=float):
        vals = out.get(key, [])
        if n == 0:
            return np.asarray([], dtype=dtype)
        a = np.asarray(vals, dtype=dtype)
        if a.size != n:
            raise ValueError(f"Output field '{key}' has length {a.size}, expected {n}.")
        return a

    ds_out = xr.Dataset(
        data_vars=dict(
            TS_comp=("target", _arr("TS_comp", float), {"units": "dB"}),
            TS_uncomp=("target", _arr("TS_uncomp", float), {"units": "dB"}),
            target_range=("target", _arr("Target_range", float), {"units": "m"}),
            target_range_disp=("target", _arr("Target_range_disp", float), {"units": "m"}),
            target_range_min=("target", _arr("Target_range_min", float), {"units": "m"}),
            target_range_max=("target", _arr("Target_range_max", float), {"units": "m"}),
            idx_r=("target", _arr("idx_r", int)),
            idx_target_lin=("target", _arr("idx_target_lin", int)),
            pulse_env_before_lin=("target", _arr("pulse_env_before_lin", int)),
            pulse_env_after_lin=("target", _arr("pulse_env_after_lin", int)),
            pulse_length_normalized_pldl=("target", _arr("PulseLength_Normalized_PLDL", float)),
            transmitted_pulse_length=("target", _arr("Transmitted_pulse_length", int)),
            angle_minor_axis=("target", _arr("Angle_minor_axis", float), {"units": "rad"}),
            angle_major_axis=("target", _arr("Angle_major_axis", float), {"units": "rad"}),
            std_angle_minor_axis=(
                "target",
                _arr("StandDev_Angles_Minor_Axis", float),
                {"units": "phase_steps"},
            ),
            std_angle_major_axis=(
                "target",
                _arr("StandDev_Angles_Major_Axis", float),
                {"units": "phase_steps"},
            ),
            heave=("target", _arr("Heave", float), {"units": "m"}),
            roll=("target", _arr("Roll", float), {"units": "rad"}),
            pitch=("target", _arr("Pitch", float), {"units": "rad"}),
            heading=("target", _arr("Heading", float), {"units": "rad"}),
            dist=("target", _arr("Dist", float), {"units": "m"}),
        ),
        coords=dict(
            target=target,
            ping_number=("target", _arr("Ping_number", int)),
            ping_time=("target", _arr("Time", "datetime64[ns]")),
        ),
        attrs=dict(
            method="matecho",
            channel=str(channel) if channel is not None else "",
            nb_valid_targets=n,
        ),
    )

    if channel is not None:
        # scalar coordinate
        ds_out = ds_out.assign_coords(channel=np.asarray(channel))

    return ds_out


def detect_matecho(ds: xr.Dataset, params: dict) -> xr.Dataset:
    """
    Matecho-inspired single-target detector (CW only).

    This is a placeholder implementation that demonstrates:
    - required signature
    - required parameter checks
    - required output structure (xr.Dataset with dim 'target')

    No detection is performed.
    """
    if params is None:
        raise ValueError("params is required.")

    channel = params.get("channel")
    if channel is None:
        raise ValueError("params['channel'] is required.")

    var_name = params.get("var_name", "Sv")

    if var_name not in ds:
        raise ValueError(f"var_name '{var_name}' not found in input dataset.")

    da = ds[var_name]

    required_dims = {"channel", "ping_time", "range_sample"}
    if not required_dims.issubset(set(da.dims)):
        raise ValueError(f"{var_name} must have dims {sorted(required_dims)}. Got: {da.dims}.")

    # channel must exist in Sv(channel, ping_time, range_sample)
    if "channel" not in da.coords:
        raise ValueError(f"{var_name} has no 'channel' coordinate.")
    if channel not in da["channel"].values:
        raise ValueError(f"Channel '{channel}' not found in {var_name}.")

    # --- optional metadata warnings (kept for API clarity)
    _missing = []
    for v in [
        "angle_alongship",
        "angle_athwartship",
        "beamwidth_alongship",
        "beamwidth_athwartship",
        "angle_offset_alongship",
        "angle_offset_athwartship",
        "angle_sensitivity_alongship",
        "angle_sensitivity_athwartship",
        "sound_speed",
    ]:
        if v not in ds:
            _missing.append(v)

    if _missing:
        warnings.warn(
            f"The following variables are missing for channel '{channel}': "
            + ", ".join(_missing)
            + ". Defaults will be used where applicable.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # ## algo ##
    # Intentionally empty: no detection performed
    # ------------------------------------------------------------------

    out = {
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
        "nb_valid_targets": 0,
    }

    return _matecho_struct_to_dataset(out, channel=channel)
