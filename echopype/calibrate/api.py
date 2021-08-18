from ..echodata import EchoData
from .calibrate_azfp import CalibrateAZFP
from .calibrate_ek import CalibrateEK60, CalibrateEK80

CALIBRATOR = {"EK60": CalibrateEK60, "EK80": CalibrateEK80, "AZFP": CalibrateAZFP}


def _compute_cal(
    cal_type,
    echodata: EchoData,
    env_params=None,
    cal_params=None,
    waveform_mode=None,
    encode_mode=None,
):
    # Sanity check on inputs
    if (waveform_mode is not None) and (waveform_mode not in ("BB", "CW")):
        raise ValueError("Input waveform_mode not recognized!")
    if (encode_mode is not None) and (encode_mode not in ("complex", "power")):
        raise ValueError("Input encode_mode not recognized!")

    # Set up calibration object
    cal_obj = CALIBRATOR[echodata.sonar_model](
        echodata,
        env_params=env_params,
        cal_params=cal_params,
        waveform_mode=waveform_mode,
        encode_mode=encode_mode,
    )
    # Perform calibration
    if cal_type == "Sv":
        return cal_obj.compute_Sv(waveform_mode=waveform_mode, encode_mode=encode_mode)
    else:
        return cal_obj.compute_Sp(waveform_mode=waveform_mode, encode_mode=encode_mode)


def compute_Sv(echodata: EchoData, **kwargs):
    """Compute volume backscattering strength (Sv) from raw data."""
    return _compute_cal(cal_type="Sv", echodata=echodata, **kwargs)


def compute_Sp(echodata: EchoData, **kwargs):
    """Compute point backscattering strength (Sp) from raw data."""
    return _compute_cal(cal_type="Sp", echodata=echodata, **kwargs)
