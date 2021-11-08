import xarray as xr

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


def compute_Sv(echodata: EchoData, **kwargs) -> xr.Dataset:
    """
    Compute volume backscattering strength (Sv) from raw data.

    The calibration routine varies depending on the sonar type.
    Currently this operation is supported for the following ``sonar_model``:
    EK60, AZFP, EK80 (see Notes below for detail).

    Parameters
    ----------
    waveform_mode : {"CW", "BB"}, optional
        Type of transmit waveform.
        Required only for data from the EK80 echosounder.

        - `"CW"` for narrowband transmission,
          returned echoes recorded either as complex or power/angle samples
        - `"BB"` for broadband transmission samples,
          returned echoes recorded as complex samples

    encode_mode : {"complex", "power"}, optional
        Type of encoded return echo data. 
        Required only for data from the EK80 echosounder.

        - `"complex"` for complex samples
        - `"power"` for power/angle samples

    Returns
    -------
    xr.Dataset
        The calibrated Sv dataset, including calibration and environmental variables
        used in the calibration operations.

    Notes
    -----
    The EK80 echosounder can be configured to transmit
    either broadband (``waveform_mode="BB"``)
    or narrowband (``waveform_mode="CW"``) signals.
    When transmitting in broadband mode, the returned echoes are
    encoded as complex samples (``encode_mode="complex"``).
    When transmitting in narrowband mode, the returned echoes can be encoded
    either as complex samples or as power/angle combinations in a format similar to
    those recorded by EK60 echosounders (``encode_mode="power"``).

    The current calibration implemented for EK80 broadband complex data
    uses band-integrated Sv with the gain computed at the center frequency
    of the transmit signal.

    """
    return _compute_cal(cal_type="Sv", echodata=echodata, **kwargs)


def compute_Sp(echodata: EchoData, **kwargs):
    """
    Compute point backscattering strength (Sp) from raw data.
    """
    return _compute_cal(cal_type="Sp", echodata=echodata, **kwargs)
