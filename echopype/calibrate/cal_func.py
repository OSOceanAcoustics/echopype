from .calibrate_ek import CalibrateEK60, CalibrateEK80
from .calibrate_azfp import CalibrateAZFP

CALIBRATOR = {
    'EK60': CalibrateEK60,
    'EK80': CalibrateEK80,
    'AZFP': CalibrateAZFP
}


def compute_Sv(echodata, env_params=None, cal_params=None, waveform_mode=None, encode_mode=None):
    """Compute volume backscattering strength (Sv) from raw data.
    """
    # Sanity check on inputs
    if (waveform_mode is not None) and (waveform_mode not in ('BB', 'CW')):
        raise ValueError('Input waveform_mode not recognized!')
    if (encode_mode is not None) and (encode_mode not in ('complex', 'power')):
        raise ValueError('Input encode_mode not recognized!')

    # Set up calibration object
    cal_obj = CALIBRATOR[echodata.sonar_model](echodata, env_params=env_params, cal_params=cal_params)
    # if env_params is None:
    #     env_params = {}
    # cal_obj.get_env_params(env_params, waveform_mode=waveform_mode)
    # if cal_params is None:
    #     cal_params = {}
    # cal_obj.get_cal_params(cal_params)

    # Perform calibration
    return cal_obj.compute_Sv(waveform_mode=waveform_mode, encode_mode=encode_mode)
