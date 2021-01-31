from .calibrate import CalibrateAZFP, CalibrateEK60, CalibrateEK80

CALIBRATOR = {
    'EK60': CalibrateEK60,
    'EK80': CalibrateEK80,
    'AZFP': CalibrateAZFP
}


def get_Sv(echodata, env_params=None, cal_params=None):
    """Compute volume backscattering strength (Sv) from raw data.
    """
    # Set up calibration object
    cal_obj = CALIBRATOR[echodata.sonar_model](echodata)
    if env_params is None:
        env_params = {}
    cal_obj.get_env_params(env_params)
    if cal_params is None:
        cal_params = {}
    cal_obj.get_cal_params(cal_params)

    # Perform calibration
    return cal_obj.get_Sv()
