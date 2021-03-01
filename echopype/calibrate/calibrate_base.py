import xarray as xr

ENV_PARAMS = (
    'temperature', 'salinity', 'pressure',
    'sound_speed', 'sound_absorption'
)

CAL_PARAMS = {
    'EK': ('sa_correction', 'gain_correction', 'equivalent_beam_angle'),
    'AZFP': ('EL', 'DS', 'TVR', 'VTX', 'equivalent_beam_angle', 'Sv_offset')
}


class CalibrateBase:
    """Class to handle calibration for all sonar models.
    """

    def __init__(self, echodata):
        self.sonar_model = None
        self.range_meter = None
        self.echodata = echodata

        # initialize all env params to None
        self.env_params = dict.fromkeys(ENV_PARAMS)

    def get_env_params(self, **kwargs):
        pass

    def get_cal_params(self, **kwargs):
        pass

    def calc_range_meter(self, **kwargs):
        """Calculate range in units meter.

        Returns
        -------
        range_meter : xr.DataArray
            range in units meter
        """
        pass

    def _cal_power(self, cal_type, **kwargs):
        """Calibrate power data for EK60, EK80, and AZFP.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'Sp' for calculating point backscattering strength
        """
        pass

    def compute_Sv(self, **kwargs):
        pass

    def compute_Sp(self, **kwargs):
        pass
