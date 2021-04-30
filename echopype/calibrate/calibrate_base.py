import xarray as xr

ENV_PARAMS = (
    'temperature',
    'salinity',
    'pressure',
    'sound_speed',
    'sound_absorption'
)

CAL_PARAMS = {
    'EK': (
        'sa_correction',
        'gain_correction',
        'equivalent_beam_angle'
    ),
    'AZFP': (
        'EL',
        'DS',
        'TVR',
        'VTX',
        'equivalent_beam_angle',
        'Sv_offset'
    )
}


class CalibrateBase:
    """Class to handle calibration for all sonar models.
    """

    def __init__(self, echodata):
        self.echodata = echodata
        self.env_params = None  # env_params are set in child class
        self.cal_params = None  # cal_params are set in child class
        self.range_meter = None  # range_meter is computed in compute_Sv/Sp in child class

    def get_env_params(self, **kwargs):
        pass

    def get_cal_params(self, **kwargs):
        pass

    def compute_range_meter(self, **kwargs):
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

    def _add_params_to_output(self, ds_out):
        """Add all cal and env parameters to output Sv dataset.
        """
        # Add env_params
        for key, val in self.env_params.items():
            ds_out[key] = val

        # Add cal_params
        for key, val in self.cal_params.items():
            ds_out[key] = val

        return ds_out
