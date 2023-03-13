import abc

from ..echodata import EchoData


class CalibrateBase(abc.ABC):
    """Class to handle calibration for all sonar models."""

    def __init__(self, echodata: EchoData, env_params=None, cal_params=None):
        self.echodata = echodata
        self.sonar_type = None

        if env_params is None:
            self.env_params = {}
        elif isinstance(env_params, dict):
            self.env_params = env_params
        else:
            raise ValueError("'env_params' has to be None or a dict")

        if cal_params is None:
            self.cal_params = {}
        elif isinstance(cal_params, dict):
            self.cal_params = cal_params
        else:
            raise ValueError("'cal_params' has to be None or a dict")

        # range_meter is computed in compute_Sv/TS in child class
        self.range_meter = None

    @abc.abstractmethod
    def compute_echo_range(self, **kwargs):
        """Calculate range (``echo_range``) in units meter.

        Returns
        -------
        range_meter : xr.DataArray
            range in units meter
        """
        pass

    @abc.abstractmethod
    def _cal_power_samples(self, cal_type, **kwargs):
        """Calibrate power data for EK60, EK80, and AZFP.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'TS' for calculating target strength
        """
        pass

    @abc.abstractmethod
    def compute_Sv(self, **kwargs):
        pass

    @abc.abstractmethod
    def compute_TS(self, **kwargs):
        pass

    def _add_params_to_output(self, ds_out):
        """Add all cal and env parameters to output Sv dataset."""
        # Add env_params
        for key, val in self.env_params.items():
            ds_out[key] = val

        # Add cal_params
        for key, val in self.cal_params.items():
            ds_out[key] = val

        return ds_out
