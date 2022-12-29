import abc

from ..echodata import EchoData
from .env_params_old import EnvParams

CAL_PARAMS = {
    "EK": ("sa_correction", "gain_correction", "equivalent_beam_angle"),
    "AZFP": ("EL", "DS", "TVR", "VTX", "equivalent_beam_angle", "Sv_offset"),
}


class CalibrateBase(abc.ABC):
    """Class to handle calibration for all sonar models."""

    def __init__(self, echodata: EchoData, env_params=None, cal_params=None):
        self.echodata = echodata

        # TODO: clean up init of env_params here: can just leave it as None
        if isinstance(env_params, EnvParams):
            env_params = env_params._apply(echodata)
        elif env_params is None:
            env_params = {}
        elif not isinstance(env_params, dict):
            raise ValueError(
                "invalid env_params type; provide an EnvParams instance, a dict, or None"
            )
        self.env_params = env_params
        self.cal_params = cal_params

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
