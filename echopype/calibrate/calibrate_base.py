import abc

import xarray as xr
import numpy as np

from typing import Literal, Optional


ENV_PARAMS = ("temperature", "salinity", "pressure", "sound_speed", "sound_absorption")

CAL_PARAMS = {
    "EK": ("sa_correction", "gain_correction", "equivalent_beam_angle"),
    "AZFP": ("EL", "DS", "TVR", "VTX", "equivalent_beam_angle", "Sv_offset"),
}

class EnvParams:
    def __init__(self, env_params, data_kind: Literal["static", "mobile", "organized"], interp_method: Literal["linear", "nearest"] = "linear", extrap_method: Optional[Literal["interp_method", "nearest"]] = None):
        self.env_params = env_params
        self.data_kind = data_kind
        self.interp_method = interp_method
        self.extrap_method = extrap_method
    
    def _apply(self, echodata) -> xr.Dataset:
        if self.data_kind == "static":
            dims = ["ping_time"]
        elif self.data_kind == "mobile":
            dims = ["latitude", "longitude"]
        elif self.data_kind == "organized":
            dims = ["ping_time", "latitude", "longitude"]
        else:
            raise ValueError("invalid data_kind")

        env_params = self.env_params

        min_max = {dim : {"min": env_params[dim].min(), "max": env_params[dim].max()} for dim in dims}
        nearest = env_params.interp({dim : echodata.beam[dim] for dim in dims}, method="nearest", kwargs={"fill_value": "extrapolate"})

        if self.extrap_method == "interp_method":
            fill_value = "extrapolate"
        else:
            fill_value = np.nan
        env_params = env_params.interp({dim : echodata.beam[dim] for dim in dims}, method=self.interp_method, kwargs={"fill_value": fill_value})

        if self.extrap_method == "nearest":
            less = nearest.sel({dim : nearest[dim][nearest[dim] < min_max[dim]["min"]] for dim in dims})
            middle = env_params.sel({dim : env_params[dim][np.logical_and(env_params[dim] >= min_max[dim]["min"], env_params[dim] <= min_max[dim]["max"])] for dim in dims})
            greater = nearest.sel({dim : nearest[dim][nearest[dim] > min_max[dim]["max"]] for dim in dims})
            env_params = xr.concat([less, middle, greater], dim="ping_time")

        return env_params

class CalibrateBase(abc.ABC):
    """Class to handle calibration for all sonar models."""

    def __init__(self, echodata):
        self.echodata = echodata
        self.env_params = None  # env_params are set in child class
        self.cal_params = None  # cal_params are set in child class

        # range_meter is computed in compute_Sv/Sp in child class
        self.range_meter = None

    @abc.abstractmethod
    def get_env_params(self, **kwargs):
        pass

    @abc.abstractmethod
    def get_cal_params(self, **kwargs):
        pass

    @abc.abstractmethod
    def compute_range_meter(self, **kwargs):
        """Calculate range in units meter.

        Returns
        -------
        range_meter : xr.DataArray
            range in units meter
        """
        pass

    @abc.abstractmethod
    def _cal_power(self, cal_type, **kwargs):
        """Calibrate power data for EK60, EK80, and AZFP.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'Sp' for calculating point backscattering strength
        """
        pass

    @abc.abstractmethod
    def compute_Sv(self, **kwargs):
        pass

    @abc.abstractmethod
    def compute_Sp(self, **kwargs):
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
