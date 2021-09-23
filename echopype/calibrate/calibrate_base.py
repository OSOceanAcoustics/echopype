import abc
from typing import Literal, Optional, Dict

import numpy as np
import xarray as xr

ENV_PARAMS = ("temperature", "salinity", "pressure", "sound_speed", "sound_absorption")

CAL_PARAMS = {
    "EK": ("sa_correction", "gain_correction", "equivalent_beam_angle"),
    "AZFP": ("EL", "DS", "TVR", "VTX", "equivalent_beam_angle", "Sv_offset"),
}


class EnvParams:
    def __init__(
        self,
        env_params,
        data_kind: Literal["stationary", "mobile", "organized"], # organized name?
        interp_method: Literal["linear", "nearest"] = "linear",
        extrap_method: Optional[Literal["linear", "nearest"]] = None,
    ):
        self.env_params = env_params
        self.data_kind = data_kind
        self.interp_method = interp_method
        self.extrap_method = extrap_method

    def _apply(self, echodata) -> Dict[str, xr.DataArray]:
        if self.data_kind == "stationary":
            dims = ["ping_time"]
        elif self.data_kind == "mobile":
            dims = ["latitude", "longitude"]
        elif self.data_kind == "organized":
            dims = ["time", "latitude", "longitude"]
        else:
            raise ValueError("invalid data_kind")

        env_params = self.env_params

        min_max = {
            dim: {"min": env_params[dim].min(), "max": env_params[dim].max()}
            for dim in dims
        }

        extrap = env_params.interp(
            {dim: echodata.platform[dim].data for dim in dims},
            method=self.extrap_method,
            kwargs={"fill_value": "extrapolate" if len(dims) == 1 else None},
        )
        interp = env_params.interp(
            {dim: echodata.platform[dim].data for dim in dims}, method=self.interp_method
        )

        if self.extrap_method is not None:
            less = extrap.sel(
                {dim: extrap[dim][extrap[dim] < min_max[dim]["min"]] for dim in dims}
            )
            middle = interp.sel(
                {
                    dim: interp[dim][
                        np.logical_and(
                            interp[dim] >= min_max[dim]["min"],
                            interp[dim] <= min_max[dim]["max"],
                        )
                    ]
                    for dim in dims
                }
            )
            greater = extrap.sel(
                {dim: extrap[dim][extrap[dim] > min_max[dim]["max"]] for dim in dims}
            )
            env_params = xr.concat([less, middle, greater], dim="ping_time")

        return {var : env_params[var] for var in env_params.data_vars}


class CalibrateBase(abc.ABC):
    """Class to handle calibration for all sonar models."""

    def __init__(self, echodata, env_params=None):
        self.echodata = echodata
        if isinstance(env_params, EnvParams):
            env_params = env_params._apply(echodata)
        elif env_params is None:
            env_params = {}
        self.env_params = env_params  # env_params are set in child class
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
