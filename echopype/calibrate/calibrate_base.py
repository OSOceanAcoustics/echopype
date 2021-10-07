import abc
from typing import Dict, List, Literal, Optional

import numpy as np
import scipy.interpolate
import xarray as xr

ENV_PARAMS = ("temperature", "salinity", "pressure", "sound_speed", "sound_absorption")

CAL_PARAMS = {
    "EK": ("sa_correction", "gain_correction", "equivalent_beam_angle"),
    "AZFP": ("EL", "DS", "TVR", "VTX", "equivalent_beam_angle", "Sv_offset"),
}

DataKind = Literal["stationary", "mobile", "organized"]
InterpMethod = Literal["linear", "nearest", "zero", "slinear", "quadratic", "cubic"]
VALID_INTERP_METHODS: Dict[DataKind, List[InterpMethod]] = {
    "stationary": ["linear", "nearest", "zero", "slinear", "quadratic", "cubic"],
    "mobile": ["linear", "nearest", "cubic"],
    "organized": ["linear", "nearest"],
}


class EnvParams:
    def __init__(
        self,
        env_params,
        data_kind: DataKind,  # organized name?
        interp_method: InterpMethod = "linear",
        extrap_method: Optional[Literal["linear", "nearest"]] = None,
    ):
        """
        Interp methods limited depending on data_kind
        Extrap only available for stationary and organized
        """
        if interp_method not in VALID_INTERP_METHODS[data_kind]:
            raise ValueError(
                f"invalid interp_method {interp_method} for data_kind {data_kind}"
            )

        self.env_params = env_params
        self.data_kind = data_kind
        self.interp_method = interp_method
        self.extrap_method = extrap_method

    def _apply(self, echodata) -> Dict[str, xr.DataArray]:
        if self.data_kind == "stationary":
            dims = ["ping_time"]
        elif self.data_kind == "mobile":
            dims = ["latitude", "longitude"]
            # dims = ["time"] # lat/lon are coords
        elif self.data_kind == "organized":
            dims = ["time", "latitude", "longitude"]
        else:
            raise ValueError("invalid data_kind")

        env_params = self.env_params

        if self.data_kind == "mobile":
            # compute_range needs indexing by ping_time
            interp_plat = echodata.platform.interp(
                {"location_time": echodata.platform["ping_time"]}
            )

            result = {}
            for var, values in env_params.data_vars.items():
                points = np.column_stack(
                    (env_params["latitude"].data, env_params["longitude"].data)
                )
                values = values.data
                xi = np.column_stack(
                    (
                        interp_plat["latitude"].data,
                        interp_plat["longitude"].data,
                    )
                )
                interp = scipy.interpolate.griddata(
                    points, values, xi, method=self.interp_method
                )
                result[var] = ("ping_time", interp)
            env_params = xr.Dataset(
                data_vars=result, coords={"ping_time": interp_plat["ping_time"]}
            )
        else:
            min_max = {
                dim: {"min": env_params[dim].min(), "max": env_params[dim].max()}
                for dim in dims
            }

            extrap = env_params.interp(
                {dim: echodata.platform[dim].data for dim in dims},
                method=self.extrap_method,
                # scipy interp uses "extrapolate" but scipy interpn uses None
                kwargs={"fill_value": "extrapolate" if len(dims) == 1 else None},
            )
            extrap_unique_idx = {
                dim: np.unique(extrap[dim], return_index=True)[1] for dim in dims
            }
            extrap = extrap.isel(**extrap_unique_idx)
            interp = env_params.interp(
                {dim: echodata.platform[dim].data for dim in dims},
                method=self.interp_method,
            )
            interp_unique_idx = {
                dim: np.unique(interp[dim], return_index=True)[1] for dim in dims
            }
            interp = interp.isel(**interp_unique_idx)

            if self.extrap_method is not None:
                less = extrap.sel(
                    {
                        dim: extrap[dim][extrap[dim] < min_max[dim]["min"]]
                        for dim in dims
                    }
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
                    {
                        dim: extrap[dim][extrap[dim] > min_max[dim]["max"]]
                        for dim in dims
                    }
                )

                # remove zero length dims (xarray does not support in combine_by_coords)
                non_zero_dims = [
                    ds
                    for ds in (less, middle, greater)
                    if all(dim_len > 0 for dim_len in ds.dims.values())
                ]
                env_params = xr.combine_by_coords(non_zero_dims)

        # if self.data_kind == "organized":
        #     # get platform latitude and longitude indexed by ping_time
        #     interp_plat = echodata.platform.interp(
        #         {"time": echodata.platform["ping_time"]}
        #     )
        #     # get env_params latitude and longitude indexed by ping_time
        #     env_params = env_params.interp(
        #         {
        #             "latitude": interp_plat["latitude"],
        #             "longitude": interp_plat["longitude"],
        #         }
        #     )

        return {var: env_params[var] for var in ("temperature", "salinity", "pressure")}
        # return {var: env_params[var] for var in env_params.data_vars}


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
