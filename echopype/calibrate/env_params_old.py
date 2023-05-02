from typing import Dict, List

import numpy as np
import scipy.interpolate
import xarray as xr
from typing_extensions import Literal

DataKind = Literal["stationary", "mobile", "organized"]
InterpMethod = Literal["linear", "nearest", "zero", "slinear", "quadratic", "cubic"]
ExtrapMethod = Literal["linear", "nearest"]
VALID_INTERP_METHODS: Dict[DataKind, List[InterpMethod]] = {
    "stationary": ["linear", "nearest", "zero", "slinear", "quadratic", "cubic"],
    "mobile": ["linear", "nearest", "cubic"],
    "organized": ["linear", "nearest"],
}


class EnvParams:
    def __init__(
        self,
        env_params: xr.Dataset,
        data_kind: DataKind,
        interp_method: InterpMethod = "linear",
        extrap_method: ExtrapMethod = "linear",
    ):
        """
        Class to hold and interpolate external environmental data for calibration purposes.

        This class can be used as the `env_params` parameter in `echopype.calibrate.compute_Sv`
        or `echopype.calibrate.compute_TS`. It is intended to be used with environmental parameters
        indexed by time. Environmental parameters will be interpolated onto dimensions within
        the Platform group of the `EchoData` object being used for calibration.

        Parameters
        ----------
        env_params : xr.Dataset
            The environmental parameters to use for calibration. This data will be interpolated with
            a provided `EchoData` object.

            When `data_kind` is `"stationary"`, env_params must have a coordinate `"time3"`.
            When `data_kind` is `"mobile"`, env_params must have coordinates `"latitude"`
            and `"longitude"`.
            When `data_kind` is `"organized"`, env_params must have coordinates `"time"`,
            `"latitude"`, and `"longitude"`. This `data_kind` is not currently supported.
        data_kind : {"stationary", "mobile", "organized"}
            The type of the environmental parameters.

            `"stationary"`: environmental parameters from a fixed location
            (for example, a single CTD).
            `"mobile"` environmental parameters from a moving location (for example, a ship).
            `"organized"`: environmental parameters from many fixed locations
            (for example, multiple CTDs).
        interp_method: {"linear", "nearest", "zero", "slinear", "quadratic", "cubic"}
            Method for interpolation of environmental parameters with the data from the
            provided `EchoData` object.

            When `data_kind` is `"stationary"`, valid `interp_method`s are `"linear"`, `"nearest"`,
            `"zero"`, `"slinear"`, `"quadratic"`, and `"cubic"`
            (see <https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.interpolate.interp1d.html>).
            When `data_kind` is `"mobile"`, valid `interp_method`s are `"linear"`, `"nearest"`, and `"cubic"`
            (see <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>).
            When `data_kind` is `"organized"`, valid `interp_method`s are `"linear"` and `"nearest"`
            (see <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html>).
        extrap_method: {"linear", "nearest"}
            Method for extrapolation of environmental parameters with the data from the
            provided `EchoData` object. Currently only supported when `data_kind` is `"stationary"`.

        Notes
        -----
        Currently cases where `data_kind` is `"organized"` are not supported; support will be added
        in a future version.

        Examples
        --------
        >>> env_params = xr.open_dataset("env_params.nc")
        >>> EnvParams(env_params, data_kind="mobile", interp_method="linear")
        >>> echopype.calibrate.compute_Sv(echodata, env_params=env_params)
        """  # noqa
        if interp_method not in VALID_INTERP_METHODS[data_kind]:
            raise ValueError(f"invalid interp_method {interp_method} for data_kind {data_kind}")

        self.env_params = env_params
        self.data_kind = data_kind
        self.interp_method = interp_method
        self.extrap_method = extrap_method

    def _apply(self, echodata) -> Dict[str, xr.DataArray]:
        if self.data_kind == "stationary":
            dims = ["time3"]
        elif self.data_kind == "mobile":
            dims = ["latitude", "longitude"]
        elif self.data_kind == "organized":
            dims = ["time", "latitude", "longitude"]
        else:
            raise ValueError("invalid data_kind")

        for dim in dims:
            if dim not in echodata["Platform"]:
                raise ValueError(
                    f"could not interpolate env_params; EchoData is missing dimension {dim}"
                )

        env_params = self.env_params

        if self.data_kind == "mobile":
            if np.isnan(echodata["Platform"]["time1"]).all():
                raise ValueError("cannot perform mobile interpolation without time1")
            # only grab needed variables for the interpolation
            platform_data = echodata["Platform"][["latitude", "longitude"]]
            # compute_range needs indexing by ping_time
            interp_plat = platform_data.interp(
                {"time1": echodata["Sonar/Beam_group1"]["ping_time"]}
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
                interp = scipy.interpolate.griddata(points, values, xi, method=self.interp_method)
                result[var] = ("time1", interp)
            env_params = xr.Dataset(
                # we expect env_params to have coordinate time1
                data_vars=result,
                coords={"time1": interp_plat["ping_time"].data},
            )
        else:
            # TODO: organized case
            min_max = {
                dim: {"min": env_params[dim].min(), "max": env_params[dim].max()} for dim in dims
            }

            extrap = env_params.interp(
                {dim: echodata["Platform"][dim].data for dim in dims},
                method=self.extrap_method,
                # scipy interp uses "extrapolate" but scipy interpn uses None
                kwargs={"fill_value": "extrapolate" if len(dims) == 1 else None},
            )
            # only keep unique indexes; xarray requires that indexes be unique
            extrap_unique_idx = {dim: np.unique(extrap[dim], return_index=True)[1] for dim in dims}
            extrap = extrap.isel(**extrap_unique_idx)
            interp = env_params.interp(
                {dim: echodata["Platform"][dim].data for dim in dims},
                method=self.interp_method,
            )
            interp_unique_idx = {dim: np.unique(interp[dim], return_index=True)[1] for dim in dims}
            interp = interp.isel(**interp_unique_idx)

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

                # remove empty datasets (xarray does not allow any dims from any datasets
                # to be length 0 in combine_by_coords)
                non_zero_dims = [
                    ds
                    for ds in (less, middle, greater)
                    if all(dim_len > 0 for dim_len in ds.dims.values())
                ]
                env_params = xr.combine_by_coords(non_zero_dims)

        # if self.data_kind == "organized":
        #     # get platform latitude and longitude indexed by ping_time
        #     interp_plat = echodata["Platform"].interp(
        #         {"time": echodata["Platform"]["ping_time"]}
        #     )
        #     # get env_params latitude and longitude indexed by ping_time
        #     env_params = env_params.interp(
        #         {
        #             "latitude": interp_plat["latitude"],
        #             "longitude": interp_plat["longitude"],
        #         }
        #     )

        if self.data_kind == "stationary":
            # renaming time3 (from Platform group) to time1 because we expect
            # environmental parameters to have dimension time1
            return {
                var: env_params[var].rename({"time3": "time1"})
                for var in ("temperature", "salinity", "pressure")
            }
        else:
            return {var: env_params[var] for var in ("temperature", "salinity", "pressure")}
