import numpy as np
import xarray as xr


def align_to_ping_time(
    external_da: xr.DataArray,
    external_time_name: str,
    ping_time_da: xr.DataArray,
    method: str = "nearest",
) -> xr.DataArray:
    """
    Aligns an external DataArray to align time-wise with the echosounder ping time DataArray.

    Parameters
    ----------
    external_da : xr.DataArray
        External non-echosounder data.
    external_time_name : str
        Time variable name of the external non-echosounder data.
    ping_time_da : xr.DataArray
        Echosounder ping time.
    method : str, default 'nearest'
        Interpolation method. Not used if external time matches ping time or external
        DataArray is a single value.
        For more interpolation methods please visit: https://docs.xarray.dev/en/stable/generated/xarray.DataArray.interp.html # noqa

    Returns
    -------
    aligned_da : xr.DataArray
        External non-echosounder data that is now aligned with the echosounder ping time.
    """
    # Rename if the external time dimension is equal to ping time
    if ping_time_da.equals(
        external_da[external_time_name].rename({external_time_name: "ping_time"})
    ):
        return external_da.rename({external_time_name: "ping_time"})
    elif len(external_da[external_time_name]) == 1:
        # Extend single, fixed-location coordinate to match ping time length
        return xr.DataArray(
            data=external_da.values[0] * np.ones(len(ping_time_da), dtype=np.float64),
            dims=["ping_time"],
            coords={"ping_time": ping_time_da.values},
            attrs=external_da.attrs,
        )
    elif len(external_da[external_time_name]) == 0:
        # Create an all NaN array matching the length of the ping time array
        data = np.full(len(ping_time_da), np.nan, dtype=np.float64)
        return xr.DataArray(
            data=data,
            dims=["ping_time"],
            coords={"ping_time": ping_time_da.values},
            attrs=external_da.attrs,
        )
    else:
        return external_da.interp(
            {external_time_name: ping_time_da},
            method=method,
            # More details for `fill_value` and `extrapolate` can be found here:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html # noqa
            kwargs={"fill_value": "extrapolate"},
        ).drop_vars(external_time_name)
