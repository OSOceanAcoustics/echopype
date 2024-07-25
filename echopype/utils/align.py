import xarray as xr


def align_to_ping_time(
    external_da: xr.DataArray,
    external_time_name: str,
    ping_time_da: xr.DataArray,
    method: str = "nearest",
) -> xr.DataArray:
    """
    Aligns an external DataArray to align time-wise with the Echosounder ping time DataArray.

    A wrapper function for https://docs.xarray.dev/en/stable/generated/xarray.DataArray.interp.html.

    Parameters
    ----------
    external_da : xr.DataArray
        External Non-Echosounder data.
    external_time_name : str, default 'nearest'
        Time variable name of the External Non-Echosounder data.
    ping_time_da : xr.DataArray
        Echosounder ping time.

    Returns
    -------
    aligned_da : xr.DataArray
        External Non-Echosounder data that is now aligned with the Echosounder ping time.
    """
    # Interpolate only if ping time and external time are not equal
    if not ping_time_da.equals(
        external_da[external_time_name].rename({external_time_name: "ping_time"})
    ):
        aligned_da = external_da.interp(
            {external_time_name: ping_time_da},
            method=method,
            # More details for `fill_value` and `extrapolate` can be found here:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html # noqa
            kwargs={"fill_value": "extrapolate"},
        ).drop_vars(external_time_name)
    else:
        aligned_da = external_da.rename({external_time_name: "ping_time"})
    return aligned_da
