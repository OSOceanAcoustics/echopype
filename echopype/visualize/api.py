import warnings
from typing import Optional, Union, List

import xarray as xr

from .plot import _plot_echogram, FacetGrid, QuadMesh
from ..echodata import EchoData


def create_echogram(
    data: Union[EchoData, xr.Dataset],
    channel: Union[str, List[str], None] = None,
    frequency: Union[str, List[str], None] = None,
    get_range: Optional[bool] = None,
    range_kwargs: dict = {},
    water_level: Union[int, float, xr.DataArray, bool, None] = None,
    **kwargs,
) -> List[Union[FacetGrid, QuadMesh]]:
    """Create an Echogram from an EchoData object or Sv and MVBS Dataset.

    Parameters
    ----------
    data : EchoData or xr.Dataset
        Echodata or Xarray Dataset to be plotted
    channel : str or list of str, optional
        The channel to be plotted.
        Otherwise all channels will be plotted.
    frequency : int, float, or list of float or ints, optional
        The frequency to be plotted.
        If not specified, all frequency will be plotted.
    get_range : bool, optional
        Flag as to whether range (``echo_range``) should be computed or not,
        by default it will just plot `range_sample`` as the yaxis.

        Note that for data that is "Sv" xarray dataset, `get_range` defaults
        to `True`.
    range_kwargs : dict
        Keyword arguments dictionary for computing range (``echo_range``).
        Keys are `env_params`, `waveform_mode`, and `encode_mode`.
    water_level : int, float, xr.DataArray, or bool, optional
        Water level data array for platform water level correction.
        Note that auto addition of water level can be performed
        when data is an EchoData object by setting this argument
        to `True`. Currently because the water level information
        is not available as part of the Sv dataset, a warning is issued
        when `water_level=True` in this case and no correction is
        performed. This behavior will change in the future when the
        default content of Sv dataset is updated to include this information.
    **kwargs: optional
        Additional keyword arguments for xarray plot pcolormesh.

    Notes
    -----
    The EK80 echosounder can be configured to transmit
    either broadband (``waveform_mode="BB"``)
    or narrowband (``waveform_mode="CW"``) signals.
    When transmitting in broadband mode, the returned echoes are
    encoded as complex samples (``encode_mode="complex"``).
    When transmitting in narrowband mode, the returned echoes can be encoded
    either as complex samples (``encode_mode="complex"``)
    or as power/angle combinations (``encode_mode="power"``) in a format
    similar to those recorded by EK60 echosounders.

    """
    range_attrs = {
        'long_name': 'Range',
        'units': 'm',
    }

    if channel and frequency:
        warnings.warn(
            "Both channel and frequency are specified. Channel filtering will be used."
        )

    if isinstance(channel, list) and len(channel) == 1:
        channel = channel[0]
    elif isinstance(frequency, list) and len(frequency) == 1:
        frequency = frequency[0]

    if isinstance(data, EchoData):
        if data.sonar_model.lower() == 'ad2cp':
            raise ValueError(
                "Visualization for AD2CP sonar model is currently unsupported."
            )
        yaxis = 'range_sample'
        variable = 'backscatter_r'
        ds = data.beam
        if 'ping_time' in ds:
            _check_ping_time(ds.ping_time)
        if get_range is True:
            yaxis = 'echo_range'

            if data.sonar_model.lower() == 'azfp':
                if 'azfp_cal_type' not in range_kwargs:
                    range_kwargs['azfp_cal_type'] = 'Sv'
                if 'env_params' not in range_kwargs:
                    raise ValueError(
                        "Please provide env_params in range_kwargs!"
                    )
            elif data.sonar_model.lower() == 'ek60':
                if 'waveform_mode' not in range_kwargs:
                    range_kwargs['waveform_mode'] = 'CW'
                elif range_kwargs['waveform_mode'] != 'CW':
                    raise ValueError(
                        f"waveform_mode {range_kwargs['waveform_mode']} is invalid. EK60 waveform_mode must be 'CW'."  # noqa
                    )

                if 'encode_mode' not in range_kwargs:
                    range_kwargs['encode_mode'] = 'power'
                elif range_kwargs['encode_mode'] != 'power':
                    raise ValueError(
                        f"encode_mode {range_kwargs['encode_mode']} is invalid. EK60 encode_mode must be 'power'."  # noqa
                    )
            elif data.sonar_model.lower() == 'ek80':
                if not all(
                    True if mode in range_kwargs else False
                    for mode in ['waveform_mode', 'encode_mode']
                ):
                    raise ValueError(
                        "Please provide waveform_mode and encode_mode in range_kwargs for EK80 sonar model."  # noqa
                    )
                waveform_mode = range_kwargs['waveform_mode']
                encode_mode = range_kwargs['encode_mode']

                if waveform_mode not in ("BB", "CW"):
                    raise ValueError(
                        f"waveform_mode {waveform_mode} is invalid. EK80 waveform_mode must be 'BB' or 'CW'."  # noqa
                    )
                elif encode_mode not in ("complex", "power"):
                    raise ValueError(
                        f"encode_mode {encode_mode} is invalid. EK80 waveform_mode must be 'complex' or 'power'."  # noqa
                    )
                elif waveform_mode == "BB" and encode_mode == "power":
                    raise ValueError(
                        "Data from broadband ('BB') transmission must be recorded as complex samples"  # noqa
                    )

            range_in_meter = data.compute_range(
                env_params=range_kwargs.get('env_params', {}),
                azfp_cal_type=range_kwargs.get('azfp_cal_type', None),
                ek_waveform_mode=range_kwargs.get('waveform_mode', 'CW'),
                ek_encode_mode=range_kwargs.get('encode_mode', 'power'),
            )
            range_in_meter.attrs = range_attrs
            if water_level is not None:
                range_in_meter = _add_water_level(
                    range_in_meter=range_in_meter,
                    water_level=water_level,
                    data_type=EchoData,
                    platform_data=data.platform,
                )
            ds = ds.assign_coords({'echo_range': range_in_meter})
            ds.echo_range.attrs = range_attrs

    elif isinstance(data, xr.Dataset):
        if 'ping_time' in data:
            _check_ping_time(data.ping_time)
        variable = 'Sv'
        ds = data
        yaxis = 'echo_range'
        if 'echo_range' not in data.dims and get_range is False:
            # Range in dims indicates that data is MVBS.
            yaxis = 'range_sample'

        # If depth is available in ds, use it.
        ds = ds.set_coords('echo_range')
        if water_level is not None:
            ds['echo_range'] = _add_water_level(
                range_in_meter=ds.echo_range,
                water_level=water_level,
                data_type=xr.Dataset,
            )
        ds.echo_range.attrs = range_attrs
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    return _plot_echogram(
        ds,
        xaxis='ping_time',
        yaxis=yaxis,
        variable=variable,
        channel=channel,
        frequency=frequency,
        **kwargs,
    )


def _check_ping_time(ping_time):
    if ping_time.shape[0] < 2:
        raise ValueError("Ping time must have a length that is greater or equal to 2")


def _add_water_level(
    range_in_meter: xr.DataArray,
    water_level: Union[int, float, xr.DataArray, bool],
    data_type: str,
    platform_data: Optional[xr.Dataset] = None,
) -> xr.DataArray:
    # Below, we rename time3 to ping_time because range_in_meter is in ping_time
    if isinstance(water_level, bool):
        if water_level is True:
            if data_type == xr.Dataset:
                warnings.warn(
                    "Boolean type found for water level. Ignored since data is an xarray dataset."
                )
                return range_in_meter
            elif data_type == EchoData:
                if (
                    isinstance(platform_data, xr.Dataset)
                    and 'water_level' in platform_data
                ):
                    return range_in_meter + platform_data.water_level.rename({'time3': 'ping_time'})
                else:
                    warnings.warn(
                        "Boolean type found for water level. Please provide platform data with water level in it or provide a separate water level data."  # noqa
                    )
                    return range_in_meter
        warnings.warn(f"Water level value of {water_level} is ignored.")
        return range_in_meter
    if isinstance(water_level, xr.DataArray):
        check_dims = range_in_meter.dims
        if 'time3' in water_level:
            water_level = water_level.rename({'time3': 'ping_time'})

        if not any(
            True if d in water_level.dims else False for d in check_dims
        ):
            raise ValueError(
                f"Water level must have any of these dimensions: {', '.join(check_dims)}"
            )
        # Adds water level to range if it exists
        return range_in_meter + water_level
    elif isinstance(water_level, (int, float)):
        return range_in_meter + water_level
