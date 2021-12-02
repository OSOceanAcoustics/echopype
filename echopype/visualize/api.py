from typing import Union, List

import xarray as xr

from echopype.visualize.plot import _plot_echogram, FacetGrid, QuadMesh, T
from echopype.echodata import EchoData


def create_echogram(
    data: Union[EchoData, xr.Dataset],
    frequency: Union[int, float, List[T], None] = None,
    get_range: bool = False,
    range_kwargs: dict = {},
    water_level: Union[int, float, xr.DataArray, None] = None,
    **kwargs,
) -> Union[FacetGrid, QuadMesh]:
    """Create an Echogram from an EchoData object or Sv and MVBS Dataset.

    Parameters
    ----------
    data : EchoData or xr.Dataset
        Echodata or Xarray Dataset to be plotted
    frequency : int, float, or list of float or ints, optional
        The frequency to be plotted.
        Otherwise all frequency will be plotted.
    get_range : bool
        Flag as to whether range should be computed or not,
        by default it will just plot range_bin as the yaxis.
    range_kwargs : dict
        Keyword arguments dictionary for computing range.
        Keys are `env_params`, `waveform_mode`, and `encode_mode`.
    water_level : xr.DataArray, optional
        Water level data array for platform water level correction.
    **kwargs: optional
        Additional keyword arguments for xarray plot pcolormesh.

    """
    range_attrs = {
        'long_name': 'Range',
        'units': 'm',
    }

    if isinstance(frequency, list) and len(frequency) == 1:
        frequency = frequency[0]

    if isinstance(data, EchoData):
        if data.sonar_model.lower() == 'ad2cp':
            raise ValueError(
                "Visualization for AD2CP sonar model is currently unsupported."
            )
        yaxis = 'range_bin'
        variable = 'backscatter_r'
        ds = data.beam
        if get_range:
            yaxis = 'range'

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

                if 'encode_mode' not in range_kwargs:
                    range_kwargs['encode_mode'] = 'power'
            elif data.sonar_model.lower() == 'ek80':
                if 'waveform_mode' not in range_kwargs:
                    raise ValueError(
                        "Please provide waveform_mode in range_kwargs for EK80 sonar model."
                    )

            range_in_meter = data.compute_range(
                env_params=range_kwargs.get('env_params', {}),
                azfp_cal_type=range_kwargs.get('azfp_cal_type', None),
                ek_waveform_mode=range_kwargs.get('waveform_mode', None),
                ek_encode_mode=range_kwargs.get('encode_mode', 'complex'),
            )
            range_in_meter.attrs = range_attrs
            if 'water_level' in data.platform:
                # Adds water level to range if it exists
                range_in_meter = range_in_meter + data.platform.water_level
            ds = ds.assign_coords({'range': range_in_meter})
            ds.range.attrs = range_attrs

    elif isinstance(data, xr.Dataset):
        variable = 'Sv'
        ds = data
        if get_range:
            yaxis = 'range'
        else:
            if 'range' in data.dims:
                # This indicates that data is MVBS.
                yaxis = 'range'
            else:
                yaxis = 'range_bin'
        # If depth is available in ds, use it.
        ds = ds.set_coords('range')
        if water_level is not None:
            if isinstance(water_level, xr.DataArray):
                check_dims = ds['range'].dims
                if not any(
                    True if d in water_level.dims else False
                    for d in check_dims
                ):
                    raise ValueError(
                        f"Water level must have any of these dimensions: {', '.join(check_dims)}"
                    )
                # Adds water level to range if it exists
                ds['range'] = ds.range + water_level
            elif isinstance(water_level, (int, float)):
                ds['range'] = ds.range + water_level
        ds.range.attrs = range_attrs
    else:
        ValueError(f"Unsupported data type: {type(data)}")

    return _plot_echogram(
        ds,
        xaxis='ping_time',
        yaxis=yaxis,
        variable=variable,
        frequency=frequency,
        **kwargs,
    )
