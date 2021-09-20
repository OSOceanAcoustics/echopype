from typing import Union, List

import xarray as xr

from echopype.visualize.plot import _plot_echogram, FacetGrid, QuadMesh, T
from echopype.echodata import EchoData
from echopype.calibrate.api import CALIBRATOR


def _compute_range(
    echodata: EchoData,
    env_params=None,
    cal_params=None,
    waveform_mode=None,
    encode_mode=None,
):
    cal_obj = CALIBRATOR[echodata.sonar_model](
        echodata,
        env_params=env_params,
        cal_params=cal_params,
        waveform_mode=waveform_mode,
        encode_mode=encode_mode,
    )
    return cal_obj.range_meter


def create_echogram(
    data: Union[EchoData, xr.Dataset],
    frequency: Union[int, float, List[T], None] = None,
    get_range: bool = False,
    water_level: Union[xr.DataArray, None] = None,
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
    water_level : xr.DataArray, optional
        Water level data array for platform water level correction.
    **kwargs: optional
        Additional keyword arguments for xarray plot pcolormesh.

    """
    range_attrs = {
        'long_name': 'Range',
        'units': 'm',
    }
    if isinstance(data, EchoData):
        yaxis = 'range_bin'
        variable = 'backscatter_r'
        ds = data.beam
        if get_range:
            yaxis = 'range'
            range_in_meter = _compute_range(data)
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
        if isinstance(water_level, xr.DataArray):
            required_dims = ['frequency', 'ping_time']
            if not all(
                True if d in water_level.dims else False for d in required_dims
            ):
                raise ValueError(
                    f"Water level must have dimensions: {', '.join(required_dims)}"
                )
            # Adds water level to range if it exists
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
