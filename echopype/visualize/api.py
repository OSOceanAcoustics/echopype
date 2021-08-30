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
    get_depth: bool = False,
    **kwargs,
) -> Union[FacetGrid, QuadMesh]:
    range_attrs = {
        'long_name': 'Depth',
        'units': 'm',
    }
    if isinstance(data, EchoData):
        yaxis = 'range_bin'
        variable = 'backscatter_r'
        ds = data.beam
        if get_depth:
            yaxis = 'depth'
            range_in_meter = _compute_range(data)
            range_in_meter.attrs = range_attrs
            ds = ds.assign_coords({'depth': range_in_meter})
    elif isinstance(data, xr.Dataset):
        variable = 'Sv'
        ds = data
        if get_depth:
            yaxis = 'depth'
        else:
            if 'range' in data.dims:
                # This indicates that data is MVBS.
                yaxis = 'range'
            else:
                yaxis = 'range_bin'

        ds = ds.assign_coords({'depth': ds.range})
        ds.depth.attrs = range_attrs
    else:
        ValueError(f"Unsupported data type: {type(data)}")

    return _plot_echogram(
        ds, yaxis=yaxis, variable=variable, frequency=frequency, **kwargs
    )
