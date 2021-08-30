from typing import Union, List

import xarray as xr

from echopype.visualize.plot import _plot_echogram, FacetGrid, QuadMesh, T
from echopype.echodata import EchoData


def create_echogram(
    data: Union[EchoData, xr.Dataset],
    frequency: Union[int, float, List[T], None] = None,
    **kwargs,
) -> Union[FacetGrid, QuadMesh]:
    if isinstance(data, EchoData):
        yaxis = 'range_bin'
        variable = 'backscatter_r'
        ds = data.beam
    elif isinstance(data, xr.Dataset):
        variable = 'Sv'
        ds = data
        if 'range' in data.dims:
            # This indicates that data is MVBS.
            yaxis = 'range'
        else:
            yaxis = 'range_bin'
    else:
        ValueError(f"Unsupported data type: {type(data)}")

    return _plot_echogram(
        ds,
        yaxis=yaxis,
        variable=variable,
        frequency=frequency,
        **kwargs
    )
