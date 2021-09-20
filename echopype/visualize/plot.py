import warnings
import matplotlib.pyplot as plt
import xarray as xr
from xarray.plot.facetgrid import FacetGrid
from matplotlib.collections import QuadMesh
from typing import Dict, Tuple, Union, List, TypeVar

T = TypeVar('T', int, float)


def _format_axis_label(axis_variable):
    return axis_variable.replace('_', " ").title()


def _set_label(
    fg: Union[FacetGrid, QuadMesh, None] = None,
    frequency: Union[int, float, None] = None,
):
    props = dict(boxstyle='square', facecolor='white', alpha=0.7)
    if isinstance(fg, FacetGrid):
        # Set each axis title
        for idx, ax in enumerate(fg.axes.flat):
            freq = fg.name_dicts[idx, 0]['frequency']
            ax.set_title("")
            ax.text(
                0.02,
                0.06,
                f"{int(freq / 1000)} kHz",
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment='bottom',
                bbox=props,
            )
    else:
        if frequency is None:
            raise ValueError(
                'Frequency value is missing for single echogram plotting.'
            )
        ax = fg.axes
        ax.text(
            0.02,
            0.04,
            f"{int(frequency / 1000)} kHz",
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment='bottom',
            bbox=props,
        )
        plt.title('')
        plt.tight_layout()


def _set_plot_defaults(kwargs):
    plot_defaults = {
        'cmap': 'jet',
        'figsize': (15, 10),
        'robust': False,
        'yincrease': False,
        'col_wrap': 1
    }

    # Set plot defaults if not passed in kwargs
    for k, v in plot_defaults.items():
        if k not in kwargs:
            kwargs[k] = v

    # Remove extra plotting attributes that should be set
    # by echopype devs
    exclude_attrs = ['x', 'y', 'col']
    for attr in exclude_attrs:
        if attr in kwargs:
            warnings.warn(f"{attr} in kwargs. Removing.")
            kwargs.pop(attr)

    return kwargs


def _plot_echogram(
    ds: xr.Dataset,
    frequency: Union[int, float, List[T], None] = None,
    variable: str = 'backscatter_r',
    xaxis: str = 'ping_time',
    yaxis: str = 'range',
    **kwargs,
) -> Union[FacetGrid, QuadMesh]:
    kwargs = _set_plot_defaults(kwargs)
    # perform frequency filtering
    if frequency:
        filtered_ds = ds[variable].sel(frequency=frequency)
    else:
        # if frequency not provided, use all
        filtered_ds = ds[variable].sel(frequency=slice(None))

    # figure out frequency size
    # to determine plotting method
    col = None
    if filtered_ds.frequency.size > 1:
        col = 'frequency'

    filtered_ds[xaxis].attrs = {
        'long_name': filtered_ds[xaxis].attrs.get(
            'long_name', _format_axis_label(xaxis)
        ),
        'units': filtered_ds[xaxis].attrs.get('units', ''),
    }
    filtered_ds[yaxis].attrs = {
        'long_name': filtered_ds[yaxis].attrs.get(
            'long_name', _format_axis_label(yaxis)
        ),
        'units': filtered_ds[yaxis].attrs.get('units', ''),
    }

    plot = filtered_ds.plot.pcolormesh(
        x=xaxis,
        y=yaxis,
        col=col,
        **kwargs,
    )
    _set_label(plot, frequency=frequency)
    return plot
