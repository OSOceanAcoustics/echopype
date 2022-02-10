import warnings
import matplotlib.pyplot as plt
import matplotlib.cm
import math
import xarray as xr
import numpy as np
from xarray.plot.facetgrid import FacetGrid
from matplotlib.collections import QuadMesh
from typing import Optional, Union, List, TypeVar
from .cm import cmap_d

T = TypeVar('T', int, float)


def _format_axis_label(axis_variable):
    return axis_variable.replace('_', " ").title()


def _set_label(
    fg: Union[FacetGrid, QuadMesh, None] = None,
    frequency: Union[int, float, None] = None,
    col: Optional[str] = None,
):
    props = {'boxstyle': 'square', 'facecolor': 'white', 'alpha': 0.7}
    if isinstance(fg, FacetGrid):
        text_pos = [0.02, 0.06]
        fontsize = 14
        if col == 'quadrant':
            if isinstance(frequency, list) or frequency is None:
                for rl in fg.row_labels:
                    if rl is not None:
                        rl.set_text('')

                for idx, cl in enumerate(fg.col_labels):
                    if cl is not None:
                        cl.set_text(f'Quadrant {fg.col_names[idx]}')

            text_pos = [0.04, 0.06]
            fontsize = 13

        for idx, ax in enumerate(fg.axes.flat):
            name_dicts = fg.name_dicts.flatten()
            if 'frequency' in name_dicts[idx]:
                freq = name_dicts[idx]['frequency']
                if col == 'frequency':
                    ax.set_title('')
            else:
                freq = frequency
                ax.set_title(f'Quadrant {fg.col_names[idx]}')
            ax.text(
                *text_pos,
                f"{int(freq / 1000)} kHz",
                transform=ax.transAxes,
                fontsize=fontsize,
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
        'col_wrap': 1,
    }

    # Set plot defaults if not passed in kwargs
    for k, v in plot_defaults.items():
        if k not in kwargs:
            kwargs[k] = v
        elif k == 'cmap' and k in kwargs:
            cmap = kwargs[k]
            try:
                if cmap in cmap_d:
                    cmap = f'ep.{cmap}'
                    kwargs[k] = cmap
                matplotlib.cm.get_cmap(cmap)
            except:
                import cmocean

                if cmap.startswith('cmo'):
                    _, cmap = cmap.split('.')

                if cmap in cmocean.cm.cmap_d:
                    kwargs[k] = f'cmo.{cmap}'
                else:
                    raise ValueError(f"{cmap} is not a valid colormap.")

    # Remove extra plotting attributes that should be set
    # by echopype devs
    exclude_attrs = ['x', 'y', 'col', 'row']
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

    row = None
    col = None

    if 'quadrant' in ds[variable].dims:
        col = 'quadrant'
        kwargs.update(
            {
                'figsize': (15, 5),
                'col_wrap': None,
            }
        )
        filtered_ds = np.abs(ds.backscatter_r + 1j * ds.backscatter_i)
    else:
        filtered_ds = ds[variable]

    # perform frequency filtering
    if frequency:
        filtered_ds = filtered_ds.sel(frequency=frequency)
    else:
        # if frequency not provided, use all
        filtered_ds = filtered_ds.sel(frequency=slice(None))

    # figure out frequency size
    # to determine plotting method
    if filtered_ds.frequency.size > 1:
        if col is None:
            col = 'frequency'
        else:
            row = 'frequency'

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

    plots = []
    if not filtered_ds.frequency.shape:
        if (
            np.any(filtered_ds.isnull()).values == np.array(True)
            and 'range' in filtered_ds.coords
            and 'range_bin' in filtered_ds.dims
            and variable in ['backscatter_r', 'Sv']
        ):
            # Handle the nans for echodata and Sv
            filtered_ds = filtered_ds.sel(
                range_bin=filtered_ds.range_bin.where(
                    ~filtered_ds.range.isel(ping_time=0).isnull()
                )
                .dropna(dim='range_bin')
                .data
            )
        plot = filtered_ds.plot.pcolormesh(
            x=xaxis,
            y=yaxis,
            col=col,
            row=row,
            **kwargs,
        )
        _set_label(plot, frequency=frequency, col=col)
        plots.append(plot)
    else:
        # Scale plots
        num_freq = len(filtered_ds.frequency)
        freq_scaling = (-0.06, -0.16)
        figsize_scale = tuple(
            [1 + (scale * num_freq) for scale in freq_scaling]
        )
        new_size = tuple(
            [
                size * figsize_scale[idx]
                for idx, size in enumerate(kwargs.get('figsize'))
            ]
        )
        kwargs.update({'figsize': new_size})

        for f in filtered_ds.frequency:
            d = filtered_ds[filtered_ds.frequency == f.values]
            if (
                np.any(d.isnull()).values == np.array(True)
                and 'range' in d.coords
                and 'range_bin' in d.dims
                and variable in ['backscatter_r', 'Sv']
            ):
                # Handle the nans for echodata and Sv
                d = d.sel(
                    range_bin=d.range_bin.where(
                        ~d.range.sel(frequency=f.values)
                        .isel(ping_time=0)
                        .isnull()
                    )
                    .dropna(dim='range_bin')
                    .data
                )
            plot = d.plot.pcolormesh(
                x=xaxis,
                y=yaxis,
                col=col,
                row=row,
                **kwargs,
            )
            _set_label(plot, frequency=frequency, col=col)
            plots.append(plot)
    return plots
