import sys
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm
import math
import xarray as xr
import numpy as np
from xarray.plot.facetgrid import FacetGrid
from matplotlib.collections import QuadMesh
from typing import Optional, Union, List
from .cm import cmap_d


def _format_axis_label(axis_variable):
    return axis_variable.replace('_', " ").title()


def _set_label(
    fg: Union[FacetGrid, QuadMesh, None] = None,
    channel: Union[str, None] = None,
    col: Optional[str] = None,
):
    props = {'boxstyle': 'square', 'facecolor': 'white', 'alpha': 0.7}
    if isinstance(fg, FacetGrid):
        text_pos = [0.02, 0.06]
        fontsize = 14
        if col == 'beam':
            if isinstance(channel, list) or channel is None:
                for rl in fg.row_labels:
                    if rl is not None:
                        rl.set_text('')

                for idx, cl in enumerate(fg.col_labels):
                    if cl is not None:
                        cl.set_text(f'Beam {fg.col_names[idx]}')

            text_pos = [0.04, 0.06]
            fontsize = 13

        for idx, ax in enumerate(fg.axes.flat):
            name_dicts = fg.name_dicts.flatten()
            if 'channel' in name_dicts[idx]:
                chan = name_dicts[idx]['channel']
                if col == 'channel':
                    ax.set_title('')
            else:
                chan = channel
                ax.set_title(f'Beam {fg.col_names[idx]}')
            ax.text(
                *text_pos,
                f"{chan} kHz",
                transform=ax.transAxes,
                fontsize=fontsize,
                verticalalignment='bottom',
                bbox=props,
            )
    else:
        if channel is None:
            raise ValueError(
                'Channel value is missing for single echogram plotting.'
            )
        ax = fg.axes
        ax.text(
            0.02,
            0.04,
            f"{channel} kHz",
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
    channel: Union[str, List[str], None] = None,
    variable: str = 'backscatter_r',
    xaxis: str = 'ping_time',
    yaxis: str = 'echo_range',
    **kwargs,
) -> Union[FacetGrid, QuadMesh]:
    kwargs = _set_plot_defaults(kwargs)

    row = None
    col = None

    if 'backscatter_i' in ds.variables:
        col = 'beam'
        kwargs.update(
            {
                'figsize': (15, 5),
                'col_wrap': None,
            }
        )
        filtered_ds = np.abs(ds.backscatter_r + 1j * ds.backscatter_i)
    else:
        filtered_ds = ds[variable]
        if 'beam' in filtered_ds.dims:
            filtered_ds = filtered_ds.isel(beam=0).drop('beam')

    # perform frequency filtering
    if channel:
        filtered_ds = filtered_ds.sel(channel=channel)
    else:
        # if channel not provided, use all
        filtered_ds = filtered_ds.sel(channel=slice(None))

    # figure out channel size
    # to determine plotting method
    if filtered_ds.channel.size > 1:
        if col is None:
            col = 'channel'
        else:
            row = 'channel'

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
    if not filtered_ds.channel.shape:
        if (
            np.any(filtered_ds.isnull()).values == np.array(True)
            and 'echo_range' in filtered_ds.coords
            and 'range_sample' in filtered_ds.dims
            and variable in ['backscatter_r', 'Sv']
        ):
            # Handle the nans for echodata and Sv
            filtered_ds = filtered_ds.sel(
                range_sample=filtered_ds.range_sample.where(
                    ~filtered_ds.echo_range.isel(ping_time=0).isnull()
                )
                .dropna(dim='range_sample')
                .data
            )
        plot = filtered_ds.plot.pcolormesh(
            x=xaxis,
            y=yaxis,
            col=col,
            row=row,
            **kwargs,
        )
        _set_label(plot, channel=channel, col=col)
        plots.append(plot)
    else:
        # Scale plots
        num_chan = len(filtered_ds.channel)
        chan_scaling = (-0.06, -0.16)
        figsize_scale = tuple(
            [1 + (scale * num_chan) for scale in chan_scaling]
        )
        new_size = tuple(
            [
                size * figsize_scale[idx]
                for idx, size in enumerate(kwargs.get('figsize'))
            ]
        )
        kwargs.update({'figsize': new_size})

        for f in filtered_ds.channel:
            d = filtered_ds[filtered_ds.channel == f.values]
            if (
                np.any(d.isnull()).values == np.array(True)
                and 'echo_range' in d.coords
                and 'range_sample' in d.dims
                and variable in ['backscatter_r', 'Sv']
            ):
                # Handle the nans for echodata and Sv
                d = d.sel(
                    range_sample=d.range_sample.where(
                        ~d.echo_range.sel(channel=f.values)
                        .isel(ping_time=0)
                        .isnull()
                    )
                    .dropna(dim='range_sample')
                    .data
                )

            plot = d.plot.pcolormesh(
                x=xaxis,
                y=yaxis,
                col=col,
                row=row,
                **kwargs,
            )
            _set_label(plot, channel=channel, col=col)
            plots.append(plot)
    return plots
