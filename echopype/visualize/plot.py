import warnings
import matplotlib.pyplot as plt
import matplotlib.cm
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
    filter_var: str = 'channel',
    filter_val: Union[str, int, float, None] = None,
    col: Optional[str] = None,
):
    props = {'boxstyle': 'square', 'facecolor': 'white', 'alpha': 0.7}
    if isinstance(fg, FacetGrid):
        text_pos = [0.02, 0.06]
        fontsize = 14
        if col == 'beam':
            if isinstance(filter_val, list) or filter_val is None:
                for rl in fg.row_labels:
                    if rl is not None:
                        rl.set_text('')

                for idx, cl in enumerate(fg.col_labels):
                    if cl is not None:
                        cl.set_text(f'Beam {fg.col_names[idx]}')

            text_pos = [0.04, 0.06]
            fontsize = 10

        for idx, ax in enumerate(fg.axes.flat):
            name_dicts = fg.name_dicts.flatten()
            if filter_var in name_dicts[idx]:
                chan = name_dicts[idx][filter_var]
                if col == filter_var:
                    ax.set_title('')
            else:
                chan = filter_val
                ax.set_title(f'Beam {fg.col_names[idx]}')
            axtext = chan
            if filter_var == 'frequency':
                axtext = f"{int(chan / 1000)} kHz"
            ax.text(
                *text_pos,
                axtext,
                transform=ax.transAxes,
                fontsize=fontsize,
                verticalalignment='bottom',
                bbox=props,
            )
    else:
        if filter_val is None:
            raise ValueError(
                f'{filter_var.title()} value is missing for single echogram plotting.'
            )
        axtext = filter_val
        if filter_var == 'frequency':
            axtext = f"{int(axtext / 1000)} kHz"
        ax = fg.axes
        ax.text(
            0.02,
            0.04,
            axtext,
            transform=ax.transAxes,
            fontsize=13,
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
    frequency: Union[str, List[str], None] = None,
    variable: str = 'backscatter_r',
    xaxis: str = 'ping_time',
    yaxis: str = 'echo_range',
    **kwargs,
) -> Union[FacetGrid, QuadMesh]:
    kwargs = _set_plot_defaults(kwargs)

    row = None
    col = None
    filter_var = 'channel'
    filter_val = None
    # perform frequency filtering
    if channel is not None:
        if 'channel' not in ds.dims:
            raise ValueError("Channel filtering is not available because channel is not a dimension for your dataset!")
        ds = ds.sel(channel=channel)
        filter_val = channel
    elif frequency is not None:
        duplicates = False
        if 'channel' in ds.dims:
            if len(np.unique(ds.frequency_nominal)) != len(ds.frequency_nominal):
                duplicates = True
            ds = ds.where(ds.frequency_nominal.isin(frequency), drop=True)
        else:
            if len(np.unique(ds.frequency)) != len(ds.frequency):
                duplicates = True
            ds = ds.sel(frequency=frequency)

        if duplicates:
            raise ValueError("Duplicate frequency found, please use channel for filtering.")
        filter_val = frequency
        filter_var = 'frequency'

    if 'backscatter_i' in ds.variables:
        col = 'beam'
        kwargs.setdefault('figsize', (15, 5))
        kwargs.update(
            {
                'col_wrap': None,
            }
        )
        filtered_ds = np.abs(ds.backscatter_r + 1j * ds.backscatter_i)
    else:
        filtered_ds = ds[variable]
        if 'beam' in filtered_ds.dims:
            filtered_ds = filtered_ds.isel(beam=0).drop('beam')

    if 'channel' in filtered_ds.dims and frequency is not None:
        filtered_ds = filtered_ds.assign_coords({'frequency': ds.frequency_nominal})
        filtered_ds = filtered_ds.swap_dims({'channel': 'frequency'})
        if filtered_ds.frequency.size == 1:
            filtered_ds = filtered_ds.isel(frequency=0)

    # figure out channel/frequency size
    # to determine plotting method
    if filtered_ds[filter_var].size > 1:
        if col is None:
            col = filter_var
        else:
            row = filter_var

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
    if not filtered_ds[filter_var].shape:
        if (
            np.any(filtered_ds.isnull()).values == np.array(True)
            and 'echo_range' in filtered_ds.coords
            and 'range_sample' in filtered_ds.dims
            and variable in ['backscatter_r', 'Sv']
        ):
            # Handle the nans for echodata and Sv
            filtered_ds = filtered_ds.sel(
                ping_time=filtered_ds.echo_range.dropna(dim='ping_time', how='all').ping_time
            )
            filtered_ds = filtered_ds.sel(
                range_sample=filtered_ds.echo_range.dropna(dim='range_sample').range_sample
            )
        plot = filtered_ds.plot.pcolormesh(
            x=xaxis,
            y=yaxis,
            col=col,
            row=row,
            **kwargs,
        )
        _set_label(plot, filter_var=filter_var, filter_val=filter_val, col=col)
        plots.append(plot)
    else:
        # Scale plots
        num_chan = len(filtered_ds[filter_var])
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

        for f in filtered_ds[filter_var]:
            d = filtered_ds[filtered_ds[filter_var] == f.values]
            if (
                np.any(d.isnull()).values == np.array(True)
                and 'echo_range' in d.coords
                and 'range_sample' in d.dims
                and variable in ['backscatter_r', 'Sv']
            ):
                # Handle the nans for echodata and Sv
                d = d.sel(
                    ping_time=d.echo_range.dropna(dim='ping_time', how='all').ping_time
                )
                d = d.sel(
                    range_sample=d.echo_range.dropna(dim='range_sample').range_sample
                )

            plot = d.plot.pcolormesh(
                x=xaxis,
                y=yaxis,
                col=col,
                row=row,
                **kwargs,
            )
            _set_label(plot, filter_var=filter_var, filter_val=filter_val, col=col)
            plots.append(plot)
    return plots
