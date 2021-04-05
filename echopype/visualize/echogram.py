import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from echopype.model import EchoData
from echopype.convert import Convert


class EchoGram:
    def __init__(self, _echo_data):
        self.echo_data = _echo_data

    def plot(self, form, frequency=None, plot_ping_number=False, infer_burst=False, col_wrap=2, **kwargs):
        """ Plots an echogram

        Parameters
        ----------
        form : str
            ``Sv``, ``TS``, or ``MVBS`` to plot corresponding echogram
        frequency : int or set of ints
            frequency or frequencies to plot. ``None`` will plot all frequencies
            default to ``None``. s
        ping_style : str
            ``number`` or ``time`` to plot ping number or ping time on the x axis
            default to ``time``
        infer burst : bool
            ``True`` will infer intervals between bursts. ``False`` will fill space with white
            default to ``False``
        **kwargs optional
            additional keyword arguments to matplotlib
        """
        data = getattr(self.echo_data, form)
        depth = self.echo_data.calc_range()

        # Plot MVBS
        if form == 'MVBS':
            # Plot one echogram
            if isinstance(frequency, int) or (frequency and len(frequency) == 1):
                if isinstance(frequency, set):
                    frequency = frequency.pop()
                data.MVBS.sel(frequency=frequency).plot(y='range_bin', **kwargs)
                return
            # Plot all echograms or select echograms
            if not frequency or (isinstance(frequency, set) and all(isinstance(f, int) for f in frequency)):
                if isinstance(frequency, set):
                    dropped_freqs = list(set(getattr(self.echo_data, form).frequency.values) - frequency)
                    to_plot = data.MVBS.drop(dropped_freqs, dim='frequency')
                else:
                    to_plot = data.MVBS
                to_plot.plot(col='frequency', col_wrap=col_wrap, y='range_bin',
                             **kwargs).set_xlabels('Ping time').set_ylabels('Range bin')
                return
            else:
                raise ValueError("frequency is not an int or a tuple of ints")

        # Plot Sv or TS

        # Plot all echograms
        if not frequency:
            data = data.assign_coords(depth=depth)
        # Plot 1 echogram
        elif isinstance(frequency, int) or len(frequency) == 1:
            if isinstance(frequency, set):
                frequency = frequency.pop()
            depth = depth.sel(frequency=frequency)
            data = data.sel(frequency=frequency).assign_coords(depth=depth)
            data = data.swap_dims({'range_bin': 'depth'})
        # Plot select echograms
        elif isinstance(frequency, set) and all(isinstance(f, int) for f in frequency):
            data = data.assign_coords(depth=depth)
            dropped_freqs = list(set(getattr(self.echo_data, form).frequency.values) - frequency)
            data = data.drop(dropped_freqs, dim='frequency')
        else:
            raise ValueError("frequency is not an int or a tuple of ints")

        # Plot ping number on x axis
        if plot_ping_number:
            ping_num = xr.DataArray(np.arange(len(data.ping_time)), coords=[data.ping_time], dims=['ping_time'])
            data = data.assign_coords(ping_num=ping_num).swap_dims({'ping_time': 'ping_num'})
            if data.ndim == 2:
                data.plot(x='ping_num', y='depth', **kwargs)
                plt.ylabel('Depth (m)')
                plt.xlabel('Ping number')
                return
            else:
                data.plot(col='frequency', col_wrap=col_wrap, x='ping_num', y='depth',
                          **kwargs).set_xlabels('Ping number').set_ylabels('Depth (m)')
                return
        # Plot ping time on x axis
        else:
            # Fill interval between bursts with white
            if not infer_burst:
                if plot_ping_number:
                    raise ValueError("Plotting with ping number cannot be done unless infer_burst = True")

                with xr.open_dataset(self.echo_data.file_path, group='Vendor') as ds_vend:
                    ping_per_profile = ds_vend.ping_per_profile  # 60 (pings) for test dataset
                    ping_period = ds_vend.ping_period  # 3 (seconds) for test dataset
                    # burst_int = ds_vend.burst_interval  # 900 (seconds) for test dataset

                if ping_per_profile != 1:
                    total_pings = len(data.ping_time)
                    pt = data.ping_time.values
                    var = data.values
                    ping = ping_per_profile
                    count = 0

                    while ping < total_pings:
                        pt = np.insert(pt, ping + count, pt[ping - 1 + count] + np.timedelta64(ping_period, 's'))
                        # If only 1 frequency, insert on the 0 axis. If multiple frequencies, insert on 1 axis
                        if data.ndim == 2:
                            var = np.insert(var, ping + count, np.nan, axis=0)
                        else:
                            var = np.insert(var, ping + count, np.nan, axis=1)
                        ping += ping_per_profile
                        count += 1

                    # If only 1 frequency, frequency is not a dimension (ndim = 2)
                    if data.ndim == 2:
                        to_plot = xr.DataArray(var, coords={'ping_time': pt, 'depth': data.depth},
                                               dims=['ping_time', 'depth'])
                        to_plot.plot(infer_intervals=False, x='ping_time', y='depth', **kwargs)
                        plt.ylabel('Depth (m)')
                        plt.xlabel('Ping time')
                        return
                    else:
                        to_plot = xr.DataArray(var, coords={'frequency': data.frequency, 'ping_time': pt,
                                                            'range_bin': data.range_bin, 'depth': data.depth},
                                               dims=['frequency', 'ping_time', 'range_bin'])
                        to_plot.plot(col='frequency', col_wrap=col_wrap, infer_intervals=False,
                                     x='ping_time', y='depth',
                                     **kwargs).set_xlabels('Ping time').set_ylabels('Depth (m)')
                        return

                # Data is not collected in bursts
                else:
                    return
            # Default interpolation for intervals between bursts
            else:
                if data.ndim == 2:
                    data.plot(x='ping_time', y='depth', **kwargs)
                    plt.ylabel('Depth (m)')
                    plt.xlabel('Ping time')
                    return
                else:
                    data.plot(col='frequency', col_wrap=col_wrap, x='ping_time', y='depth',
                              **kwargs).set_xlabels('Ping time').set_ylabels('Depth (m)')
                    return

# azfp_xml_path = './echopype/test_data/azfp/17041823.XML'
# azfp_01a_path = './echopype/test_data/azfp/17082117.01A'

# # Convert to .nc file
# tmp_convert = Convert(azfp_01a_path, azfp_xml_path)
# tmp_convert.raw2nc()

# tmp_echo = EchoData(tmp_convert.nc_path)
# tmp_echo.calibrate(save=False)
# tmp_echo.calibrate_ts(save=False)
# tmp_echo.get_MVBS()

# tmp_plot = EchoGram(tmp_echo)
# tmp_plot.plot('Sv', frequency=38000, infer_burst=False, robust=True, cmap='jet')
# plt.show()

#EK60 TESTING
# ek60_raw_path = './echopype/test_data/ek60/DY1801_EK60-D20180211-T164025.raw'
# import os
# nc_path = os.path.join(os.path.dirname(ek60_raw_path),
#                        os.path.splitext(os.path.basename(ek60_raw_path))[0] + '.nc')
# tmp = Convert(ek60_raw_path)
# tmp.raw2nc()

# # Read .nc file into an EchoData object and calibrate
# tmp_echo = EchoData(nc_path)
# tmp_echo.calibrate(save=False)
# tmp_plot = EchoGram(tmp_echo)
# tmp_plot.plot('Sv', frequency=38000, infer_burst=True, robust=True, cmap='jet')
# plt.show()