"""
echopype data model inherited from based class Process for EK80 data.
"""

import os
import datetime as dt
import numpy as np
import xarray as xr
import dask
from scipy import signal
from ..utils import uwa
from .processbase import ProcessBase


class ProcessEK80(ProcessBase):
    """Class for manipulating EK80 echo data already converted to netCDF.
    """
    def __init__(self, file_path=""):
        ProcessBase.__init__(self, file_path)
        self._acidity = None
        self._salinity = None
        self._temperature = None
        self._pressure = None
        self._beam_params = None
        self._ch_ids = None
        self._tau_effective = None
        self.transmit_signal = []
        self.backscatter_compressed = []
        self._beam_params = self.get_beam_params()
        self._sound_speed = self.calc_sound_speed()
        self._salinity = self.get_salinity()
        self._temperature = self.get_temperature()
        self._pressure = self.get_pressure()
        self._sample_thickness = self.calc_sample_thickness()
        self._seawater_absorption = self.calc_seawater_absorption()
        self._range = self.calc_range()

    @property
    def ch_ids(self):
        if self._ch_ids is None:
            with self._open_dataset(self.file_path, group="Beam") as ds_beam:
                self._ch_ids = ds_beam.channel_id.data
        return self._ch_ids

    @property
    def tau_effective(self):
        return self._tau_effective

    def get_beam_params(self, unique_only=False):
        self._beam_params = dict()
        params = ['transmit_duration_nominal', 'sample_interval', 'transmit_power', 'slope']
        with self._open_dataset(self.file_path, group='Beam') as ds_beam:
            for param in params:
                if unique_only:
                    val, idx = np.unique(ds_beam[param], return_index=True, axis=1)
                    self._beam_params[param + '_indices'] = idx
                else:
                    val = ds_beam[param]
                self._beam_params[param] = val
        return self._beam_params

    def get_salinity(self):
        if self._salinity is None:
            with self._open_dataset(self.file_path, group="Environment") as ds_env:
                return ds_env.salinity

    def get_temperature(self, path=''):
        path = path if path else self.file_path
        if self._temperature is None:
            with self._open_dataset(path, group="Environment") as ds_env:
                return ds_env.temperature

    def get_pressure(self):
        if self._pressure is None:
            with self._open_dataset(self.file_path, group="Environment") as ds_env:
                return ds_env.depth

    def calc_sound_speed(self, src='file'):
        """gets sound speed [m/s] using parameters stored in the .nc file.
        Will use a custom path if one is provided
        """
        if src == 'file':
            with self._open_dataset(self.file_path, group="Environment") as ds_env:
                return ds_env.sound_speed_indicative
        elif src == 'user':
            ss = uwa.calc_sound_speed(salinity=self.salinity,
                                      temperature=self.temperature,
                                      pressure=self.pressure)
            return ss * np.ones(self.sound_speed.size)
        else:
            ValueError('Not sure how to update sound speed!')

    def calc_seawater_absorption(self, src='user', path=''):
        """Returns the seawater absorption

        Parameters
        ----------
        src : str
            'file' will return the seawater absoption recorded in the .nc file
            'user' will calculate the seawater absorption. Default (Francois and Garrison, 1982).

        Returns
        -------
        Seawater absorption value
        """
        if src == 'user':
            path = path if path else self.file_path
            with self._open_dataset(path, group='Beam') as ds_beam:
                try:
                    f0 = ds_beam.frequency_start
                    f1 = ds_beam.frequency_end
                    f = (f0 + f1) / 2
                except AttributeError:
                    f = ds_beam.frequency
            sea_abs = uwa.calc_seawater_absorption(f,
                                                   salinity=self.salinity,
                                                   temperature=self.temperature,
                                                   pressure=self.pressure,
                                                   formula_source='FG')
        else:
            ValueError('Not sure how to update seawater absorption!')
        return sea_abs

    def calc_sample_thickness(self, path=''):
        """gets sample thickness using parameters stored in the .nc file.
        Will use a custom path if one is provided
        """
        path = path if path else self.file_path
        with self._open_dataset(path, group="Beam") as ds_beam:
            return self.sound_speed * ds_beam.sample_interval / 2  # sample thickness

    def calc_range(self, range_bins=None, path=''):
        """Calculates range [m] using parameters stored in the .nc file.
        Will use a custom path if one is provided
        """
        st = self.calc_sample_thickness(path) if path else self.sample_thickness
        path = path if path else self.file_path
        with self._open_dataset(path, group="Beam") as ds_beam:
            if range_bins:
                range_bin = np.arange(range_bins)
                range_bin = xr.DataArray(range_bin, coords=[('range_bin', range_bin)])
            else:
                range_bin = ds_beam.range_bin
            tdn = ds_beam.transmit_duration_nominal
            range_meter = st * range_bin - tdn * self.sound_speed / 2  # DataArray [frequency x range_bin]
            range_meter = range_meter.where(range_meter > 0, other=0)
            return range_meter

    def calc_transmit_signal(self):
        """Generate transmit signal as replica for pulse compression.
        """
        def chirp_linear(t, f0, f1, tau):
            beta = (f1 - f0) * (tau ** -1)
            return np.cos(2 * np.pi * (beta / 2 * (t ** 2) + f0 * t))

        # Retrieve filter coefficients
        with self._open_dataset(self.file_path, group="Vendor") as ds_fil, \
            self._open_dataset(self.file_path, group="Beam") as ds_beam:

            # Get various parameters
            Ztrd = 75  # Transducer quadrant nominal impedance [Ohms] (Supplied by Simrad)
            delta = 1 / 1.5e6   # Hard-coded EK80 sample interval
            # tau = ds_beam.transmit_duration_nominal.data            # TODO make getters matrix for different values
            # tx_power = ds_beam.transmit_power.data
            # slope = ds_beam.slope.data  # Use slope of first ping
            tau = self._beam_params['transmit_duration_nominal'].values
            tx_power = self._beam_params['transmit_power'].values
            slope = self._beam_params['slope'].values

            # Find indexes with a unique transmitted signal.
            # Get outer product of 3 parameters
            beam_params = np.einsum('i,j,k->ijk', tau[0], slope[0], tx_power[0])
            # Get diagonal of 3D matix
            _, unique_signal_idx, counts = np.unique(beam_params[np.diag_indices(tau.shape[1], ndim=3)],
                                                     return_index=True, return_counts=True)
            # Re order beam parameters because np.unique sorts the output
            sort_idx = np.argsort(unique_signal_idx)
            unique_signal_idx = unique_signal_idx[sort_idx]
            counts = counts[sort_idx]
            tau = tau[:, unique_signal_idx]
            tx_power = tx_power[:, unique_signal_idx]
            slope = slope[:, unique_signal_idx]

            amp = np.sqrt((tx_power / 4) * (2 * Ztrd))
            f0 = ds_beam.frequency_start.data
            f1 = ds_beam.frequency_end.data

            ytx = []
            for i in range(tau.shape[1]):
                ytx_ping = []
                # Create transmit signal
                for ch in range(ds_beam.frequency.size):
                    t = np.arange(0, tau[ch][i], delta)
                    nt = len(t)
                    nwtx = (int(2 * np.floor(slope[ch][i] * nt)))
                    wtx_tmp = np.hanning(nwtx)
                    nwtxh = (int(np.round(nwtx / 2)))
                    wtx = np.concatenate([wtx_tmp[0:nwtxh], np.ones((nt - nwtx)), wtx_tmp[nwtxh:]])
                    y_tmp = amp[ch][i] * chirp_linear(t, f0[ch], f1[ch], tau[ch][i]) * wtx
                    # The transmit signal must have a max amplitude of 1
                    y = (y_tmp / np.max(np.abs(y_tmp)))

                    # filter and decimation
                    wbt_fil = ds_fil.attrs[self.ch_ids[ch] + '_WBT_filter_r'] + 1j * \
                        ds_fil.attrs[self.ch_ids[ch] + "_WBT_filter_i"]
                    pc_fil = ds_fil.attrs[self.ch_ids[ch] + '_PC_filter_r'] + 1j * \
                        ds_fil.attrs[self.ch_ids[ch] + '_PC_filter_i']

                    # Apply WBT filter and downsample
                    ytx_tmp = np.convolve(y, wbt_fil)
                    ytx_tmp = ytx_tmp[0::ds_fil.attrs[self.ch_ids[ch] + "_WBT_decimation"]]

                    # Apply PC filter and downsample
                    ytx_tmp = np.convolve(ytx_tmp, pc_fil)
                    ytx_tmp = ytx_tmp[0::ds_fil.attrs[self.ch_ids[ch] + "_PC_decimation"]]
                    ytx_ping.append(ytx_tmp)
                    del nwtx, wtx_tmp, nwtxh, wtx, y_tmp, y, ytx_tmp
                ytx.extend([ytx_ping for n in range(counts[i])])

            #  TODO: package the sampling interval together with the signal
            self.transmit_signal = np.array(ytx).T

    def pulse_compression(self):
        """Pulse compression using transmit signal as replica.
        """
        with self._open_dataset(self.file_path, group="Beam") as ds_beam:
            backscatter = ds_beam.backscatter_r + ds_beam.backscatter_i * 1j  # Construct complex backscatter
            nfreq, _, npings, _ = backscatter.shape
            backscatter_compressed = []
            tau_constants = []

            def rectangularize(v):
                lens = np.array([len(item) for item in v])
                mask = lens[:, None] > np.arange(lens.max())
                out = np.full(mask.shape, np.nan, dtype='complex128')
                out[mask] = np.concatenate(v)
                return out

            def compress(ping_idx):
                # Convolve tx signal with backscatter. atol=1e-7 between fft and direct convolution
                compressed = np.apply_along_axis(signal.convolve, axis=1,
                                                 arr=tmp_b[:, ping_idx, :],
                                                 in2=np.flipud(tmp_y[ping_idx])) / \
                    np.linalg.norm(self.transmit_signal[ch][ping_idx]) ** 2
                return np.mean(compressed, axis=0)

            def calc_effective_pulse_length(ping_idx):
                ptxa = np.square(np.abs(signal.convolve(self.transmit_signal[ch][ping_idx],
                                        np.flipud(tmp_y[ping_idx]), method='direct') /
                                        np.linalg.norm(self.transmit_signal[ch][ping_idx]) ** 2))
                return np.sum(ptxa) / (np.max(ptxa))

            # Loop over channels
            for ch in range(nfreq):
                # tmp_x = np.fft.fft(backscatter[i].dropna('range_bin'))
                # tmp_y = np.fft.fft(np.flipud(np.conj(ytx[i])))
                # remove quadrants that are nans across all samples
                tmp_b = backscatter[ch].dropna('range_bin', how='all')
                # remove samples that are nans across all quadrants
                tmp_b = tmp_b.dropna('quadrant', how='all')
                # tmp_b = tmp_b[:, 0, :]        # 1 ping
                tmp_y = np.conj(self.transmit_signal[ch])

                # backscatter_compressed.append(compressed)
                # TODO: try out xrft for convolution
                backscatter_lazy = [dask.delayed(compress)(i) for i in range(npings)]
                backscatter_compressed.append(dask.compute(*backscatter_lazy))
                backscatter_compressed[-1] = rectangularize(np.array(backscatter_compressed[-1]))
                backscatter_compressed[-1] = xr.DataArray(
                    np.expand_dims(backscatter_compressed[-1], 0),
                    coords={
                        'frequency': [backscatter[ch].frequency],
                        'ping_time': backscatter.ping_time,
                        'range_bin': np.arange(backscatter_compressed[-1].shape[1]),
                        'frequency_start': backscatter[ch].frequency_start,
                        'frequency_end': backscatter[ch].frequency_end
                    },
                    dims=['frequency', 'ping_time', 'range_bin'])

                # Effective pulse length TODO: calculate unique
                ptxa_lazy = [dask.delayed(calc_effective_pulse_length)(i) for i in range(npings)]
                tau_constants.append(dask.compute(*ptxa_lazy))

            self._tau_effective = tau_constants * ds_beam.sample_interval
            self.backscatter_compressed = xr.concat(backscatter_compressed, dim='frequency')

    def calibrate(self, mode='Sv', save=False, save_path=None, save_postfix=None):
        """Perform echo-integration to get volume backscattering strength (Sv)
        or target strength (TS) from EK80 power data.

        Parameters
        -----------
        mode : str
            'Sv' for volume backscattering strength calibration (default)
            'TS' for target strength calibration
        save : bool, optional
            whether to save calibrated output
            default to ``False``
        save_path : str
            Full filename to save to, overwriting the RAWFILENAME_Sv.nc default
        save_postfix : str
            Filename postfix, default to '_Sv' or '_TS'
        """

        ds_beam = self._open_dataset(self.file_path, group="Beam")

        # Check for cw data file
        split = os.path.splitext(self.file_path)
        cw_path = split[0] + '_cw' + split[1]
        if save_postfix is None:
            save_postfix = '_' + mode
        if os.path.exists(cw_path):
            self.calibrate_cw(mode, cw_path, save, save_path, save_postfix)
        elif 'backscatter_i' not in ds_beam:
            self.calibrate_cw(mode, self.file_path, save, save_path, save_postfix)

        # Calibrate bb data
        if 'backscatter_i' in ds_beam:
            Ztrd = 75       # Transducer quadrant nominal impedance [Ohms] (Supplied by Simrad)
            Rwbtrx = 1000   # Wideband transceiver impedance [Ohms] (Supplied by Simrad)
            self.calc_transmit_signal()  # Get transmit signal
            self.pulse_compression()    # Perform pulse compression

            c = self.sound_speed
            tx_power = ds_beam.transmit_power
            f_nominal = ds_beam.frequency
            f_center = (ds_beam.frequency_start.data + ds_beam.frequency_end.data) / 2
            psifc = ds_beam.equivalent_beam_angle + 20 * np.log10(f_nominal / f_center)
            la2 = (c / f_center) ** 2
            Sv = []
            TS = []

            # Take cvm,xabsolute value of complex backscatter
            prx = np.abs(self.backscatter_compressed)
            prx = prx * prx / 2 * (np.abs(Rwbtrx + Ztrd) / Rwbtrx) ** 2 / np.abs(Ztrd)
            # TODO Gfc should be gain interpolated at the center frequency
            # Only 1 gain value is given provided per channel
            Gfc = ds_beam.gain_correction
            ranges = self.calc_range(range_bins=prx.shape[2])
            ranges = ranges.where(ranges >= 1, other=1)
            if mode == 'Sv':
                Sv = (
                    10 * np.log10(prx) + 20 * np.log10(ranges) +
                    2 * self.seawater_absorption * ranges -
                    10 * np.log10(tx_power * la2[:, None] * c / (32 * np.pi * np.pi)) -
                    2 * Gfc - 10 * np.log10(self.tau_effective) - psifc
                )
            if mode == 'TS':
                TS = (
                    10 * np.log10(prx) + 40 * np.log10(ranges) +
                    2 * self.seawater_absorption * ranges -
                    10 * np.log10(tx_power * la2[:, None] / (16 * np.pi * np.pi)) -
                    2 * Gfc
                )
            ds_beam.close()     # Close opened dataset
            # Save Sv calibrated data
            if mode == 'Sv':
                Sv.name = 'Sv'
                Sv = Sv.to_dataset()
                Sv['range'] = (('frequency', 'ping_time', 'range_bin'), ranges)
                self.Sv = Sv
                if save:
                    self.Sv_path = self.validate_path(save_path, save_postfix)
                    print('%s  saving calibrated Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
                    self._save_dataset(Sv, self.Sv_path, mode="w")
            # Save TS calibrated data
            elif mode == 'TS':
                TS.name = 'TS'
                TS = TS.to_dataset()
                TS['range'] = (('frequency', 'range_bin', 'range_bin'), ranges)
                self.TS = TS
                if save:
                    self.TS_path = self.validate_path(save_path, save_postfix)
                    print('%s  saving calibrated TS to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.TS_path))
                    self._save_dataset(self.TS, self.TS_path, mode="w")

    def calibrate_TS(self, save=False, save_path=None, save_postfix=None):
        self.calibrate(mode='TS', save=save, save_path=save_path, save_postfix=save_postfix)

    def calibrate_cw(self, mode='Sv', file_path='', save=False, save_path=None, save_postfix=None):
        """Perform echo-integration to get volume backscattering strength (Sv) from EK80 power data.

        Parameters
        -----------
        mode : str
            'Sv' for volume backscattering strength (default)
            'TS' for target strength
        file_path : str
            Path to CW data
        save : bool, optional
            whether to save calibrated Sv output
            default to ``False``
        save_path : str
            Full filename to save to, overwriting the RAWFILENAME_Sv.nc default
        save_postfix : str
            Filename postfix
        """
        # Open data set for and Beam groups
        if file_path and os.path.exists(file_path):
            ds_beam = self._open_dataset(file_path, group="Beam")
        else:
            file_path = self.file_path
            ds_beam = self._open_dataset(self.file_path, group="Beam")

        # Derived params
        wavelength = self.sound_speed / ds_beam.frequency  # wavelength

        # Retrieved params
        backscatter_r = ds_beam['backscatter_r'].load()
        range_meter = self.calc_range(path=file_path)
        sea_abs = self.calc_seawater_absorption(path=file_path)

        if mode == 'Sv':
            # Calc gain
            CSv = 10 * np.log10((ds_beam.transmit_power * (10 ** (ds_beam.gain_correction / 10)) ** 2 *
                                wavelength ** 2 * self.sound_speed * ds_beam.transmit_duration_nominal *
                                10 ** (ds_beam.equivalent_beam_angle / 10)) /
                                (32 * np.pi ** 2))

            # Get TVG and absorption
            TVG = np.real(20 * np.log10(range_meter.where(range_meter >= 1, other=1)))
            ABS = 2 * sea_abs * range_meter

            # Calibration and echo integration
            Sv = backscatter_r + TVG + ABS - CSv - 2 * ds_beam.sa_correction
            Sv.name = 'Sv'
            Sv = Sv.to_dataset()

            # Attach calculated range into data set
            Sv['range'] = (('frequency', 'ping_time', 'range_bin'), range_meter)

            # Save calibrated data into the calling instance and
            #  to a separate .nc file in the same directory as the data filef.Sv = Sv
            self.Sv = Sv
            if save:
                if save_postfix is None:
                    save_postfix = '_' + mode
                self.Sv_path = self.validate_path(save_path, save_postfix, file_path)
                print('%s  saving calibrated Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
                self._save_dataset(Sv, self.Sv_path, mode="w")
        elif mode == 'TS':
            CSp = 10 * np.log10((ds_beam.transmit_power * (10 ** (ds_beam.gain_correction / 10)) ** 2 *
                                wavelength ** 2) / (16 * np.pi ** 2))
            TVG = np.real(40 * np.log10(range_meter.where(range_meter >= 1, other=1)))
            ABS = 2 * self.seawater_absorption * range_meter

            # Calibration and echo integration
            TS = backscatter_r + TVG + ABS - CSp
            TS.name = 'TS'
            TS = TS.to_dataset()

            # Attach calculated range into data set
            TS['range'] = (('frequency', 'range_bin'), range_meter)

            # Save calibrated data into the calling instance and
            #  to a separate .nc file in the same directory as the data filef.Sv = Sv
            self.TS = TS
            if save:
                self.TS_path = self.validate_path(save_path, save_postfix)
                print('%s  saving calibrated TS to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.TS_path))
                self._save_dataset(TS, self.TS_path, mode="w")

        # Close opened resources
        ds_beam.close()
