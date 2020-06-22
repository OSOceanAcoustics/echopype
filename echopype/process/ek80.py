"""
echopype data model inherited from based class Process for EK80 data.
"""

import os
import datetime as dt
import numpy as np
import xarray as xr
from scipy import signal
from echopype.utils import uwa
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
        self._ch_ids = None
        self._tau_effective = None
        self.ytx = []
        self.backscatter_compressed = []
        self._sound_speed = self.get_sound_speed()
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

    def get_sound_speed(self):
        """gets sound speed [m/s] using parameters stored in the .nc file.
        Will use a custom path if one is provided
        """
        with self._open_dataset(self.file_path, group="Environment") as ds_env:
            return ds_env.sound_speed_indicative

    def calc_seawater_absorption(self, src='FG', path=''):
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
        return sea_abs

    def calc_sample_thickness(self, path=''):
        """gets sample thickness using parameters stored in the .nc file.
        Will use a custom path if one is provided
        """
        path = path if path else self.file_path
        with self._open_dataset(path, group="Beam") as ds_beam:
            sth = self.sound_speed * ds_beam.sample_interval / 2  # sample thickness
            return sth

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
            range_meter = range_bin * st - \
                ds_beam.transmit_duration_nominal * self.sound_speed / 2  # DataArray [frequency x range_bin]
            range_meter = range_meter.where(range_meter > 0, other=0).transpose()
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
            tau = ds_beam.transmit_duration_nominal.data
            txpower = ds_beam.transmit_power.data
            f0 = ds_beam.frequency_start.data
            f1 = ds_beam.frequency_end.data
            slope = ds_beam.slope[:, 0].data  # Use slope of first ping
            amp = np.sqrt((txpower / 4) * (2 * Ztrd))

            # Create transmit signal
            ytx = []
            for ch in range(ds_beam.frequency.size):
                t = np.arange(0, tau[ch], delta)
                nt = len(t)
                nwtx = (int(2 * np.floor(slope[ch] * nt)))
                wtx_tmp = np.hanning(nwtx)
                nwtxh = (int(np.round(nwtx / 2)))
                wtx = np.concatenate([wtx_tmp[0:nwtxh], np.ones((nt - nwtx)), wtx_tmp[nwtxh:]])
                y_tmp = amp[ch] * chirp_linear(t, f0[ch], f1[ch], tau[ch]) * wtx
                # The transmit signal must have a max amplitude of 1
                y = (y_tmp / np.max(np.abs(y_tmp)))

                # filter and decimation
                wbt_fil = ds_fil[self.ch_ids[ch] + "_WBT_filter"].data
                pc_fil = ds_fil[self.ch_ids[ch] + "_PC_filter"].data
                # if saved as netCDF4, convert compound complex datatype to complex64
                if wbt_fil.ndim == 1:
                    wbt_fil = np.array([complex(n[0], n[1]) for n in wbt_fil], dtype='complex64')
                    pc_fil = np.array([complex(n[0], n[1]) for n in pc_fil], dtype='complex64')

                # Apply WBT filter and downsample
                ytx_tmp = np.convolve(y, wbt_fil)
                ytx_tmp = ytx_tmp[0::ds_fil.attrs[self.ch_ids[ch] + "_WBT_decimation"]]

                # Apply PC filter and downsample
                ytx_tmp = np.convolve(ytx_tmp, pc_fil)
                ytx_tmp = ytx_tmp[0::ds_fil.attrs[self.ch_ids[ch] + "_PC_decimation"]]
                ytx.append(ytx_tmp)
                del nwtx, wtx_tmp, nwtxh, wtx, y_tmp, y, ytx_tmp

            # TODO: rename ytx into something like 'transmit_signal' and
            #  also package the sampling interval together with the signal
            self.ytx = ytx

    def pulse_compression(self):
        """Pulse compression using transmit signal as replica.
        """
        with self._open_dataset(self.file_path, group="Beam") as ds_beam:
            sample_interval = ds_beam.sample_interval
            backscatter = ds_beam.backscatter_r + ds_beam.backscatter_i * 1j  # Construct complex backscatter

            backscatter_compressed = []
            tau_constants = []
            # Loop over channels
            for ch in range(ds_beam.frequency.size):
                # tmp_x = np.fft.fft(backscatter[i].dropna('range_bin'))
                # tmp_y = np.fft.fft(np.flipud(np.conj(ytx[i])))
                tmp_b = backscatter[ch].dropna('range_bin')
                # tmp_b = tmp_b[:, 0, :]        # 1 ping
                tmp_y = np.flipud(np.conj(self.ytx[ch]))

                # Convolve tx signal with backscatter. atol=1e-7 between fft and direct convolution
                compressed = xr.apply_ufunc(lambda m: np.apply_along_axis(
                                            lambda m: signal.convolve(m, tmp_y), axis=2, arr=m),
                                            tmp_b,
                                            input_core_dims=[['range_bin']],
                                            output_core_dims=[['range_bin']],
                                            exclude_dims={'range_bin'}) / np.linalg.norm(self.ytx[ch]) ** 2
                # Average across quadrants
                backscatter_compressed.append(compressed)

                # Effective pulse length
                ptxa = np.square(np.abs(signal.convolve(self.ytx[ch], tmp_y, method='direct') /
                                        np.linalg.norm(self.ytx[ch]) ** 2))
                tau_constants.append(np.sum(ptxa) / (np.max(ptxa)))
            self._tau_effective = np.array(tau_constants) * sample_interval
            # Pad nans so that each channel has the same range_bin length
            largest_range_bin = max(bc.shape for bc in backscatter_compressed)[2]
            for i, ds in enumerate(backscatter_compressed):
                pad_width = largest_range_bin - ds.shape[2]
                backscatter_compressed[i] = xr.apply_ufunc(lambda x: np.pad(x, ((0,0), (0,0), (0,pad_width)),
                                                                            constant_values=np.nan),
                                                           ds,
                                                           input_core_dims=[['range_bin']],
                                                           output_core_dims=[['range_bin']],
                                                           exclude_dims={'range_bin'})
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
            f_nominal = ds_beam.frequency
            f_center = (ds_beam.frequency_start.data + ds_beam.frequency_end.data) / 2
            psifc = ds_beam.equivalent_beam_angle + 20 * np.log10(f_nominal / f_center)
            la2 = (c / f_center) ** 2
            Sv = []
            TS = []

            # Average accross quadrants and take the absolute value of complex backscatter
            prx = np.abs(np.mean(self.backscatter_compressed, axis=1))
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
                      10 * np.log10(ds_beam.transmit_power * la2 * c / (32 * np.pi * np.pi)) -
                      2 * Gfc - 10 * np.log10(self.tau_effective) - psifc
                )
            if mode == 'TS':
                TS = (
                      10 * np.log10(prx) + 40 * np.log10(ranges) +
                      2 * self.seawater_absorption * ranges -
                      10 * np.log10(ds_beam.transmit_power * la2 / (16 * np.pi * np.pi)) -
                      2 * Gfc
                )
            ds_beam.close()     # Close opened dataset
            # Save Sv calibrated data
            if mode == 'Sv':
                Sv.name = 'Sv'
                Sv = Sv.to_dataset()
                Sv['range'] = (('frequency', 'range_bin'), ranges)
                self.Sv = Sv
                if save:
                    self.Sv_path = self.validate_path(save_path, save_postfix)
                    print('%s  saving calibrated Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
                    self._save_dataset(Sv, self.Sv_path, mode="w")
            # Save TS calibrated data
            elif mode == 'TS':
                TS.name = 'TS'
                TS = TS.to_dataset()
                TS['ranges'] = (('frequency', 'range_bin'), ranges)
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
            Sv['range'] = (('frequency', 'range_bin'), range_meter)

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
