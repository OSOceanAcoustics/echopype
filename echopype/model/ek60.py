"""
echopype data model that keeps tracks of echo data and
its connection to data files.
"""

import os
import datetime as dt
import numpy as np
import xarray as xr


class EchoData(object):
    """Base class for echo data."""

    def __init__(self, file_path=""):
        self.file_path = file_path         # this passes the input through file name test
        self.tvg_correction_factor = 2     # range bin offset factor for calculating time-varying gain in EK60
        self.noise_est_range_bin_size = 10  # meters per tile for noise estimation
        self.noise_est_ping_size = 40       # number of pings per tile for noise estimation

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, p):
        self._file_path = p


        # Load netCDF groups if file format is correct
        pp = os.path.basename(p)
        _, ext = os.path.splitext(pp)

        if ext == '.raw':
            print('Data file in manufacturer format, please convert to .nc first.')
        elif ext == '.nc':
            self.toplevel = xr.open_dataset(self.file_path)
            if self.toplevel.sonar_convention_name == 'SONAR-netCDF4':
                # Point to various groups to .nc file converted from .raw
                self.provenance = xr.open_dataset(self.file_path, group="Provenance")
                self.environment = xr.open_dataset(self.file_path, group="Environment")
                self.platform = xr.open_dataset(self.file_path, group="Platform")
                self.sonar = xr.open_dataset(self.file_path, group="Sonar")
                self.beam = xr.open_dataset(self.file_path, group="Beam")

                # Get a separate .nc filename for storing processed data if computation is performed
                self.proc_path = os.path.join(os.path.dirname(self.file_path),
                                              os.path.splitext(os.path.basename(self.file_path))[0]+'_proc.nc')
            else:
                print('netCDF file convention not recognized.')
                # Groups for .nc file converted from .raw
                self.toplevel = ''
                self.provenance = ''
                self.environment = ''
                self.platform = ''
                self.sonar = ''
                self.beam = ''

                # Processed data
                self.proc_path = ''
        else:
            print('Data file format not recognized.')

    def calibrate(self):
        """Perform echo-integration to get volume backscattering strength (Sv) from EK60 power data.
        """

        # Loop through each frequency for calibration
        Sv = np.zeros(self.beam.backscatter_r.shape)
        for f_seq, freq in enumerate(self.beam.frequency.values):
            # Params from env group
            c = self.environment.sound_speed_indicative.sel(frequency=freq).values
            alpha = self.environment.absorption_indicative.sel(frequency=freq).values

            # Params from beam group
            t = self.beam.sample_interval.sel(frequency=freq).values
            gain = self.beam.gain_correction.sel(frequency=freq).values
            phi = self.beam.equivalent_beam_angle.sel(frequency=freq).values
            pt = self.beam.transmit_power.sel(frequency=freq).values
            tau = self.beam.transmit_duration_nominal.sel(frequency=freq).values
            Sac = 2 * self.beam.sa_correction.sel(frequency=freq).values

            # Derived params
            dR = c*t/2  # sample thickness
            wvlen = c/freq  # wavelength

            # Calc gain
            CSv = 10 * np.log10((pt * (10 ** (gain / 10))**2 *
                                 wvlen**2 * c * tau * 10**(phi / 10)) /
                                (32 * np.pi ** 2))

            # Get TVG
            range_vec = np.arange(self.beam.range_bin.size) * dR
            range_vec = range_vec - (self.tvg_correction_factor * dR)
            range_vec[range_vec < 0] = 0

            TVG = np.empty(range_vec.shape)
            TVG[range_vec != 0] = np.real(20 * np.log10(range_vec[range_vec != 0]))
            TVG[range_vec == 0] = 0

            # Get absorption
            ABS = 2 * alpha * range_vec

            # Compute Sv
            Sv[f_seq, :, :] = self.beam.backscatter_r.sel(frequency=freq).values + TVG + ABS - CSv - Sac

        # Assemble an xarray DataArray
        ping_time = (self.beam.ping_time.data - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')
        ds = xr.Dataset(
            {'Sv': (['frequency', 'ping_time', 'range_bin'], Sv,
                    {'long_name': 'Platform pitch',
                     'standard_name': 'platform_pitch_angle',
                     'units': 'arc_degree',
                     'valid_range': (-90.0, 90.0)}),
             },
            coords={'frequency': (['frequency'], self.beam.frequency,
                                  {'units': 'Hz',
                                   'valid_min': 0.0}),
                    'ping_time': (['ping_time'], ping_time,
                                  {'axis': 'T',
                                   'calendar': 'gregorian',
                                   'long_name': 'Timestamp of each ping',
                                   'standard_name': 'time',
                                   'units': 'seconds since 1900-01-01'}),
                    'range_bin': (['range_bin'], self.beam.range_bin)},
            attrs={'tvg_correction_factor': self.tvg_correction_factor})

        # Save calibrated data to a separate .nc file in the same directory as the data file
        print('Saving calibrated Sv into %s' % self.proc_path)
        ds.to_netcdf(path=self.proc_path, mode="w")
        ds.close()

    def get_noise_est(self, noise_est_range_bin_size=None, noise_est_ping_size=None):
        """
        Estimate noise level by getting the minimum value for bins of averaged ping.

        This function is called internally by remove_noise()
        Reference: De Robertis & Higginbottom, 2017, ICES Journal of Marine Sciences

        Parameters
        ------------
        noise_est_range_bin_size : meters per tile for noise estimation [m]
        noise_est_ping_size : number of pings per tile for noise estimation
        """

        # Check params
        if (noise_est_range_bin_size is not None) and (self.noise_est_range_bin_size != noise_est_range_bin_size):
            self.noise_est_range_bin_size = noise_est_range_bin_size
        if (noise_est_ping_size is not None) and (self.noise_est_ping_size != noise_est_ping_size):
            self.noise_est_ping_size = noise_est_ping_size

        # Loop through each frequency to estimate noise
        for f_seq, freq in enumerate(self.beam.frequency.values):

            # Get range bin size
            c = self.environment.sound_speed_indicative.sel(frequency=freq).values
            t = self.beam.sample_interval.sel(frequency=freq).values
            range_bin_size = c * t / 2

            # Adjust noise_est_range_bin_size because range_bin_size may be an inconvenient value
            num_range_bin_per_tile = np.round(self.noise_est_range_bin_size /
                                              range_bin_size).astype(int)  # number of range_bin per tile
            self.noise_est_range_bin_size = num_range_bin_per_tile * range_bin_size

            # Number of tiles along range_bin
            if np.mod(self.beam.range_bin.size, num_range_bin_per_tile) != 0:
                N_range_bin = np.floor(self.beam.range_bin.size / num_range_bin_per_tile).astype(int) + 1
            else:
                N_range_bin = np.int(self.beam.range_bin.size / num_range_bin_per_tile)
            if np.mod(self.beam.ping_time.size, self.noise_est_ping_size) != 0:
                N_ping = np.floor(self.beam.ping_time.size / self.noise_est_ping_size).astype(int) + 1
            else:
                N_ping = np.int(self.beam.ping_time.size / self.noise_est_ping_size)

            # Get noise estimates over N_range_bin x N_ping tile
            idx_p_base = np.arange(self.noise_est_ping_size, dtype=int)
            idx_r_base = np.arange(num_range_bin_per_tile, dtype=int)
            noise_est = np.empty((self.beam.frequency.size, N_ping, N_range_bin))
            for p_seq in range(N_ping):  # loop through all pings
                for r_seq in range(N_range_bin):  # loop through all range bins
                    idx_p = idx_p_base[[0, -1]] + self.noise_est_ping_size * p_seq
                    idx_r = idx_r_base[[0, -1]] + num_range_bin_per_tile * r_seq
                    noise_est[:, p_seq, r_seq] = \
                        np.mean(10 ** (self.beam.backscatter_r.isel(ping_time=slice(idx_p[0], idx_p[1]),
                                                                    range_bin=slice(idx_r[0], idx_r[1])).values / 10))
            noise_est = noise_est.min(axis=2)
            ping_time_reduce = self.beam.ping_time.isel(ping_time=np.arange(0, self.beam.ping_time.size,
                                                                            self.noise_est_ping_size)).data
            # Assemble an xarray DataArray
            ping_time_reduce = (ping_time_reduce - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')
            ds = xr.Dataset(
                {'noise_est': (['frequency', 'ping_time_noise_est'], noise_est,),
                 },
                coords={'frequency': (['frequency'], self.beam.frequency,
                                      {'units': 'Hz',
                                       'valid_min': 0.0}),
                        'ping_time_noise_est': (['ping_time_noise_est'], ping_time_reduce,
                                      {'axis': 'T',
                                       'calendar': 'gregorian',
                                       'long_name': 'Timestamp of each ping',
                                       'standard_name': 'time',
                                       'units': 'seconds since 1900-01-01'})},
                attrs={'tvg_correction_factor': self.tvg_correction_factor,
                       'noise_est_range_bin_size': self.noise_est_range_bin_size,
                       'noise_est_ping_size': self.noise_est_ping_size})

            # Save noise estimates to _proc.nc
            print('Saving calibrated Sv into %s' % self.proc_path)
            ds.to_netcdf(path=self.proc_path, mode="a")
            ds.close()

            # Average uncompensated power over each tile and find minimum value of power for each averaged bin
            # TODO: Let's use xarray dataArray/dataSet.coarsen for this average

    # @file_path.setter
    # def file_path(self, p):
    #     pp = os.path.basename(p)
    #     _, ext = os.path.splitext(pp)
    #     if ext == '.raw':
    #         raise ValueError('Data file in manufacturer format, please convert to .nc format first.')
    #     elif ext == '.nc':
    #         print('Got an .nc file! can start processing!')
    #         # print('Let us try to set some attributes')
    #     elif ext == '':
    #         print('Do nothing with empty file_path')
    #         self.toplevel = []
    #         self.provenance = []
    #         self.environment = []
    #         self.platform = []
    #         self.sonar = []
    #         self.beam = []
    #     else:
    #         # print('Not sure what file this is. EchoData only accepts .nc file as inputs.)
    #         raise ValueError('Not sure what file this is?? try to find a .nc file??')
    #     self._file_path = p




