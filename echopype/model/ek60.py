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
        self.file_path = file_path  # this passes the input through file name test

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
                # Point to various groups
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
                self.proc_path = ''
        else:
            print('Data file format not recognized.')

    def calibrate(self, tvg_correction_factor=2):
        """Perform echo-integration to get volume backscattering strength (Sv) from EK60 power data.

        Parameters
        -----------
        tvg_correction_factor : range bin offset factor for calculating time-varying gain in EK60.
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
            range_vec = range_vec - (tvg_correction_factor * dR)
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
            attrs={'tvg_correction_factor': tvg_correction_factor})

        # Save calibrated data to a separate .nc file in the same directory as the data file
        print('Saving calibrated Sv into %s' % self.proc_path)
        ds.to_netcdf(path=self.proc_path, mode="w")
        ds.close()

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




