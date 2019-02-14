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
        self.tvg_correction_factor = 2  # range bin offset factor for calculating time-varying gain in EK60
        self.TVG = None  # time varying gain along range
        self.ABS = None  # absorption along range
        self.sample_thickness = None  # sample thickness for each frequency
        self.noise_est_range_bin_size = 5  # meters per tile for noise estimation
        self.noise_est_ping_size = 30  # number of pings per tile for noise estimation
        self.MVBS_range_bin_size = 5  # meters per tile for MVBS
        self.MVBS_ping_size = 30  # number of pings per tile for MVBS
        self.Sv = None  # calibrated volume backscattering strength
        self.Sv_clean = None  # denoised volume backscattering strength
        self.MVBS = None  # mean volume backscattering strength

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

            # Get .nc filenames for storing processed data if computation is performed
            self.Sv_path = os.path.join(os.path.dirname(self.file_path),
                                        os.path.splitext(os.path.basename(self.file_path))[0] + '_Sv.nc')
            self.Sv_clean_path = os.path.join(os.path.dirname(self.file_path), '_Sv_clean.nc')
            self.MVBS_path = os.path.join(os.path.dirname(self.file_path),
                                          os.path.splitext(os.path.basename(self.file_path))[0] + '_MVBS.nc')

            # Raise error if the file format convention does not match
            if self.toplevel.sonar_convention_name != 'SONAR-netCDF4':
                raise ValueError('netCDF file convention not recognized.')
        else:
            raise ValueError('Data file format not recognized.')

    def calibrate(self):
        """Perform echo-integration to get volume backscattering strength (Sv) from EK60 power data.
        """

        # Open data set for Environment and Beam groups
        ds_env = xr.open_dataset(self.file_path, group="Environment")
        ds_beam = xr.open_dataset(self.file_path, group="Beam")

        # Derived params
        sample_thickness = ds_env.sound_speed_indicative * ds_beam.sample_interval / 2  # sample thickness
        wavelength = ds_env.sound_speed_indicative / ds_env.frequency  # wavelength

        # Calc gain
        CSv = 10 * np.log10((ds_beam.transmit_power * (10 ** (ds_beam.gain_correction / 10)) ** 2 *
                             wavelength ** 2 * ds_env.sound_speed_indicative * ds_beam.transmit_duration_nominal *
                             10 ** (ds_beam.equivalent_beam_angle / 10)) /
                            (32 * np.pi ** 2))

        # Get TVG and absorption
        range_meter = ds_beam.range_bin * sample_thickness - \
                      self.tvg_correction_factor * sample_thickness  # DataArray [frequency x range_bin]
        range_meter = range_meter.where(range_meter > 0, other=0)  # set all negative elements to 0
        TVG = np.real(20 * np.log10(range_meter.where(range_meter != 0, other=1)))
        ABS = 2 * ds_env.absorption_indicative * range_meter

        # Save TVG and ABS for noise estimation use
        self.sample_thickness = sample_thickness
        self.TVG = TVG
        self.ABS = ABS

        # Calibration and echo integration
        Sv = ds_beam.backscatter_r + TVG + ABS - CSv - 2 * ds_beam.sa_correction
        Sv.name = 'Sv'

        # Save calibrated data to a separate .nc file in the same directory as the data file
        print('%s  saving calibrated Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
        Sv.to_netcdf(path=self.Sv_path, mode="w")
        ds_env.close()
        ds_beam.close()

    def remove_noise(self, noise_est_range_bin_size=None, noise_est_ping_size=None):
        """Remove noise by using noise estimates obtained from the minimum mean calibrated power level
        along each column of tiles.

        See method noise_estimates() for details of noise estimation.
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

        # Get calibrated power
        proc_data = xr.open_dataset(self.Sv_path)

        # Adjust noise_est_range_bin_size because range_bin_size may be an inconvenient value
        num_range_bin_per_tile = (np.round(self.noise_est_range_bin_size /
                                           self.sample_thickness).astype(int)).values.max()  # num of range_bin per tile
        self.noise_est_range_bin_size = (num_range_bin_per_tile * self.sample_thickness).values

        # Number of tiles along range_bin
        if np.mod(proc_data.range_bin.size, num_range_bin_per_tile) == 0:
            num_tile_range_bin = np.ceil(proc_data.range_bin.size / num_range_bin_per_tile).astype(int) + 1
        else:
            num_tile_range_bin = np.ceil(proc_data.range_bin.size / num_range_bin_per_tile).astype(int)

        # Produce a new coordinate for groupby operation
        if np.mod(proc_data.ping_time.size, self.noise_est_ping_size) == 0:
            num_tile_ping = np.ceil(proc_data.ping_time.size / self.noise_est_ping_size).astype(int) + 1
            z = np.array([np.arange(num_tile_ping - 1)] * self.noise_est_ping_size).squeeze().T.ravel()
        else:
            num_tile_ping = np.ceil(proc_data.ping_time.size / self.noise_est_ping_size).astype(int)
            pad = np.ones(proc_data.ping_time.size - (num_tile_ping - 1) * self.noise_est_ping_size, dtype=int) \
                  * (num_tile_ping - 1)
            z = np.hstack((np.array([np.arange(num_tile_ping - 1)]
                                    * self.noise_est_ping_size).squeeze().T.ravel(), pad))

        # Tile bin edges along range
        # ... -1 to make sure each bin has the same size because of the right-inclusive and left-exclusive bins
        range_bin_tile_bin = np.arange(num_tile_range_bin) * num_range_bin_per_tile - 1

        # Function for use with apply
        def remove_n(x):
            p_c_lin = 10 ** ((x - self.ABS - self.TVG) / 10)
            nn = 10 * np.log10(p_c_lin.groupby_bins('range_bin', range_bin_tile_bin).mean('range_bin').
                               groupby('frequency').mean('ping_time').min(dim='range_bin_bins'))
            return x.where(x > nn, other=np.nan)

        # Groupby noise removal operation
        proc_data.coords['add_idx'] = ('ping_time', z)
        Sv_clean = proc_data.Sv.groupby('add_idx').apply(remove_n)
        Sv_clean.name = 'Sv_clean'

        # Save as a netCDF file
        print('%s  saving denoised Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_clean_path))
        Sv_clean.to_netcdf(self.Sv_clean_path)
        proc_data.close()

    def noise_estimates(self, noise_est_range_bin_size=None, noise_est_ping_size=None):
        """
        Obtain noise estimates from the minimum mean calibrated power level along each column of tiles.

        The tiles here are defined by class attributes noise_est_range_bin_size and noise_est_ping_size.
        This method contains redundant pieces of code that also appear in method remove_noise(),
        but this method can be used separately to determine the exact tile size for noise removal before
        noise removal is actually performed.

        Parameters
        ------------
        noise_est_range_bin_size : float
            meters per tile for noise estimation [m]
        noise_est_ping_size : int
            number of pings per tile for noise estimation

        Returns
        ---------
        noise_est : float
            noise estimates as a numpy array with dimension [ping_bin x frequency]
            ping_bin is the number of tiles along ping_time calculated from attributes noise_est_ping_size
        """

        # Check params
        if (noise_est_range_bin_size is not None) and (self.noise_est_range_bin_size != noise_est_range_bin_size):
            self.noise_est_range_bin_size = noise_est_range_bin_size
        if (noise_est_ping_size is not None) and (self.noise_est_ping_size != noise_est_ping_size):
            self.noise_est_ping_size = noise_est_ping_size

        # Open data set for Environment and Beam groups
        ds_beam = xr.open_dataset(self.file_path, group="Beam")

        # Use calibrated data to calculate noise removal
        proc_data = xr.open_dataset(self.Sv_path)

        # Adjust noise_est_range_bin_size because range_bin_size may be an inconvenient value
        num_range_bin_per_tile = (np.round(self.noise_est_range_bin_size /
                                           self.sample_thickness).astype(int)).values.max()  # num of range_bin per tile
        self.noise_est_range_bin_size = (num_range_bin_per_tile * self.sample_thickness).values

        # Number of tiles along range_bin
        if np.mod(ds_beam.range_bin.size, num_range_bin_per_tile) == 0:
            num_tile_range_bin = np.ceil(ds_beam.range_bin.size / num_range_bin_per_tile).astype(int) + 1
        else:
            num_tile_range_bin = np.ceil(ds_beam.range_bin.size / num_range_bin_per_tile).astype(int)

        # Produce a new coordinate for groupby operation
        if np.mod(ds_beam.ping_time.size, self.noise_est_ping_size) == 0:
            num_tile_ping = np.ceil(ds_beam.ping_time.size / self.noise_est_ping_size).astype(int) + 1
            z = np.array([np.arange(num_tile_ping - 1)] * self.noise_est_ping_size).squeeze().T.ravel()
        else:
            num_tile_ping = np.ceil(ds_beam.ping_time.size / self.noise_est_ping_size).astype(int)
            pad = np.ones(ds_beam.ping_time.size - (num_tile_ping - 1) * self.noise_est_ping_size, dtype=int) \
                  * (num_tile_ping - 1)
            z = np.hstack(
                (np.array([np.arange(num_tile_ping - 1)] * self.noise_est_ping_size).squeeze().T.ravel(), pad))

        # Tile bin edges along range
        # ... -1 to make sure each bin has the same size because of the right-inclusive and left-exclusive bins
        range_bin_tile_bin = np.arange(num_tile_range_bin+1) * num_range_bin_per_tile - 1

        # Get noise estimates
        proc_data.coords['add_idx'] = ('ping_time', z)
        proc_data.Sv.groupby('add_idx').groupby_bins('range_bin', range_bin_tile_bin)

        # Function for use with apply
        noise_est = []

        def est_n(x):
            p_c_lin = 10 ** ((x - self.ABS - self.TVG) / 10)
            nn = 10 * np.log10(p_c_lin.groupby_bins('range_bin', range_bin_tile_bin).mean('range_bin').
                               groupby('frequency').mean('ping_time').min(dim='range_bin_bins'))
            noise_est.append(nn.values)
            return x

        # Groupby operation for estimating noise
        proc_data.coords['add_idx'] = ('ping_time', z)
        _ = proc_data.Sv.groupby('add_idx').apply(est_n)

        # Close opened resources
        ds_beam.close()

        return noise_est

    def get_MVBS(self, source='Sv', MVBS_range_bin_size=None, MVBS_ping_size=None):
        """Calculate Mean Volume Backscattering Strength (MVBS).

        The calculation uses class attributes MVBS_ping_size and MVBS_range_bin_size.

        Parameters
        ------------
        source : str
            source used to calculate MVBS, can be ``Sv`` or ``Sv_clean``,
            where ``Sv`` and ``Sv_clean`` are the original and denoised volume
            backscattering strengths, respectively. Both are calibrated.
        MVBS_range_bin_size : float
            meters per tile for calculating MVBS [m]
        MVBS_ping_size : int
            number of pings per tile for calculating MVBS

        Returns
        ---------
        MVBS : xarray DataArray
            MVBS has dimensions [ping_bin and range_bin_bin]
            range_bin_bin is the number of tiles along range_bin calculated from attributes MVBS_range_bin_size
            ping_bin is the number of tiles along ping_time calculated from attributes MVBS_ping_size
        """
        # Check params
        if (MVBS_range_bin_size is not None) and (self.MVBS_range_bin_size != MVBS_range_bin_size):
            self.MVBS_range_bin_size = MVBS_range_bin_size
        if (MVBS_ping_size is not None) and (self.MVBS_ping_size != MVBS_ping_size):
            self.MVBS_ping_size = MVBS_ping_size

        # Use calibrated data to calculate noise removal
        if source == 'Sv':
            proc_data = xr.open_dataset(self.Sv_path)
        elif source == 'Sv_clean':
            if self.Sv_clean is not None:
                proc_data = xr.open_dataset(self.Sv_clean_path)
            else:
                raise ValueError('Need to obtain Sv_clean first by calling remove_noise()')
        else:
            raise ValueError('Unknown source, cannot calculate MVBS')

        # Adjust noise_est_range_bin_size because range_bin_size may be an inconvenient value
        num_range_bin_per_tile = (np.round(self.MVBS_range_bin_size /
                                           self.sample_thickness).astype(int)).values.max()  # num of range_bin per tile
        self.MVBS_range_bin_size = (num_range_bin_per_tile * self.sample_thickness).values

        # Number of tiles along range_bin
        if np.mod(proc_data.range_bin.size, num_range_bin_per_tile) == 0:
            num_tile_range_bin = np.ceil(proc_data.range_bin.size / num_range_bin_per_tile).astype(int) + 1
        else:
            num_tile_range_bin = np.ceil(proc_data.range_bin.size / num_range_bin_per_tile).astype(int)

        # Produce a new coordinate for groupby operation
        if np.mod(proc_data.ping_time.size, self.MVBS_ping_size) == 0:
            num_tile_ping = np.ceil(proc_data.ping_time.size / self.MVBS_ping_size).astype(int) + 1
            z = np.array([np.arange(num_tile_ping - 1)] * self.MVBS_ping_size).squeeze().T.ravel()
        else:
            num_tile_ping = np.ceil(proc_data.ping_time.size / self.MVBS_ping_size).astype(int)
            pad = np.ones(proc_data.ping_time.size - (num_tile_ping - 1) * self.MVBS_ping_size, dtype=int) \
                  * (num_tile_ping - 1)
            z = np.hstack(
                (np.array([np.arange(num_tile_ping - 1)] * self.MVBS_ping_size).squeeze().T.ravel(), pad))

        # Tile bin edges along range
        # ... -1 to make sure each bin has the same size because of the right-inclusive and left-exclusive bins
        range_bin_tile_bin = np.arange(num_tile_range_bin+1) * num_range_bin_per_tile - 1

        # Calculate MVBS
        proc_data.coords['add_idx'] = ('ping_time', z)
        MVBS = proc_data.Sv.groupby('add_idx').mean('ping_time').\
            groupby_bins('range_bin', range_bin_tile_bin).mean(['range_bin'])

        # Set MVBS coordinates
        ping_time = proc_data.ping_time[list(map(lambda x: x[0], list(proc_data.ping_time.groupby('add_idx').groups.values())))]
        MVBS.coords['ping_time'] = ('add_idx', ping_time)
        MVBS.coords['ping_time'] = ('add_idx', ping_time)
        range_bin = list(map(lambda x: x[0], list(proc_data.range_bin.
                                                       groupby_bins('range_bin', range_bin_tile_bin).groups.values())))
        MVBS.coords['range_bin'] = ('range_bin_bins', range_bin)
        MVBS.swap_dims({'range_bin_bins': 'range_bin'})

        # Save results in object and as a netCDF file
        print('%s  saving MVBS to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.MVBS_path))
        self.MVBS = MVBS
        MVBS.to_netcdf(self.MVBS_path)
        proc_data.close()
