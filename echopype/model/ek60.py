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
            self.Sv_clean_path = os.path.join(os.path.dirname(self.file_path),
                                              os.path.splitext(os.path.basename(self.file_path))[0] + '_Sv_clean.nc')
            self.MVBS_path = os.path.join(os.path.dirname(self.file_path),
                                          os.path.splitext(os.path.basename(self.file_path))[0] + '_MVBS.nc')

            # Raise error if the file format convention does not match
            if self.toplevel.sonar_convention_name != 'SONAR-netCDF4':
                raise ValueError('netCDF file convention not recognized.')
        else:
            raise ValueError('Data file format not recognized.')

    def calibrate(self, save=True):
        """Perform echo-integration to get volume backscattering strength (Sv) from EK60 power data.

        Parameters
        -----------
        save : bool, optional
            whether to save calibrated Sv output
            default to ``True``
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

        # Save calibrated data into the calling instance and
        # ... to a separate .nc file in the same directory as the data file
        self.Sv = Sv
        if save:
            print('%s  saving calibrated Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
            Sv.to_netcdf(path=self.Sv_path, mode="w")

        # Close opened resources
        ds_env.close()
        ds_beam.close()

    @staticmethod
    def get_tile_params(r_data_sz, p_data_sz, r_tile_sz, p_tile_sz, sample_thickness):
        """Obtain ping_time and range_bin parameters associated with groupby and groupby_bins operations.

        These parameters are used in methods remove_noise(), noise_estimates(), get_MVBS().

        Parameters
        -----------
        r_data_sz : int
            number of range_bin entries in data
        p_data_sz : int
            number of ping_time entries in data
        r_tile_sz : float
            tile size along the range_bin dimension [m]
        p_tile_sz : int
            tile size along the ping_time dimension [number of pings]
        sample_thickness : float
            thickness of each data sample, determined by sound speed and pulse duration

        Returns
        --------
        r_tile_sz : int
            modified tile size along the range dimension [m], determined by sample_thickness
        p_idx : list of int
            indices along the ping_time dimension for :py:func:`xarray.DataArray.groupby` operation
        r_tile_bin_edge : list of int
            bin edges along the range_bin dimension for :py:func:`xarray.DataArray.groupby_bins` operation
        """
        # Adjust noise_est_range_bin_size because range_bin_size may be an inconvenient value
        num_r_per_tile = (np.round(r_tile_sz / sample_thickness).astype(int)).values.max()  # num of range_bin per tile
        r_tile_sz = (num_r_per_tile * sample_thickness).values

        # Number of tiles along range_bin
        if np.mod(r_data_sz, num_r_per_tile) == 0:
            num_tile_range_bin = np.ceil(r_data_sz / num_r_per_tile).astype(int) + 1
        else:
            num_tile_range_bin = np.ceil(r_data_sz / num_r_per_tile).astype(int)

        # Produce a new coordinate for groupby operation
        if np.mod(p_data_sz, p_tile_sz) == 0:
            num_tile_ping = np.ceil(p_data_sz / p_tile_sz).astype(int) + 1
            p_idx = np.array([np.arange(num_tile_ping - 1)] * p_tile_sz).squeeze().T.ravel()
        else:
            num_tile_ping = np.ceil(p_data_sz / p_tile_sz).astype(int)
            pad = np.ones(p_data_sz - (num_tile_ping - 1) * p_tile_sz, dtype=int) \
                  * (num_tile_ping - 1)
            p_idx = np.hstack(
                (np.array([np.arange(num_tile_ping - 1)] * p_tile_sz).squeeze().T.ravel(), pad))

        # Tile bin edges along range
        # ... -1 to make sure each bin has the same size because of the right-inclusive and left-exclusive bins
        r_tile_bin_edge = np.arange(num_tile_range_bin+1) * num_r_per_tile - 1

        return r_tile_sz, p_idx, r_tile_bin_edge

    def remove_noise(self, noise_est_range_bin_size=None, noise_est_ping_size=None, save=False):
        """Remove noise by using noise estimates obtained from the minimum mean calibrated power level
        along each column of tiles.

        See method noise_estimates() for details of noise estimation.
        Reference: De Robertis & Higginbottom, 2017, ICES Journal of Marine Sciences

        Parameters
        ------------
        noise_est_range_bin_size : float, optional
            meters per tile for noise estimation [m]
        noise_est_ping_size : int, optional
            number of pings per tile for noise estimation
        save : bool, optional
            whether to save the denoised Sv (``Sv_clean``) into a new .nc file
            default to ``False``
        """

        # Check params
        if (noise_est_range_bin_size is not None) and (self.noise_est_range_bin_size != noise_est_range_bin_size):
            self.noise_est_range_bin_size = noise_est_range_bin_size
        if (noise_est_ping_size is not None) and (self.noise_est_ping_size != noise_est_ping_size):
            self.noise_est_ping_size = noise_est_ping_size

        # Get calibrated power
        proc_data = xr.open_dataset(self.Sv_path)

        # Get tile indexing parameters
        self.noise_est_range_bin_size, add_idx, range_bin_tile_bin_edge = \
            self.get_tile_params(r_data_sz=proc_data.range_bin.size,
                                 p_data_sz=proc_data.ping_time.size,
                                 r_tile_sz=self.noise_est_range_bin_size,
                                 p_tile_sz=self.noise_est_ping_size,
                                 sample_thickness=self.sample_thickness)

        # Function for use with apply
        def remove_n(x):
            p_c_lin = 10 ** ((x - self.ABS - self.TVG) / 10)
            nn = 10 * np.log10(p_c_lin.groupby_bins('range_bin', range_bin_tile_bin_edge).mean('range_bin').
                               groupby('frequency').mean('ping_time').min(dim='range_bin_bins')) \
                 + self.ABS + self.TVG
            return x.where(x > nn, other=np.nan)

        # Groupby noise removal operation
        proc_data.coords['add_idx'] = ('ping_time', add_idx)
        Sv_clean = proc_data.Sv.groupby('add_idx').apply(remove_n)

        # Set up DataArray
        Sv_clean.name = 'Sv_clean'
        Sv_clean = Sv_clean.drop('add_idx')
        Sv_clean = Sv_clean.to_dataset()
        Sv_clean['noise_est_range_bin_size'] = ('frequency', self.noise_est_range_bin_size)
        Sv_clean.attrs['noise_est_ping_size'] = self.noise_est_ping_size
        Sv_clean['sample_thickness'] = ('frequency', self.sample_thickness)

        # Save as object attributes as a netCDF file
        self.Sv_clean = Sv_clean
        if save:
            print('%s  saving denoised Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_clean_path))
            Sv_clean.to_netcdf(self.Sv_clean_path)

        # Close opened resources
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
        noise_est : xarray DataSet
            noise estimates as a DataArray with dimension [ping_time x range_bin]
            ping_time and range_bin are taken from the first element of each tile along each of the dimensions
        """

        # Check params
        if (noise_est_range_bin_size is not None) and (self.noise_est_range_bin_size != noise_est_range_bin_size):
            self.noise_est_range_bin_size = noise_est_range_bin_size
        if (noise_est_ping_size is not None) and (self.noise_est_ping_size != noise_est_ping_size):
            self.noise_est_ping_size = noise_est_ping_size

        # Use calibrated data to calculate noise removal
        proc_data = xr.open_dataset(self.Sv_path)

        # Get tile indexing parameters
        self.noise_est_range_bin_size, add_idx, range_bin_tile_bin_edge = \
            self.get_tile_params(r_data_sz=proc_data.range_bin.size,
                                 p_data_sz=proc_data.ping_time.size,
                                 r_tile_sz=self.noise_est_range_bin_size,
                                 p_tile_sz=self.noise_est_ping_size,
                                 sample_thickness=self.sample_thickness)

        # Noise estimates
        proc_data['power_cal'] = 10 ** ((proc_data.Sv - self.ABS - self.TVG) / 10)
        proc_data.coords['add_idx'] = ('ping_time', add_idx)
        noise_est = 10 * np.log10(proc_data.power_cal.groupby('add_idx').mean('ping_time').
                                  groupby_bins('range_bin', range_bin_tile_bin_edge).mean(['range_bin']).
                                  min('range_bin_bins'))

        # Set noise estimates coordinates and other attributes
        ping_time = proc_data.ping_time[list(map(lambda x: x[0],
                                                 list(proc_data.ping_time.groupby('add_idx').groups.values())))]
        noise_est.coords['ping_time'] = ('add_idx', ping_time)
        noise_est = noise_est.swap_dims({'add_idx': 'ping_time'}).drop({'add_idx'})
        noise_est = noise_est.to_dataset(name='noise_est')
        noise_est['noise_est_range_bin_size'] = ('frequency', self.noise_est_range_bin_size)
        noise_est.attrs['noise_est_ping_size'] = self.noise_est_ping_size
        noise_est['sample_thickness'] = ('frequency', self.sample_thickness)

        # Close opened resources
        proc_data.close()

        return noise_est

    def get_MVBS(self, source='Sv', MVBS_range_bin_size=None, MVBS_ping_size=None, save=False):
        """Calculate Mean Volume Backscattering Strength (MVBS).

        The calculation uses class attributes MVBS_ping_size and MVBS_range_bin_size to
        calculate and save MVBS as a new attribute to the calling EchoData instance.
        MVBS is an xarray DataArray with dimensions ``ping_time`` and ``range_bin``
        that are from the first elements of each tile along the corresponding dimensions
        in the original Sv or Sv_clean DataArray.

        Parameters
        ------------
        source : str
            source used to calculate MVBS, can be ``Sv`` or ``Sv_clean``,
            where ``Sv`` and ``Sv_clean`` are the original and denoised volume
            backscattering strengths, respectively. Both are calibrated.
        MVBS_range_bin_size : float, optional
            meters per tile for calculating MVBS [m]
        MVBS_ping_size : int, optional
            number of pings per tile for calculating MVBS
        save : bool, optional
            whether to save the denoised Sv (``Sv_clean``) into a new .nc file
            default to ``False``
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

        # Get tile indexing parameters
        self.MVBS_range_bin_size, add_idx, range_bin_tile_bin_edge = \
            self.get_tile_params(r_data_sz=proc_data.range_bin.size,
                                 p_data_sz=proc_data.ping_time.size,
                                 r_tile_sz=self.MVBS_range_bin_size,
                                 p_tile_sz=self.MVBS_ping_size,
                                 sample_thickness=self.sample_thickness)

        # Calculate MVBS
        proc_data.coords['add_idx'] = ('ping_time', add_idx)
        MVBS = proc_data.Sv.groupby('add_idx').mean('ping_time').\
            groupby_bins('range_bin', range_bin_tile_bin_edge).mean(['range_bin'])

        # Set MVBS coordinates
        ping_time = proc_data.ping_time[list(map(lambda x: x[0],
                                                 list(proc_data.ping_time.groupby('add_idx').groups.values())))]
        MVBS.coords['ping_time'] = ('add_idx', ping_time)
        range_bin = list(map(lambda x: x[0], list(proc_data.range_bin.
                                                  groupby_bins('range_bin', range_bin_tile_bin_edge).groups.values())))
        MVBS.coords['range_bin'] = ('range_bin_bins', range_bin)
        MVBS = MVBS.swap_dims({'range_bin_bins': 'range_bin', 'add_idx': 'ping_time'}).\
            drop({'add_idx', 'range_bin_bins'})

        # Set MVBS attributes
        MVBS.name = 'MVBS'
        MVBS = MVBS.to_dataset()
        MVBS['noise_est_range_bin_size'] = ('frequency', self.MVBS_range_bin_size)
        MVBS.attrs['noise_est_ping_size'] = self.MVBS_ping_size
        MVBS['sample_thickness'] = ('frequency', self.sample_thickness)

        # Save results in object and as a netCDF file
        self.MVBS = MVBS
        if save:
            print('%s  saving MVBS to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.MVBS_path))
            MVBS.to_netcdf(self.MVBS_path)

        # Close opened resources
        proc_data.close()
