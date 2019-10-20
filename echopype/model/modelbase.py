"""
echopype data model that keeps tracks of echo data and
its connection to data files.
"""

import os
import warnings
import datetime as dt

import numpy as np
import xarray as xr


class ModelBase(object):
    """Class for manipulating echo data that is already converted to netCDF."""

    def __init__(self, file_path=""):
        self.file_path = file_path  # this passes the input through file name test
        self.noise_est_range_bin_size = 5  # meters per tile for noise estimation
        self.noise_est_ping_size = 30  # number of pings per tile for noise estimation
        self.MVBS_range_bin_size = 5  # meters per tile for MVBS
        self.MVBS_ping_size = 30  # number of pings per tile for MVBS
        self.TVG = None  # time varying gain along range
        self.ABS = None  # absorption along range
        self.Sv = None  # calibrated volume backscattering strength
        self.Sv_clean = None  # denoised volume backscattering strength
        self.TS = None  # calibrated target strength
        self.MVBS = None  # mean volume backscattering strength
        self._sample_thickness = None
        self._range = None

    # TODO: Set noise_est_range_bin_size, noise_est_ping_size,
    #  MVBS_range_bin_size, and MVBS_ping_size all to be properties
    #  and provide getter/setter

    @property
    def sample_thickness(self):
        if self._sample_thickness is None:  # if this is empty
            self._sample_thickness = self.calc_sample_thickness()
        return self._sample_thickness

    @sample_thickness.setter
    def sample_thickness(self, sth):
        self._sample_thickness = sth

    @property
    def range(self):
        if self._range is None:  # if this is empty
            self._range = self.calc_range()
        return self._range

    @range.setter
    def range(self, rr):
        self._range = rr

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, p):
        self._file_path = p

        # Load netCDF groups if file format is correct
        pp = os.path.basename(p)
        _, ext = os.path.splitext(pp)

        supported_ext_list = ['.raw', '.01A']
        if ext in supported_ext_list:
            print('Data file in manufacturer format, please convert to .nc first.')
        elif ext == '.nc':
            self.toplevel = xr.open_dataset(self.file_path)

            # Get .nc filenames for storing processed data if computation is performed
            self.Sv_path = os.path.join(os.path.dirname(self.file_path),
                                        os.path.splitext(os.path.basename(self.file_path))[0] + '_Sv.nc')
            self.Sv_clean_path = os.path.join(os.path.dirname(self.file_path),
                                              os.path.splitext(os.path.basename(self.file_path))[0] + '_Sv_clean.nc')
            self.TS_path = os.path.join(os.path.dirname(self.file_path),
                                        os.path.splitext(os.path.basename(self.file_path))[0] + '_TS.nc')
            self.MVBS_path = os.path.join(os.path.dirname(self.file_path),
                                          os.path.splitext(os.path.basename(self.file_path))[0] + '_MVBS.nc')
            # Raise error if the file format convention does not match
            if self.toplevel.sonar_convention_name != 'SONAR-netCDF4':
                raise ValueError('netCDF file convention not recognized.')
            self.toplevel.close()
        else:
            raise ValueError('Data file format not recognized.')

    def calc_sample_thickness(self):
        """Base method to be overridden for calculating sample_thickness for different sonar models.
        """
        # issue warning when subclass methods not available
        print('Sample thickness calculation has not been implemented for this sonar model!')

    def calc_range(self):
        """Base method to be overridden for calculating range for different sonar models.
        """
        # issue warning when subclass methods not available
        print('Range calculation has not been implemented for this sonar model!')

    def calibrate(self):
        """Base method to be overridden for calibration and echo-integration for different sonar models.
        """
        # issue warning when subclass methods not available
        print('Calibration has not been implemented for this sonar model!')

    @staticmethod
    def get_tile_params(r_data_sz, p_data_sz, r_tile_sz, p_tile_sz, sample_thickness):
        """Obtain ping_time and range_bin parameters associated with groupby and groupby_bins operations.

        These parameters are used in methods remove_noise(), noise_estimates(), get_MVBS().

        Parameters
        ----------
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
        -------
        r_tile_sz : int
            modified tile size along the range dimension [m], determined by sample_thickness
        p_idx : list of int
            indices along the ping_time dimension for :py:func:`xarray.DataArray.groupby` operation
        r_tile_bin_edge : list of int
            bin edges along the range_bin dimension for :py:func:`xarray.DataArray.groupby_bins` operation
        """
        # TODO: Need to make this compatible with the possibly different sample_thickness
        #  for each frequency channel. The difference will show up in num_r_per_tile and
        #  propagates down to r_tile_sz, num_tile_range_bin, and r_tile_bin_edge; all of
        #  these will also become a DataArray or a list of list instead of just a number
        #  or a list of numbers.

        # Adjust noise_est_range_bin_size because range_bin_size may be an inconvenient value
        num_r_per_tile = (np.round(r_tile_sz / sample_thickness).astype(int)).values.max()  # num of range_bin per tile
        r_tile_sz = (num_r_per_tile * sample_thickness).values

        # TODO: double check this, but edits from @cyrf0006 seems correct
        num_tile_range_bin = np.ceil(r_data_sz / num_r_per_tile).astype(int)
        # Number of tiles along range_bin <--- old routine
        # if np.mod(r_data_sz, num_r_per_tile) == 0:
        #     num_tile_range_bin = np.ceil(r_data_sz / num_r_per_tile).astype(int) + 1
        # else:
        #     num_tile_range_bin = np.ceil(r_data_sz / num_r_per_tile).astype(int)

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

    def _get_proc_Sv(self):
        """Private method to return calibrated Sv either from memory or _Sv.nc file.

        This method is called by remove_noise(), noise_estimates() and get_MVBS().
        """
        if self.Sv is None:  # if don't have Sv as attribute
            if os.path.exists(self.Sv_path):  # but have _Sv.nc file
                self.Sv = xr.open_dataset(self.Sv_path)   # load file
            else:  # if also don't have _Sv.nc file
                print('Data has not been calibrated. Performing calibration now.')
                self.calibrate()  # then calibrate
        return self.Sv

    def _get_proc_Sv_clean(self):
        """Private method to return calibrated Sv_clean either from memory or _Sv_clean.nc file.

        This method is called get_MVBS().
        """
        if self.Sv_clean is None:  # if don't have Sv_clean as attribute
            if os.path.exists(self.Sv_clean_path):  # but have _Sv_clean.nc file
                self.Sv_clean = xr.open_dataset(self.Sv_clean_path)  # load file
            else:  # if also don't have _Sv_clean.nc file
                if self.Sv is None:   # if hasn't performed calibration yet
                    print('Data has not been calibrated. Performing calibration now.')
                    self.calibrate()      # then calibrate
                print('Noise removal has not been performed. Performing noise removal now.')
                self.remove_noise()   # and then remove noise
        return self.Sv_clean  # and point to results

    def remove_noise(self, noise_est_range_bin_size=None, noise_est_ping_size=None,
                     SNR=0, Sv_threshold=None, save=False):
        """Remove noise by using noise estimates obtained from the minimum mean calibrated power level
        along each column of tiles.

        See method noise_estimates() for details of noise estimation.
        Reference: De Robertis & Higginbottom, 2017, ICES Journal of Marine Sciences

        Parameters
        ----------
        noise_est_range_bin_size : float, optional
            Meters per tile for noise estimation [m]
        noise_est_ping_size : int, optional
            Number of pings per tile for noise estimation
        SNR : int, optional
            Minimum signal-to-noise ratio (remove values below this after general noise removal).
        Sv_threshold : int, optional
            Minimum Sv threshold [dB] (remove values below this after general noise removal)
        save : bool, optional
            Whether to save the denoised Sv (``Sv_clean``) into a new .nc file.
            Default to ``False``.
        """

        # Check params
        if (noise_est_range_bin_size is not None) and (self.noise_est_range_bin_size != noise_est_range_bin_size):
            self.noise_est_range_bin_size = noise_est_range_bin_size
        if (noise_est_ping_size is not None) and (self.noise_est_ping_size != noise_est_ping_size):
            self.noise_est_ping_size = noise_est_ping_size

        # Get calibrated power
        proc_data = self._get_proc_Sv()

        # Get tile indexing parameters
        self.noise_est_range_bin_size, add_idx, range_bin_tile_bin_edge = \
            self.get_tile_params(r_data_sz=proc_data.range_bin.size,
                                 p_data_sz=proc_data.ping_time.size,
                                 r_tile_sz=self.noise_est_range_bin_size,
                                 p_tile_sz=self.noise_est_ping_size,
                                 sample_thickness=self.sample_thickness)

        # TODO: this right now will break when _get_proc_Sv() gets self.Sv from file.
        #  This is also why the calculation of ABS and TVG should be in the parent
        #  class methods instead of being done under calibration() in the child class
        ABS = self.ABS
        TVG = self.TVG

        # Function for use with apply
        def remove_n(x):
            p_c_lin = 10 ** ((x - ABS - TVG) / 10)
            nn = 10 * np.log10(p_c_lin.groupby_bins('range_bin', range_bin_tile_bin_edge).mean('range_bin').
                               groupby('frequency').mean('ping_time').min(dim='range_bin_bins')) \
                 + ABS + TVG
            # Return values where signal is [SNR] dB above noise and at least [Sv_threshold] dB
            if not Sv_threshold:
                return x.where(x > (nn + SNR), other=np.nan)
            else:
                return x.where((x > (nn + SNR)) & (x > Sv_threshold), other=np.nan)

            # # Noise calculation
            # if (self.ABS is None) & (self.TVG is None):
            #     p_c_lin = 10 ** (x / 10)
            #     nn = 10 * np.log10(p_c_lin.groupby_bins('range_bin', range_bin_tile_bin_edge).mean('range_bin').
            #                        groupby('frequency').mean('ping_time').min(dim='range_bin_bins'))
            # else:
            #     p_c_lin = 10 ** ((x - self.ABS - self.TVG) / 10)
            #     nn = 10 * np.log10(p_c_lin.groupby_bins('range_bin', range_bin_tile_bin_edge).mean('range_bin').
            #                        groupby('frequency').mean('ping_time').min(dim='range_bin_bins')) \
            #          + self.ABS + self.TVG

        # Groupby noise removal operation
        proc_data.coords['add_idx'] = ('ping_time', add_idx)
        Sv_clean = proc_data.Sv.groupby('add_idx').apply(remove_n)

        # Set up DataSet
        Sv_clean.name = 'Sv_clean'
        Sv_clean = Sv_clean.drop('add_idx')
        Sv_clean = Sv_clean.to_dataset()
        Sv_clean['noise_est_range_bin_size'] = ('frequency', self.noise_est_range_bin_size)
        Sv_clean.attrs['noise_est_ping_size'] = self.noise_est_ping_size

        # Attach calculated range into data set
        Sv_clean['range'] = (('frequency', 'range_bin'), self.range.T)

        # Save as object attributes as a netCDF file
        self.Sv_clean = Sv_clean
        if save:
            print('%s  saving denoised Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_clean_path))
            Sv_clean.to_netcdf(self.Sv_clean_path)

        # Close opened resources
        proc_data.close()

    def noise_estimates(self, noise_est_range_bin_size=None, noise_est_ping_size=None):
        """Obtain noise estimates from the minimum mean calibrated power level along each column of tiles.

        The tiles here are defined by class attributes noise_est_range_bin_size and noise_est_ping_size.
        This method contains redundant pieces of code that also appear in method remove_noise(),
        but this method can be used separately to determine the exact tile size for noise removal before
        noise removal is actually performed.

        Parameters
        ----------
        noise_est_range_bin_size : float
            meters per tile for noise estimation [m]
        noise_est_ping_size : int
            number of pings per tile for noise estimation

        Returns
        -------
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
        proc_data = self._get_proc_Sv()

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
                                  groupby_bins('range_bin', range_bin_tile_bin_edge).mean('range_bin').
                                  min('range_bin_bins'))

        # Set noise estimates coordinates and other attributes
        ping_time = proc_data.ping_time[list(map(lambda x: x[0],
                                                 list(proc_data.ping_time.groupby('add_idx').groups.values())))]
        noise_est.coords['ping_time'] = ('add_idx', ping_time)
        noise_est = noise_est.swap_dims({'add_idx': 'ping_time'}).drop({'add_idx'})
        noise_est = noise_est.to_dataset(name='noise_est')
        noise_est['noise_est_range_bin_size'] = ('frequency', self.noise_est_range_bin_size)
        noise_est.attrs['noise_est_ping_size'] = self.noise_est_ping_size

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
        ----------
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
        # TODO: Not sure what @cyrf0006 meant below, but need to resolve the issues surrounding
        #  potentially having different sample_thickness for each frequency. This is the same
        #  issue that needs to be resolved in ``get_tile_params`` and all calling methods.
        #  --- Below are comments from @cyfr0006 ---
        #  -FC here problem because self.MVBS_range_bin_size is size 4 while MVBS_range_bin_size is size 1
        #  if (MVBS_range_bin_size is not None) and (self.MVBS_range_bin_size != MVBS_range_bin_size):
        if (MVBS_range_bin_size is not None) and (self.MVBS_range_bin_size != MVBS_range_bin_size):
            self.MVBS_range_bin_size = MVBS_range_bin_size
        if (MVBS_ping_size is not None) and (self.MVBS_ping_size != MVBS_ping_size):
            self.MVBS_ping_size = MVBS_ping_size

        # Use calibrated data to calculate noise removal
        if source == 'Sv':
            proc_data = self._get_proc_Sv()
        elif source == 'Sv_clean':
            proc_data = self._get_proc_Sv_clean()
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
        if source == 'Sv':
            MVBS = proc_data.Sv.groupby('add_idx').mean('ping_time').\
                groupby_bins('range_bin', range_bin_tile_bin_edge).mean('range_bin')
        elif source == 'Sv_clean':
            # TODO: the calculation below issues warnings when encountering all NaN slices.
            #  This is an open issue in dask: https://github.com/dask/dask/issues/3245
            #  Suppress this warning for now.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                MVBS = proc_data.Sv_clean.groupby('add_idx').mean('ping_time').\
                    groupby_bins('range_bin', range_bin_tile_bin_edge).mean('range_bin')
        else:
            raise ValueError('Unknown source, cannot calculate MVBS')

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
        MVBS['MVBS_range_bin_size'] = ('frequency', self.MVBS_range_bin_size)
        MVBS.attrs['MVBS_ping_size'] = self.MVBS_ping_size

        # Attach calculated range to MVBS
        MVBS['range'] = self.Sv.range.sel(range_bin=MVBS.range_bin)

        # TODO: need to save noise_est_range_bin_size and noise_est_ping_size if source='Sv_clean'
        #  and also save an additional attribute that specifies the source
        # MVBS.attrs['noise_est_range_bin_size'] = self.noise_est_range_bin_size
        # MVBS.attrs['noise_est_ping_size'] = self.noise_est_ping_size

        # Drop add_idx added to Sv
        # TODO: somehow this still doesn't work and self.Sv or self.Sv_clean
        #  will have this additional dimension attached
        proc_data = proc_data.drop('add_idx')

        # Save results in object and as a netCDF file
        self.MVBS = MVBS
        if save:
            print('%s  saving MVBS to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.MVBS_path))
            MVBS.to_netcdf(self.MVBS_path)

        # Close opened resources
        proc_data.close()
