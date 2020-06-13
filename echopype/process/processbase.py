"""
echopype data model that keeps tracks of echo data and
its connection to data files.
"""

import os
import warnings
import datetime as dt
from echopype.utils import uwa

import numpy as np
import xarray as xr
import zarr


class ProcessBase(object):
    """Class for manipulating echo data that is already converted to netCDF."""

    def __init__(self, file_path=""):
        self.file_path = file_path  # this passes the input through file name test
        self.noise_est_range_bin_size = 5  # meters per tile for noise estimation
        self.noise_est_ping_size = 30  # number of pings per tile for noise estimation
        self.MVBS_range_bin_size = 5  # meters per tile for MVBS
        self.MVBS_ping_size = 30  # number of pings per tile for MVBS
        self.Sv = None            # calibrated volume backscattering strength
        self.Sv_path = None       # path to save calibrated results
        self.Sv_clean = None      # denoised volume backscattering strength
        self.TS = None            # calibrated target strength
        self.TS_path = None       # path to save TS calculation results
        self.MVBS = None          # mean volume backscattering strength
        self._file_format = None
        self._open_dataset = None
        self._salinity = None
        self._temperature = None
        self._pressure = None
        self._sound_speed = None
        self._sample_thickness = None
        self._range = None
        self._seawater_absorption = None

        self._set_file_format()
        self._set_open_dataset()

    @property
    def salinity(self):
        return self._salinity

    @salinity.setter
    def salinity(self, sal):
        self._salinity = sal

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, pres):
        self._pressure = pres

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, t):
        self._temperature = t

    @property
    def sample_thickness(self):
        return self._sample_thickness

    @sample_thickness.setter
    def sample_thickness(self, sth):
        self._sample_thickness = sth

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, rr):
        self._range = rr

    @property
    def seawater_absorption(self):
        return self._seawater_absorption

    @seawater_absorption.setter
    def seawater_absorption(self, absorption):
        self._seawater_absorption.values = absorption

    @property
    def sound_speed(self):
        return self._sound_speed

    @sound_speed.setter
    def sound_speed(self, ss):
        if isinstance(self._sound_speed, xr.DataArray):
            self._sound_speed.values = ss
        else:
            self._sound_speed = ss

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
        elif ext == '.zarr':
            self.toplevel = zarr.open(self._file_path)

            # Get .zarr filenames for storing processed data if computation is performed
            self.Sv_path = os.path.join(os.path.dirname(self.file_path),
                                        os.path.splitext(os.path.basename(self.file_path))[0] + '_Sv.zarr')
            self.Sv_clean_path = os.path.join(os.path.dirname(self.file_path),
                                              os.path.splitext(os.path.basename(self.file_path))[0] + '_Sv_clean.zarr')
            self.TS_path = os.path.join(os.path.dirname(self.file_path),
                                        os.path.splitext(os.path.basename(self.file_path))[0] + '_TS.zarr')
            self.MVBS_path = os.path.join(os.path.dirname(self.file_path),
                                          os.path.splitext(os.path.basename(self.file_path))[0] + '_MVBS.zarr')

            # Raise error if the file format convention does not match
            if self.toplevel.attrs['sonar_convention_name'] != 'SONAR-netCDF4':
                raise ValueError('netCDF file convention not recognized.')
        else:
            raise ValueError('Data file format not recognized.')
    
    def _set_file_format(self):
        if self.file_path.endswith('.nc'):
            self._file_format = 'netcdf'
        elif self.file_path.endswith('.zarr'):
            self._file_format = 'zarr'

    def calc_sound_speed(self, src='file'):
        """Base method to be overridden for calculating sound_speed for different sonar models
        """
        # issue warning when subclass methods not available
        print("Sound speed calculation has not been implemented for this sonar model!")

    def calc_seawater_absorption(self, src='file'):
        """Base method to be overridden for calculating seawater_absorption for different sonar models
        """
        # issue warning when subclass methods not available
        print("Seawater absorption calculation has not been implemented for this sonar model!")

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

    def recalculate_environment(self, ss=True, sa=True, st=True, r=True):
        """ Recalculates sound speed, seawater absorption, sample thickness, and range using
        salinity, temperature, and pressure

        Parameters
        ----------
        ss : bool
            Whether to calcualte sound speed. Defaults to `True`
        sa : bool
            Whether to calcualte seawater absorption. Defaults to `True`
        st : bool
            Whether to calcualte sample thickness. Defaults to `True`
        r : bool
            Whether to calcualte range. Defaults to `True`
        """
        s, t, p = self.salinity, self.temperature, self.pressure
        if s is not None and t is not None and p is not None:
            if ss:
                self.sound_speed = self.calc_sound_speed(src='user')
            if sa:
                self.seawater_absorption = self.calc_seawater_absorption(src='user')
            if st:
                self.sample_thickness = self.calc_sample_thickness()
            if r:
                self.range = self.calc_range()
        elif s is None:
            print("Salinity was not provided. Environment was not recalculated")
        elif t is None:
            print("Temperature was not provided. Environment was not recalculated")
        else:
            print("Pressure was not provided. Environment was not recalculated")

    def calibrate(self):
        """Base method to be overridden for volume backscatter calibration and echo-integration for different sonar models.
        """
        # issue warning when subclass methods not available
        print('Calibration has not been implemented for this sonar model!')

    def calibrate_TS(self):
        """Base method to be overridden for target strength calibration and echo-integration for different sonar models.
        """
        # issue warning when subclass methods not available
        print('Target strength calibration has not been implemented for this sonar model!')

    def validate_path(self, save_path=None, save_postfix='_Sv', file_path=''):
        """Creates a directory if it doesnt exist. Returns a valid save path.
        """
        def _assemble_path():
            file_in = os.path.basename(file_path)
            file_name, file_ext = os.path.splitext(file_in)
            return file_name + save_postfix + file_ext

        file_path = file_path if file_path else self.file_path
        if save_path is None:
            save_dir = os.path.dirname(file_path)
            file_out = _assemble_path()
        else:
            path_ext = os.path.splitext(save_path)[1]
            # If given save_path is file, split into directory and file
            if path_ext != '':
                save_dir, file_out = os.path.split(save_path)
                if save_dir == '':  # save_path is only a filename without directory
                    save_dir = os.path.dirname(file_path)  # use directory from input file
            # If given save_path is a directory, get a filename from input .nc file
            else:
                save_dir = save_path
                file_out = _assemble_path()

        # Create folder if not already exists
        if save_dir == '':
            # TODO: should we use '.' instead of os.getcwd()?
            save_dir = os.getcwd()  # explicit about path to current directory
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        return os.path.join(save_dir, file_out)

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
        r_tile_bin_edge : list of int
            bin edges along the range_bin dimension for :py:func:`xarray.DataArray.groupby_bins` operation
        p_tile_bin_edge : list of int
            bin edges along the ping_time dimension for :py:func:`xarray.DataArray.groupby_bins` operation
        """

        # Adjust noise_est_range_bin_size because range_bin_size may be an inconvenient value
        num_r_per_tile = np.round(r_tile_sz / sample_thickness).astype(int)  # num of range_bin per tile
        r_tile_sz = num_r_per_tile * sample_thickness

        # Total number of range_bin and ping tiles
        num_tile_range_bin = np.ceil(r_data_sz / num_r_per_tile).astype(int)
        if np.mod(p_data_sz, p_tile_sz) == 0:
            num_tile_ping = np.ceil(p_data_sz / p_tile_sz).astype(int) + 1
        else:
            num_tile_ping = np.ceil(p_data_sz / p_tile_sz).astype(int)

        # Tile bin edges along range
        # ... -1 to make sure each bin has the same size because of the right-inclusive and left-exclusive bins
        r_tile_bin_edge = [np.arange(x.values + 1) * y.values - 1 for x, y in zip(num_tile_range_bin, num_r_per_tile)]
        p_tile_bin_edge = np.arange(num_tile_ping + 1) * p_tile_sz - 1

        return r_tile_sz, r_tile_bin_edge, p_tile_bin_edge

    def _get_proc_Sv(self, source_path=None, source_postfix='_Sv'):
        """Private method to return calibrated Sv either from memory or _Sv.nc file.

        This method is called by remove_noise(), noise_estimates() and get_MVBS().
        """
        if self.Sv is None:  # calibration not yet performed
            Sv_path = self.validate_path(save_path=source_path,  # wrangle _Sv path
                                              save_postfix=source_postfix)
            if os.path.exists(Sv_path):  # _Sv exists
                self.Sv = self._open_dataset(Sv_path)  # load _Sv file
            else:
                # if path specification given but file do not exist:
                if (source_path is not None) or (source_postfix != '_Sv'):
                    print('%s  no calibrated data found in specified path: %s' %
                          (dt.datetime.now().strftime('%H:%M:%S'), Sv_path))
                else:
                    print('%s  data has not been calibrated. ' % dt.datetime.now().strftime('%H:%M:%S'))
                print('          performing calibration now and operate from Sv in memory.')
                self.calibrate()  # calibrate, have Sv in memory
        return self.Sv

    def remove_noise(self, source_postfix='_Sv', source_path=None,
                     noise_est_range_bin_size=None, noise_est_ping_size=None,
                     SNR=0, Sv_threshold=None,
                     save=False, save_postfix='_Sv_clean', save_path=None):
        """Remove noise by using noise estimates obtained from the minimum mean calibrated power level
        along each column of tiles.

        See method noise_estimates() for details of noise estimation.
        Reference: De Robertis & Higginbottom, 2007, ICES Journal of Marine Sciences

        Parameters
        ----------
        source_postfix : str
            postfix of the Sv file used to remove noise from, default to '_Sv'
        source_path : str
            path of Sv file used to remove noise from, can be one of the following:
            - None (default):
                    use Sv in RAWFILENAME_Sv.nc in the same folder as the raw data file,
                    or when RAWFILENAME_Sv.nc doesn't exist, perform self.calibrate() and use the resulted self.Sv
            - path to a directory: RAWFILENAME_Sv.nc in the specified directory
            - path to a specific file: the specified file, e.g., ./another_directory/some_other_filename.nc
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
        save_postfix : str
            Filename postfix, default to '_Sv_clean'
        save_path : str
            Full filename to save to, overwriting the RAWFILENAME_Sv_clean.nc default
        """

        # Check params
        if (noise_est_range_bin_size is not None) and (self.noise_est_range_bin_size != noise_est_range_bin_size):
            self.noise_est_range_bin_size = noise_est_range_bin_size
        if (noise_est_ping_size is not None) and (self.noise_est_ping_size != noise_est_ping_size):
            self.noise_est_ping_size = noise_est_ping_size

        # Get calibrated Sv
        if self.Sv is not None:
            print('%s  Remove noise from Sv stored in memory.' % dt.datetime.now().strftime('%H:%M:%S'))
            print_src = False
        else:
            print_src = True

        proc_data = self._get_proc_Sv(source_path=source_path, source_postfix=source_postfix)

        if print_src:
            print('%s  Remove noise from Sv stored in: %s' %
                  (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))

        # Get tile indexing parameters
        self.noise_est_range_bin_size, range_bin_tile_bin_edge, ping_tile_bin_edge = \
            self.get_tile_params(r_data_sz=proc_data.range_bin.size,
                                 p_data_sz=proc_data.ping_time.size,
                                 r_tile_sz=self.noise_est_range_bin_size,
                                 p_tile_sz=self.noise_est_ping_size,
                                 sample_thickness=self.sample_thickness)

        # Get TVG and ABS for compensating for transmission loss
        range_meter = self.range
        TVG = np.real(20 * np.log10(range_meter.where(range_meter >= 1, other=1)))
        ABS = 2 * self.seawater_absorption * range_meter

        # Function for use with apply
        def remove_n(x, rr):
            p_c_lin = 10 ** ((x.Sv - x.ABS - x.TVG) / 10)
            nn = 10 * np.log10(p_c_lin.mean(dim='ping_time').groupby_bins('range_bin', rr).mean().min(
                dim='range_bin_bins')) + x.ABS + x.TVG
            # Return values where signal is [SNR] dB above noise and at least [Sv_threshold] dB
            if not Sv_threshold:
                return x.Sv.where(x.Sv > (nn + SNR), other=np.nan)
            else:
                return x.Sv.where((x.Sv > (nn + SNR)) & (x > Sv_threshold), other=np.nan)

        # Groupby noise removal operation
        proc_data.coords['ping_idx'] = ('ping_time', np.arange(proc_data.Sv['ping_time'].size))
        ABS.name = 'ABS'
        TVG.name = 'TVG'
        pp = xr.merge([proc_data, ABS])
        pp = xr.merge([pp, TVG])
        # check if number of range_bin per tile the same for all freq channels
        if np.unique([np.array(x).size for x in range_bin_tile_bin_edge]).size == 1:
            Sv_clean = pp.groupby_bins('ping_idx', ping_tile_bin_edge).\
                            map(remove_n, rr=range_bin_tile_bin_edge[0])
            Sv_clean = Sv_clean.drop_vars(['ping_idx'])
        else:
            tmp_clean = []
            cnt = 0
            for key, val in pp.groupby('frequency'):  # iterate over different frequency channel
                tmp = val.groupby_bins('ping_idx', ping_tile_bin_edge). \
                    map(remove_n, rr=range_bin_tile_bin_edge[cnt])
                cnt += 1
                tmp_clean.append(tmp)
            clean_val = np.array([zz.values for zz in xr.align(*tmp_clean, join='outer')])
            Sv_clean = xr.DataArray(clean_val,
                                    coords={'frequency': proc_data['frequency'].values,
                                            'ping_time': tmp_clean[0]['ping_time'].values,
                                            'range_bin': tmp_clean[0]['range_bin'].values},
                                    dims=['frequency', 'ping_time', 'range_bin'])

        # Set up DataSet
        Sv_clean.name = 'Sv'
        Sv_clean = Sv_clean.to_dataset()
        Sv_clean['noise_est_range_bin_size'] = ('frequency', self.noise_est_range_bin_size)
        Sv_clean.attrs['noise_est_ping_size'] = self.noise_est_ping_size

        # Attach calculated range into data set
        Sv_clean['range'] = (('frequency', 'range_bin'), self.range)

        # Save as object attributes as a netCDF file
        self.Sv_clean = Sv_clean

        if save:
            self.Sv_clean_path = self.validate_path(save_path=save_path, save_postfix=save_postfix)
            print('%s  saving denoised Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_clean_path))
            self._save_dataset(Sv_clean, self.Sv_clean_path)

        # Close opened resources
        proc_data.close()

    def noise_estimates(self, source_postfix='_Sv', source_path=None,
                        noise_est_range_bin_size=None, noise_est_ping_size=None):
        """Obtain noise estimates from the minimum mean calibrated power level along each column of tiles.

        The tiles here are defined by class attributes noise_est_range_bin_size and noise_est_ping_size.
        This method contains redundant pieces of code that also appear in method remove_noise(),
        but this method can be used separately to determine the exact tile size for noise removal before
        noise removal is actually performed.

        Parameters
        ----------
        source_postfix : str
            postfix of the Sv file used to calculate noise estimates from, default to '_Sv'
        source_path : str
            path of Sv file used to calculate noise estimates from, can be one of the following:
            - None (default):
                    use Sv in RAWFILENAME_Sv.nc in the same folder as the raw data file,
                    or when RAWFILENAME_Sv.nc doesn't exist, perform self.calibrate() and use the resulted self.Sv
            - path to a directory: RAWFILENAME_Sv.nc in the specified directory
            - path to a specific file: the specified file, e.g., ./another_directory/some_other_filename.nc
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
        self.noise_est_range_bin_size, range_bin_tile_bin_edge, ping_tile_bin_edge = \
            self.get_tile_params(r_data_sz=proc_data.range_bin.size,
                                 p_data_sz=proc_data.ping_time.size,
                                 r_tile_sz=self.noise_est_range_bin_size,
                                 p_tile_sz=self.noise_est_ping_size,
                                 sample_thickness=self.sample_thickness)

        # Values for noise estimates
        range_meter = self.range
        TVG = np.real(20 * np.log10(range_meter.where(range_meter >= 1, other=1)))
        ABS = 2 * self.seawater_absorption * range_meter

        # Noise estimates
        proc_data['power_cal'] = 10 ** ((proc_data.Sv - ABS - TVG) / 10)
        # check if number of range_bin per tile the same for all freq channels
        if np.unique([np.array(x).size for x in range_bin_tile_bin_edge]).size == 1:
            noise_est = 10 * np.log10(proc_data['power_cal'].coarsen(
                ping_time=self.noise_est_ping_size,
                range_bin=int(np.unique(self.noise_est_range_bin_size / self.sample_thickness)),
                boundary='pad').mean().min(dim='range_bin'))
        else:
            range_bin_coarsen_idx = (self.noise_est_range_bin_size / self.sample_thickness).astype(int)
            tmp_noise = []
            for r_bin in range_bin_coarsen_idx:
                freq = r_bin.frequency.values
                tmp_da = 10 * np.log10(proc_data['power_cal'].sel(frequency=freq).coarsen(
                    ping_time=self.noise_est_ping_size,
                    range_bin=r_bin.values,
                    boundary='pad').mean().min(dim='range_bin'))
                tmp_da.name = 'noise_est'
                tmp_noise.append(tmp_da)

            # Construct a dataArray  TODO: this can probably be done smarter using xarray native functions
            noise_val = np.array([zz.values for zz in xr.align(*tmp_noise, join='outer')])
            noise_est = xr.DataArray(noise_val,
                                coords={'frequency': proc_data['frequency'].values,
                                        'ping_time': tmp_noise[0]['ping_time'].values},
                                dims=['frequency', 'ping_time'])
        noise_est = noise_est.to_dataset(name='noise_est')
        noise_est['noise_est_range_bin_size'] = ('frequency', self.noise_est_range_bin_size)
        noise_est.attrs['noise_est_ping_size'] = self.noise_est_ping_size

        # Close opened resources
        proc_data.close()

        return noise_est

    def get_MVBS(self, source_postfix='_Sv', source_path=None,
                 MVBS_range_bin_size=None, MVBS_ping_size=None,
                 save=False, save_postfix='_MVBS', save_path=None):
        """Calculate Mean Volume Backscattering Strength (MVBS).

        The calculation uses class attributes MVBS_ping_size and MVBS_range_bin_size to
        calculate and save MVBS as a new attribute to the calling Process instance.
        MVBS is an xarray DataArray with dimensions ``ping_time`` and ``range_bin``
        that are from the first elements of each tile along the corresponding dimensions
        in the original Sv or Sv_clean DataArray.

        Parameters
        ----------
        source_postfix : str
            postfix of the Sv file used to calculate MVBS, can be '_Sv' or '_Sv_clean'. Default to '_Sv'
        source_path : str
            path of Sv file used to calculate MVBS, can be one of the following:
            - None (default):
                    use Sv in RAWFILENAME_Sv.nc in the same folder as the raw data file,
                    or when RAWFILENAME_Sv.nc doesn't exist, perform self.calibrate() and use the resulted self.Sv
            - path to a directory: RAWFILENAME_Sv.nc in the specified directory
            - path to a specific file: the specified file, e.g., ./another_directory/some_other_filename.nc
        MVBS_range_bin_size : float, optional
            meters per tile for calculating MVBS [m]
        MVBS_ping_size : int, optional
            number of pings per tile for calculating MVBS
        save : bool, optional
            whether to save the calculated MVBS into a new .nc file, default to ``False``
        save_postfix : str
            Filename postfix, default to '_MVBS'
        save_path : str
            Full filename to save to, overwriting the RAWFILENAME_MVBS.nc default
        """
        # Check params
        if (MVBS_range_bin_size is not None) and (self.MVBS_range_bin_size != MVBS_range_bin_size):
            self.MVBS_range_bin_size = MVBS_range_bin_size
        if (MVBS_ping_size is not None) and (self.MVBS_ping_size != MVBS_ping_size):
            self.MVBS_ping_size = MVBS_ping_size

        # Get Sv by validating path and calibrate if not already done
        if self.Sv is not None:
            print('%s  use Sv stored in memory to calculate MVBS' % dt.datetime.now().strftime('%H:%M:%S'))
            print_src = False
        else:
            print_src = True

        if source_postfix == '_Sv_clean':
            if self.Sv_clean is None:
                self.remove_noise()
            proc_data = self.Sv_clean
        else:
            proc_data = self._get_proc_Sv(source_path=source_path, source_postfix=source_postfix)

        if print_src:
            if self.Sv_path is not None:
                print('%s  Sv source used to calculate MVBS: %s' %
                      (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
            else:
                print('%s  Sv source used to calculate MVBS: memory' %
                      dt.datetime.now().strftime('%H:%M:%S'))

        # Get tile indexing parameters
        self.MVBS_range_bin_size, range_bin_tile_bin_edge, ping_tile_bin_edge = \
            self.get_tile_params(r_data_sz=proc_data.range_bin.size,
                                 p_data_sz=proc_data.ping_time.size,
                                 r_tile_sz=self.MVBS_range_bin_size,
                                 p_tile_sz=self.MVBS_ping_size,
                                 sample_thickness=self.sample_thickness)
        # Calculate MVBS
        Sv_linear = 10 ** (proc_data.Sv / 10)  # convert to linear domain before averaging
        # check if number of range_bin per tile the same for all freq channels
        if np.unique([np.array(x).size for x in range_bin_tile_bin_edge]).size == 1:
            MVBS = 10 * np.log10(Sv_linear.coarsen(
                ping_time=self.MVBS_ping_size,
                range_bin=int(np.unique(self.MVBS_range_bin_size / self.sample_thickness)),
                boundary='pad').mean())
            MVBS.coords['range_bin'] = ('range_bin', np.arange(MVBS['range_bin'].size))
        else:
            range_bin_coarsen_idx = (self.MVBS_range_bin_size / self.sample_thickness).astype(int)
            tmp_MVBS = []
            for r_bin in range_bin_coarsen_idx:
                freq = r_bin.frequency.values
                tmp_da = 10 * np.log10(Sv_linear.sel(frequency=freq).coarsen(
                        ping_time=self.MVBS_ping_size,
                        range_bin=r_bin.values,
                        boundary='pad').mean())
                tmp_da.coords['range_bin'] = ('range_bin', np.arange(tmp_da['range_bin'].size))
                tmp_da.name = 'MVBS'
                tmp_MVBS.append(tmp_da)

            # Construct a dataArray  TODO: this can probably be done smarter using xarray native functions
            MVBS_val = np.array([zz.values for zz in xr.align(*tmp_MVBS, join='outer')])
            MVBS = xr.DataArray(MVBS_val,
                                coords={'frequency': Sv_linear['frequency'].values,
                                        'ping_time': tmp_MVBS[0]['ping_time'].values,
                                        'range_bin': np.arange(MVBS_val.shape[2])},
                                dims=['frequency', 'ping_time', 'range_bin']).dropna(dim='range_bin', how='all')

        # Set MVBS attributes
        MVBS.name = 'MVBS'
        MVBS = MVBS.to_dataset()
        MVBS['MVBS_range_bin_size'] = ('frequency', self.MVBS_range_bin_size)
        MVBS.attrs['MVBS_ping_size'] = self.MVBS_ping_size

        # Save results in object and as a netCDF file
        self.MVBS = MVBS
        if save:
            self.MVBS_path = self.validate_path(save_path=save_path, save_postfix=save_postfix)
            print('%s  saving MVBS to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.MVBS_path))
            self._save_dataset(MVBS, self.MVBS_path)

        # Close opened resources
        proc_data.close()
    
    def _save_dataset(self, ds, path, mode="w"):
        """Save dataset to the appropriate formats.

        A utility method to use the correct function to save the dataset,
        based on the input file format.

        Parameters
        ----------
        ds : xr.Dataset
            xarray dataset object
        path : str
            output file
        """
        if self._file_format == 'netcdf':
            ds.to_netcdf(path, mode=mode)
        elif self._file_format == 'zarr':
            ds.to_zarr(path, mode=mode)

    def _set_open_dataset(self):
        if self._file_format == 'netcdf':
            self._open_dataset = xr.open_dataset
        elif self._file_format == 'zarr':
            self._open_dataset = xr.open_zarr
