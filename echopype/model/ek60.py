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
        self.noise_est_range_bin_size = 5  # meters per tile for noise estimation
        self.noise_est_ping_size = 30      # number of pings per tile for noise estimation

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

            # Get a separate .nc filename for storing processed data if computation is performed
            self.proc_path = os.path.join(os.path.dirname(self.file_path),
                                          os.path.splitext(os.path.basename(self.file_path))[0] + '_proc.nc')

            if self.toplevel.sonar_convention_name == 'SONAR-netCDF4':
                # # Point to various groups to .nc file converted from .raw
                # self.provenance = xr.open_dataset(self.file_path, group="Provenance")
                # self.environment = xr.open_dataset(self.file_path, group="Environment")
                # self.platform = xr.open_dataset(self.file_path, group="Platform")
                # self.sonar = xr.open_dataset(self.file_path, group="Sonar")
                # self.beam = xr.open_dataset(self.file_path, group="Beam")

                # Get a separate .nc filename for storing processed data if computation is performed
                self.proc_path = os.path.join(os.path.dirname(self.file_path),
                                              os.path.splitext(os.path.basename(self.file_path))[0]+'_proc.nc')
            else:
                print('netCDF file convention not recognized.')
                # # Groups for .nc file converted from .raw
                # self.toplevel = ''
                # self.provenance = ''
                # self.environment = ''
                # self.platform = ''
                # self.sonar = ''
                # self.beam = ''

                # Processed data
                self.proc_path = ''
        else:
            print('Data file format not recognized.')

    def calibrate(self):
        """Perform echo-integration to get volume backscattering strength (Sv) from EK60 power data.
        """

        # Open data set for Environment and Beam groups
        ds_env = xr.open_dataset(self.file_path, group="Environment")
        ds_beam = xr.open_dataset(self.file_path, group="Beam")

        # Params from env group
        c = ds_env.sound_speed_indicative
        alpha = ds_env.absorption_indicative

        # Params from beam group
        t = ds_beam.sample_interval
        gain = ds_beam.gain_correction
        phi = ds_beam.equivalent_beam_angle
        pt = ds_beam.transmit_power
        tau = ds_beam.transmit_duration_nominal
        Sac = 2 * ds_beam.sa_correction

        # Derived params
        dR = c * t / 2  # sample thickness
        wavelength = c / ds_env.frequency  # wavelength

        # Calc gain
        CSv = 10 * np.log10((ds_beam.transmit_power * (10 ** (ds_beam.gain_correction / 10)) ** 2 *
                             wavelength ** 2 * c * ds_beam.transmit_duration_nominal *
                             10 ** (ds_beam.equivalent_beam_angle / 10)) /
                            (32 * np.pi ** 2))

        # Get TVG and absorption
        range_meter = ds_beam.range_bin * dR - self.tvg_correction_factor * dR  # DataArray [frequency x range_bin]
        range_meter = range_meter.where(range_meter > 0, other=0)  # set all negative elements to 0
        TVG = np.real(20 * np.log10(range_meter.where(range_meter != 0, other=1)))
        ABS = 2 * alpha * range_meter

        # Calibration and echo integration
        Sv = ds_beam.backscatter_r + TVG + ABS - CSv - 2 * ds_beam.sa_correction

        # Save calibrated data to a separate .nc file in the same directory as the data file
        print('%s  saving calibrated Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.proc_path))
        Sv.to_dataset().to_netcdf(path=self.proc_path, mode="w")
        ds_env.close()
        ds_beam.close()

    def remove_noise(self, noise_est_range_bin_size=None, noise_est_ping_size=None):
        """
        Remove noise by using noise estimates obtained from the minimum mean calibrated power level
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

        # Open data set for Environment and Beam groups
        ds_env = xr.open_dataset(self.file_path, group="Environment")
        ds_beam = xr.open_dataset(self.file_path, group="Beam")

        # Get TVG and absorption
        c = ds_env.sound_speed_indicative
        t = ds_beam.sample_interval
        dR = c * t / 2  # sample thickness
        range_meter = ds_beam.range_bin * dR - self.tvg_correction_factor * dR  # DataArray [frequency x range_bin]
        range_meter = range_meter.where(range_meter > 0, other=0)  # set all negative elements to 0
        TVG = np.real(20 * np.log10(range_meter.where(range_meter != 0, other=1)))

        alpha = ds_env.absorption_indicative
        ABS = 2 * alpha * range_meter

        # Get calibrated power
        proc_data = xr.open_dataset(self.proc_path)
        # power_cal_lin = 10 ** ((proc_data.Sv - TVG - ABS) / 10)  # calibrated power, based on which noise is estimated

        # Adjust noise_est_range_bin_size because range_bin_size may be an inconvenient value
        num_range_bin_per_tile = (np.round(self.noise_est_range_bin_size /
                                           dR).astype(int)).values.max()  # number of range_bin per tile
        self.noise_est_range_bin_size = (num_range_bin_per_tile * dR).values

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
            z = np.hstack((np.array([np.arange(num_tile_ping - 1)] * self.noise_est_ping_size).squeeze().T.ravel(), pad))

        # Tile bin edges along range
        # ... -1 to make sure each bin has the same size because of the right-inclusive and left-exclusive bins
        range_bin_tile_bin = np.arange(num_tile_range_bin) * num_range_bin_per_tile - 1

        # Function for use with apply
        def remove_n(x):
            p_c_lin = 10 ** ((x - ABS - TVG) / 10)
            nn = 10 * np.log10(p_c_lin.groupby_bins('range_bin', range_bin_tile_bin).mean('range_bin').
                               groupby('frequency').mean('ping_time').min(dim='range_bin_bins'))
            return x.where(x > nn, other=np.nan)

        # Groupby noise removal operation
        proc_data.coords['add_idx'] = ('ping_time', z)
        Sv_clean = proc_data.Sv.groupby('add_idx').apply(remove_n)
        Sv_clean.name = 'Sv_clean'

        # Save as a netCDF file
        Sv_clean.to_dataset().to_netcdf(self.proc_path)
        proc_data.close()
        ds_env.close()
        ds_beam.close()

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
        ds_env = xr.open_dataset(self.file_path, group="Environment")
        ds_beam = xr.open_dataset(self.file_path, group="Beam")

        # Get TVG and absorption
        c = ds_env.sound_speed_indicative
        t = ds_beam.sample_interval
        dR = c * t / 2  # sample thickness
        range_meter = ds_beam.range_bin * dR - self.tvg_correction_factor * dR  # DataArray [frequency x range_bin]
        range_meter = range_meter.where(range_meter > 0, other=0)  # set all negative elements to 0
        TVG = np.real(20 * np.log10(range_meter.where(range_meter != 0, other=1)))

        alpha = ds_env.absorption_indicative
        ABS = 2 * alpha * range_meter

        # Use calibrated data to calculate noise removal
        proc_data = xr.open_dataset(self.proc_path)
        # power_cal_lin = 10 ** ((proc_data.Sv - TVG - ABS) / 10)  # calibrated power, based on which noise is estimated

        # Adjust noise_est_range_bin_size because range_bin_size may be an inconvenient value
        num_range_bin_per_tile = (np.round(self.noise_est_range_bin_size /
                                           dR).astype(int)).values.max()  # number of range_bin per tile
        self.noise_est_range_bin_size = (num_range_bin_per_tile * dR).values

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
            p_c_lin = 10 ** ((x - ABS - TVG) / 10)
            nn = 10 * np.log10(p_c_lin.groupby_bins('range_bin', range_bin_tile_bin).mean('range_bin').
                               groupby('frequency').mean('ping_time').min(dim='range_bin_bins'))
            noise_est.append(nn.values)
            return x

        # Groupby operation for estimating noise
        proc_data.coords['add_idx'] = ('ping_time', z)
        _ = proc_data.Sv.groupby('add_idx').apply(est_n)

        # Close opened resources
        ds_env.close()
        ds_beam.close()

        return noise_est

        # if (N_ping-1)*self.noise_est_ping_size >= self.beam.ping_time.size:
        #     ping_bin = self.beam.ping_time.isel(ping_time=np.hstack((np.arange(N_ping-1) * self.noise_est_ping_size,
        #                                                              self.beam.ping_time.size-1)))
        # else:
        #     ping_bin = self.beam.ping_time.isel(ping_time=np.arange(N_ping) * self.noise_est_ping_size)
        #
        # noise_est_r = power_cal_lin.groupby_bins('range_bin', range_bin_bin).mean('range_bin')
        # noise_est_rp = 10*np.log10(noise_est_r.groupby_bins('ping_time', ping_bin).mean('ping_time'))
        # noise_est = noise_est_rp.groupby('ping_time_bins').min('range_bin_bins')


        # # Get noise estimates over N_range_bin x N_ping tile
        # idx_r_base = np.arange(num_range_bin_per_tile, dtype=int)
        # idx_p_base = np.arange(self.noise_est_ping_size, dtype=int)
        # noise_est = np.empty((self.beam.frequency.size, N_ping, N_range_bin))
        # # for f_seq in range(self.beam.frequency.size):  # loop through all freuqency
        # for p_seq in range(N_ping):  # loop through all pings
        #     for r_seq in range(N_range_bin):  # loop through all range bins
        #         idx_p = idx_p_base[[0, -1]] + self.noise_est_ping_size * p_seq
        #         idx_r = idx_r_base[[0, -1]] + num_range_bin_per_tile * r_seq
        #         noise_est[:, p_seq, r_seq] = \
        #             10 * np.log10(np.mean(10 ** (self.beam.backscatter_r.isel(
        #                 ping_time=slice(idx_p[0], idx_p[1]),
        #                 range_bin=slice(idx_r[0], idx_r[1])).values.reshape((self.beam.frequency.size, -1)) / 10),
        #                                   axis=1))
        #     noise_est = noise_est.min(axis=2)
        #
        # # Loop through each frequency to estimate noise
        # for f_seq, freq in enumerate(self.beam.frequency.values):
        #
        #     # Get range bin size
        #     c = self.environment.sound_speed_indicative.sel(frequency=freq).values
        #     t = self.beam.sample_interval.sel(frequency=freq).values
        #     range_bin_size = c * t / 2
        #
        #     # Adjust noise_est_range_bin_size because range_bin_size may be an inconvenient value
        #     num_range_bin_per_tile = np.round(self.noise_est_range_bin_size /
        #                                       range_bin_size).astype(int)  # number of range_bin per tile
        #     self.noise_est_range_bin_size = num_range_bin_per_tile * range_bin_size
        #
        #     # Number of tiles along range_bin
        #     if np.mod(self.beam.range_bin.size, num_range_bin_per_tile) != 0:
        #         N_range_bin = np.floor(self.beam.range_bin.size / num_range_bin_per_tile).astype(int) + 1
        #     else:
        #         N_range_bin = np.int(self.beam.range_bin.size / num_range_bin_per_tile)
        #     if np.mod(self.beam.ping_time.size, self.noise_est_ping_size) != 0:
        #         N_ping = np.floor(self.beam.ping_time.size / self.noise_est_ping_size).astype(int) + 1
        #     else:
        #         N_ping = np.int(self.beam.ping_time.size / self.noise_est_ping_size)
        #
        #     # Get noise estimates over N_range_bin x N_ping tile
        #     idx_p_base = np.arange(self.noise_est_ping_size, dtype=int)
        #     idx_r_base = np.arange(num_range_bin_per_tile, dtype=int)
        #     noise_est = np.empty((self.beam.frequency.size, N_ping, N_range_bin))
        #     for p_seq in range(N_ping):  # loop through all pings
        #         for r_seq in range(N_range_bin):  # loop through all range bins
        #             idx_p = idx_p_base[[0, -1]] + self.noise_est_ping_size * p_seq
        #             idx_r = idx_r_base[[0, -1]] + num_range_bin_per_tile * r_seq
        #             noise_est[:, p_seq, r_seq] = \
        #                 np.mean(10 ** (self.beam.backscatter_r.isel(ping_time=slice(idx_p[0], idx_p[1]),
        #                                                             range_bin=slice(idx_r[0], idx_r[1])).values / 10))
        #     noise_est = noise_est.min(axis=2)
        #     ping_time_reduce = self.beam.ping_time.isel(ping_time=np.arange(0, self.beam.ping_time.size,
        #                                                                     self.noise_est_ping_size)).data
        #     # Assemble an xarray DataArray
        #     ping_time_reduce = (ping_time_reduce - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')
        #     ds = xr.Dataset(
        #         {'noise_est': (['frequency', 'ping_time_noise_est'], noise_est,),
        #          },
        #         coords={'frequency': (['frequency'], self.beam.frequency,
        #                               {'units': 'Hz',
        #                                'valid_min': 0.0}),
        #                 'ping_time_noise_est': (['ping_time_noise_est'], ping_time_reduce,
        #                               {'axis': 'T',
        #                                'calendar': 'gregorian',
        #                                'long_name': 'Timestamp of each ping',
        #                                'standard_name': 'time',
        #                                'units': 'seconds since 1900-01-01'})},
        #         attrs={'tvg_correction_factor': self.tvg_correction_factor,
        #                'noise_est_range_bin_size': self.noise_est_range_bin_size,
        #                'noise_est_ping_size': self.noise_est_ping_size})
        #
        #     # Save noise estimates to _proc.nc
        #     print('%s  saving noise estimates to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.proc_path))
        #     ds.to_netcdf(path=self.proc_path, mode="a")
        #     ds.close()

            # Average uncompensated power over each tile and find minimum value of power for each averaged bin
            # TODO: Let's use xarray dataArray/dataSet.coarsen for this average

    # def remove_noise(self):
    #     """
    #     Noise removal and TVG + absorption compensation
    #     This method will call `get_noise` to make sure to have attribute `noise_est`
    #     [Reference] De Robertis & Higginbottom, 2017, ICES JMR
    #     INPUT:
    #         const_noise   use a single value of const noise for all frequencies
    #     """
    #
    #     # Get noise estimation
    #     if const_noise!=[]:
    #         for freq_str in self.cal_params.keys():
    #             self.noise_est[freq_str] = const_noise
    #     else:
    #         self.get_noise()
    #
    #     # Initialize arrays
    #     Sv_raw = defaultdict(list)
    #     Sv_corrected = defaultdict(list)
    #     Sv_noise = defaultdict(list)
    #
    #     # Remove noise
    #     for (freq_str,vals) in self.cal_params.items():  # Loop through all transducers
    #         # Get cal params
    #         f = self.cal_params[freq_str]['frequency']
    #         c = self.cal_params[freq_str]['soundvelocity']
    #         t = self.cal_params[freq_str]['sampleinterval']
    #         alpha = self.cal_params[freq_str]['absorptioncoefficient']
    #         G = self.cal_params[freq_str]['gain']
    #         phi = self.cal_params[freq_str]['equivalentbeamangle']
    #         pt = self.cal_params[freq_str]['transmitpower']
    #         tau = self.cal_params[freq_str]['pulselength']
    #
    #         # key derived params
    #         dR = c*t/2   # sample thickness
    #         wvlen = c/f  # wavelength
    #
    #         # Calc gains
    #         CSv = 10 * np.log10((pt * (10**(G/10))**2 * wvlen**2 * c * tau * 10**(phi/10)) / (32 * np.pi**2))
    #
    #         # calculate Sa Correction
    #         idx = [i for i,dd in enumerate(self.cal_params[freq_str]['pulselengthtable']) if dd==tau]
    #         Sac = 2 * self.cal_params[freq_str]['sacorrectiontable'][idx]
    #
    #         # Get TVG
    #         range_vec = np.arange(self.hdf5_handle['power_data'][freq_str].shape[0]) * dR
    #         range_corrected = range_vec - (self.tvg_correction_factor * dR)
    #         range_corrected[range_corrected<0] = 0
    #
    #         TVG = np.empty(range_corrected.shape)
    #         # TVG = real(20 * log10(range_corrected));
    #         TVG[range_corrected!=0] = np.real( 20*np.log10(range_corrected[range_corrected!=0]) )
    #         TVG[range_corrected==0] = 0
    #
    #         # Get absorption
    #         ABS = 2*alpha*range_corrected
    #
    #         # Remove noise and compensate measurement for transmission loss
    #         # also estimate Sv_noise for subsequent SNR check
    #         if isinstance(self.noise_est[freq_str],(int,float)):  # if noise_est is a single element
    #             subtract = 10**(self.hdf5_handle['power_data'][freq_str][:]/10)-self.noise_est[freq_str]
    #             tmp = 10*np.log10(np.ma.masked_less_equal(subtract,0))
    #             tmp.set_fill_value(-999)
    #             Sv_corrected[freq_str] = (tmp.T+TVG+ABS-CSv-Sac).T
    #             Sv_noise[freq_str] = 10*np.log10(self.noise_est[freq_str])+TVG+ABS-CSv-Sac
    #         else:
    #             sz = self.hdf5_handle['power_data'][freq_str].shape
    #             ping_bin_num = int(np.floor(sz[1]/self.ping_bin))
    #             Sv_corrected[freq_str] = np.ma.empty(sz)  # log domain corrected Sv
    #             Sv_noise[freq_str] = np.empty(sz)    # Sv_noise
    #             for iP in range(ping_bin_num):
    #                 ping_idx = np.arange(self.ping_bin) +iP*self.ping_bin
    #                 subtract = 10**(self.hdf5_handle['power_data'][freq_str][:,ping_idx]/10) -self.noise_est[freq_str][iP]
    #                 tmp = 10*np.log10(np.ma.masked_less_equal(subtract,0))
    #                 tmp.set_fill_value(-999)
    #                 Sv_corrected[freq_str][:,ping_idx] = (tmp.T +TVG+ABS-CSv-Sac).T
    #                 Sv_noise[freq_str][:,ping_idx] = np.array([10*np.log10(self.noise_est[freq_str][iP])+TVG+ABS-CSv-Sac]*self.ping_bin).T
    #
    #         # Raw Sv withour noise removal but with TVG/absorption compensation
    #         Sv_raw[freq_str] = (self.hdf5_handle['power_data'][freq_str][:].T+TVG+ABS-CSv-Sac).T
    #
    #     # Save results
    #     self.Sv_raw = Sv_raw
    #     self.Sv_corrected = Sv_corrected
    #     self.Sv_noise = Sv_noise

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




