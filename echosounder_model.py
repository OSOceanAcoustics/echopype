"""
Model for loading echosounder files from:
- Simrad EK60 (now)
- Simrad EK80 (future)
- ASL Env Sci AFZP (future)


"""

import numpy as np
import h5py
import datetime as dt
from collections import defaultdict
from matplotlib.dates import date2num


class EchoDataRaw(object):
    '''
    Class for loading manipulating raw echosounder data (raw = not subsetted and not MVBS)
    '''
    def __init__(self,filepath='',ping_bin=40,depth_bin=5,tvg_correction_factor=2):
        self.filepath = filepath
        self.bin_size = float('nan')
        self.ping_bin = ping_bin
        self.depth_bin = depth_bin
        self.tvg_correction_factor = tvg_correction_factor

        if self.filepath=='':  # only initialize the object
            self.hdf5_handle = []
            self.cal_params = defaultdict(list)
        else:  # load echo data from HDF5 file
            self.load_hdf5()

        # Emtpy attributes that will only be evaluated when associated methods called by user
        self.noise_est = defaultdict(list)
        self.Sv_raw = defaultdict(list)        # Sv without noise removal but with TVG & absorption compensation
        self.Sv_corrected = defaultdict(list)  # Sv with noise removed and with TVG & absorption compensation
        self.Sv_noise = defaultdict(list)      # the noise component with TVG & absorption compensation

    # Methods to set critical params
    def set_ping_bin(self,ping_bin):
        self.ping_bin = ping_bin

    def set_depth_bin(self,depth_bin):
        self.depth_bin = depth_bin

    def set_tvg_correction_factor(self,tvg_correction_factor):
        self.tvg_correction_factor = tvg_correction_factor

    # Methods to load and manipulate data
    def load_hdf5(self):
        self.hdf5_handle = h5py.File(self.filepath,'r')  # read-only
        self.bin_size = self.hdf5_handle['metadata/bin_size'][0]  # minimun bin size in depth
        self.get_cal_params()

    def find_freq_seq(self,freq):
        '''Find the sequence of transducer of a particular freq'''
        return int(np.where(np.array(self.hdf5_handle['metadata/zplsc_frequency'][:])==freq)[0])

    def get_cal_params(self):
        '''
        Pull calibration paramteres from metadata
        '''
        cal_params = defaultdict(list)
        # get group names from HDF5
        fh_keys = []
        for p in self.hdf5_handle.keys():
            fh_keys.append(p)
        # get cal params from HDF5: loop through all transducers
        for tx_name in list(filter(lambda x: x[0:10]=='transducer', fh_keys)):  # group name = transducer%02d
            freq = self.hdf5_handle[tx_name]['frequency'][0]  # frequency of this transducer, type = str
            freq_seq = self.find_freq_seq(int(freq))
            freq_str = str(freq)  # convert to str for compatibility with how self.power_data is indexed
            cal_params[freq_str] = defaultdict(list)
            cal_params[freq_str]['frequency'] = freq
            cal_params[freq_str]['soundvelocity'] = self.hdf5_handle['metadata/zplsc_sound_velocity'][freq_seq]
            cal_params[freq_str]['sampleinterval'] = self.hdf5_handle['metadata/zplsc_sample_interval'][freq_seq]
            cal_params[freq_str]['absorptioncoefficient'] = self.hdf5_handle['metadata/zplsc_absorption_coeff'][freq_seq]
            cal_params[freq_str]['gain'] = self.hdf5_handle[tx_name]['gain'][0]
            cal_params[freq_str]['equivalentbeamangle'] = self.hdf5_handle[tx_name]['equiv_beam_angle'][0]
            cal_params[freq_str]['transmitpower'] = self.hdf5_handle['metadata/zplsc_transmit_power'][freq_seq]
            cal_params[freq_str]['pulselength'] = self.hdf5_handle['metadata/zplsc_pulse_length'][freq_seq]
            cal_params[freq_str]['pulselengthtable'] = self.hdf5_handle[tx_name]['pulse_length_table'][:]
            cal_params[freq_str]['sacorrectiontable'] = self.hdf5_handle[tx_name]['sa_correction_table'][:]
        self.cal_params = cal_params

    def get_noise(self):
        '''
        Get minimum value for bins of averaged ping (= noise)
        This method is called internally by `remove_noise`
        [Reference] De Robertis & Higginbottom, 2017, ICES JMR
        '''
        N = int(np.floor(self.depth_bin/self.bin_size))  # rough number of depth bins

        # Average uncompensated power over M pings and N depth bins
        # and find minimum value of power for each averaged bin
        noise_est = defaultdict(list)
        for (freq_str,vals) in self.hdf5_handle['power_data'].items():
            sz = vals.shape
            power = vals[:]  # access as a numpy ndarray
            depth_bin_num = int(np.floor((sz[0]-self.tvg_correction_factor)/N))  # number of depth bins
            ping_bin_num = int(np.floor(sz[1]/self.ping_bin))                    # number of ping bins
            power_bin = np.empty([depth_bin_num,ping_bin_num])
            for iD in range(depth_bin_num):
                for iP in range(ping_bin_num):
                    depth_idx = np.arange(N) + N*iD + self.tvg_correction_factor  # match the 2-sample offset
                    ping_idx = np.arange(self.ping_bin) + self.ping_bin*iP
                    power_bin[iD,iP] = np.mean(10**(power[np.ix_(depth_idx,ping_idx)]/10))
            noise_est[freq_str] = np.min(power_bin,0)  # noise = minimum value for each averaged ping
        self.noise_est = noise_est

    def remove_noise(self):
        '''
        Noise removal and TVG + absorption compensation
        This method will call `get_noise` to make sure to have attribute `noise_est`
        [Reference] De Robertis & Higginbottom, 2017, ICES JMR
        '''
        # Get noise estimation
        self.get_noise()

        # Initialize arrays
        Sv_raw = defaultdict(list)
        Sv_corrected = defaultdict(list)
        Sv_noise = defaultdict(list)

        # Remove noise
        for (freq_str,vals) in self.cal_params.items():  # Loop through all transducers
            # Get cal params
            f = self.cal_params[freq_str]['frequency']
            c = self.cal_params[freq_str]['soundvelocity']
            t = self.cal_params[freq_str]['sampleinterval']
            alpha = self.cal_params[freq_str]['absorptioncoefficient']
            G = self.cal_params[freq_str]['gain']
            phi = self.cal_params[freq_str]['equivalentbeamangle']
            pt = self.cal_params[freq_str]['transmitpower']
            tau = self.cal_params[freq_str]['pulselength']

            # key derived params
            dR = c*t/2   # sample thickness
            wvlen = c/f  # wavelength

            # Calc gains
            CSv = 10 * np.log10((pt * (10**(G/10))**2 * wvlen**2 * c * tau * 10**(phi/10)) / (32 * np.pi**2))

            # calculate Sa Correction
            idx = [i for i,dd in enumerate(self.cal_params[freq_str]['pulselengthtable']) if dd==tau]
            Sac = 2 * self.cal_params[freq_str]['sacorrectiontable'][idx]

            # Get TVG
            range_vec = np.arange(self.hdf5_handle['power_data'][freq_str].shape[0]) * dR
            range_corrected = range_vec - (self.tvg_correction_factor * dR)
            range_corrected[range_corrected<0] = 0

            TVG = np.empty(range_corrected.shape)
            # TVG = real(20 * log10(range_corrected));
            TVG[range_corrected!=0] = np.real( 20*np.log10(range_corrected[range_corrected!=0]) )
            TVG[range_corrected==0] = 0

            # Get absorption
            ABS = 2*alpha*range_corrected

            # Remove noise and compensate measurement for transmission loss
            # also estimate Sv_noise for subsequent SNR check
            if self.noise_est[freq_str].shape==():  # if noise_est is a single element
                subtract = 10**(self.hdf5_handle['power_data'][freq_str]/10)-self.noise_est[freq_str]
                tmp = 10*np.log10(np.ma.masked_less_equal(subtract,0))
                tmp.set_fill_value(-999)
                Sv_corrected[freq_str] = (tmp.T+TVG+ABS-CSv-Sac).T
                Sv_noise[freq_str] = 10*np.log10(self.noise_est[freq_str])+TVG+ABS-CSv-Sac
            else:
                sz = self.hdf5_handle['power_data'][freq_str].shape
                ping_bin_num = int(np.floor(sz[1]/self.ping_bin))
                Sv_corrected[freq_str] = np.ma.empty(sz)  # log domain corrected Sv
                Sv_noise[freq_str] = np.empty(sz)    # Sv_noise
                for iP in range(ping_bin_num):
                    ping_idx = np.arange(self.ping_bin) +iP*self.ping_bin
                    subtract = 10**(self.hdf5_handle['power_data'][freq_str][:,ping_idx]/10) -self.noise_est[freq_str][iP]
                    tmp = 10*np.log10(np.ma.masked_less_equal(subtract,0))
                    tmp.set_fill_value(-999)
                    Sv_corrected[freq_str][:,ping_idx] = (tmp.T +TVG+ABS-CSv-Sac).T
                    Sv_noise[freq_str][:,ping_idx] = np.array([10*np.log10(self.noise_est[freq_str][iP])+TVG+ABS-CSv-Sac]*self.ping_bin).T

            # Raw Sv withour noise removal but with TVG/absorption compensation
            Sv_raw[freq_str] = (self.hdf5_handle['power_data'][freq_str][:].T+TVG+ABS-CSv-Sac).T

        # Save results
        self.Sv_raw = Sv_raw
        self.Sv_corrected = Sv_corrected
        self.Sv_noise = Sv_noise

    def subset_data(self,date_wanted,subset_params):
        '''
        Subset echo data with datetime object `date_wanted`
        '''
        # total number of subsetted pings per day
        ping_per_day = len(subset_params['hour_all'])*\
                       len(subset_params['min_all'])*\
                       len(subset_params['sec_all'])
        subset_ping_time = np.empty((ping_per_day*len(date_wanted),))
        subset_Sv_raw = defaultdict(list)
        subset_Sv_corrected = defaultdict(list)
        subset_Sv_noise = defaultdict(list)
        for (freq_str,vals) in self.hdf5_handle['power_data'].items():  # loop through all transducers
            sz = vals.shape
            subset_Sv_raw[freq_str] = np.ma.empty((sz[0],ping_per_day*len(date_wanted)))
            subset_Sv_corrected[freq_str] = np.ma.empty((sz[0],ping_per_day*len(date_wanted)))
            subset_Sv_noise[freq_str] = np.ma.empty((sz[0],ping_per_day*len(date_wanted)))

            for iD,dd_curr in zip(range(len(date_wanted)),date_wanted):  # loop through all dates wanted
                # set up indexing to get wanted pings
                dd = dt.datetime.strptime(dd_curr,'%Y%m%d')
                time_wanted = [dt.datetime(dd.year,dd.month,dd.day,hh,mm,ss) \
                               for hh in subset_params['hour_all']\
                               for mm in subset_params['min_all'] \
                               for ss in subset_params['sec_all']]

                # ping sequence index in subsetted data
                ping_idx = iD*ping_per_day + np.arange(ping_per_day)
                subset_ping_time[ping_idx] = date2num(time_wanted)   # fill in subsetted ping_time

                # fine closest ping index in raw data
                idx_wanted = [self.find_nearest_time_idx(tt,2) for tt in time_wanted]
                notnanidx = np.argwhere(~np.isnan(idx_wanted)).flatten()
                notnanidx_in_all = np.array(idx_wanted)[notnanidx].astype(int)

                # fill in Sv values
                notnanidx = notnanidx + iD*ping_per_day   # adjust to global index in subsetted data
                subset_Sv_raw[freq_str][:,notnanidx]       = self.Sv_raw[freq_str][:,notnanidx_in_all]
                subset_Sv_corrected[freq_str][:,notnanidx] = self.Sv_corrected[freq_str][:,notnanidx_in_all]
                subset_Sv_noise[freq_str][:,notnanidx]     = self.Sv_noise[freq_str][:,notnanidx_in_all]

                idx_save_mask = np.argwhere(np.isnan(idx_wanted)) + iD*ping_per_day   # adjust to global index in subsetted data
                subset_Sv_raw[freq_str][:,idx_save_mask]       = np.ma.masked
                subset_Sv_corrected[freq_str][:,idx_save_mask] = np.ma.masked
                subset_Sv_noise[freq_str][:,idx_save_mask]     = np.ma.masked

        # Update Sv using subsetted values
        self.Sv_raw = subset_Sv_raw
        self.Sv_corrected = subset_Sv_corrected
        self.Sv_noise = subset_Sv_noise

    def find_nearest_time_idx(self,time_wanted,tolerance):
        '''
        Method to find nearest element
        This method is called by `subset_data`
        INPUT:
            time_wanted   a datetime object
            tolerance     the max tolerance in second allowed between `time_wanted` and `all_timestamp`
        '''
        time_wanted_num = date2num(time_wanted)
        idx = np.searchsorted(self.hdf5_handle['ping_time'], time_wanted_num, side="left")
        if idx > 0 and (idx == len(self.hdf5_handle['ping_time']) or \
           np.abs(time_wanted_num - self.hdf5_handle['ping_time'][idx-1]) < \
           np.abs(time_wanted_num - self.hdf5_handle['ping_time'][idx])):
            idx -= 1

        # If interval between the selected index and time wanted > `tolerance` seconds
        sec_diff = dt.timedelta(self.hdf5_handle['ping_time'][idx]-time_wanted_num).total_seconds()
        if np.abs(sec_diff)>tolerance:
            return np.nan
        else:
            return idx


    def get_mvbs(self):
        '''
        Obtain Mean Volume Backscattering Strength (MVBS) from `Sv_corrected`
        '''
        N = int(np.floor(self.depth_bin/self.bin_size))  # rough number of depth bins

        # Average Sv over M pings and N depth bins
        depth_bin_num = int(np.floor(Sv.shape[1]/N))
        ping_bin_num = int(np.floor(Sv.shape[2]/ping_bin_range))
        MVBS = np.ma.empty([Sv.shape[0],depth_bin_num,ping_bin_num])
        for iF in range(Sv.shape[0]):
            for iD in range(depth_bin_num):
                for iP in range(ping_bin_num):
                    depth_idx = np.arange(N) + N*iD
                    ping_idx = np.arange(ping_bin_range) + ping_bin_range*iP
                    MVBS[iF,iD,iP] = 10*np.log10( np.nanmean(10**(Sv[np.ix_((iF,),depth_idx,ping_idx)]/10)) )
        return MVBS

#     def save_mvbs2hdf5(self):
#
#
# class EchoDataMVBS(EchoDataRaw):
#
#     def load_hdf5(self):  # overload this function from EchoDataRaw
#
#     # Get all attributes
