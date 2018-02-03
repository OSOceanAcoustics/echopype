

import os, sys, glob
import numpy as np
import datetime as dt
import h5py
import matplotlib.pyplot as plt
from matplotlib.dates import date2num,num2date
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0,'../mi_instrument')
from mi.instrument.kut.ek60.ooicore import zplsc_b

#from datetime import datetime
#sys.path.insert(0,'/home/wu-jung/code_git/mi-instrument')

#from mi.instrument.kut.ek60.ooicore.zplsc_b import *
#from concat_raw import *



# Colormap from Jech & Michaels 2006
multifreq_th_colors = np.array([[0,0,0],\
                                [86,25,148],\
                                [28,33,179],\
                                [0,207,239],\
                                [41,171,71],\
                                [51,204,51],\
                                [255,239,0],\
                                [255,51,0]])/255.
mf_cmap = colors.ListedColormap(multifreq_th_colors)
mf_bounds = range(9)
mf_norm = colors.BoundaryNorm(mf_bounds,mf_cmap.N)



# Colormap: standard EK60
e_cmap_colors = np.array([[255, 255, 255],\
                          [159, 159, 159],\
                          [ 95,  95,  95],\
                          [  0,   0, 255],\
                          [  0,   0, 127],\
                          [  0, 191,   0],\
                          [  0, 127,   0],\
                          [255, 255,   0],\
                          [255, 127,   0],\
                          [255,   0, 191],\
                          [255,   0,   0],\
                          [166,  83,  60],\
                          [120,  60,  40]])/255.
e_cmap_th = [-80,-30]
e_cmap = colors.ListedColormap(e_cmap_colors)
e_bounds = np.linspace(e_cmap_th[0],e_cmap_th[1],e_cmap.N+1)
e_norm = colors.BoundaryNorm(e_bounds,e_cmap.N)



def get_cal_params(power_data_dict,particle_data,config_header,config_transducer):
    """
    Get calibration params from the unpacked file
    Parameters come from config_header and config_transducer (both from header),
                                      as well as particle_data (from .RAW file)
    """
    cal_params = []
    for ii in range(len(power_data_dict)):
        cal_params_tmp = {}
        cal_params_tmp['soundername'] = config_header['sounder_name'];
        cal_params_tmp['frequency'] = config_transducer[ii]['frequency'];
        cal_params_tmp['soundvelocity'] = particle_data[0]['zplsc_sound_velocity'][ii];
        cal_params_tmp['sampleinterval'] = particle_data[0]['zplsc_sample_interval'][ii];
        cal_params_tmp['absorptioncoefficient'] = particle_data[0]['zplsc_absorption_coeff'][ii];
        cal_params_tmp['gain'] = config_transducer[ii]['gain']    # data.config(n).gain;
        cal_params_tmp['equivalentbeamangle'] = config_transducer[ii]['equiv_beam_angle']   # data.config(n).equivalentbeamangle;
        cal_params_tmp['pulselengthtable'] = config_transducer[ii]['pulse_length_table']   # data.config(n).pulselengthtable;
        cal_params_tmp['gaintable']  = config_transducer[ii]['gain_table']   # data.config(n).gaintable;
        cal_params_tmp['sacorrectiontable'] = config_transducer[ii]['sa_correction_table']   # data.config(n).sacorrectiontable;
        cal_params_tmp['transmitpower'] = particle_data[0]['zplsc_transmit_power'][ii]   # data.pings(n).transmitpower(pingNo);
        cal_params_tmp['pulselength'] = particle_data[0]['zplsc_pulse_length'][ii]   # data.pings(n).pulselength(pingNo);
        cal_params_tmp['anglesensitivityalongship'] = config_transducer[ii]['angle_sensitivity_alongship']  # data.config(n).anglesensitivityalongship;
        cal_params_tmp['anglesensitivityathwartship'] = config_transducer[ii]['angle_sensitivity_athwartship']   #data.config(n).anglesensitivityathwartship;
        cal_params_tmp['anglesoffsetalongship'] = config_transducer[ii]['angle_offset_alongship']   # data.config(n).anglesoffsetalongship;
        cal_params_tmp['angleoffsetathwartship'] = config_transducer[ii]['angle_offset_athwart']   # data.config(n).angleoffsetathwartship;
        cal_params_tmp['transducerdepth'] = particle_data[0]['zplsc_transducer_depth'][ii]   # data.pings(n).transducerdepth(pingNo);
        cal_params.append(cal_params_tmp)
    return cal_params



def get_noise(power_data,depth_bin_size,ping_bin_range,depth_bin_range,tvgCorrectionFactor=2):
    '''
    INPUT:
        power_data            2D mtx of power data [depth x ping num]
        ping_bin_range        average over M pings
        depth_bin_range       average over depth_bin_range [m]
        tvgCorrectionFactor   default (=2) is to apply TVG correction with offset of 2 samples
                              note this factor is important in TVG compensation
                              and therefore in how power_bin is obtained as well
    OUTPUT:
        minimum value for bins of averaged ping
    '''
    N = int(np.floor(depth_bin_range/depth_bin_size))
    
    # Average uncompensated power over M pings and N depth bins
    depth_bin_num = int(np.floor((power_data.shape[0]-tvgCorrectionFactor)/N))
    ping_bin_num = int(np.floor(power_data.shape[1]/ping_bin_range))
    power_bin = np.empty([depth_bin_num,ping_bin_num])
    for iD in range(depth_bin_num):
        for iP in range(ping_bin_num):
            depth_idx = np.arange(N)+N*iD+tvgCorrectionFactor  # match the 2-sample offset
            ping_idx = np.arange(ping_bin_range)+ping_bin_range*iP
            power_bin[iD,iP] = np.mean(10**(power_data[np.ix_(depth_idx,ping_idx)]/10))

    # Noise = minimum value for each averaged ping
    return np.min(power_bin,0),ping_bin_num



def remove_noise(power_data,cal,noise_est,ping_bin_range=40,tvg_correction_factor=2):
    '''
    Function for noise removal and TVG + absorption compensation
    Ref: De Robertis et al. 2010
    
    INPUT:
        power_data      2D mtx of power data [depth x ping num]
        noise_est       results from `get_noise`
                        if pass in only a scalar, it will be treated as the noise estimate for all data
                        if pass in a vecotr, the noise will be removed adaptively using values in the vector
        ping_bin_range          average over M pings
        depth_bin_range         average over depth_bin_range [m]
        tvg_correction_factor   default(=2) for converting power_data to Sv
    OUTPUT:
        Sv_raw       TVG and absorption compensated Sv data, no noise removal
        Sv_corr      TVG and absorption compensated Sv data, no noise removal        
        Sv_noise     TVG and absorption compensated noise estimation
    '''

    # Get cal params
    f = cal['frequency']
    c = cal['soundvelocity']
    t = cal['sampleinterval']
    alpha = cal['absorptioncoefficient']
    G = cal['gain']
    phi = cal['equivalentbeamangle']
    pt = cal['transmitpower']
    tau = cal['pulselength']

    # key derived params
    dR = c*t/2   # sample thickness
    wvlen = c/f  # wavelength

    # Calc gains
    CSv = 10 * np.log10((pt * (10**(G/10))**2 * wvlen**2 * c * tau * 10**(phi/10)) / (32 * np.pi**2))

    # calculate Sa Correction
    idx = [i for i,dd in enumerate(cal['pulselengthtable']) if dd==tau]
    Sac = 2 * cal['sacorrectiontable'][idx]

    # Get TVG
    range_vec = np.arange(power_data.shape[0]) * dR
    rangeCorrected = range_vec - (tvg_correction_factor * dR)
    rangeCorrected[rangeCorrected<0] = 0

    TVG = np.empty(rangeCorrected.shape)
    TVG[rangeCorrected!=0] = \
        np.real( 20*np.log10(rangeCorrected[rangeCorrected!=0]) )  # TVG = real(20 * log10(rangeCorrected));
    TVG[rangeCorrected==0] = 0

    # Get absorption
    ABS = 2*alpha*rangeCorrected

    # Compensate measurement for noise and corrected for transmission loss
    # also estimate Sv_noise component for subsequent SNR check

    # Noise removal
    if noise_est.shape==():  # if noise_est is single element
        subtract = 10**(power_data/10)-noise_est
        tmp = 10*np.log10(np.ma.masked_less_equal(subtract,0))
        tmp.set_fill_value(-999)
        Sv_corr = (tmp.T+TVG+ABS-CSv-Sac).T
        Sv_noise = 10*np.log10(noise_est)+TVG+ABS-CSv-Sac

    else:
        ping_bin_num = int(np.floor(power_data.shape[1]/ping_bin_range))
        Sv_corr = np.ma.empty(power_data.shape)   # log domain corrected Sv
        Sv_noise = np.empty(power_data.shape)  # Sv_noise
        for iP in range(ping_bin_num):
            ping_idx = np.arange(ping_bin_range) +iP*ping_bin_range
            subtract = 10**(power_data[:,ping_idx]/10) -noise_est[iP]
            tmp = 10*np.log10(np.ma.masked_less_equal(subtract,0))
            tmp.set_fill_value(-999)
            Sv_corr[:,ping_idx] = (tmp.T +TVG+ABS-CSv-Sac).T
            Sv_noise[:,ping_idx] = np.array([10*np.log10(noise_est[iP])+TVG+ABS-CSv-Sac]*ping_bin_range).T
    
    # Raw Sv withour noise removal but with TVG/absorption compensation
    Sv_raw = (power_data.T+TVG+ABS-CSv-Sac).T
    
    return Sv_raw,Sv_corr,Sv_noise



def get_MVBS(Sv,depth_bin_size,ping_bin_range,depth_bin_range):
    '''
    Obtain mean MVBS
    
    INPUT:
        th                Sv threshold: discard Sv values below th during averaging
        depth_bin_size    depth bin size from unpacked data
        ping_bin_range    average over M pings
        depth_bin_range   average over depth_bin_range [m]
    OUTPUT:
        smoothed Sv data
    '''

    N = int(np.floor(depth_bin_range/depth_bin_size))  # total number of depth bins
    
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



def raw2MVBS_daterange(date_wanted,data_path,save_path,save_fname,\
                       ping_time_param,ping_bin_range,depth_bin_range,tvg_correction_factor):
    '''
    Unpack .raw files within a certain date range, estimate noise, clean up data,
    and calculate MVBS using specific params
    '''

    for iD,dd_curr in zip(range(len(date_wanted)),date_wanted):

        # load files and get calibration params
        fname = glob.glob(os.path.join(data_path,'OOI-D%s*.raw' %dd_curr))[0]
        particle_data, data_times, power_data, freq, bin_size, config_header, config_transducer = \
            zplsc_b.parse_echogram_file(fname)
        cal_params = get_cal_params(power_data,particle_data,config_header,config_transducer)

        # swap sequence of 120 kHz and 38 kHz cal_params and data
        cal_params = [cal_params[fn] for fn in [1,0,2]]
        power_data = get_power_data_mtx(power_data,freq)

        # clean data
        Sv_raw_tmp = np.ma.empty((power_data.shape))
        Sv_corr_tmp = np.ma.empty((power_data.shape))
        Sv_noise_tmp = np.ma.empty((power_data.shape[0:2]))
        for fn in range(power_data.shape[0]):
            noise_est,ping_bin_num = get_noise(power_data[fn,:,:],bin_size,ping_bin_range,depth_bin_range)
            Sv_raw_tmp[fn,:,:],Sv_corr_tmp[fn,:,:],Sv_noise_tmp[fn,:] = \
                    remove_noise(power_data[fn,:,:],cal_params[fn],noise_est.min(),ping_bin_range,tvg_correction_factor)

        # set up indexing to get wanted pings
        dd = dt.datetime.strptime(dd_curr,'%Y%m%d')
        time_wanted = [dt.datetime(dd.year,dd.month,dd.day,hh,mm,ss) \
                       for hh in ping_time_param['hour_all']\
                       for mm in ping_time_param['min_all'] \
                       for ss in ping_time_param['sec_all']]
        idx_wanted = [find_nearest_time_idx(data_times,tt,2) for tt in time_wanted]
        notnanidx = np.argwhere(~np.isnan(idx_wanted)).flatten()
        notnanidx_in_all = np.array(idx_wanted)[notnanidx].astype(int)

        # get data to be saved
        ping_per_day = len(ping_time_param['hour_all'])*len(ping_time_param['min_all'])*len(ping_time_param['sec_all'])
        Sv_raw = np.ma.empty((Sv_raw_tmp.shape[0],Sv_raw_tmp.shape[1],ping_per_day))
        Sv_corr = np.ma.empty((Sv_raw_tmp.shape[0],Sv_raw_tmp.shape[1],ping_per_day))
        Sv_noise = np.ma.empty((Sv_raw_tmp.shape[0],Sv_raw_tmp.shape[1],1))

        Sv_raw[:,:,notnanidx] = Sv_raw_tmp[:,:,notnanidx_in_all]
        Sv_corr[:,:,notnanidx] = Sv_corr_tmp[:,:,notnanidx_in_all]
        Sv_noise[:,:,0] = Sv_noise_tmp
        ping_time = date2num(time_wanted)

        idx_save_mask = np.argwhere(np.isnan(idx_wanted))
        Sv_raw[:,:,idx_save_mask] = np.ma.masked
        Sv_corr[:,:,idx_save_mask] = np.ma.masked

        # save into h5 file
        sz = Sv_raw.shape
        f = h5py.File(os.path.join(save_path,'%s_Sv.h5' %save_fname),"a")
        if "Sv_raw" in f:  # if file alread exist and contains Sv mtx
            print '%s  unpacking file: %s' % (dt.datetime.now().strftime('%H:%M:%S'),\
                                              'file exists, append new data mtx...')
            # append new data
            sz_exist = f['Sv_raw'].shape  # shape of existing Sv mtx
            f['Sv_raw'].resize((sz_exist[0],sz_exist[1],sz_exist[2]+sz[2]))
            f['Sv_raw'][:,:,sz_exist[2]:] = Sv_raw
            f['Sv_corr'].resize((sz_exist[0],sz_exist[1],sz_exist[2]+sz[2]))
            f['Sv_corr'][:,:,sz_exist[2]:] = Sv_corr
            f['Sv_noise'].resize((sz_exist[0],sz_exist[1],sz_exist[2]+1))
            f['Sv_noise'][:,:,sz_exist[2]:] = Sv_noise
            f['ping_time'].resize((sz_exist[2]+sz[2],))
            f['ping_time'][sz_exist[2]:] = ping_time
        else:
            print '%s  unpacking file: %s' % (dt.datetime.now().strftime('%H:%M:%S'),\
                                              'new H5 file, create new dataset...')
            # create dataset and save
            f.create_dataset("Sv_raw", sz, maxshape=(sz[0],sz[1],None), data=Sv_raw, chunks=True)
            f.create_dataset("Sv_corr", sz, maxshape=(sz[0],sz[1],None), data=Sv_corr, chunks=True)
            f.create_dataset("Sv_noise", (Sv_raw.shape[0],Sv_raw.shape[1],1), maxshape=(sz[0],sz[1],None),\
                             data=Sv_noise, chunks=True)
            f.create_dataset("ping_time", (sz[2],), maxshape=(None,), data=ping_time, chunks=True)
            f.create_dataset("depth_bin_size",data=bin_size)
        f.close()

        # get MVBS
        MVBS = get_MVBS(Sv_corr,bin_size,ping_bin_range,depth_bin_range)

        # save into h5 file
        sz = MVBS.shape
        ping_time_MVBS = ping_time[0::ping_bin_range]  # get ping time every ping_bin_range for MVBS

        f = h5py.File(os.path.join(save_path,'%s_MVBS.h5' %save_fname),"a")
        if "MVBS" in f:  # if file alread exist and contains Sv mtx
            print '-- H5 file exists, append new data mtx...'
            # append new data
            sz_exist = f['MVBS'].shape  # shape of existing Sv mtx
            f['MVBS'].resize((sz_exist[0],sz_exist[1],sz_exist[2]+sz[2]))
            f['MVBS'][:,:,sz_exist[2]:] = MVBS
            f['ping_time'].resize((sz_exist[2]+sz[2],))
            f['ping_time'][sz_exist[2]:] = ping_time_MVBS
        else:
            print '-- New H5 file, create new dataset...'
            # create dataset and save
            f.create_dataset("MVBS", sz, maxshape=(sz[0],sz[1],None), data=MVBS, chunks=True)
            f.create_dataset("ping_time", (sz[2],), maxshape=(None,), data=ping_time_MVBS, chunks=True)
            f.create_dataset("depth_bin_size",data=depth_bin_range)
        f.close()
        


def multifreq_color_code(Sv38,Sv120,Sv200):
    '''
    Multi-frequency color-coding regiem
    Ref: Jech and Michaels 2006
    
    INPUT:
        Sv at 38, 120, and 200 kHz
        
    OUTPUT:
        Sv_mf   numerical indicator mtx of combination of
                presence/absence of multiple freq
    '''
    yes_1 = ~np.isnan(Sv38)
    yes_2 = ~np.isnan(Sv120)
    yes_3 = ~np.isnan(Sv200)

    Sv_mf = np.empty(Sv38)
    Sv_mf[~yes_1 & ~yes_2 & ~yes_3] = 0
    Sv_mf[ yes_1 & ~yes_2 & ~yes_3] = 1
    Sv_mf[ yes_1 &  yes_2 & ~yes_3] = 2
    Sv_mf[ yes_1 &  yes_2 &  yes_3] = 3
    Sv_mf[ yes_1 & ~yes_2 &  yes_3] = 4
    Sv_mf[~yes_1 &  yes_2 & ~yes_3] = 5
    Sv_mf[~yes_1 &  yes_2 &  yes_3] = 6
    Sv_mf[~yes_1 & ~yes_2 &  yes_3] = 7

    return Sv_mf



def get_power_data_mtx(data_dict,frequencies):
    '''
    Convert data_dict to numpy array
    Input:
        data_dict     power_data_dict or Sv from power2Sv()
        frequencies   unpacked dict from parse_echogram_file()
    '''
    fval = frequencies.values()
    fidx = sorted(range(len(fval)), key=lambda k: fval[k])   # get key sequence for low to high freq
    fidx = [x+1 for x in fidx]
    return np.array((data_dict[fidx[0]],\
                     data_dict[fidx[1]],\
                     data_dict[fidx[2]]))  # organize all values into matrix



def find_nearest_time_idx(all_timestamp_num,time_wanted,tolerance):
    '''
    Function to find nearest element
    time_wanted is a datetime object
    tolerance is the max tolerance in second allowed between `time_wanted` and `all_timestamp`
    all_timestamp_num is a numerical date object (i.e., output from `date2num`)
    '''
    time_wanted_num = date2num(time_wanted)
    idx = np.searchsorted(all_timestamp_num, time_wanted_num, side="left")
    if idx > 0 and (idx == len(all_timestamp_num) or \
        np.abs(time_wanted_num - all_timestamp_num[idx-1]) < np.abs(time_wanted_num - all_timestamp_num[idx])):
        idx -= 1

    # If interval between the selected index and time wanted > `tolerance` seconds
    sec_diff = dt.timedelta(all_timestamp_num[idx]-time_wanted_num).total_seconds()
    if np.abs(sec_diff)>tolerance:
        return np.nan
    else:
        return idx


def plot_echogram(V,plot_start_day,plot_range_day,plot_param,fig_size=(16,7),cmap_name='viridis'):

    x_ticks_spacing = plot_param["x_ticks_spacing"]  # spacing: in num of days
    y_ticks_num = plot_param["y_ticks_num"]
    y_start_idx = plot_param["y_start_idx"]
    y_end_idx = plot_param["y_end_idx"]
    y_offset_idx = plot_param["y_offset_idx"]
    c_min = plot_param["c_min"]
    c_max = plot_param["c_max"]
    c_ticks_spacing = plot_param["c_ticks_spacing"]
    ping_per_day_mvbs = plot_param["ping_per_day_mvbs"]
    depth_bin_size = plot_param["depth_bin_size"]
    ping_time = plot_param["ping_time"]

    v_mtx = V[:,y_start_idx:(V.shape[1]+y_end_idx),\
                 ping_per_day_mvbs*(plot_start_day-1)+np.arange(ping_per_day_mvbs*plot_range_day)]

    y_ticks_spacing = np.floor(v_mtx.shape[1]/(y_ticks_num-1)).astype(int)
    y_ticks = np.arange(0,v_mtx.shape[1],y_ticks_spacing)
    y_ticklabels = y_ticks*depth_bin_size + (y_start_idx+y_offset_idx)*depth_bin_size

    x_ticks = np.arange(0,plot_range_day,x_ticks_spacing)*ping_per_day_mvbs
    x_ticks_in_ping_time = np.arange(plot_start_day-1,plot_start_day-1+plot_range_day,x_ticks_spacing)*ping_per_day_mvbs
    x_ticklabels = [num2date(xx).strftime('%m/%d') for xx in ping_time[x_ticks_in_ping_time]]
    #x_ticklabels = [num2date(xx).strftime('%m/%d') for xx in ping_time[x_ticks[1:]]]
    #x_ticklabels.insert(0,num2date(ping_time[x_ticks[0]]).strftime('%b-%d'))

    c_ticks = np.arange(c_min,c_max+c_ticks_spacing,c_ticks_spacing)

    fig,ax = plt.subplots(3,1,figsize=fig_size,sharex=True)
    for iX in range(3):
        im = ax[iX].imshow(v_mtx[iX,::-1,:],aspect='auto',\
                           vmax=c_max,vmin=c_min,cmap=cmap_name)
        divider = make_axes_locatable(ax[iX])
        cax = divider.append_axes("right", size="1%", pad=0.1)
        cbar = plt.colorbar(im,cax=cax,ticks=c_ticks)
        ax[iX].set_yticks(y_ticks)
        ax[iX].set_yticklabels(y_ticklabels,fontsize=14)
        ax[iX].set_ylabel('Depth (m)',fontsize=16)
        if iX==2:
            ax[iX].set_xticks(x_ticks)
            ax[iX].set_xticklabels(x_ticklabels,fontsize=14)
            ax[iX].set_xlabel('Date',fontsize=16)
        #if iX==0:
        #    ax[iX].set_title('38 kHz',fontsize=14)
        #elif iX==1:
        #    ax[iX].set_title('120 kHz',fontsize=14)
        #else:
        #    ax[iX].set_title('200 kHz',fontsize=14)
        if plot_range_day<=20:  # if short time plot day separator
            for dd in range(1,plot_range_day):
                ax[iX].plot(np.array((dd,dd))*ping_per_day_mvbs,(0,v_mtx.shape[1]),'--',color=(0.8,0.8,0.8))
    plt.tight_layout(h_pad=0.1)