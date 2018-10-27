
import glob, os, sys
import datetime as dt  # quick fix to avoid datetime and datetime.datetime confusion
from matplotlib.dates import date2num, num2date
from calendar import monthrange
import h5py
import matplotlib.pylab as plt
# from modest_image import imshow
# import numpy as np   # already imported in zplsc_b

sys.path.append('/Users/wujung/code/mi-instrument/')
from mi.instrument.kut.ek60.ooicore.zplsc_b import *



def get_data_mtx(data_dict,frequencies):
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


def get_date_idx(date_wanted,fname_all):
    '''
    Index the files in the wanted date range
    '''
    raw_file_times = [FILE_NAME_MATCHER.match(x) for x in fname_all];
    # idx = [x for x in range(len(X)) if X[x].group('Date')=='20150912']  # solution 1
    date_list = map(lambda x: x.group('Date'),raw_file_times)  # if in python 3 need to do list(map()) to convert map object to list
    time_list = map(lambda x: x.group('Time'),raw_file_times)

    if len(date_wanted)==1:
        idx_date = [i for i,dd in enumerate(date_list) if dd==date_wanted[0]]
    elif len(date_wanted)==2:
        idx_date = [i for i,dd in enumerate(date_list) if dd>=date_wanted[0] and dd<=date_wanted[-1]]
    else:
        print 'Invalid range: date_wanted!'
        idx_date = []

    if len(idx_date)!=0:
        # if midnight was recorded in the previous file
        # AND if the previous record is the day before
        day_diff = dt.datetime.strptime(date_list[idx_date[0]], "%Y%m%d").toordinal() -\
                   dt.datetime.strptime(date_list[idx_date[0]-1], "%Y%m%d").toordinal()
        if time_list[idx_date[0]]>'000000' and day_diff==1:
            idx_date.insert(0,idx_date[0]-1)
    else:
        print 'date wanted does not exist!'
    return idx_date


def unpack_raw_to_h5(fname,h5_fname,deci_len=[]):
    '''
    Unpack *.raw files and save directly into h5 files
    INPUT:
        fname      file to be unpacked
        h5_fname   hdf5 file to be written in to
        deci_len   number of pings to skip over,
                   default=[], i.e., no skipping
    '''
    # Unpack data
    particle_data, data_times, power_data_dict, freq, bin_size, config_header, config_transducer = \
        parse_echogram_file(fname)

    # Open hdf5 file
    f = h5py.File(h5_fname,"a")

    # Convert from power to Sv
    cal_params = get_cal_params(power_data_dict,particle_data,config_header,config_transducer)
    Sv = power2Sv(power_data_dict,cal_params)  # convert from power to Sv
    Sv_mtx = get_data_mtx(Sv,freq)

    if deci_len:
        Sv_mtx = Sv_mtx[:,:,::deci_len]
        data_times = data_times[::deci_len]

    sz = Sv_mtx.shape

    # Write to hdf5 file
    if "Sv" in f:  # if file alread exist and contains Sv mtx
        # Check if start time of this file is before last time point of last file
        # Is yes, discard current file and break out from function
        time_diff = dt.timedelta(data_times[0]-f['data_times'][-1])
        hr_diff = (time_diff.days*86400+time_diff.seconds)/3600
        if hr_diff<0:
            print '-- New file time bad'
            return
        else:
            print '-- H5 file exists, append new data mtx...'
            # append new data
            sz_exist = f['Sv'].shape  # shape of existing Sv mtx
            f['Sv'].resize((sz_exist[0],sz_exist[1],sz_exist[2]+sz[2]))
            f['Sv'][:,:,sz_exist[2]:] = Sv_mtx
            f['data_times'].resize((sz_exist[2]+sz[2],))
            f['data_times'][sz_exist[2]:] = data_times
    else:
        print '-- New H5 file, create new dataset...'
        # create dataset and save Sv
        f.create_dataset("Sv", sz, maxshape=(sz[0],sz[1],None), data=Sv_mtx, chunks=True)
        f.create_dataset("data_times", (sz[2],), maxshape=(None,), data=data_times, chunks=True)
        # label dimensions
        f['Sv'].dims[0].label = 'frequency'
        f['Sv'].dims[1].label = 'depth'
        f['Sv'].dims[2].label = 'time'
        # create frequency dimension scale, use f['Sv'].dims[0][0][0] to access
        f['frequency'] = freq.values()
        f['Sv'].dims.create_scale(f['frequency'])
        f['Sv'].dims[0].attach_scale(f['frequency'])
        # save bin_size
        f.create_dataset("bin_size",data=bin_size)
    f.close()


# print 'original size is ' + str({k: v.shape for (k,v) in power_data_dict.items()})
# freq = frequencies.value()  # get freuqency values
# freq = {k: str(int(v/1E3))+'k' for (k,v) in frequencies.items()}  # get frequencies


# plotting
def plot_Sv(h5_fname,save_path):

    f = h5py.File(h5_fname,"r")

    # Get time stamp
    time_format = '%Y-%m-%d\n%H:%M:%S'
    time_length = f['data_times'].size
    # X axis label
    # subset the xticks so that we don't plot every one
    xticks = np.linspace(0, time_length, 11)
    xstep = int(round(xticks[1]))
    # format trans_array_time array so that it can be used to label the x-axis
    xticklabels = [i for i in num2date(f['data_times'][::xstep])] + [num2date(f['data_times'][-1])]
    xticklabels = [i.strftime(time_format) for i in xticklabels]

    # Plot figure
    print 'plotting figure...'
    fig, ax = plt.subplots(3, sharex=True)
    for ff in range(f['Sv'].shape[0]):
        imshow(ax[ff],f['Sv'][ff,:,:],aspect='auto',vmax=-34,vmin=-80,interpolation='none')
    ax[-1].set_xlabel('time (UTC)')
    ax[-1].set_xticks(xticks)
    ax[-1].set_xticklabels(xticklabels, rotation=45, horizontalalignment='center')
    #ax[-1].set_xlim(0, time_length)
    fig.set_figwidth(16)
    fig.set_figheight(10)

    # Save figure
    save_fname = os.path.join(save_path,h5_fname+'.png')
    print 'saving figure...'
    fig.savefig(save_fname)
    plt.close(fig)



def get_num_days_pings(h5_fname):
    ''' Get the total number of days and number of pings per day for the given year and month '''
    H5_FILENAME_MATCHER = re.compile('(?P<SITE_CODE>\S*)_(?P<YearMonth>\S*)\.\S*')

    # Get month and day range
    ym = datetime.datetime.strptime(H5_FILENAME_MATCHER.match(h5_fname).group('YearMonth'),'%Y%m')
    year = ym.year
    month = ym.month
    _,daynum = monthrange(year,month)

    # Get datetime object for on the hour every hour in all days in the month
    all_day = range(1,daynum+1)  # list of all days
    all_hr = range(24)  # list of all hour: 0-23
    all_minutes = range(1,11)  # list of all minutes: 0-9
    every_ping = [datetime.datetime(year,month,day,hr,minutes,0) \
                for day in all_day for hr in all_hr for minutes in all_minutes]
    pings_per_day = len(all_hr)*len(all_minutes)
    return pings_per_day,daynum


def get_data_from_h5(data_path,h5_fname):
    ''' Retrieve data from h5 files '''
    f = h5py.File(os.path.join(data_path,h5_fname),'r')

    # Get month and day range
    ym = datetime.datetime.strptime(H5_FILENAME_MATCHER.match(h5_fname).group('YearMonth'),'%Y%m')
    year = ym.year
    month = ym.month
    _,daynum = monthrange(year,month)

    # Get datetime object for on the hour every hour in all days in the month
    all_day = range(1,daynum+1)  # list of all days
    all_hr = range(24)  # list of all hour: 0-23
    all_minutes = range(1,11)  # list of all minutes: 0-9
    every_ping = [datetime.datetime(year,month,day,hr,minutes,0) \
                for day in all_day for hr in all_hr for minutes in all_minutes]
    pings_per_day = len(all_hr)*len(all_minutes)

    # Get f['data_times'] idx for every hour in all days in the month
    all_idx = [find_nearest_time_idx(f['data_times'],hr) for hr in every_ping]
    all_idx = np.array(all_idx)  # to allow numpy operation

    # Clean up all_idx
    # --> throw away days with more than 5 pings missing
    # --> fill in occasional missing pings with neighboring values
    all_idx_rshp = np.reshape(all_idx,(-1,pings_per_day))
    num_nan_of_day = np.sum(np.isnan(all_idx_rshp),1)
    for day in range(len(num_nan_of_day)):
        if num_nan_of_day[day]>5:
            all_idx_rshp[day,:] = np.nan
    all_idx = np.reshape(all_idx_rshp,-1)

    # Extract timing and Sv data
    notnanidx = np.int_(all_idx[~np.isnan(all_idx)])
    data_times = np.empty(all_idx.shape)  # initialize empty array
    data_times[~np.isnan(all_idx)] = f['data_times'][notnanidx.tolist()]
  
    Sv_tmp = f['Sv'][:,:,0]
    Sv_mtx = np.empty((Sv_tmp.shape[0],Sv_tmp.shape[1],all_idx.shape[0]))
    Sv_mtx[:,:,~np.isnan(all_idx)] = f['Sv'][:,:,notnanidx.tolist()]

    bin_size = f['bin_size'][0]      # size of each depth bin

    return Sv_mtx,data_times,bin_size,pings_per_day,all_idx


