"""
Classes for visualizing echo data
"""

import numpy as np
import matplotlib.colors as colors
# import datetime as dt
from matplotlib.dates import date2num
from collections import defaultdict
import echopype_model


# Colormap: multi-frequency availability from Jech & Michaels 2006
MF_COLORS = np.array([[0,0,0],\
                      [86,25,148],\
                      [28,33,179],\
                      [0,207,239],\
                      [41,171,71],\
                      [51,204,51],\
                      [255,239,0],\
                      [255,51,0]])/255.
MF_CMAP_TMP = colors.ListedColormap(MF_COLORS)
MF_CMAP = colors.BoundaryNorm(range(MF_COLORS.shape[0]+1),MF_CMAP_TMP.N)


# Colormap: standard EK60
EK60_COLORS = np.array([[255, 255, 255],\
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
EK60_CMAP_TH = [-80,-30]
EK60_CMAP_TMP = colors.ListedColormap(EK60_COLORS)
EK60_CMAP_BOUNDS = np.linspace(EK60_CMAP_TH[0],EK60_CMAP_TH[1],EK60_CMAP_TMP.N+1)
EK60_CMAP = colors.BoundaryNorm(EK60_CMAP_BOUNDS,EK60_CMAP_TMP.N)


# Default plot echogram params
ECHOGRAM_PARAMS = defaultdict(list)
ECHOGRAM_PARAMS['hour_spacing'] = 1      # spacing: in [hour]
ECHOGRAM_PARAMS['depth_ticks_num'] = 5   # number of tick marks on y-axis
ECHOGRAM_PARAMS['depth_min'] = 0         # min depth to plot [m]
ECHOGRAM_PARAMS['depth_max'] = 200       # max depth to plot [m]
ECHOGRAM_PARAMS['cmap_min'] = -80        # min of color scale
ECHOGRAM_PARAMS['cmap_max'] = -30        # max of color scale
ECHOGRAM_PARAMS['c_ticks_spacing']       # number of ticks in colorbar


class EchoDataViewer(object):
    '''
    Class to view echo data.
    Input can be a EchoDataRaw (now) or EchoDataMVBS (future)
    '''
    def __init__(self,echo_object='',fig_size=[15,8],cmap='viridis'):
        self.fig_size = fig_size
        self.cmap = cmap
        if echo_object=='':
            self.echo_data_type = ''
            self.echo_vals = []
            self.frequency = []
            self.depth_bin = []
            self.ping_bin = []
            self.ping_time = []
        else:
            self.echo_object = echo_object
            self.load_echo_info()
        # Emtpy attributes that will only be evaluated when associated methods called by user
        self.date_range = []

    # Methods to set/get critical attributes
    def load_echo_info(self):
        if isinstance(self.echo_object,(echopype_model.EchoDataRaw)):
            self.echo_data_type = 'EchoDataRaw'
            self.echo_vals = self.echo_object.Sv_corrected
            self.depth_bin = self.echo_object.bin_size  # this is for Sv data
        else:
            self.echo_data_type = 'others'  # add other types later
            self.echo_vals = []
            self.depth_bin = self.echo_object.depth_bin  # this is for MVBS
        self.frequency = [float(x) for x in self.echo_object.Sv_corrected.keys()]  # get frequency info
        self.ping_bin = self.echo_object.ping_bin
        self.ping_time = self.echo_object.ping_time

    def set_date_range(self,date_range):
        self.date_range = date_range

    def set_fig_size(self,fig_size):
        self.fig_size = fig_size

    def set_cmap(self,cmap):
        self.cmap = cmap

    # Methods for visualization
    def echogram(self,ax,echogram_params,freq_select):
        '''
        Plot echogram for selected frequencies
        INPUT:
            ax    axis the echogram to be plot on
            echogram_params   plotting parameters
            freq_select       selected frequency (dtype=float)
        '''
        freq_idx = self.find_freq_seq(freq_select)
        sz = self.echo_vals[str(self.frequency[freq_idx])].shape
        # Getting start and end ping indices
        if self.date_range=='':
            print('No data range set. Use set_date_range to select date range to plot')
            return
        else:
            date_num_range = date2num(self.date_range)
            ping_idx_start = np.searchsorted(self.ping_time, date_num_range[0], side="left")
            ping_idx_end = np.searchsorted(self.ping_time, date_num_range[1], side="right")
            if ping_idx_end>=self.ping_time.shape[0]:
                ping_idx_end = self.ping_time.shape[0]-1

        # Getting start and end depth indices
        depth_vec = np.arange(sz[0])*self.depth_bin  # list of depth bins [m]
        depth_idx_start = np.searchsorted(depth_vec, echogram_params['depth_min'], side="left")
        depth_idx_end = np.searchsorted(depth_vec, echogram_params['depth_max'], side="right")
        if depth_idx_end>=depth_vec.shape[0]:
            depth_idx_end = depth_vec.shape[0]-1

        # Set up xticks -- day
        del_time = (self.date_range[1]-self.date_range[0])
        x_ticks_num = (del_time.days*24+del_time.seconds/60/60)/echogram_params['hour_spacing']
        x_ticks_spacing = sz[1]/(x_ticks_num)
        x_ticks = np.arange(0,sz[1],x_ticks_spacing)
        x_ticks_label = np.arange(x_ticks.shape[0])*echogram_params['hour_spacing']  # this probably should be outside of the function

        # Set up yticks -- depth
        y_ticks_spacing = sz[0]/(echogram_params['depth_ticks_num']-1)
        y_ticks = np.arange(echogram_params['depth_ticks_num'])*y_ticks_spacing
        depth_spacing = np.around((echogram_params['depth_max']-\
                        echogram_params['depth_min'])/(echogram_params['depth_ticks_num']-1),decimals=1)
        depth_label = np.around(np.arange(echogram_params['depth_ticks_num'])*depth_spacing,decimals=1)

        # Plot
        # -- plot echo_vals upside-down
        axim = ax.imshow(self.echo_vals[str(self.frequency[freq_idx])][::-1,:],aspect='auto',\
                  vmax=echogram_params['cmap_max'],vmin=echogram_params['cmap_min'],cmap=self.cmap)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(depth_label)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks_label)

        return axim


    def find_freq_seq(self,freq_select):
        '''Find the sequence of transducer of a particular freq'''
        return int(np.where(np.array(self.frequency)==freq_select)[0])
