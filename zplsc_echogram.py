"""
@package mi.instrument.kut.ek60.ooicore.driver
@file marine-integrations/mi/instrument/kut/ek60/ooicore/driver.py
@author Craig Risien
@brief ZPLSC Echogram generation for the ooicore

Release notes:

This class supports the generation of ZPLSC echograms. It needs matplotlib version 1.3.1 for the code to display the
colorbar at the bottom of the figure. If matplotlib version 1.1.1 is used, the colorbar is plotted over the
figure instead of at the bottom of it.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date
from modest_image import imshow

from datetime import datetime

import re
import numpy as np

from struct import unpack

__author__ = 'Craig Risien from OSU'
__license__ = 'Apache 2.0'

LENGTH_SIZE = 4
DATAGRAM_HEADER_SIZE = 12
CONFIG_HEADER_SIZE = 516
CONFIG_TRANSDUCER_SIZE = 320
TRANSDUCER_1 = 'Transducer # 1: '
TRANSDUCER_2 = 'Transducer # 2: '
TRANSDUCER_3 = 'Transducer # 3: '

# Reference time "seconds since 1900-01-01 00:00:00"
REF_TIME = date2num(datetime(1900, 1, 1, 0, 0, 0))

# set global regex expressions to find all sample, annotation and NMEA sentences
SAMPLE_REGEX = r'RAW\d{1}'
SAMPLE_MATCHER = re.compile(SAMPLE_REGEX, re.DOTALL)

ANNOTATE_REGEX = r'TAG\d{1}'
ANNOTATE_MATCHER = re.compile(ANNOTATE_REGEX, re.DOTALL)

NMEA_REGEX = r'NME\d{1}'
NMEA_MATCHER = re.compile(NMEA_REGEX, re.DOTALL)


###########################################################################
# ZPLSC Echogram
###########################################################################

####################################################################################
# Create functions to read the datagrams contained in the raw file. The
# code below was developed using example Matlab code produced by Lars Nonboe
# Andersen of Simrad and provided by Dr. Kelly Benoit-Bird and the
# raw data file format specification in the Simrad EK60 manual, with reference
# to code in Rick Towler's readEKraw toolbox.
def read_datagram_header(chunk):
    """
    Reads the EK60 raw data file datagram header
    @param chunk data chunk to read the datagram header from
    @return: datagram header
    """
    # setup unpack structure and field names
    field_names = ('datagram_type', 'internal_time')
    fmt = '<4sll'

    # read in the values from the byte string chunk
    values = unpack(fmt, chunk)

    # the internal date time structure represents the number of 100
    # nanosecond intervals since January 1, 1601. this is known as the
    # Windows NT Time Format.
    internal = values[2] * (2 ** 32) + values[1]

    # create the datagram header dictionary
    datagram_header = dict(zip(field_names, [values[0], internal]))
    return datagram_header


def read_config_header(chunk):
    """
    Reads the EK60 raw data file configuration header information
    from the byte string passed in as a chunk
    @param chunk data chunk to read the config header from
    @return: configuration header
    """
    # setup unpack structure and field names
    field_names = ('survey_name', 'transect_name', 'sounder_name',
                   'version', 'transducer_count')
    fmt = '<128s128s128s30s98sl'

    # read in the values from the byte string chunk
    values = list(unpack(fmt, chunk))
    values.pop(4)  # drop the spare field

    # strip the trailing zero byte padding from the strings
    # for i in [0, 1, 2, 3]:
    for i in xrange(4):
        values[i] = values[i].strip('\x00')

    # create the configuration header dictionary
    config_header = dict(zip(field_names, values))
    return config_header


def read_config_transducer(chunk):
    """
    Reads the EK60 raw data file configuration transducer information
    from the byte string passed in as a chunk
    @param chunk data chunk to read the configuration transducer information from
    @return: configuration transducer information
    """

    # setup unpack structure and field names
    field_names = ('channel_id', 'beam_type', 'frequency', 'gain',
                   'equiv_beam_angle', 'beam_width_alongship', 'beam_width_athwartship',
                   'angle_sensitivity_alongship', 'angle_sensitivity_athwartship',
                   'angle_offset_alongship', 'angle_offset_athwart', 'pos_x', 'pos_y',
                   'pos_z', 'dir_x', 'dir_y', 'dir_z', 'pulse_length_table', 'gain_table',
                   'sa_correction_table', 'gpt_software_version')
    fmt = '<128sl15f5f8s5f8s5f8s16s28s'

    # read in the values from the byte string chunk
    values = list(unpack(fmt, chunk))

    # convert some of the values to arrays
    pulse_length_table = np.array(values[17:22])
    gain_table = np.array(values[23:28])
    sa_correction_table = np.array(values[29:34])

    # strip the trailing zero byte padding from the strings
    for i in [0, 35]:
        values[i] = values[i].strip('\x00')

    # put it back together, dropping the spare strings
    config_transducer = dict(zip(field_names[0:17], values[0:17]))
    config_transducer[field_names[17]] = pulse_length_table
    config_transducer[field_names[18]] = gain_table
    config_transducer[field_names[19]] = sa_correction_table
    config_transducer[field_names[20]] = values[35]
    return config_transducer


class ZPLSPlot(object):
    font_size_small = 14
    font_size_large = 18
    num_xticks = 25
    num_yticks = 7
    interplot_spacing = 0.1
    lower_percentile = 5
    upper_percentile = 95

    def __init__(self, data_times, power_data_dict, frequency_dict, bin_size):
        self.power_data_dict = self._transpose_and_flip(power_data_dict)
        self.min_db, self.max_db = self._get_power_range(power_data_dict)
        self.frequency_dict = frequency_dict

        # convert ntp time, i.e. seconds since 1900-01-01 00:00:00 to matplotlib time
        self.data_times = (data_times / (60 * 60 * 24)) + REF_TIME
        max_depth, _ = self.power_data_dict[1].shape
        self._setup_plot(bin_size, max_depth)

    def generate_plots(self):
        """
        Generate plots for all transducers in data set
        """
        freq_to_channel = {v: k for k, v in self.frequency_dict.iteritems()}
        data_axes = None
        for index, frequency in enumerate(sorted(freq_to_channel)):
            channel = freq_to_channel[frequency]
            td_f = self.frequency_dict[channel]
            title = 'Power: Transducer #%d: Frequency: %0.1f kHz' % (channel, td_f / 1000)
            data_axes = self._generate_plot(self.ax[index], self.power_data_dict[channel], title,
                                            self.min_db, self.max_db)

        if data_axes:
            self._display_x_labels(self.ax[2], self.data_times)
            self.fig.tight_layout(rect=[0, 0.0, 0.97, 1.0])
            self._display_colorbar(self.fig, data_axes)

    def write_image(self, filename):
        self.fig.savefig(filename)
        plt.close(self.fig)
        self.fig = None

    def _setup_plot(self, bin_size, max_depth):
        # subset the yticks so that we don't plot every one
        yticks = np.linspace(0, max_depth, self.num_yticks)
        # create range vector (depth in meters)
        yticklabels = np.round(np.linspace(0, max_depth * bin_size, self.num_yticks)).astype(int)

        self.fig, self.ax = plt.subplots(len(self.frequency_dict), sharex=True, sharey=True)
        self.fig.subplots_adjust(hspace=self.interplot_spacing)
        self.fig.set_size_inches(40, 19)

        for axes in self.ax:
            axes.grid(False)
            axes.set_ylabel('depth (m)', fontsize=self.font_size_small)
            axes.set_yticks(yticks)
            axes.set_yticklabels(yticklabels, fontsize=self.font_size_small)
            axes.tick_params(axis="both", labelcolor="k", pad=4, direction='out', length=5, width=2)
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.spines['bottom'].set_visible(False)
            axes.spines['left'].set_visible(False)

    @staticmethod
    def _get_power_range(power_dict):
        # Calculate the power data range across all channels
        all_power_data = np.concatenate(power_dict.values())
        max_db = np.nanpercentile(all_power_data, ZPLSPlot.upper_percentile)
        min_db = np.nanpercentile(all_power_data, ZPLSPlot.lower_percentile)
        return min_db, max_db

    @staticmethod
    def _transpose_and_flip(power_dict):
        for channel in power_dict:
            # Transpose array data so we have time on the x-axis and depth on the y-axis
            power_dict[channel] = power_dict[channel].transpose()
            # reverse the Y axis (so depth is measured from the surface (at the top) to the ZPLS (at the bottom)
            power_dict[channel] = power_dict[channel][::-1]
        return power_dict

    @staticmethod
    def _generate_plot(ax, power_data, title, min_db, max_db):
        """
        Generate a ZPLS plot for an individual channel
        :param ax:  matplotlib axis to receive the plot image
        :param power_data:  Transducer data array
        :param data_times:  Transducer internal time array
        :param title:  plot title
        :param min_db: minimum power level
        :param max_db: maximum power level
        """
        # only generate plots for the transducers that have data
        if power_data.size <= 0:
            return

        ax.set_title(title, fontsize=ZPLSPlot.font_size_large)
        return imshow(ax, power_data, interpolation='none', aspect='auto', cmap='jet', vmin=min_db, vmax=max_db)

    @staticmethod
    def _display_x_labels(ax, data_times):
        time_format = '%Y-%m-%d\n%H:%M:%S'
        time_length = data_times.size
        # X axis label
        # subset the xticks so that we don't plot every one
        xticks = np.linspace(0, time_length, ZPLSPlot.num_xticks)
        xstep = int(round(xticks[1]))
        # format trans_array_time array so that it can be used to label the x-axis
        xticklabels = [i for i in num2date(data_times[::xstep])] + [num2date(data_times[-1])]
        xticklabels = [i.strftime(time_format) for i in xticklabels]

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        ax.set_xlabel('time (UTC)', fontsize=ZPLSPlot.font_size_small)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, horizontalalignment='center', fontsize=ZPLSPlot.font_size_small)
        ax.set_xlim(0, time_length)

    @staticmethod
    def _display_colorbar(fig, data_axes):
        # Add a colorbar to the specified figure using the data from the given axes
        ax = fig.add_axes([0.965, 0.12, 0.01, 0.775])
        cb = fig.colorbar(data_axes, cax=ax, use_gridspec=True)
        cb.set_label('dB', fontsize=ZPLSPlot.font_size_large)
        cb.ax.tick_params(labelsize=ZPLSPlot.font_size_small)
