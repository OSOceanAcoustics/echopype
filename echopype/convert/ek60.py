"""
Functions to unpack Simrad EK60 .raw data file and save to .nc.
"""


import re
import os
from collections import defaultdict
import numpy as np
from datetime import datetime as dt
import pytz
import pynmea2

from echopype.convert.utils.ek_raw_io import RawSimradFile, SimradEOF
from echopype.convert.utils.nmea_data import NMEAData
from echopype.convert.utils.set_groups import SetGroups
from echopype._version import get_versions
from .convertbase import ConvertBase
ECHOPYPE_VERSION = get_versions()['version']
del get_versions


# Create a constant to convert indexed power to power.
INDEX2POWER = (10.0 * np.log10(2.0) / 256.0)

# Create a constant to convert from indexed angles to electrical angles.
INDEX2ELEC = 180.0 / 128.0


class ConvertEK60(ConvertBase):
    """Class for converting EK60 .raw files."""
    def __init__(self, _filename=""):
        ConvertBase.__init__(self)
        self.filename = _filename  # path to EK60 .raw filename to be parsed

        # Initialize file parsing storage variables
        self.config_datagram = None
        self.nmea_data = NMEAData()  # object for NMEA data
        self.ping_data_dict = {}   # dictionary to store metadata
        self.power_dict = {}   # dictionary to store power data
        self.angle_dict = {}   # dictionary to store angle data
        self.ping_time = []    # list to store ping time
        self.CON1_datagram = None    # storage for CON1 datagram for ME70
        self.nc_path = None
        self.zarr_path = None

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, p):
        pp = os.path.basename(p)
        _, ext = os.path.splitext(pp)
        if ext != '.raw':
            raise ValueError('Please specify a .raw file.')
            # print('Data file in manufacturer format, please convert first.')
            # print('To convert data, follow the steps below:')
        else:
            self._filename = p

    def _append_channel_ping_data(self, ch_num, datagram):
        """ Append ping-by-ping channel metadata extracted from the newly read datagram of type 'RAW'.

        Parameters
        ----------
        ch_num : int
            number of the channel to append metadata to
        datagram : dict
            the newly read datagram of type 'RAW'
        """
        self.ping_data_dict[ch_num]['mode'].append(datagram['mode'])
        self.ping_data_dict[ch_num]['transducer_depth'].append(datagram['transducer_depth'])
        self.ping_data_dict[ch_num]['transmit_power'].append(datagram['transmit_power'])
        self.ping_data_dict[ch_num]['pulse_length'].append(datagram['pulse_length'])
        self.ping_data_dict[ch_num]['bandwidth'].append(datagram['bandwidth'])
        self.ping_data_dict[ch_num]['sample_interval'].append(datagram['sample_interval'])
        self.ping_data_dict[ch_num]['sound_velocity'].append(datagram['sound_velocity'])
        self.ping_data_dict[ch_num]['absorption_coefficient'].append(datagram['absorption_coefficient'])
        self.ping_data_dict[ch_num]['heave'].append(datagram['heave'])
        self.ping_data_dict[ch_num]['roll'].append(datagram['roll'])
        self.ping_data_dict[ch_num]['pitch'].append(datagram['pitch'])
        self.ping_data_dict[ch_num]['temperature'].append(datagram['temperature'])
        self.ping_data_dict[ch_num]['heading'].append(datagram['heading'])

    def _read_datagrams(self, fid):
        """
        Read various datagrams until the end of a ``.raw`` file.

        Only includes code for storing RAW and NMEA datagrams and
        ignoring the TAG, BOT, and DEP datagrams.

        Parameters
        ----------
        fid
            a RawSimradFile file object opened in ``self.load_ek60_raw()``
        """
        num_datagrams_parsed = 0
        tmp_num_ch_per_ping_parsed = 0  # number of channels of the same ping parsed
                                        # this is used to control saving only pings
                                        # that have all freq channels present
        tmp_datagram_dict = []  # tmp list of datagrams, only saved to actual output
                                # structure if data from all freq channels are present

        while True:
            try:
                new_datagram = fid.read(1)
            except SimradEOF:
                break

            # Convert the timestamp to a datetime64 object.
            new_datagram['timestamp'] = np.datetime64(new_datagram['timestamp'].replace(tzinfo=None), '[ms]')

            num_datagrams_parsed += 1

            # RAW datagrams store raw acoustic data for a channel
            if new_datagram['type'].startswith('RAW'):
                curr_ch_num = new_datagram['channel']

                # Reset counter and storage for parsed number of channels
                # if encountering datagram from the first channel
                if curr_ch_num == 1:
                    tmp_num_ch_per_ping_parsed = 0
                    tmp_datagram_dict = []

                # Save datagram temporarily before knowing if all freq channels are present
                tmp_num_ch_per_ping_parsed += 1
                tmp_datagram_dict.append(new_datagram)

                # Actually save datagram when all freq channels are present
                if np.all(np.array([curr_ch_num, tmp_num_ch_per_ping_parsed])
                          == self.config_datagram['transceiver_count']):

                    # append ping time from first channel
                    self.ping_time.append(tmp_datagram_dict[0]['timestamp'])

                    for ch_seq in range(self.config_datagram['transceiver_count']):
                        # If frequency matches for this channel, actually store data
                        # Note all storage structure indices are 1-based since they are indexed by
                        # the channel number as stored in config_datagram['transceivers'].keys()
                        if self.config_datagram['transceivers'][ch_seq+1]['frequency'] \
                                == tmp_datagram_dict[ch_seq]['frequency']:
                            self._append_channel_ping_data(ch_seq+1, tmp_datagram_dict[ch_seq])  # ping-by-ping metadata
                            self.power_dict[ch_seq+1].append(tmp_datagram_dict[ch_seq]['power'])  # append power data
                            self.angle_dict[ch_seq+1].append(tmp_datagram_dict[ch_seq]['angle'])  # append angle data
                        else:
                            # TODO: need error-handling code here
                            print('Frequency mismatch for data from the same channel number!')

            # NME datagrams store ancillary data as NMEA-0817 style ASCII data.
            elif new_datagram['type'].startswith('NME'):
                # Add the datagram to our nmea_data object.
                self.nmea_data.add_datagram(new_datagram['timestamp'],
                                            new_datagram['nmea_string'])

            # TAG datagrams contain time-stamped annotations inserted via the recording software
            elif new_datagram['type'].startswith('TAG'):
                print('TAG datagram encountered.')

            # BOT datagrams contain sounder detected bottom depths from .bot files
            elif new_datagram['type'].startswith('BOT'):
                print('BOT datagram encountered.')

            # DEP datagrams contain sounder detected bottom depths from .out files
            # as well as reflectivity data
            elif new_datagram['type'].startswith('DEP'):
                print('DEP datagram encountered.')

            else:
                print("Unknown datagram type: " + str(new_datagram['type']))

    def load_ek60_raw(self):
        """Method to parse the EK60 ``.raw`` data file.

        This method parses the ``.raw`` file and saves the parsed data
        to the ConvertEK60 instance.
        """
        print('%s  converting file: %s' % (dt.now().strftime('%H:%M:%S'), os.path.basename(self.filename)))

        with RawSimradFile(self.filename, 'r') as fid:

            # Read the CON0 configuration datagram
            self.config_datagram = fid.read(1)
            self.config_datagram['timestamp'] = np.datetime64(
                self.config_datagram['timestamp'].replace(tzinfo=None), '[ms]')

            # Check if reading an ME70 file with a CON1 datagram.
            next_datagram = fid.peek()
            if next_datagram == 'CON1':
                self.CON1_datagram = fid.read(1)
            else:
                self.CON1_datagram = None

            for ch_num in self.config_datagram['transceivers'].keys():
                self.ping_data_dict[ch_num] = defaultdict(list)
                self.ping_data_dict[ch_num]['frequency'] = \
                    self.config_datagram['transceivers'][ch_num]['frequency']
                self.power_dict[ch_num] = []
                self.angle_dict[ch_num] = []

            # Read the rest of datagrams
            self._read_datagrams(fid)

            # Check lengths of power data and only store those with the majority sizes (for now)
            # TODO: figure out how to handle this better when there's a clear switch in the middle

            # Find indices of unwanted pings
            lens = [len(l) for l in self.power_dict[1]]
            uni, uni_inv, uni_cnt = np.unique(lens, return_inverse=True, return_counts=True)

            # Dictionary with keys = length of range bin and value being the indexes for the pings with that range
            indices = {num: [i for i, x in enumerate(lens) if x == num] for num in uni}

            # Initialize dictionaries. keys are index for ranges. values are dictionaries with keys for each freq
            self.ping_time_split = {}
            self.power_dict_split = {}
            self.angle_dict_split = {}
            for i, length in enumerate(uni):
                self.ping_time_split[i] = self.ping_time[:uni_cnt[i]]
                self.power_dict_split[i] = {ch_num: [] for ch_num in self.config_datagram['transceivers'].keys()}
                self.angle_dict_split[i] = {ch_num: [] for ch_num in self.config_datagram['transceivers'].keys()}

            for ch_num in self.config_datagram['transceivers'].keys():
                # r_b represents index for range_bin (how many different range_bins there are).
                # r is the list of indexes that correspond to that range
                for r_b, r in enumerate(indices.values()):
                    for r_idx in r:
                        self.power_dict_split[r_b][ch_num].append(self.power_dict[ch_num][r_idx])
                        self.angle_dict_split[r_b][ch_num].append(self.power_dict[ch_num][r_idx])
                    self.power_dict_split[r_b][ch_num] = np.array(self.power_dict_split[r_b][ch_num]) * INDEX2POWER
                    self.angle_dict_split[r_b][ch_num] = np.array(self.power_dict_split[r_b][ch_num])

        # TODO: need to convert angle data too
        self.ping_time = np.array(self.ping_time)
        self.range_lengths = uni

        # Trim excess data from NMEA object
        self.nmea_data.trim()

    def save(self, file_format):
        """Save data from EK60 `.raw` to netCDF format.
        """

        # Subfunctions to set various dictionaries
        def _set_toplevel_dict():
            out_dict = dict(Conventions='CF-1.7, SONAR-netCDF4, ACDD-1.3',
                            keywords='EK60',
                            sonar_convention_authority='ICES',
                            sonar_convention_name='SONAR-netCDF4',
                            sonar_convention_version='1.7',
                            summary='',
                            title='')
            out_dict['date_created'] = dt.strptime(filedate + '-' + filetime,'%Y%m%d-%H%M%S').isoformat() + 'Z'
            return out_dict

        def _set_env_dict():
            return dict(frequency=freq,
                        absorption_coeff=abs_val,
                        sound_speed=ss_val)

        def _set_prov_dict():
            return dict(conversion_software_name='echopype',
                        conversion_software_version=ECHOPYPE_VERSION,
                        conversion_time=dt.now(tz=pytz.utc).isoformat(timespec='seconds'))  # use UTC time

        def _set_sonar_dict():
            return dict(sonar_manufacturer='Simrad',
                        sonar_model=self.config_datagram['sounder_name'],
                        sonar_serial_number='',
                        sonar_software_name='',
                        sonar_software_version=self.config_datagram['version'],
                        sonar_type='echosounder')

        def _set_platform_dict():
            out_dict = dict()
            # TODO: Need to reconcile the logic between using the unpacked "survey_name"
            #  and the user-supplied platform_name
            # self.platform_name = self.config_datagram['survey_name']
            out_dict['platform_name'] = self.platform_name
            out_dict['platform_type'] = self.platform_type
            out_dict['platform_code_ICES'] = self.platform_code_ICES

            # Read pitch/roll/heave from ping data
            out_dict['ping_time'] = self.ping_time  # [seconds since 1900-01-01] for xarray.to_netcdf conversion
            out_dict['pitch'] = np.array(self.ping_data_dict[1]['pitch'], dtype='float32')
            out_dict['roll'] = np.array(self.ping_data_dict[1]['roll'], dtype='float32')
            out_dict['heave'] = np.array(self.ping_data_dict[1]['heave'], dtype='float32')
            # water_level is set to 0 for EK60 since this is not separately recorded
            # and is part of transducer_depth
            out_dict['water_level'] = np.int32(0)

            # Read lat/long from NMEA datagram
            idx_loc = np.argwhere(np.isin(self.nmea_data.messages, ['GGA', 'GLL', 'RMC'])).squeeze()
            nmea_msg = []
            [nmea_msg.append(pynmea2.parse(self.nmea_data.raw_datagrams[x])) for x in idx_loc]
            out_dict['lat'] = np.array([x.latitude for x in nmea_msg])
            out_dict['lon'] = np.array([x.longitude for x in nmea_msg])
            out_dict['location_time'] = self.nmea_data.nmea_times[idx_loc]
            return out_dict

        def _set_nmea_dict():
            # Assemble dict for saving to groups
            out_dict = dict()
            out_dict['nmea_time'] = self.nmea_data.nmea_times
            out_dict['nmea_datagram'] = self.nmea_data.raw_datagrams
            return out_dict

        def _set_beam_dict():
            beam_dict = dict()
            beam_dict['beam_mode'] = 'vertical'
            beam_dict['conversion_equation_t'] = 'type_3'  # type_3 is EK60 conversion
            beam_dict['ping_time'] = self.ping_time   # [seconds since 1900-01-01] for xarray.to_netcdf conversion
            # beam_dict['backscatter_r'] = np.array([self.power_dict[x] for x in self.power_dict.keys()])
            beam_dict['range_lengths'] = self.range_lengths
            beam_dict['power_dict'] = self.power_dict_split
            beam_dict['angle_dict'] = self.angle_dict_split
            beam_dict['ping_time_split'] = self.ping_time_split
            # Additional coordinate variables added by echopype for storing data as a cube with
            # dimensions [frequency x ping_time x range_bin]
            beam_dict['frequency'] = freq
            # beam_dict['range_bin'] = np.arange(self.power_dict[1].shape[1])  # added by echopype, not in convention

            beam_dict['range_bin'] = dict()
            for ranges in self.ping_time_split.keys():
                beam_dict['range_bin'][ranges] = np.arange(beam_dict['power_dict'][ranges][1].shape[1])

            # Loop through each transducer for channel-specific variables
            bm_width = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            bm_dir = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            bm_angle = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            tx_pos = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            beam_dict['equivalent_beam_angle'] = np.zeros(shape=(tx_num,), dtype='float32')
            beam_dict['gain_correction'] = np.zeros(shape=(tx_num,), dtype='float32')
            beam_dict['gpt_software_version'] = []
            beam_dict['channel_id'] = []

            for c_seq, c in self.config_datagram['transceivers'].items():
                c_seq -= 1
                bm_width['beamwidth_receive_major'][c_seq] = c['beamwidth_alongship']
                bm_width['beamwidth_receive_minor'][c_seq] = c['beamwidth_athwartship']
                bm_width['beamwidth_transmit_major'][c_seq] = c['beamwidth_alongship']
                bm_width['beamwidth_transmit_minor'][c_seq] = c['beamwidth_athwartship']
                bm_dir['beam_direction_x'][c_seq] = c['dir_x']
                bm_dir['beam_direction_y'][c_seq] = c['dir_y']
                bm_dir['beam_direction_z'][c_seq] = c['dir_z']
                bm_angle['angle_offset_alongship'][c_seq] = c['angle_offset_alongship']
                bm_angle['angle_offset_athwartship'][c_seq] = c['angle_offset_athwartship']
                bm_angle['angle_sensitivity_alongship'][c_seq] = c['angle_sensitivity_alongship']
                bm_angle['angle_sensitivity_athwartship'][c_seq] = c['angle_sensitivity_athwartship']
                tx_pos['transducer_offset_x'][c_seq] = c['pos_x']
                tx_pos['transducer_offset_y'][c_seq] = c['pos_y']
                tx_pos['transducer_offset_z'][c_seq] = c['pos_z'] + self.ping_data_dict[c_seq+1]['transducer_depth'][0]
                beam_dict['equivalent_beam_angle'][c_seq] = c['equivalent_beam_angle']
                beam_dict['gain_correction'][c_seq] = c['gain']
                beam_dict['gpt_software_version'].append(c['gpt_software_version'])
                beam_dict['channel_id'].append(c['channel_id'])

            beam_dict['beam_width'] = bm_width
            beam_dict['beam_direction'] = bm_dir
            beam_dict['beam_angle'] = bm_angle
            beam_dict['transducer_position'] = tx_pos

            # Loop through each transducer for variables that may vary at each ping
            # -- this rarely is the case for EK60 so we check first before saving
            pl_tmp = np.unique(self.ping_data_dict[1]['pulse_length']).size
            pw_tmp = np.unique(self.ping_data_dict[1]['transmit_power']).size
            bw_tmp = np.unique(self.ping_data_dict[1]['bandwidth']).size
            si_tmp = np.unique(self.ping_data_dict[1]['sample_interval']).size
            if np.all(np.array([pl_tmp, pw_tmp, bw_tmp, si_tmp]) == 1):
                tx_sig = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
                beam_dict['sample_interval'] = np.zeros(shape=(tx_num,), dtype='float32')
                for t_seq in range(tx_num):
                    tx_sig['transmit_duration_nominal'][t_seq] = \
                        np.float32(self.ping_data_dict[t_seq + 1]['pulse_length'][0])
                    tx_sig['transmit_power'][t_seq] = \
                        np.float32(self.ping_data_dict[t_seq + 1]['transmit_power'][0])
                    tx_sig['transmit_bandwidth'][t_seq] = \
                        np.float32(self.ping_data_dict[t_seq + 1]['bandwidth'][0])
                    beam_dict['sample_interval'][t_seq] = \
                        np.float32(self.ping_data_dict[t_seq + 1]['sample_interval'][0])
            else:
                tx_sig = defaultdict(lambda: np.zeros(shape=(tx_num, ping_num), dtype='float32'))
                beam_dict['sample_interval'] = np.zeros(shape=(tx_num, ping_num), dtype='float32')
                for t_seq in range(tx_num):
                    tx_sig['transmit_duration_nominal'][t_seq, :] = \
                        np.array(self.ping_data_dict[t_seq + 1]['pulse_length'], dtype='float32')
                    tx_sig['transmit_power'][t_seq, :] = \
                        np.array(self.ping_data_dict[t_seq + 1]['transmit_power'], dtype='float32')
                    tx_sig['transmit_bandwidth'][t_seq, :] = \
                        np.array(self.ping_data_dict[t_seq + 1]['bandwidth'], dtype='float32')
                    beam_dict['sample_interval'][t_seq, :] = \
                        np.array(self.ping_data_dict[t_seq + 1]['sample_interval'], dtype='float32')

            beam_dict['transmit_signal'] = tx_sig
            # Build other parameters
            beam_dict['non_quantitative_processing'] = np.array([0, ] * freq.size, dtype='int32')
            # -- sample_time_offset is set to 2 for EK60 data, this value is NOT from sample_data['offset']
            beam_dict['sample_time_offset'] = np.array([2, ] * freq.size, dtype='int32')

            idx = [np.argwhere(np.isclose(tx_sig['transmit_duration_nominal'][x - 1],
                                          self.config_datagram['transceivers'][x]['pulse_length_table'])).squeeze()
                   for x in self.config_datagram['transceivers'].keys()]
            beam_dict['sa_correction'] = \
                np.array([x['sa_correction_table'][y]
                          for x, y in zip(self.config_datagram['transceivers'].values(), np.array(idx))])

            return beam_dict

        # Load data from RAW file
        if not bool(self.power_dict):  # if haven't parsed .raw file
            self.load_ek60_raw()

        # Get exported filename
        filename = os.path.splitext(os.path.basename(self.filename))[0]
        self.save_path = os.path.join(os.path.split(self.filename)[0], filename + file_format)
        self.nc_path = os.path.join(os.path.split(self.filename)[0], filename + '.nc')
        self.zarr_path = os.path.join(os.path.split(self.filename)[0], filename + '.zarr')
        # filename must have "-" as the field separator for the last 2 fields
        filename_tup = filename.split("-")
        filedate = filename_tup[len(filename_tup)-2].replace("D","")
        filetime = filename_tup[len(filename_tup)-1].replace("T","")        
        

        # Check if nc file already exists
        # ... if yes, abort conversion and issue warning
        # ... if not, continue with conversion
        if os.path.exists(self.save_path):
            print(f'          ... this file has already been converted to {file_format}, conversion not executed.')
        else:
            # Retrieve variables
            tx_num = self.config_datagram['transceiver_count']
            ping_num = self.ping_time.size
            freq = np.array([self.config_datagram['transceivers'][x]['frequency']
                             for x in self.config_datagram['transceivers'].keys()], dtype='float32')

            # Extract absorption and sound speed depending on if the values are identical for all pings
            abs_tmp = np.unique(self.ping_data_dict[1]['absorption_coefficient']).size
            ss_tmp = np.unique(self.ping_data_dict[1]['sound_velocity']).size
            # --- if identical for all pings, save only values from the first ping
            if np.all(np.array([abs_tmp, ss_tmp]) == 1):
                abs_val = np.array([self.ping_data_dict[x]['absorption_coefficient'][0]
                                    for x in self.config_datagram['transceivers'].keys()], dtype='float32')
                ss_val = np.array([self.ping_data_dict[x]['sound_velocity'][0]
                                   for x in self.config_datagram['transceivers'].keys()], dtype='float32')
            # --- if NOT identical for all pings, save as array of dimension [frequency x ping_time]
            else:
                abs_val = np.array([self.ping_data_dict[x]['absorption_coefficient']
                                    for x in self.config_datagram['transceivers'].keys()],
                                   dtype='float32')
                ss_val = np.array([self.ping_data_dict[x]['sound_velocity']
                                   for x in self.config_datagram['transceivers'].keys()],
                                  dtype='float32')

            # Create SetGroups object
            grp = SetGroups(file_path=self.save_path, echo_type='EK60')
            grp.set_toplevel(_set_toplevel_dict())  # top-level group
            grp.set_env(_set_env_dict())            # environment group
            grp.set_provenance(os.path.basename(self.filename), _set_prov_dict())    # provenance group
            grp.set_platform(_set_platform_dict())  # platform group
            grp.set_nmea(_set_nmea_dict())          # platform/NMEA group
            grp.set_sonar(_set_sonar_dict())        # sonar group
            grp.set_beam(_set_beam_dict())          # beam group
