from collections import defaultdict
from .utils.ek_raw_io import SimradEOF
from datetime import datetime as dt
import numpy as np
import os

FILENAME_DATETIME_EK60 = '(?P<survey>.+)?-?D(?P<date>\\w{1,8})-T(?P<time>\\w{1,6})-?(?P<postfix>\\w+)?.raw'
NMEA_GPS_SENTENCE = 'GGA'


class ParseBase:
    """Parent class for all convert classes.
    """
    def __init__(self, file):
        self.source_file = file
        self.timestamp_pattern = None  # regex pattern used to grab datetime embedded in filename
        self.nmea_gps_sentence = None  # select GPS datagram in _set_platform_dict()
        self.ping_time = []            # list to store ping time

    def _print_status(self):
        """Prints message to console giving information about the raw file being parsed.
        """


class ParseEK(ParseBase):
    """Class for converting data from Simrad echosounders.
    """
    def __init__(self, file, params):
        super().__init__(file)

        # Parent class attributes
        self.timestamp_pattern = FILENAME_DATETIME_EK60  # regex pattern used to grab datetime embedded in filename
        self.nmea_gps_sentence = NMEA_GPS_SENTENCE  # select GPS datagram in _set_platform_dict()

        # Class attributes
        self.config_datagram = None
        self.ping_data_dict = self.ping_data_dict = defaultdict(lambda: defaultdict(list))
        self.ping_time = defaultdict(list)  # store ping time according to channel
        self.num_range_bin_groups = None  # number of range_bin groups
        self.ch_ping_idx = []
        self.data_type = self._select_datagrams(params)

        self.nmea = defaultdict(list)   # Dictionary to store NMEA data(timestamp and string)
        self.mru = defaultdict(list)  # Dictionary to store MRU data (heading, pitch, roll, heave)
        self.fil_coeffs = defaultdict(dict)  # Dictionary to store PC and WBT coefficients
        self.fil_df = defaultdict(dict)  # Dictionary to store filter decimation factors

    def _print_status(self):
        time = self.config_datagram['timestamp'].astype(dt).strftime("%Y-%b-%d %H:%M:%S")
        print(f"{dt.now().strftime('%H:%M:%S')} converting file {os.path.basename(self.source_file)}, "
              f"time of first ping: {time}")

    def parse_raw(self):
        """This method calls private functions to parse the raw data file.
        """

    def _read_datagrams(self, fid):
        """Read all datagrams.

        A sample EK60 RAW0 datagram:
            {'type': 'RAW0',
            'low_date': 71406392,
            'high_date': 30647127,
            'channel': 1,
            'mode': 3,
            'transducer_depth': 9.149999618530273,
            'frequency': 18000.0,
            'transmit_power': 2000.0,
            'pulse_length': 0.0010239999974146485,
            'bandwidth': 1573.66552734375,
            'sample_interval': 0.00025599999935366213,
            'sound_velocity': 1466.0,
            'absorption_coefficient': 0.0030043544247746468,
            'heave': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'temperature': 4.0,
            'heading': 0.0,
            'transmit_mode': 1,
            'spare0': '\x00\x00\x00\x00\x00\x00',
            'offset': 0,
            'count': 1386,
            'timestamp': numpy.datetime64('2018-02-11T16:40:25.276'),
            'bytes_read': 5648,
            'power': array([ -6876,  -8726, -11086, ..., -11913, -12522, -11799], dtype=int16),
            'angle': array([[ 110,   13],
                [   3,   -4],
                [ -54,  -65],
                ...,
                [ -92, -107],
                [-104, -122],
                [  82,   74]], dtype=int8)}

        A sample EK80 XML-parameter datagram:
            {'channel_id': 'WBT 545612-15 ES200-7C',
             'channel_mode': 0,
             'pulse_form': 1,
             'frequency_start': '160000',
             'frequency_end': '260000',
             'pulse_duration': 0.001024,
             'sample_interval': 5.33333333333333e-06,
             'transmit_power': 15.0,
             'slope': 0.01220703125}

        A sample EK80 XML-environment datagram:
            {'type': 'XML0',
             'low_date': 3137819385,
             'high_date': 30616609,
             'timestamp': numpy.datetime64('2017-09-12T23:49:10.723'),
             'bytes_read': 448,
             'subtype': 'environment',
             'environment': {'depth': 240.0,
              'acidity': 8.0,
              'salinity': 33.7,
              'sound_speed': 1486.4,
              'temperature': 6.9,
              'latitude': 45.0,
              'sound_velocity_profile': [1.0, 1486.4, 1000.0, 1486.4],
              'sound_velocity_source': 'Manual',
              'drop_keel_offset': 0.0,
              'drop_keel_offset_is_manual': 0,
              'water_level_draft': 0.0,
              'water_level_draft_is_manual': 0,
              'transducer_name': 'Unknown',
              'transducer_sound_speed': 1490.0},
             'xml': '<?xml version="1.0" encoding="utf-8"?>\r\n<Environment Depth="240" ... />\r\n</Environment>'}
        """
        num_datagrams_parsed = 0
        tmp_num_ch_per_ping_parsed = 0  # number of channels of the same ping parsed
                                        # this is used to control saving only pings
                                        # that have all freq channels present
        tmp_datagram_dict = []  # tmp list of datagrams, only saved to actual output
                                # structure if data from all freq channels are present

        while True:
            try:
                # TODO: @ngkvain: what I need in the code to not PARSE the raw0/3 datagram
                #  when users only want CONFIG or ENV, but the way this is implemented
                #  the raw0/3 datagrams are still parsed, you are just not saving them
                new_datagram = fid.read(1)

            except SimradEOF:
                break

            # Convert the timestamp to a datetime64 object.
            new_datagram['timestamp'] = np.datetime64(new_datagram['timestamp'].replace(tzinfo=None), '[ms]')

            num_datagrams_parsed += 1

            # Skip any datagram that the user does not want to save

            # TODO: @ngkvain: what does this first if check do? Why don't you just check for 'ALL'?
            if (not any(new_datagram['type'].startswith(dgram) for dgram in self.data_type) and
               'ALL' not in self.data_type):
                continue
            # XML datagrams store environment or instrument parameters for EK80
            if new_datagram['type'].startswith("XML"):
                if new_datagram['subtype'] == 'environment' and ('ENV' in self.data_type or 'ALL' in self.data_type):
                    self.environment = new_datagram['environment']
                    self.environment['xml'] = new_datagram['xml']
                elif new_datagram['subtype'] == 'parameter' and ('ALL' in self.data_type):
                    current_parameters = new_datagram['parameter']

            # RAW0 datagrams store raw acoustic data for a channel for EK60
            # TODO: change saving of RAW0 datagrams in the same way as RAW3 datagrams:
            #   - keeping all the ping_time
            #   - do not assume that all pings are transmitted simultaneously
            #   - do not assume that the pings from different channels come in in a particular sequence.

            elif new_datagram['type'].startswith('RAW0'):
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
                if np.all(np.array([curr_ch_num, tmp_num_ch_per_ping_parsed]) ==
                          self.config_datagram['transceiver_count']):

                    # append ping time from first channel
                    self.ping_time.append(tmp_datagram_dict[0]['timestamp'])
                    for ch_seq in range(self.config_datagram['transceiver_count']):
                        # If frequency matches for this channel, actually store data
                        # Note all storage structure indices are 1-based since they are indexed by
                        # the channel number as stored in config_datagram['transceivers'].keys()
                        if self.config_datagram['transceivers'][ch_seq + 1]['frequency'] \
                                == tmp_datagram_dict[ch_seq]['frequency']:
                            self._append_channel_ping_data(tmp_datagram_dict[ch_seq])   # metadata per ping
                        else:
                            # TODO: need error-handling code here
                            print('Frequency mismatch for data from the same channel number!')

            # RAW3 datagrams store raw acoustic data for a channel for EK80
            elif new_datagram['type'].startswith('RAW3'):
                curr_ch_id = new_datagram['channel_id']
                # Check if the proceeding Parameter XML does not match with data in this RAW3 datagram
                if current_parameters['channel_id'] != curr_ch_id:
                    raise ValueError("Parameter ID does not match RAW")

                # append ping time from first channel
                # self.ping_time.append(new_datagram['timestamp'])
                self.ping_time[curr_ch_id].append(new_datagram['timestamp'])

                # Append ping by ping data
                new_datagram.update(current_parameters)
                self._append_channel_ping_data(new_datagram)
                if self.n_complex_dict[curr_ch_id] is None:
                    # update number of complex samples for each channel
                    self.n_complex_dict[curr_ch_id] = new_datagram['n_complex']

            # NME datagrams store ancillary data as NMEA-0817 style ASCII data.
            elif new_datagram['type'].startswith('NME'):
                self.nmea['timestamp'].append(new_datagram['timestamp'])
                self.nmea['nmea_string'].append(new_datagram['nmea_string'])

            # MRU datagrams contain motion data for each ping for EK80
            elif new_datagram['type'].startswith("MRU"):
                self.mru['heading'].append(new_datagram['heading'])
                self.mru['pitch'].append(new_datagram['pitch'])
                self.mru['roll'].append(new_datagram['roll'])
                self.mru['heave'].append(new_datagram['heave'])
                self.mru['timestamp'].append(new_datagram['timestamp'])

            # FIL datagrams contain filters for proccessing bascatter data for EK80
            elif new_datagram['type'].startswith("FIL"):
                self.fil_coeffs[new_datagram['channel_id']][new_datagram['stage']] = new_datagram['coefficients']
                self.fil_df[new_datagram['channel_id']][new_datagram['stage']] = new_datagram['decimation_factor']

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

    def _append_channel_ping_data(self, datagram):
        """Append ping by ping data.
        """
        unsaved = ['channel', 'channel_id', 'offset', 'low_date', 'high_date', #'frequency',
                   'transmit_mode', 'spare0', 'bytes_read', 'type'] #, 'n_complex']
        ch_id = datagram['channel_id'] if 'channel_id' in datagram else datagram['channel']
        for k, v in datagram.items():
            if k not in unsaved:
                self.ping_data_dict[k][ch_id].append(v)

    def _find_range_group(self, data_dict):
        """Find the pings at which range_bin changes.
        """
        if all([p is None for p in data_dict.values()]):
            return None, None
        uni, uni_cnt, range_lengths = [], [], []
        # Find the channel with the most range length changes
        for val in data_dict.values():
            range_bin_lens = [len(l) for l in val]
            uni_tmp, uni_cnt_tmp = np.unique(range_bin_lens, return_counts=True)
            # used in looping when saving files with different range_bin numbers
            range_lengths = uni_tmp if len(uni_tmp) > len(uni) else range_lengths
            uni_cnt = uni_cnt_tmp if len(uni_cnt_tmp) > len(uni_cnt) else uni_cnt
        return range_lengths, np.cumsum(np.insert(uni_cnt, 0, 0))

    def _match_ch_ping_time(self):
        # Match timestamp of each ping in power data with ping_time for each channel
        # If all channels ping at the same time then ch_indices equals the ping_time
        self.ch_ping_idx = {ch: np.searchsorted(self.ping_time, timestamp) for
                            ch, timestamp in self.ping_data_dict['timestamp'].items()}

    def _rectangularize(self, data_dict, is_power=True):
        """ Takes a potentially irregular dictionary with frequency as keys and returns
        a rectangular numpy array padded with nans.
        """
        # Find where the range changes
        range_lengths, uni_cnt_insert = self._find_range_group(data_dict)
        # Exit function if no power data is collected on any of the channels
        if uni_cnt_insert is None:
            return None, None

        # TODO: @ngkavin:
        #  I think it will be cleaner to assemble one DataArray for each channel (pad the shorter pings with NaN),
        #  and then concat or merge along the frequency dimension with NaN padding
        #  to directly assemble a cube DataSet with ping_time x range_bin x frequency dimensions.
        # Slice out which ping times correspond to which ping in each channel
        ch_indices = [self.ch_ping_idx[ch] for ch in data_dict.keys()]

        # Find the largest range length across channels and range groups
        largest_range = 0
        for ch, power in data_dict.items():
            if ch is not None:
                ch_size = len(max(power, key=len))
                largest_range = ch_size if ch_size > largest_range else largest_range
        assert max(range_lengths) <= largest_range
        # Must define a power-type of either float32 or complex64 because np.nan cannot be int
        power_type = np.complex64 if list(data_dict.values())[0][0].dtype == np.complex64 else np.float32
        # Check if data is angle or power. Angle data has 1 extra dimension
        if is_power:
            tmp_data = np.full((len(data_dict), len(self.ping_time),
                                largest_range), np.nan, dtype=power_type)
        else:
            tmp_data = np.full((len(data_dict), len(self.ping_time),
                                largest_range, 2), np.nan)
        # Pad range groups and channels
        for i in range(len(range_lengths)):
            # List of all channels sliced into a range group
            grouped_indices = [np.array(ch[uni_cnt_insert[i]:uni_cnt_insert[i + 1]])
                               for ch in ch_indices]
            # Group pings by range length
            grouped_data = [np.array(ch[uni_cnt_insert[i]:uni_cnt_insert[i + 1]])
                            for ch in data_dict.values() if ch is not None]

            for ch in range(len(grouped_data)):
                # Exception occurs when only one channel records angle data.
                # In that case, skip channel (filled with nan)
                if not is_power:
                    if grouped_data[ch].ndim == 1:
                        continue
                # Fill in nan array with power/angle data
                tmp_data[ch, grouped_indices[ch], :grouped_data[ch].shape[1]] = grouped_data[ch]
        return tmp_data

    def _select_datagrams(self, params):
        """ Translates user input into specific datagrams or ALL

        Valid use cases:
        # get GPS info only (EK60, EK80)
        # ec.to_netcdf(data_type='GPS')

        # get configuration XML only (EK80)
        # ec.to_netcdf(data_type='CONFIG')

        # get environment XML only (EK80)
        # ec.to_netcdf(data_type='ENV')
        """
