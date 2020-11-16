from collections import defaultdict
from .utils.ek_raw_io import SimradEOF
from datetime import datetime as dt
import numpy as np
import os

FILENAME_DATETIME_EK60 = '(?P<survey>.+)?-?D(?P<date>\\w{1,8})-T(?P<time>\\w{1,6})-?(?P<postfix>\\w+)?.raw'
NMEA_GPS_SENTECE = 'GGA'


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
        time = self.config_datagram['timestamp'].astype(dt).strftime("%Y-%b-%d %H:%M:%S")
        print(f"{dt.now().strftime('%H:%M:%S')} converting file {os.path.basename(self.source_file)}, "
              f"time of first ping: {time}")


class ParseEK(ParseBase):
    """Class for converting data from Simrad echosounders.
    """
    def __init__(self, file, params):
        super().__init__(file)

        # Parent class attributes
        self.timestamp_pattern = FILENAME_DATETIME_EK60  # regex pattern used to grab datetime embedded in filename
        self.nmea_gps_sentence = NMEA_GPS_SENTECE  # select GPS datagram in _set_platform_dict()

        # Class attributes
        self.config_datagram = None
        self.nmea_time = []
        self.raw_nmea_string = []
        self.ping_data_dict = defaultdict()  # dictionary to store metadata
        self.num_range_bin_groups = None  # number of range_bin groups

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
                new_datagram = fid.read(1)
            except SimradEOF:
                break

            # Convert the timestamp to a datetime64 object.
            new_datagram['timestamp'] = np.datetime64(new_datagram['timestamp'].replace(tzinfo=None), '[ms]')

            num_datagrams_parsed += 1

            # Skip any datagram that the user does not want to save
            # TODO: WJ: what does this check do? if user only selects 'ENV' the 'RAW0/3' are still parsed?
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

            # RAW datagrams store raw acoustic data for a channel for EK60
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

            # RAW datagrams store raw acoustic data for a channel for EK80
            elif new_datagram['type'].startswith('RAW3'):
                curr_ch_id = new_datagram['channel_id']
                if current_parameters['channel_id'] != curr_ch_id:
                    raise ValueError("Parameter ID does not match RAW")

                # tmp_num_ch_per_ping_parsed += 1
                if curr_ch_id not in self.recorded_ch_ids:
                    self.recorded_ch_ids.append(curr_ch_id)

                # append ping time from first channel
                if curr_ch_id == self.recorded_ch_ids[0]:
                    self.ping_time.append(new_datagram['timestamp'])

                # Append ping by ping data
                new_datagram.update(current_parameters)
                self._append_channel_ping_data(new_datagram)
                if self.n_complex_dict[curr_ch_id] < 0:
                    self.n_complex_dict[curr_ch_id] = new_datagram['n_complex']  # update n_complex data

            # NME datagrams store ancillary data as NMEA-0817 style ASCII data.
            elif new_datagram['type'].startswith('NME'):
                # Add the datagram to our nmea_data object.
                # TODO: Look at duplicate time
                # Check duplicate time
                # duplicate_idx = (self.nmea_time == new_datagram['timestamp'])
                # if np.any(duplicate_idx):
                #     # Check for duplicate message.
                #     string = np.array(self.raw_nmea_string)[duplicate_idx][0]
                #     # If duplicate, skip saving the datagram
                #     if new_datagram['nmea_string'][1:6] == string[1:6]:
                #         continue
                self.nmea_time.append(new_datagram['timestamp'])
                self.raw_nmea_string.append(new_datagram['nmea_string'])

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
                if 'ALL' in self.data_type:
                    print("Unknown datagram type: " + str(new_datagram['type']))

    def _append_channel_ping_data(self, datagram):
        """Append ping by ping data.
        """
        unsaved = ['channel', 'channel_id', 'offset', 'low_date', 'high_date', 'frequency',
                   'transmit_mode', 'spare0', 'bytes_read', 'type', 'n_complex']
        ch_id = datagram['channel_id'] if 'channel_id' in datagram else datagram['channel']
        for k, v in datagram.items():
            if k not in unsaved:
                self.ping_data_dict[k][ch_id].append(v)

    def _find_range_group(self, power_dict):
        """Find the pings at which range_bin changes.
        """
        if all([p is None for p in power_dict.values()]):
            return None, None
        range_bin_lens = [len(l) for l in list(power_dict.values())[0]]
        uni, uni_cnt = np.unique(range_bin_lens, return_counts=True)
        self.range_lengths = uni  # used in looping when saving files with different range_bin numbers

        return uni, np.cumsum(np.insert(uni_cnt, 0, 0))

    def _check_ping_channel_match(self):
        """Check if the number of RAW datagrams loaded are integer multiples of the number of channels.
        """
        # Check line 312 of convert/ek60.py

    def _clean_channel(self):
        """Remove channels that do not record any pings.
        """
        # TODO Look at ek60 save only when all channels are present
        if len(self.ch_ids) != len(self.recorded_ch_ids):
            self.ch_ids = self.recorded_ch_ids

    def _rectangularize(self, power_dict, angle_dict=None):
        """ Takes a potentially irregular dictionary with frequency as keys and returns
        a rectangular numpy array padded with nans.
        """
        INDEX2POWER = (10.0 * np.log10(2.0) / 256.0) if self.sonar_type == 'EK60' else 1

        power_dict = {k: v for k, v in power_dict.items() if v is not None}
        uni, uni_cnt_insert = self._find_range_group(power_dict)
        if uni is None:
            return None, None
        if angle_dict is not None:
            angle_dict = {k: v for k, v in angle_dict.items() if v is not None}

        largest_range = 0
        # Find the largest range length across channels and range groups
        for ch in power_dict.values():
            if ch is not None:
                ch_size = len(max(ch, key=len))
                largest_range = ch_size if ch_size > largest_range else largest_range
        assert max(uni) <= largest_range
        power_type = 'complex64' if np.any(np.iscomplex(list(power_dict.values())[0])) else 'float64'
        tmp_power = np.full((len(power_dict), uni_cnt_insert[-1],
                             largest_range), np.nan, dtype=power_type)
        tmp_angle = np.full((len(power_dict), uni_cnt_insert[-1],
                            largest_range, 2), np.nan) if angle_dict is not None else None
        # Pad range groups and channels
        for i in range(len(uni)):
            # List of all channels sliced into a range group
            grouped_power = [np.array(p[uni_cnt_insert[i]:uni_cnt_insert[i + 1]])
                             for p in power_dict.values() if p is not None]
            if angle_dict is not None:
                grouped_angle = [np.array(a[uni_cnt_insert[i]:uni_cnt_insert[i + 1]])
                                 for a in angle_dict.values() if a is not None]

            for ch in range(len(grouped_power)):
                # Pad power data
                tmp_power[ch, uni_cnt_insert[i]:uni_cnt_insert[i + 1],
                          :grouped_power[ch].shape[1]] = grouped_power[ch]
                # Pad angle data
                if angle_dict is not None:
                    if grouped_angle[ch].ndim == 1:
                        continue
                    # Exception occurs when only one channel records angle data.
                    # In that case, skip channel (filled with nan)
                    tmp_angle[ch, uni_cnt_insert[i]:uni_cnt_insert[i + 1],
                                :grouped_angle[ch].shape[1], :] = grouped_angle[ch]
        return tmp_power * INDEX2POWER, tmp_angle

    def _select_datagrams(self, params):
        # get GPS info only (EK60, EK80)
        # ec.to_netcdf(data_type='GPS')

        # get configuration XML only (EK80)
        # ec.to_netcdf(data_type='CONFIG_XML')

        # get environment XML only (EK80)
        # ec.to_netcdf(data_type='ENV_XML')
        def translate_to_dgram(s):
            if s == 'ALL':
                return ['ALL']
            elif s == 'GPS':
                if self.sonar_type == 'EK60':
                    return ['NME', 'GPS']
                elif self.sonar_type == 'EK80':
                    return ['NME', 'MRU', 'GPS']
            # TODO: WJ: I don't see how the 'ENV_XML' case is working,
            #  and why do you need both CONFIG+ENV and EXPORT?
            elif s == 'CONFIG_XML':
                return ['CONFIG']
            elif s == 'ENV_XML':
                return ['XML', 'ENV']
            elif s == 'EXPORT':
                return ['EXPORT']
        if isinstance(params, str):
            dgrams = translate_to_dgram(params)
        else:
            dgrams = []
            for p in params:
                dgrams += translate_to_dgram(p)
        return dgrams
