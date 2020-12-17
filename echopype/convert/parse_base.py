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
        # TODO: @ngkvain: do you have config_datagram for AZPF?
        time = self.config_datagram['timestamp'].astype(dt).strftime("%Y-%b-%d %H:%M:%S")
        print(f"{dt.now().strftime('%H:%M:%S')} converting file {os.path.basename(self.source_file)}, "
              f"time of first ping: {time}")


class ParseEK(ParseBase):
    """Class for converting data from Simrad echosounders.
    """
    def __init__(self, file, params):  # TODO: @ngkvain: you pass in param from convert.py but it is not used
        super().__init__(file)

        # Parent class attributes
        self.timestamp_pattern = FILENAME_DATETIME_EK60  # regex pattern used to grab datetime embedded in filename
        self.nmea_gps_sentence = NMEA_GPS_SENTECE  # select GPS datagram in _set_platform_dict()

        # Class attributes
        self.config_datagram = None
        self.ping_time = defaultdict(list)  # store ping time according to channel
        self.nmea_time = []
        self.raw_nmea_string = []
        self.ping_data_dict = defaultdict()  # dictionary to store metadata
        self.num_range_bin_groups = None  # number of range_bin groups
        self.ch_ping_idx = []

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
            # TODO: @ngkvain: Where is self.data_type initialized? I only see you pass in param, which is not used

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

                    # TODO: @ngkavin: please change the saving of RAW0 datagram content and timestamp
                    #  in the same style as RAW3 (i.e., save channel-specific ping time),
                    #   - keeping all the ping_time
                    #   - do not assume that all pings are transmitted simultaneously
                    #   - do not assume that the pings from different channels come in a particular sequence.
                    #  and make downstream changes in ParseEK60 and SetGroupsEK60 in the same style as
                    #  the new SetGroupsEK80.set_beam.
                    #  Note there are additional variables encoded in RAW0 compared to RAW3
                    #  so you will have to change other groups in addition to Beam, but otherwise
                    #  EK60 raw formats are very similar to EK80.
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

                # tmp_num_ch_per_ping_parsed += 1
                if curr_ch_id not in self.recorded_ch_ids:
                    self.recorded_ch_ids.append(curr_ch_id)

                # Save channel-specific ping time
                self.ping_time[curr_ch_id].append(new_datagram['timestamp'])

                # Append ping by ping data
                new_datagram.update(current_parameters)
                self._append_channel_ping_data(new_datagram)
                # if self.n_complex_dict[curr_ch_id] < 0:   # TODO: @ngkvain: why test <0 ?
                #     self.n_complex_dict[curr_ch_id] = new_datagram['n_complex']  # update n_complex data

            # NME datagrams store ancillary data as NMEA-0817 style ASCII data.
            elif new_datagram['type'].startswith('NME'):
                # TODO: @ngkvain: change to use the same structure as mru,
                #  so have a field for the nmea timestamp and a field for raw strings
                self.nmea_time.append(new_datagram['timestamp'])
                self.raw_nmea_string.append(new_datagram['nmea_string'])

            # MRU datagrams contain motion data for each ping for EK80
            elif new_datagram['type'].startswith("MRU"):   # TODO: @ngkvain: is self.mru initialized somewhere?
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
                if 'ALL' in self.data_type:   # TODO: @ngkvain: why do you do this here?
                    print("Unknown datagram type: " + str(new_datagram['type']))

    def _append_channel_ping_data(self, datagram):
        """Append ping by ping data.
        """
        # TODO: do a thorough check with the convention and processing
        unsaved = ['channel', 'channel_id', 'low_date', 'high_date', # 'offset', 'frequency' ,
                   #'transmit_mode', 'spare0', 'bytes_read', 'type'] #, 'n_complex']
        ch_id = datagram['channel_id'] if 'channel_id' in datagram else datagram['channel']
        for k, v in datagram.items():
            if k not in unsaved:
                self.ping_data_dict[k][ch_id].append(v)

    def _select_datagrams(self, params):
        # TODO: @ngkavin: Why do you need this translation?
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
                # TODO: @ngkavin:
                #  What you have breaks the class inheritance.
                #  If you find yourself using sonar model specific things,
                #  then that thing should be in the child class.
                if self.sonar_type == 'EK60':
                    return ['NME', 'GPS']
                elif self.sonar_type == 'EK80':
                    return ['NME', 'MRU', 'GPS']
            elif s == 'CONFIG':
                return ['CONFIG']
            elif s == 'ENV':
                return ['XML', 'ENV']
            # TODO: @ngkavin:
            #  'EXPORT' is not a good variable name as it does not reflect what this flag controls.
            #  also explain in what circumstance EXPORT will be passed in?
            elif s == 'EXPORT':
                return ['EXPORT']
        if isinstance(params, str):
            dgrams = translate_to_dgram(params)
        else:
            dgrams = []
            for p in params:
                dgrams += translate_to_dgram(p)
        return dgrams
