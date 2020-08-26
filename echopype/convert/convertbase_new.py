from collections import defaultdict
from .utils.nmea_data import NMEAData
import xml.dom.minidom
from struct import unpack
from datetime import datetime as dt
from datetime import timezone
import math
import numpy as np
import os

FILENAME_DATETIME_EK60 = '(?P<survey>.+)?-?D(?P<date>\w{1,8})-T(?P<time>\w{1,6})-?(?P<postfix>\w+)?.raw'
FILENAME_DATETIME_AZFP = '\w+.raw'
NMEA_GPS_SENTECE = 'GGA'


class ParseBase:
    """Parent class for all convert classes.
    """
    def __init__(self, file, params=None, compress=True, overwrite=False):
        self.source_file = file
        self.ui_param = params
        self.compress = compress
        self.overwrite = overwrite
        self.timestamp_pattern = None  # regex pattern used to grab datetime embedded in filename
        self.nmea_gps_sentence = None  # select GPS datagram in _set_platform_dict()

    def __str__(self):
        """Overload the print function to allow user to print basic properties of this object.

        Print out should include: source file name, source folder location, echosounder model.
        """

    def _print_status(self):
        """Prints message to console giving information about the raw file being parsed.
        """
        # TODO: this currently is only in convert/azfp.py and
        #  embedded in other methods for EK60 and EK80.
        #  Let's consolidate the same operation in one method.


class ParseEK(ParseBase):
    """Class for converting data from Simrad echosounders.
    """
    def __init__(self, file, params=None):
        super().__init__(file, params)

        # Parent class attributes
        self.timestamp_pattern = FILENAME_DATETIME_EK60  # regex pattern used to grab datetime embedded in filename
        self.nmea_gps_sentence = NMEA_GPS_SENTECE  # select GPS datagram in _set_platform_dict()

        # Class attributes
        self.config_datagram = None
        self.nmea_data = NMEAData()  # object for NMEA data
        self.ping_data_dict = {}  # dictionary to store metadata
        self.power_dict = {}     # dictionary to store power data
        self.angle_dict = {}     # dictionary to store angle data
        self.ping_time = []      # list to store ping time
        self.num_range_bin_groups = None  # number of range_bin groups

    def parse_raw(self):
        """This method calls private functions to parse the raw data file.
        """

    def _check_env_param_uniqueness(self):
        """Check if env parameters are unique throughout the file.
        """
        # Right now we don't handle changes in the environmental parameters and
        # just store the first set of unpacked env parameters.
        # We should have this function to check this formally, and save the env
        # parameters along the ping_time dimension if they change.
        # NOTE: discuss this in light of what AZFP has.

        # TODO:
        #  EK80: there is only 1 XML-environment datagram for each file
        #        so no need to check for uniqueness for the variables.
        #  EK60: the environment variables are wrapped in RAW0 datagrams
        #        for each ping -- see docstring _read_datagrams()
        #        so will have to check for uniqueness.
        #        variables to check are:
        #           - sound_velocity
        #        	- absorption_coefficient
        # 	        - temperature

    def _check_tx_param_uniqueness(self):
        """Check if transmit parameters are unique throughout the file.
        """
        # Right now we don't handle the case if any transmit parameters change in the middle of the file.
        # We also haven't run into any file that has that happening?
        # At minimum we should spit out an error message.
        # For EK60 the params are: pulse_length, transmit_power, bandwidth, sample_interval
        # For EK80 the params will include frequency-related params

        # TODO:
        #  EK80: each RAW3 datagrams is preceded by an XML-parameter datagram,
        #        which contains tx parameters for the following channel data
        #        -- see docstring _read_datagrams() for detail
        #        we will have to check for uniqueness for these variables.
        #  EK60: the environment variables are wrapped in RAW0 datagrams
        #        for each ping -- see docstring _read_datagrams()
        #        we will have to check for uniqueness for these parameters
        #        variables to check are:
        #           - transmit_power
        #           - pulse_length
        #           - bandwidth
        #           - sample_interval
        #           - transmit_mod(  # 0 = Active, 1 = Passive, 2 = Test, -1 = Unknown)
        #           - offset

    def _read_datagrams(self):
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

    def _append_channel_ping_data(self):
        """Append non-backscatter data for each ping.
        """
        # This is currently implemented for EK60, but not for EK80.
        # line 94-123 in convert/ek80.py can be made into this method:
        # TODO: simplify the calls `current_parameters['channel_id']` to
        #  `ch_id = current_parameters['channel_id']`
        #  and then just use ch_id, to increase code readbility

    def _find_range_group(self):
        """Find the pings at which range_bin changes.
        """
        # Use this function to check if all pings in this file contains the same range_bin.
        # If not, find out the index of which changes happen.

    def _check_ping_channel_match(self):
        """Check if the number of RAW datagrams loaded are integer multiples of the number of channels.
        """
        # Check line 312 of convert/ek60.py

    def _clean_channel(self):
        """Remove channels that do not record any pings.
        """
        # This seems to be what line 211 of convert/ek80.py is doing?


class ParseEK60(ParseEK):
    """Class for converting data from Simrad EK60 echosounders.
    """
    def parse_raw(self):
        """Parse raw data file from Simrad EK60 echosounder.
        """


class ParseEK80(ParseEK):
    """Class for converting data from Simrad EK60 echosounders.
    """
    def __init__(self, file, params):
        super().__init__(file, params)
        self.complex_dict = {}  # dictionary to store complex data
        self.n_complex_dict = {}  # dictionary to store the number of beams in split-beam complex data
        self.environment = {}  # dictionary to store environment data
        # self.parameters = defaultdict(dict)  # Dictionary to hold parameter data --> use self.ping_data_dict
        self.mru = defaultdict(list)  # Dictionary to store MRU data (heading, pitch, roll, heave)
        self.fil_coeffs = defaultdict(dict)  # Dictionary to store PC and WBT coefficients
        self.fil_df = defaultdict(dict)  # Dictionary to store filter decimation factors
        self.ch_ids = []  # List of all channel ids
        self.recorded_ch_ids = []  # TODO: what is the difference between this and ch_ids?

        # TODO: we shouldn't need `power_dict_split`, `angle_dict_split` and `ping_time_split`,
        #  and can just get the values from the original corresponding storage values
        #  `power_dict`, `angle_dict`, `ping_time`

    def parse_raw(self):
        """Parse raw data file from Simrad EK80 echosounder.
        """
        # TODO: line 191 block can be substitute by `self.parameters[ch_id] = defaultdict(list)`

    def _sort_ch_bb_cw(self):
        """Sort which channels are broadband (BB) and continuous wave (CW).
        """


class ParseAZFP(ParseBase):
    """Class for converting data from ASL Environmental Sciences AZFP echosounder.
    """
    def __init__(self, file, params=None):
        super().__init__(file, params)
        # Parent class attributes
        self.timestamp_pattern = FILENAME_DATETIME_EK60  # regex pattern used to grab datetime embedded in filename

        # Class attributes
        self.parameters = dict()
        self.unpacked_data = defaultdict(list)

    def load_AZFP_xml(self):
        """Parse XML file to get params for reading AZFP data."""
        """Parses the AZFP  XML file.
        """
        def get_value_by_tag_name(tag_name, element=0):
            """Returns the value in an XML tag given the tag name and the number of occurrences."""
            return px.getElementsByTagName(tag_name)[element].childNodes[0].data

        # TODO: consider writing a ParamAZFPxml class for storing parameters
        px = xml.dom.minidom.parse(self.ui_param['xml_path'])
        self.parameters['num_freq'] = int(get_value_by_tag_name('NumFreq'))
        self.parameters['serial_number'] = int(get_value_by_tag_name('SerialNumber'))
        self.parameters['burst_interval'] = float(get_value_by_tag_name('BurstInterval'))
        self.parameters['pings_per_burst'] = int(get_value_by_tag_name('PingsPerBurst'))
        self.parameters['average_burst_pings'] = int(get_value_by_tag_name('AverageBurstPings'))

        # Temperature coeff
        self.parameters['ka'] = float(get_value_by_tag_name('ka'))
        self.parameters['kb'] = float(get_value_by_tag_name('kb'))
        self.parameters['kc'] = float(get_value_by_tag_name('kc'))
        self.parameters['A'] = float(get_value_by_tag_name('A'))
        self.parameters['B'] = float(get_value_by_tag_name('B'))
        self.parameters['C'] = float(get_value_by_tag_name('C'))

        # tilts
        self.parameters['X_a'] = float(get_value_by_tag_name('X_a'))
        self.parameters['X_b'] = float(get_value_by_tag_name('X_b'))
        self.parameters['X_c'] = float(get_value_by_tag_name('X_c'))
        self.parameters['X_d'] = float(get_value_by_tag_name('X_d'))
        self.parameters['Y_a'] = float(get_value_by_tag_name('Y_a'))
        self.parameters['Y_b'] = float(get_value_by_tag_name('Y_b'))
        self.parameters['Y_c'] = float(get_value_by_tag_name('Y_c'))
        self.parameters['Y_d'] = float(get_value_by_tag_name('Y_d'))

        # Initializing fields for each transducer frequency
        self.parameters['dig_rate'] = []
        self.parameters['lock_out_index'] = []
        self.parameters['gain'] = []
        self.parameters['pulse_length'] = []
        self.parameters['DS'] = []
        self.parameters['EL'] = []
        self.parameters['TVR'] = []
        self.parameters['VTX'] = []
        self.parameters['BP'] = []
        self.parameters['range_samples'] = []
        self.parameters['range_averaging_samples'] = []
        # Get parameters for each transducer frequency
        for ch in range(self.parameters['num_freq']):
            self.parameters['range_samples'].append(int(get_value_by_tag_name('RangeSamples', ch)))
            self.parameters['range_averaging_samples'].append(int(get_value_by_tag_name('RangeAveragingSamples', ch)))
            self.parameters['dig_rate'].append(float(get_value_by_tag_name('DigRate', ch)))
            self.parameters['lock_out_index'].append(float(get_value_by_tag_name('LockOutIndex', ch)))
            self.parameters['gain'].append(float(get_value_by_tag_name('Gain', ch)))
            self.parameters['pulse_length'].append(float(get_value_by_tag_name('PulseLen', ch)))
            self.parameters['DS'].append(float(get_value_by_tag_name('DS', ch)))
            self.parameters['EL'].append(float(get_value_by_tag_name('EL', ch)))
            self.parameters['TVR'].append(float(get_value_by_tag_name('TVR', ch)))
            self.parameters['VTX'].append(float(get_value_by_tag_name('VTX0', ch)))
            self.parameters['BP'].append(float(get_value_by_tag_name('BP', ch)))
        self.parameters['sensors_flag'] = float(get_value_by_tag_name('SensorsFlag'))

    def parse_raw(self):
        """Parse raw data file from AZFP echosounder.

        Parameters
        ----------
        raw : list
            raw filename
        """

        # Start of computation subfunctions
        def compute_temp(counts):
            """Returns the temperature in celsius given from xml data and the counts from ancillary"""
            v_in = 2.5 * (counts / 65535)
            R = (self.parameters['ka'] + self.parameters['kb'] * v_in) / (self.parameters['kc'] - v_in)
            T = 1 / (self.parameters['A'] + self.parameters['B'] * (math.log(R)) +
                     self.parameters['C'] * (math.log(R) ** 3)) - 273
            return T

        def compute_tilt(N, a, b, c, d):
            return a + b * N + c * N**2 + d * N**3

        def compute_battery(N):
            USL5_BAT_CONSTANT = (2.5 / 65536.0) * (86.6 + 475.0) / 86.6
            return N * USL5_BAT_CONSTANT

        # Instrument specific constants
        HEADER_SIZE = 124
        HEADER_FORMAT = ">HHHHIHHHHHHHHHHHHHHHHHHHHHHHHHHHHHBBBBHBBBBBBBBHHHHHHHHHHHHHHHHHHHH"

        # Read xml file into dict
        self.load_AZFP_xml()

        with open(self.source_file, 'rb') as file:
            ping_num = 0
            eof = False
            while not eof:
                header_chunk = file.read(HEADER_SIZE)
                if header_chunk:
                    header_unpacked = unpack(HEADER_FORMAT, header_chunk)

                    # Reading will stop if the file contains an unexpected flag
                    if self._split_header(file, header_unpacked):
                        # Appends the actual 'data values' to unpacked_data
                        self._add_counts(file, ping_num)
                        if ping_num == 0:
                            # Display information about the file that was loaded in
                            self._print_status()
                        # Compute temperature from unpacked_data[ii]['ancillary][4]
                        self.unpacked_data['temperature'].append(compute_temp(self.unpacked_data['ancillary'][ping_num][4]))
                        # compute x tilt from unpacked_data[ii]['ancillary][0]
                        self.unpacked_data['tilt_x'].append(
                            compute_tilt(self.unpacked_data['ancillary'][ping_num][0],
                                         self.parameters['X_a'], self.parameters['X_b'],
                                         self.parameters['X_c'], self.parameters['X_d']))
                        # Compute y tilt from unpacked_data[ii]['ancillary][1]
                        self.unpacked_data['tilt_y'].append(
                            compute_tilt(self.unpacked_data['ancillary'][ping_num][1],
                                         self.parameters['Y_a'], self.parameters['Y_b'],
                                         self.parameters['Y_c'], self.parameters['Y_d']))
                        # Compute cos tilt magnitude from tilt x and y values
                        self.unpacked_data['cos_tilt_mag'].append(
                            math.cos((math.sqrt(self.unpacked_data['tilt_x'][ping_num] ** 2 +
                                                self.unpacked_data['tilt_y'][ping_num] ** 2)) * math.pi / 180))
                        # Calculate voltage of main battery pack
                        self.unpacked_data['battery_main'].append(
                            compute_battery(self.unpacked_data['ancillary'][ping_num][2]))
                        # If there is a Tx battery pack
                        self.unpacked_data['battery_tx'].append(
                            compute_battery(self.unpacked_data['ad'][ping_num][0]))
                    else:
                        break
                else:
                    # End of file
                    eof = True
                ping_num += 1
        self._check_uniqueness()

    @staticmethod
    def _get_fields():
        """Returns the fields contained in each header of the raw file.
        """
        _fields = (
            ('profile_flag', 'u2'),
            ('profile_number', 'u2'),
            ('serial_number', 'u2'),
            ('ping_status', 'u2'),
            ('burst_int', 'u4'),
            ('year', 'u2'),                 # Year
            ('month', 'u2'),                # Month
            ('day', 'u2'),                  # Day
            ('hour', 'u2'),                 # Hour
            ('minute', 'u2'),               # Minute
            ('second', 'u2'),               # Second
            ('hundredths', 'u2'),           # Hundredths of a second
            ('dig_rate', 'u2', 4),          # Digitalization rate for each channel
            ('lockout_index', 'u2', 4),     # Lockout index for each channel
            ('num_bins', 'u2', 4),          # Number of bins for each channel
            ('range_samples_per_bin', 'u2', 4),     # Range samples per bin for each channel
            ('ping_per_profile', 'u2'),     # Number of pings per profile
            ('avg_pings', 'u2'),            # Flag indicating whether the pings average in time
            ('num_acq_pings', 'u2'),        # Pings acquired in the burst
            ('ping_period', 'u2'),          # Ping period in seconds
            ('first_ping', 'u2'),
            ('last_ping', 'u2'),
            ('data_type', "u1", 4),         # Datatype for each channel 1=Avg unpacked_data (5bytes), 0=raw (2bytes)
            ('data_error', 'u2'),           # Error number is an error occurred
            ('phase', 'u1'),                # Phase number used to acquire this profile
            ('overrun', 'u1'),              # 1 if an overrun occurred
            ('num_chan', 'u1'),             # 1, 2, 3, or 4
            ('gain', 'u1', 4),              # gain channel 1-4
            ('spare_chan', 'u1'),           # spare channel
            ('pulse_length', 'u2', 4),      # Pulse length chan 1-4 uS
            ('board_num', 'u2', 4),         # The board the data came from channel 1-4
            ('frequency', 'u2', 4),         # frequency for channel 1-4 in kHz
            ('sensor_flag', 'u2'),          # Flag indicating if pressure sensor or temperature sensor is available
            ('ancillary', 'u2', 5),         # Tilt-X, Y, Battery, Pressure, Temperature
            ('ad', 'u2', 2)                 # AD channel 6 and 7
        )
        return _fields

    def _print_status(self):
        """Prints message to console giving information about the raw file being parsed.
        """
        filename = os.path.basename(self.source_file)
        timestamp = dt(self.unpacked_data['year'][0], self.unpacked_data['month'][0], self.unpacked_data['day'][0],
                       self.unpacked_data['hour'][0], self.unpacked_data['minute'][0],
                       int(self.unpacked_data['second'][0] + self.unpacked_data['hundredths'][0] / 100))
        timestr = timestamp.strftime("%Y-%b-%d %H:%M:%S")
        pathstr, xml_name = os.path.split(self.ui_param['xml_path'])
        print(f"{dt.now().strftime('%H:%M:%S')} converting file {filename} with {xml_name}, "
              f"time of first ping {timestr}")

    def _split_header(self, raw, header_unpacked):
        """Splits the header information into a dictionary.
        Modifies self.unpacked_data

        Parameters
        ----------
        raw
            open binary file
        header_unpacked
            output of struct unpack of raw file

        Returns
        -------
            True or False depending on whether the unpacking was successful
        """
        FILE_TYPE = 64770       # Instrument specific constant
        fields = self._get_fields()
        if header_unpacked[0] != FILE_TYPE:  # first field should match hard-coded FILE_TYPE from manufacturer
            check_eof = raw.read(1)
            if check_eof:
                print("Error: Unknown file type")
                return False
        header_byte_cnt = 0
        firmware_freq_len = 4   # fields with num_freq data still takes 4 bytes, the extra bytes contain random numbers
        field_w_freq = ('dig_rate', 'lockout_index', 'num_bins', 'range_samples_per_bin',  # fields with num_freq data
                        'data_type', 'gain', 'pulse_length', 'board_num', 'frequency')
        for field in fields:
            if field[0] in field_w_freq:  # fields with num_freq data
                self.unpacked_data[field[0]].append(
                    header_unpacked[header_byte_cnt:header_byte_cnt + self.parameters['num_freq']])
                header_byte_cnt += firmware_freq_len
            elif len(field) == 3:  # other longer fields ('ancillary' and 'ad')
                self.unpacked_data[field[0]].append(header_unpacked[header_byte_cnt:header_byte_cnt + field[2]])
                header_byte_cnt += field[2]
            else:
                self.unpacked_data[field[0]].append(header_unpacked[header_byte_cnt])
                header_byte_cnt += 1
        return True

    def _add_counts(self, raw, ping_num):
        """Unpacks the echosounder raw data. Modifies self.unpacked_data.
        """
        vv_tmp = [[]] * self.unpacked_data['num_chan'][ping_num]
        for freq_ch in range(self.unpacked_data['num_chan'][ping_num]):
            counts_byte_size = self.unpacked_data['num_bins'][ping_num][freq_ch]
            if self.unpacked_data['data_type'][ping_num][freq_ch]:
                if self.unpacked_data['avg_pings'][ping_num]:  # if pings are averaged over time
                    divisor = self.unpacked_data['ping_per_profile'][ping_num] * \
                        self.unpacked_data['range_samples_per_bin'][ping_num][freq_ch]
                else:
                    divisor = self.unpacked_data['range_samples_per_bin'][ping_num][freq_ch]
                ls = unpack(">" + "I" * counts_byte_size, raw.read(counts_byte_size * 4))     # Linear sum
                lso = unpack(">" + "B" * counts_byte_size, raw.read(counts_byte_size * 1))    # linear sum overflow
                v = (np.array(ls) + np.array(lso) * 4294967295) / divisor
                v = (np.log10(v) - 2.5) * (8 * 65535) * self.parameters['DS'][freq_ch]
                v[np.isinf(v)] = 0
                vv_tmp[freq_ch] = v
            else:
                counts_chunk = raw.read(counts_byte_size * 2)
                counts_unpacked = unpack(">" + "H" * counts_byte_size, counts_chunk)
                vv_tmp[freq_ch] = counts_unpacked
        self.unpacked_data['counts'].append(vv_tmp)

    def _check_uniqueness(self):
        """Check for ping-by-ping consistency of sampling parameters and reduce if identical.
        """
        if not self.unpacked_data:
            self.parse_raw()

        if np.array(self.unpacked_data['profile_flag']).size != 1:    # Only check uniqueness once. Will error if done twice
            # fields with num_freq data
            field_w_freq = ('dig_rate', 'lockout_index', 'num_bins', 'range_samples_per_bin',
                            'data_type', 'gain', 'pulse_length', 'board_num', 'frequency')
            # fields to reduce size if the same for all pings
            field_include = ('profile_flag', 'serial_number',
                             'burst_int', 'ping_per_profile', 'avg_pings', 'ping_period',
                             'phase', 'num_chan', 'spare_chan')
            for field in field_w_freq:
                uniq = np.unique(self.unpacked_data[field], axis=0)
                if uniq.shape[0] == 1:
                    self.unpacked_data[field] = uniq.squeeze()
                else:
                    raise ValueError(f"Header value {field} is not constant for each ping")
            for field in field_include:
                uniq = np.unique(self.unpacked_data[field])
                if uniq.shape[0] == 1:
                    self.unpacked_data[field] = uniq.squeeze()
                else:
                    raise ValueError(f"Header value {field} is not constant for each ping")

    def _get_ping_time(self):
        """Assemble ping time from parsed values.
        """

        if not self.unpacked_data:
            self.parse_raw()

        ping_time = []
        for ping_num, year in enumerate(self.unpacked_data['year']):
            ping_time.append(dt(year,
                                self.unpacked_data['month'][ping_num],
                                self.unpacked_data['day'][ping_num],
                                self.unpacked_data['hour'][ping_num],
                                self.unpacked_data['minute'][ping_num],
                                int(self.unpacked_data['second'][ping_num] +
                                    self.unpacked_data['hundredths'][ping_num] / 100)
                                ).replace(tzinfo=timezone.utc).timestamp())
        return ping_time

    @staticmethod
    def _calc_Sv_offset(f, pulse_length):
        """Calculate the compensation factor for Sv calculation.
        """
        # TODO: this method seems should be in echopype.process
        if f > 38000:
            if pulse_length == 300:
                return 1.1
            elif pulse_length == 500:
                return 0.8
            elif pulse_length == 700:
                return 0.5
            elif pulse_length == 900:
                return 0.3
            elif pulse_length == 1000:
                return 0.3
        else:
            if pulse_length == 500:
                return 1.1
            elif pulse_length == 1000:
                return 0.7
