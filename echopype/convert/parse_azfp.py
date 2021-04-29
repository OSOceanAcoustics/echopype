from collections import defaultdict
import xml.dom.minidom
from struct import unpack
from datetime import datetime as dt
from datetime import timezone
import math
import numpy as np
import os
import fsspec
from .parse_base import ParseBase

FILENAME_DATETIME_AZFP = '\\w+.01A'


class ParseAZFP(ParseBase):
    """Class for converting data from ASL Environmental Sciences AZFP echosounder.
    """
    def __init__(self, file, params, storage_options={}):
        super().__init__(file, storage_options)
        # Parent class attributes
        self.timestamp_pattern = FILENAME_DATETIME_AZFP  # regex pattern used to grab datetime embedded in filename
        self.xml_path = params

        # Class attributes
        self.parameters = dict()
        self.unpacked_data = defaultdict(list)
        self.sonar_type = 'AZFP'

    def load_AZFP_xml(self):
        """Parse XML file to get params for reading AZFP data."""
        """Parses the AZFP  XML file.
        """
        def get_value_by_tag_name(tag_name, element=0):
            """Returns the value in an XML tag given the tag name and the number of occurrences."""
            return px.getElementsByTagName(tag_name)[element].childNodes[0].data
        # TODO: consider writing a ParamAZFPxml class for storing parameters

        xmlmap = fsspec.get_mapper(self.xml_path, **self.storage_options)
        px = xml.dom.minidom.parse(xmlmap.fs.open(xmlmap.root))

        int_params = {'NumFreq': 'num_freq', 'SerialNumber': 'serial_number',
                      'BurstInterval': 'burst_interval',
                      'PingsPerBurst': 'pings_per_burst', 'AverageBurstPings': 'average_burst_pings',
                      'SensorsFlag': 'sensors_flag'}
        float_params = ['ka', 'kb', 'kc', 'A', 'B', 'C',                            # Temperature coeffs
                        'X_a', 'X_b', 'X_c', 'X_d', 'Y_a', 'Y_b', 'Y_c', 'Y_d']     # Tilt coeffs]
        freq_params = {'RangeSamples': 'range_samples', 'RangeAveragingSamples': 'range_averaging_samples',
                       'DigRate': 'dig_rate', 'LockOutIndex': 'lockout_index',
                       'Gain': 'gain', 'PulseLen': 'pulse_length', 'DS': 'DS',
                       'EL': 'EL', 'TVR': 'TVR', 'VTX0': 'VTX', 'BP': 'BP'}

        # Retreive integer parameters from the xml file
        for old_name, new_name in int_params.items():
            self.parameters[new_name] = int(get_value_by_tag_name(old_name))
        # Retreive floating point parameters from the xml file
        for param in float_params:
            self.parameters[param] = float(get_value_by_tag_name(param))
        # Retrieve frequency dependent parameters from the xml file
        for old_name, new_name in freq_params.items():
            self.parameters[new_name] = [float(get_value_by_tag_name(old_name, ch)) for
                                         ch in range(self.parameters['num_freq'])]

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
        fmap = fsspec.get_mapper(self.source_file, **self.storage_options)

        with fmap.fs.open(fmap.root, 'rb') as file:
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
                        self.unpacked_data['temperature'].append(
                            compute_temp(self.unpacked_data['ancillary'][ping_num][4]))
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
        self._get_ping_time()
        # Explicitly cast frequency to a float in accordance with the SONAR-netCDF4 convention
        self.unpacked_data['frequency'] = self.unpacked_data['frequency'].astype(np.float64)

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
        pathstr, xml_name = os.path.split(self.xml_path)
        print(f"{dt.now().strftime('%H:%M:%S')}  parsing file {filename} with {xml_name}, "
              f"time of first ping: {timestr}")

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

        if np.array(self.unpacked_data['profile_flag']).size != 1:    # Only check uniqueness once.
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
            ping_time.append(np.datetime64(dt(year,
                             self.unpacked_data['month'][ping_num],
                             self.unpacked_data['day'][ping_num],
                             self.unpacked_data['hour'][ping_num],
                             self.unpacked_data['minute'][ping_num],
                             int(self.unpacked_data['second'][ping_num] +
                                 self.unpacked_data['hundredths'][ping_num] / 100)).replace(tzinfo=None), '[ms]'))
        self.ping_time = ping_time

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
