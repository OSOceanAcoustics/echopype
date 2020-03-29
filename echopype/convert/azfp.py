import os
from collections import defaultdict
import numpy as np
import xml.dom.minidom
import math
from datetime import datetime as dt
from datetime import timezone
from echopype.convert.utils.set_groups import SetGroups
from struct import unpack
from echopype._version import get_versions
from .convertbase import ConvertBase
ECHOPYPE_VERSION = get_versions()['version']
del get_versions


class ConvertAZFP(ConvertBase):
    """Class for converting AZFP `.01A` files """

    def __init__(self, _filename='', _xml_path=''):
        ConvertBase.__init__(self)
        self.filename = _filename
        self.xml_path = _xml_path
        # self.file_name = os.path.basename(self.filename)
        self.FILE_TYPE = 64770
        self.HEADER_SIZE = 124
        self.HEADER_FORMAT = ">HHHHIHHHHHHHHHHHHHHHHHHHHHHHHHHHHHBBBBHBBBBBBBBHHHHHHHHHHHHHHHHHHHH"
        self.parameters = dict()

        # Adds to self.parameters the contents of the xml file
        self.loadAZFPxml()

        # Initialize variables that'll be filled later
        self.unpacked_data = None
        self._checked_unique = False

    def loadAZFPxml(self):
        """Parses the AZFP  XML file.
        """
        def get_value_by_tag_name(tag_name, element=0):
            """Returns the value in an XML tag given the tag name and the number of occurrences."""
            return px.getElementsByTagName(tag_name)[element].childNodes[0].data

        # TODO: consider writing a ParamAZFPxml class for storing parameters
        px = xml.dom.minidom.parse(self.xml_path)
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
        for freq_ch in range(self.parameters['num_freq']):
            self.parameters['range_samples'].append(int(get_value_by_tag_name('RangeSamples', freq_ch)))
            self.parameters['range_averaging_samples'].append(int(get_value_by_tag_name('RangeAveragingSamples', freq_ch)))
            self.parameters['dig_rate'].append(float(get_value_by_tag_name('DigRate', freq_ch)))
            self.parameters['lock_out_index'].append(float(get_value_by_tag_name('LockOutIndex', freq_ch)))
            self.parameters['gain'].append(float(get_value_by_tag_name('Gain', freq_ch)))
            self.parameters['pulse_length'].append(float(get_value_by_tag_name('PulseLen', freq_ch)))
            self.parameters['DS'].append(float(get_value_by_tag_name('DS', freq_ch)))
            self.parameters['EL'].append(float(get_value_by_tag_name('EL', freq_ch)))
            self.parameters['TVR'].append(float(get_value_by_tag_name('TVR', freq_ch)))
            self.parameters['VTX'].append(float(get_value_by_tag_name('VTX0', freq_ch)))
            self.parameters['BP'].append(float(get_value_by_tag_name('BP', freq_ch)))
        self.parameters['sensors_flag'] = float(get_value_by_tag_name('SensorsFlag'))

    @staticmethod
    def get_fields():
        """Returns the fields contained in each header of the raw file."""
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

    def _split_header(self, raw, header_unpacked, unpacked_data, fields):
        """Splits the header information into a dictionary.

        Parameters
        ----------
        raw
            open binary file
        header_unpacked
            output of struct unpack of raw file
        unpacked_data
            current unpacked data
        fields
            fields to be unpacked for each ping, defined in ``get_fields``

        Returns
        -------
            True or False depending on whether the unpacking was successful
        """
        if header_unpacked[0] != self.FILE_TYPE:  # first field should match hard-coded FILE_TYPE from manufacturer
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
                unpacked_data[field[0]].append(
                    header_unpacked[header_byte_cnt:header_byte_cnt + self.parameters['num_freq']])
                # unpacked_data[ping_num][field[0]] = \
                #     header_unpacked[header_byte_cnt:header_byte_cnt + self.parameters['num_freq']]
                header_byte_cnt += firmware_freq_len
            elif len(field) == 3:  # other longer fields ('ancillary' and 'ad')
                unpacked_data[field[0]].append(header_unpacked[header_byte_cnt:header_byte_cnt + field[2]])
                # unpacked_data[ping_num][field[0]] = \
                #     header_unpacked[header_byte_cnt:header_byte_cnt + field[2]]
                header_byte_cnt += field[2]
            else:
                unpacked_data[field[0]].append(header_unpacked[header_byte_cnt])
                # unpacked_data[ping_num][field[0]] = header_unpacked[header_byte_cnt]
                header_byte_cnt += 1
        return True

    def _add_counts(self, raw, ping_num, unpacked_data):
        """Unpacks the echosounder raw data. Modifies unpacked_data in place.

        Parameters
        ----------
        raw
            open binary file
        ping_num
            ping number
        unpacked_data
            current unpacked data
        """
        vv_tmp = [[]] * unpacked_data['num_chan'][ping_num]
        for freq_ch in range(unpacked_data['num_chan'][ping_num]):
            counts_byte_size = unpacked_data['num_bins'][ping_num][freq_ch]
            if unpacked_data['data_type'][ping_num][freq_ch]:
                if unpacked_data['avg_pings'][ping_num]:  # if pings are averaged over time
                    divisor = unpacked_data['ping_per_profile'][ping_num] * \
                              unpacked_data['range_samples_per_bin'][ping_num][freq_ch]
                else:
                    divisor = unpacked_data['range_samples_per_bin'][ping_num][freq_ch]
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
        unpacked_data['counts'].append(vv_tmp)

    def _print_status(self, path, unpacked_data):
        """Prints message to console giving information about the raw file being parsed

        Parameters
        ----------
        path
            path to the 01A file
        unpacked_data
            current unpacked data
        """
        filename = os.path.basename(path)
        timestamp = dt(unpacked_data['year'][0], unpacked_data['month'][0], unpacked_data['day'][0],
                       unpacked_data['hour'][0], unpacked_data['minute'][0],
                       int(unpacked_data['second'][0] + unpacked_data['hundredths'][0] / 100))
        timestr = timestamp.strftime("%Y-%b-%d %H:%M:%S")
        (pathstr, xml_name) = os.path.split(self.xml_path)
        print(f"{dt.now().strftime('%H:%M:%S')} converting file {filename} with {xml_name}, "
              f"time of first ping {timestr}")

    def check_uniqueness(self):
        """Check for ping-by-ping consistency of sampling parameters and reduce if identical.

        Those included in this function should be identical throughout all pings.
        Therefore raise error if not identical.
        """
        if not self.unpacked_data:
            self.parse_raw()

        if not self._checked_unique:    # Only check uniqueness once. Will error if done twice
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
        self._checked_unique = True

    def parse_raw(self, raw):
        """Parses a raw AZFP file of the 01A file format

        Parameters
        ----------
        raw : list
            raw filenames
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

        unpacked_data = defaultdict(list)
        fields = self.get_fields()
        for file in raw:
            with open(file, 'rb') as raw:
                ping_num = 0
                eof = False
                while not eof:
                    header_chunk = raw.read(self.HEADER_SIZE)
                    if header_chunk:
                        header_unpacked = unpack(self.HEADER_FORMAT, header_chunk)

                        # Reading will stop if the file contains an unexpected flag
                        if self._split_header(raw, header_unpacked, unpacked_data, fields):
                            # Appends the actual 'data values' to unpacked_data
                            self._add_counts(raw, ping_num, unpacked_data)
                            if ping_num == 0:
                                # Display information about the file that was loaded in
                                self._print_status(file, unpacked_data)
                            # Compute temperature from unpacked_data[ii]['ancillary][4]
                            unpacked_data['temperature'].append(compute_temp(unpacked_data['ancillary'][ping_num][4]))
                            # compute x tilt from unpacked_data[ii]['ancillary][0]
                            unpacked_data['tilt_x'].append(
                                compute_tilt(unpacked_data['ancillary'][ping_num][0],
                                             self.parameters['X_a'], self.parameters['X_b'],
                                             self.parameters['X_c'], self.parameters['X_d']))
                            # Compute y tilt from unpacked_data[ii]['ancillary][1]
                            unpacked_data['tilt_y'].append(
                                compute_tilt(unpacked_data['ancillary'][ping_num][1],
                                             self.parameters['Y_a'], self.parameters['Y_b'],
                                             self.parameters['Y_c'], self.parameters['Y_d']))
                            # Compute cos tilt magnitude from tilt x and y values
                            unpacked_data['cos_tilt_mag'].append(
                                math.cos((math.sqrt(unpacked_data['tilt_x'][ping_num] ** 2 +
                                                    unpacked_data['tilt_y'][ping_num] ** 2)) * math.pi / 180))
                            # Calculate voltage of main battery pack
                            unpacked_data['battery_main'].append(
                                compute_battery(unpacked_data['ancillary'][ping_num][2]))
                            # If there is a Tx battery pack
                            unpacked_data['battery_tx'].append(
                                compute_battery(unpacked_data['ad'][ping_num][0]))
                        else:
                            break
                    else:
                        # End of file
                        eof = True
                    ping_num += 1

        self.unpacked_data = unpacked_data

    def get_ping_time(self):
        """Returns the ping times"""

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

    def save(self, file_format, save_path=None, combine_opt=False, overwrite=False, compress=True):
        """Save data from raw 01A format to a netCDF4 or Zarr file

        Parameters
        ----------
        file_format : str
            format of output file. ".nc" for netCDF4 or ".zarr" for Zarr
        save_path : str
            Path to save output to. Must be a directory if converting multiple files.
            Must be a filename if combining multiple files.
            If `False`, outputs in the same location as the input raw file.
        combine_opt : bool
            Whether or not to combine a list of input raw files.
            Raises error if combine_opt is true and there is only one file being converted.
        overwrite : bool
            Whether or not to overwrite the file if the output path already exists.
        compress : bool
            Whether or not to compress backscatter data. Defaults to `True`
        """

        # Subfunctions to set various dictionaries
        def export(file_idx=None):
            def calc_Sv_offset(f, pulse_length):
                """Calculate a compensation for the effects of finite response
                times of both the receiving and transmitting parts of the transducer.
                The correction magnitude depends on the length of the transmitted pulse
                and the response time (transmission and reception) of the transducer.
                The numbers used below are documented on p.91 in GU-100-AZFP-01-R50 Operator's Manual.
                This subfunction is called by ``_set_beam_dict()``.

                Parameters
                ----------
                f
                    frequency in Hz
                pulse_length
                    pulse length in ms
                """
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

            def _set_toplevel_dict():
                out_dict = dict(conventions='CF-1.7, SONAR-netCDF4-1.0, ACDD-1.3',
                                keywords='AZFP',
                                sonar_convention_authority='ICES',
                                sonar_convention_name='SONAR-netCDF4',
                                sonar_convention_version='1.0',
                                summary='',
                                title='')
                return out_dict

            def _set_env_dict():
                out_dict = dict(temperature=self.unpacked_data['temperature'],  # temperature measured at instrument
                                ping_time=ping_time)
                return out_dict

            def _set_platform_dict():
                out_dict = dict(platform_name=self.platform_name,
                                platform_type=self.platform_type,
                                platform_code_ICES=self.platform_code_ICES)
                return out_dict

            def _set_prov_dict():
                attrs = ('conversion_software_name', 'conversion_software_version', 'conversion_time')
                vals = ('echopype', ECHOPYPE_VERSION, dt.utcnow().isoformat(timespec='seconds') + 'Z')  # use UTC time
                return dict(zip(attrs, vals))

            def _set_sonar_dict():
                attrs = ('sonar_manufacturer', 'sonar_model', 'sonar_serial_number',
                         'sonar_software_name', 'sonar_software_version', 'sonar_type')
                vals = ('ASL Environmental Sciences', 'Acoustic Zooplankton Fish Profiler',
                        int(self.unpacked_data['serial_number']),   # should have only 1 value (identical for all pings)
                        'Based on AZFP Matlab Toolbox', '1.4', 'echosounder')
                return dict(zip(attrs, vals))

            def _set_beam_dict():
                anc = np.array(self.unpacked_data['ancillary'])   # convert to np array for easy slicing
                dig_rate = self.unpacked_data['dig_rate']         # dim: freq

                # Build variables in the output xarray Dataset
                N = []   # for storing backscatter_r values for each frequency
                Sv_offset = np.zeros(freq.shape)
                for ich in range(len(freq)):
                    Sv_offset[ich] = calc_Sv_offset(freq[ich], self.unpacked_data['pulse_length'][ich])
                    N.append(np.array([self.unpacked_data['counts'][p][ich]
                                       for p in range(len(self.unpacked_data['year']))]))

                tdn = self.unpacked_data['pulse_length'] / 1e6  # Convert microseconds to seconds
                range_samples_xml = np.array(self.parameters['range_samples'])         # from xml file
                range_samples_per_bin = self.unpacked_data['range_samples_per_bin']    # from data header

                # Calculate sample interval in seconds
                if len(dig_rate) == len(range_samples_per_bin):
                    sample_int = range_samples_per_bin / dig_rate
                else:
                    raise ValueError("dig_rate and range_samples not unique across frequencies")

                # Largest number of counts along the range dimension among the different channels
                longest_range_bin = np.max(self.unpacked_data['num_bins'])
                range_bin = np.arange(longest_range_bin)
                # TODO: replace the following with an explicit check of length of range across channels
                try:
                    np.array(N)
                # Exception occurs when N is not rectangular,
                #  so it must be padded with nan values to make it rectangular
                except ValueError:
                    N = [np.pad(n, ((0, 0), (0, longest_range_bin - n.shape[1])),
                                mode='constant', constant_values=np.nan)
                         for n in N]

                beam_dict = dict()

                # Dimensions
                beam_dict['frequency'] = freq
                beam_dict['ping_time'] = ping_time
                beam_dict['range_bin'] = range_bin

                beam_dict['backscatter_r'] = N                                   # dim: freq x ping_time x range_bin
                beam_dict['gain_correction'] = self.parameters['gain']           # dim: freq
                beam_dict['sample_interval'] = sample_int                        # dim: freq
                beam_dict['transmit_duration_nominal'] = tdn                     # dim: freq
                beam_dict['temperature_counts'] = anc[:, 4]                      # dim: ping_time
                beam_dict['tilt_x_count'] = anc[:, 0]                            # dim: ping_time
                beam_dict['tilt_y_count'] = anc[:, 1]                            # dim: ping_time
                beam_dict['tilt_x'] = self.unpacked_data['tilt_x']               # dim: ping_time
                beam_dict['tilt_y'] = self.unpacked_data['tilt_y']               # dim: ping_time
                beam_dict['cos_tilt_mag'] = self.unpacked_data['cos_tilt_mag']   # dim: ping_time
                beam_dict['EBA'] = self.parameters['BP']          # dim: freq
                beam_dict['DS'] = self.parameters['DS']           # dim: freq
                beam_dict['EL'] = self.parameters['EL']           # dim: freq
                beam_dict['TVR'] = self.parameters['TVR']         # dim: freq
                beam_dict['VTX'] = self.parameters['VTX']         # dim: freq
                beam_dict['Sv_offset'] = Sv_offset                # dim: freq
                beam_dict['range_samples'] = range_samples_xml    # dim: freq
                beam_dict['range_averaging_samples'] = self.parameters['range_averaging_samples']   # dim: freq
                beam_dict['number_of_frequency'] = self.parameters['num_freq']
                beam_dict['number_of_pings_per_burst'] = self.parameters['pings_per_burst']
                beam_dict['average_burst_pings_flag'] = self.parameters['average_burst_pings']

                # Temperature coefficients
                beam_dict['temperature_ka'] = self.parameters['ka']
                beam_dict['temperature_kb'] = self.parameters['kb']
                beam_dict['temperature_kc'] = self.parameters['kc']
                beam_dict['temperature_A'] = self.parameters['A']
                beam_dict['temperature_B'] = self.parameters['B']
                beam_dict['temperature_C'] = self.parameters['C']

                # Tilt coefficients
                beam_dict['tilt_X_a'] = self.parameters['X_a']
                beam_dict['tilt_X_b'] = self.parameters['X_b']
                beam_dict['tilt_X_c'] = self.parameters['X_c']
                beam_dict['tilt_X_d'] = self.parameters['X_d']
                beam_dict['tilt_Y_a'] = self.parameters['Y_a']
                beam_dict['tilt_Y_b'] = self.parameters['Y_b']
                beam_dict['tilt_Y_c'] = self.parameters['Y_c']
                beam_dict['tilt_Y_d'] = self.parameters['Y_d']

                return beam_dict

            def _set_vendor_specific_dict():
                out_dict = {
                    'ping_time': ping_time,
                    'frequency': freq,
                    'profile_flag': self.unpacked_data['profile_flag'],
                    'profile_number': self.unpacked_data['profile_number'],
                    'ping_status': self.unpacked_data['ping_status'],
                    'burst_interval': self.unpacked_data['burst_int'],
                    'digitization_rate': self.unpacked_data['dig_rate'],    # dim: frequency
                    'lockout_index': self.unpacked_data['lockout_index'],   # dim: frequency
                    'num_bins': self.unpacked_data['num_bins'],             # dim: frequency
                    'range_samples_per_bin': self.unpacked_data['range_samples_per_bin'],   # dim: frequency
                    'ping_per_profile': self.unpacked_data['ping_per_profile'],
                    'average_pings_flag': self.unpacked_data['avg_pings'],
                    'number_of_acquired_pings': self.unpacked_data['num_acq_pings'],   # dim: ping_time
                    'ping_period': self.unpacked_data['ping_period'],
                    'first_ping': self.unpacked_data['first_ping'],      # dim: ping_time
                    'last_ping': self.unpacked_data['last_ping'],        # dim: ping_time
                    'data_type': self.unpacked_data['data_type'],        # dim: frequency
                    'data_error': self.unpacked_data['data_error'],      # dim: frequency
                    'phase': self.unpacked_data['phase'],
                    'number_of_channels': self.unpacked_data['num_chan'],
                    'spare_channel': self.unpacked_data['spare_chan'],
                    'board_number': self.unpacked_data['board_num'],     # dim: frequency
                    'sensor_flag': self.unpacked_data['sensor_flag'],    # dim: ping_time
                    'ancillary': self.unpacked_data['ancillary'],        # dim: ping_time x 5 values
                    'ad_channels': self.unpacked_data['ad'],             # dim: ping_time x 2 values
                    'battery_main': self.unpacked_data['battery_main'],
                    'battery_tx': self.unpacked_data['battery_tx']
                }
                out_dict['ancillary_len'] = list(range(len(out_dict['ancillary'][0])))
                out_dict['ad_len'] = list(range(len(out_dict['ad_channels'][0])))
                return out_dict

            # Parse raw data if haven't already
            if self.unpacked_data is None:
                self.parse_raw(self.filename)
            # Check variables that should not vary with ping time
            self.check_uniqueness()

            freq = np.array(self.unpacked_data['frequency']) * 1000    # Frequency in Hz
            ping_time = self.get_ping_time()

            if file_idx is None:
                out_file = self.save_path
                raw_file = self.filename
            else:
                out_file = self.save_path[file_idx]
                raw_file = [self.filename[file_idx]]

            # Check if nc file already exists and deletes it if overwrite is true
            if os.path.exists(out_file) and overwrite:
                print("          overwriting: " + out_file)
                os.remove(out_file)
            # Check if nc file already exists
            # ... if yes, abort conversion and issue warning
            # ... if not, continue with conversion
            if os.path.exists(out_file):
                print(f'          ... this file has already been converted to {file_format}, conversion not executed.')
            else:
                # Create SetGroups object
                grp = SetGroups(file_path=out_file, echo_type='AZFP', compress=compress)
                grp.set_toplevel(_set_toplevel_dict())      # top-level group
                grp.set_env(_set_env_dict())                # environment group
                grp.set_provenance(raw_file, _set_prov_dict())        # provenance group
                grp.set_platform(_set_platform_dict())      # platform group
                grp.set_sonar(_set_sonar_dict())            # sonar group
                grp.set_beam(_set_beam_dict())              # beam group
                grp.set_vendor_specific(_set_vendor_specific_dict())    # AZFP Vendor specific group

        self.validate_path(save_path, file_format, combine_opt)
        if len(self.filename) == 1 or combine_opt:
            export()
        else:
            for file_seq, file in enumerate(self.filename):
                if file_seq > 0:
                    self._checked_unique = False
                    self.unpacked_data = None
                self.parse_raw([file])
                export(file_seq)
