"""
Functions to unpack Simrad EK60 .raw data file and save to .nc.

TODO: remove old functions once verify everything is absolutely correct
"""


import re
import os
from collections import defaultdict
from struct import unpack_from, unpack
import numpy as np
from datetime import datetime as dt
from matplotlib.dates import date2num
import pytz
import pynmea2

from echopype.convert.utils.ek60_raw_io import RawSimradFile, SimradEOF
from echopype.convert.utils.nmea_data import NMEAData
from echopype.convert.utils.set_groups import SetGroups
from echopype._version import get_versions
ECHOPYPE_VERSION = get_versions()['version']
del get_versions


# Create a constant to convert indexed power to power.
INDEX2POWER = (10.0 * np.log10(2.0) / 256.0)

# Create a constant to convert from indexed angles to electrical angles.
INDEX2ELEC = 180.0 / 128.0


class ConvertEK60:
    """Class for converting EK60 `.raw` files."""

    def __init__(self, _filename=""):

        self.filename = _filename  # path to EK60 .raw filename to be parsed

        # Initialize file parsing storage variables
        self.config_datagram = None
        self.nmea_data = NMEAData()  # object for NMEA data
        self.ping_data_dict = {}   # dictionary to store metadata
        self.power_dict = {}   # dictionary to store power data
        self.angle_dict = {}   # dictionary to store angle data
        self.ping_time = []    # list to store ping time
        self.first_ch_flag = False   # flag to help check ping time
        self.CON1_datagram = None    # storage for CON1 datagram for ME70

        # Constants for unpacking .raw files
        self.BLOCK_SIZE = 1024*4             # Block size read in from binary file to search for token
        self.LENGTH_SIZE = 4
        self.DATAGRAM_HEADER_SIZE = 12
        self.CONFIG_HEADER_SIZE = 516
        self.CONFIG_TRANSDUCER_SIZE = 320

        # Set global regex expressions to find all sample, annotation and NMEA sentences
        self.SAMPLE_REGEX = b'RAW\d{1}'
        self.SAMPLE_MATCHER = re.compile(self.SAMPLE_REGEX, re.DOTALL)
        self.FILENAME_REGEX = r'(?P<prefix>\S*)-D(?P<date>\d{1,})-T(?P<time>\d{1,})'
        self.FILENAME_MATCHER = re.compile(self.FILENAME_REGEX, re.DOTALL)

        # Reference time "seconds since 1900-01-01 00:00:00"
        self.REF_TIME = date2num(dt(1900, 1, 1, 0, 0, 0))
        self.WINDOWS_EPOCH = dt(1601, 1, 1)
        self.NTP_EPOCH = dt(1900, 1, 1)
        self.NTP_WINDOWS_DELTA = (self.NTP_EPOCH - self.WINDOWS_EPOCH).total_seconds()

        # Numpy data type object for unpacking the Sample datagram including the header from binary *.raw
        sample_dtype = np.dtype([('length1', 'i4'),  # 4 byte int (long)
                                 # Datagram header
                                 ('datagram_type', 'a4'),  # 4 byte string
                                 ('low_date_time', 'u4'),  # 4 byte int (long)
                                 ('high_date_time', 'u4'),  # 4 byte int (long)
                                 # Sample datagram
                                 ('channel_number', 'i2'),  # 2 byte int (short)
                                 ('mode', 'i2'),  # 2 byte int (short): whether split-beam or single-beam
                                 ('transducer_depth', 'f4'),  # 4 byte float
                                 ('frequency', 'f4'),  # 4 byte float
                                 ('transmit_power', 'f4'),  # 4 byte float
                                 ('pulse_length', 'f4'),  # 4 byte float
                                 ('bandwidth', 'f4'),  # 4 byte float
                                 ('sample_interval', 'f4'),  # 4 byte float
                                 ('sound_velocity', 'f4'),  # 4 byte float
                                 ('absorption_coefficient', 'f4'),  # 4 byte float
                                 ('heave', 'f4'),  # 4 byte float
                                 ('roll', 'f4'),  # 4 byte float
                                 ('pitch', 'f4'),  # 4 byte float
                                 ('temperature', 'f4'),  # 4 byte float
                                 ('trawl_upper_depth_valid', 'i2'),  # 2 byte int (short)
                                 ('trawl_opening_valid', 'i2'),  # 2 byte int (short)
                                 ('trawl_upper_depth', 'f4'),  # 4 byte float
                                 ('trawl_opening', 'f4'),  # 4 byte float
                                 ('offset', 'i4'),  # 4 byte int (long)
                                 ('count', 'i4')])  # 4 byte int (long): number of items to unpack for power_data
        self.sample_dtype = sample_dtype.newbyteorder('<')
        self.power_dtype = np.dtype([('power_data', '<i2')])     # 2 byte int (short)
        self.angle_dtype = np.dtype([('athwartship', '<i1'), ('alongship', '<i1')])     # 1 byte ints

        # Initialize other params that will be unpacked from data
        self.config_header = None
        self.config_transducer = None
        self.first_ping_metadata = None
        self.data_times = None
        self.motion = None
        self.power_data_dict = None
        self.angle_data_dict = None
        self.tr_data_dict = None
        self.nc_path = None

        # Other params to be input by user
        self.platform_name = ""
        self.platform_type = ""
        self.platform_code_ICES = ""

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

    @property
    def platform_name(self):
        return self._platform_name

    @platform_name.setter
    def platform_name(self, pname):
        self._platform_name = pname

    @property
    def platform_type(self):
        return self._platform_type

    @platform_type.setter
    def platform_type(self, ptype):
        self._platform_type = ptype

    @property
    def platform_code_ICES(self):
        return self._platform_code_ICES

    @platform_code_ICES.setter
    def platform_code_ICES(self, pcode):
        self._platform_code_ICES = pcode

    @staticmethod
    def _read_config_header(chunk):
        """Reading EK60 `.raw` configuration header information from the byte string passed in as a chunk.

        This method unpacks info from configuration header into self.config_header

        Parameters
        ----------
        chunk : int
            data chunk to read the config header from
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
        for i in range(4):
            values[i] = values[i].strip(b'\x00')

        # create the configuration header dictionary
        return dict(zip(field_names, values))

    @staticmethod
    def _read_config_transducer(chunk):
        """Reading EK60 `.raw` configuration transducer information from the byte string passed in as a chunk.

        This method unpacks info from transducer header info self.config_transducer

        Parameters
        ----------
        chunk : int
            data chunk to read the configuration transducer information from
        """

        # setup unpack structure and field names
        field_names = ('channel_id', 'beam_type', 'frequency', 'gain',
                       'equiv_beam_angle', 'beam_width_alongship', 'beam_width_athwartship',
                       'angle_sensitivity_alongship', 'angle_sensitivity_athwartship',
                       'angle_offset_alongship', 'angle_offset_athwartship', 'pos_x', 'pos_y',
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
            values[i] = values[i].strip(b'\x00')

        # put it back together, dropping the spare strings
        config_transducer = dict(zip(field_names[0:17], values[0:17]))
        config_transducer[field_names[17]] = pulse_length_table
        config_transducer[field_names[18]] = gain_table
        config_transducer[field_names[19]] = sa_correction_table
        config_transducer[field_names[20]] = values[35]
        return config_transducer

    def read_header(self, file_handle):
        """Read header and transducer config from EK60 raw data file.

        This method calls private methods `_read_config_header()` and `_read_config_transducer()` to
        populate self.config_header and self.config_transducer.
        """

        # Read binary file a block at a time
        raw = file_handle.read(self.BLOCK_SIZE)

        # Read the configuration datagram, output at the beginning of the file
        length1, = unpack_from('<l', raw)
        byte_cnt = self.LENGTH_SIZE

        # Configuration datagram header
        byte_cnt += self.DATAGRAM_HEADER_SIZE

        # Configuration: header
        config_header = self._read_config_header(raw[byte_cnt:byte_cnt + self.CONFIG_HEADER_SIZE])
        byte_cnt += self.CONFIG_HEADER_SIZE
        config_transducer = []
        for num_transducer in range(config_header['transducer_count']):
            config_transducer.append(self._read_config_transducer(raw[byte_cnt:byte_cnt + self.CONFIG_TRANSDUCER_SIZE]))
            byte_cnt += self.CONFIG_TRANSDUCER_SIZE

        # Compare length1 (from beginning of datagram) to length2 (from the end of datagram) to
        # the actual number of bytes read. A mismatch can indicate an invalid, corrupt, misaligned,
        # or missing configuration datagram or a reverse byte order binary data file.
        # A bad/missing configuration datagram header is a significant error.
        length2, = unpack_from('<l', raw, byte_cnt)
        if not (length1 == length2 == byte_cnt-self.LENGTH_SIZE):
            print('Possible file corruption or format incompatibility.')
    #         raise InstrumentDataException(
    #             "Length of configuration datagram and number of bytes read do not match: length1: %s"
    #             ", length2: %s, byte_cnt: %s. Possible file corruption or format incompatibility." %
    #             (length1, length2, byte_cnt+LENGTH_SIZE))
        byte_cnt += self.LENGTH_SIZE
        file_handle.seek(byte_cnt)

        # Populate class attributes
        self.config_header = config_header
        self.config_transducer = config_transducer

    def _windows_to_ntp(self, windows_time):
        """Convert a windows file timestamp into Network Time Protocol.

        Parameters
        ----------
        windows_time
            100ns since Windows time epoch

        Returns
        -------
            timestamp into Network Time Protocol (NTP).
        """
        return windows_time / 1e7 - self.NTP_WINDOWS_DELTA

    @staticmethod
    def _build_windows_time(high_word, low_word):
        """Generate Windows time value from high and low date times.

        Parameters
        ----------
        high_word
            high word portion of the Windows datetime
        low_word
            low word portion of the Windows datetime

        Returns
        -------
            time in 100ns since 1601/01/01 00:00:00 UTC
        """
        return (high_word << 32) + low_word

    def process_sample(self, input_file, transducer_count):
        """Processing one sample at a time from input_file.

        Parameters
        ----------
        input_file : str
            EK60 raw data file name
        transducer_count : int
            number of transducers

        Returns
        -------
        channel : int
            number of channels in the sample
        ntp_time : float
            Network Time Protocol time
        sample_data, power_data, angle_data : list
            data and metadata contained in each sample
        """
        # log.trace('Processing one sample from input_file: %r', input_file)
        # print('Processing one sample from input_file')

        # Read and unpack the Sample Datagram into numpy array
        sample_data = np.fromfile(input_file, dtype=self.sample_dtype, count=1)
        channel = sample_data['channel_number'][0]

        # Check for a valid channel number that is within the number of transducers config
        # to prevent incorrectly indexing into the dictionaries.
        # An out of bounds channel number can indicate invalid, corrupt,
        # or misaligned datagram or a reverse byte order binary data file.
        # Log warning and continue to try and process the rest of the file.
        if channel < 0 or channel > transducer_count:
            print('Invalid channel: %s for transducer count: %s. \n\
            Possible file corruption or format incompatibility.' % (channel, transducer_count))

        # Convert high and low bytes to internal time
        windows_time = self._build_windows_time(sample_data['high_date_time'][0], sample_data['low_date_time'][0])
        ntp_time = self._windows_to_ntp(windows_time)

        count = sample_data['count'][0]

        # Extract array of power data
        power_data = np.fromfile(input_file, dtype=self.power_dtype, count=count).astype('f8')

        # Read the athwartship and alongship angle measurements
        if sample_data['mode'][0] > 1:
            angle_data = np.fromfile(input_file, dtype=self.angle_dtype, count=count)
        else:
            angle_data = []

        # Read and compare length1 (from beginning of datagram) to length2
        # (from the end of datagram). A mismatch can indicate an invalid, corrupt,
        # or misaligned datagram or a reverse byte order binary data file.
        # Log warning and continue to try and process the rest of the file.
        len_dtype = np.dtype([('length2', '<i4')])  # 4 byte int (long)
        length2_data = np.fromfile(input_file, dtype=len_dtype, count=1)
        if not (sample_data['length1'][0] == length2_data['length2'][0]):
            print('Mismatching beginning and end length values in sample datagram: \n\
            length1: %d, length2: %d.\n\
            Possible file corruption or format incompatibility.'
                  % (sample_data['length1'][0], length2_data['length2'][0]))

        return channel, ntp_time, sample_data, power_data, angle_data

    @staticmethod
    def append_metadata(metadata, channel, sample_data):
        """Store metadata when reading the first ping of all channels.

        Parameters
        ----------
        metadata
            first_ping_metadata[channel] to be saved to
        channel
            channel from which metadata is being read
        sample_data
            unpacked sample data from process_sample()
        """
        # Fixed across ping
        metadata['channel'].append(channel)
        metadata['transducer_depth'].append(sample_data['transducer_depth'][0])          # [meters]
        metadata['frequency'].append(sample_data['frequency'][0])                        # [Hz]
        metadata['sound_velocity'].append(sample_data['sound_velocity'][0])              # [m/s]
        metadata['absorption_coeff'].append(sample_data['absorption_coefficient'][0])    # [dB/m]
        metadata['temperature'].append(sample_data['temperature'][0])                    # [degC]
        metadata['mode'].append(sample_data['mode'][0])             # >1: split-beam, 0: single-beam

        return metadata  # this may be removed?

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
            a RawSimradFile file object opened in ``self.load_ek60.raw()``
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
            new_datagram['timestamp'] = \
                np.datetime64(new_datagram['timestamp'], '[ms]')

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
            self.config_datagram['timestamp'] = np.datetime64(self.config_datagram['timestamp'], '[ms]')

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
            for i, length in enumerate(uni):
                self.ping_time_split[i] = self.ping_time[:uni_cnt[i]]
                self.power_dict_split[i] = {ch_num: [] for ch_num in self.config_datagram['transceivers'].keys()}

            for ch_num in self.config_datagram['transceivers'].keys():
                # r_b represents index for range_bin (how many different range_bins there are).
                # r is the list of indexes that correspond to that range
                for r_b, r in enumerate(indices.values()):
                    for r_idx in r:
                        self.power_dict_split[r_b][ch_num].append(self.power_dict[ch_num][r_idx])
                    self.power_dict_split[r_b][ch_num] = np.array(self.power_dict_split[r_b][ch_num]) * INDEX2POWER

        # TODO: need to convert angle data too
        self.ping_time = np.array(self.ping_time)
        self.range_lengths = uni

        # Trim excess data from NMEA object
        self.nmea_data.trim()

    def load_ek60_raw_old(self):
        with open(self.filename, 'rb') as input_file:  # read ('r') input file using binary mode ('b')

            self.read_header(input_file)  # unpack info to self.config_header and self.config_transducer
            position = input_file.tell()

            # *_data_temp_dict are for storing different channels within each ping
            # content of *_temp_dict are saved to *_data_dict whenever all channels of the same ping are unpacked
            # see below comment "Check if we have enough records to produce a new row of data"

            # Initialize output structure
            first_ping_metadata = defaultdict(list)  # metadata for each channel
            power_data_dict = defaultdict(list)      # echo power
            angle_data_dict = defaultdict(list)      # alongship and athwartship electronic angle
            tr_data_dict = defaultdict(list)         # transmit signal metadata
            data_times = []                          # ping time
            motion = []                              # pitch, roll, heave

            # Read binary file a block at a time
            raw = input_file.read(self.BLOCK_SIZE)

            # Flag used to check if data are from the same ping
            last_time = None

            while len(raw) > 4:
                # We only care for the Sample datagrams, skip over all the other datagrams
                match = self.SAMPLE_MATCHER.search(raw)

                if match:
                    # Offset by size of length value
                    match_start = match.start() - self.LENGTH_SIZE

                    # Seek to the position of the length data before the token to read into numpy array
                    input_file.seek(position + match_start)

                    # try:
                    next_channel, next_time, next_sample, next_power, next_angle = \
                        self.process_sample(input_file, self.config_header['transducer_count'])  # read each sample

                    # Check if it's from different channels within the same ping
                    # next_time=last_time when it's the same ping but different channel
                    if next_time != last_time:   # if data is from a new ping
                        # Clear out our temporary dictionaries and set the last time to this time
                        sample_temp_dict = defaultdict(list)
                        power_temp_dict = defaultdict(list)
                        angle_temp_dict = defaultdict(list)    # include both alongship and athwartship angle
                        last_time = next_time    # update ping time

                    # Store this data
                    sample_temp_dict[next_channel] = next_sample
                    power_temp_dict[next_channel] = next_power
                    angle_temp_dict[next_channel] = next_angle

                    # Check if we have enough records to produce a new row of data
                    # if yes this means that data from all transducer channels have been read for a particular ping
                    # a new row of data means all channels of data from one ping
                    # if only 2 channels of data were received but there are a total of 3 transducers,
                    # the data are not stored in the final power_data_dict
                    if len(sample_temp_dict) == len(power_temp_dict) == \
                            len(angle_temp_dict) == self.config_header['transducer_count']:

                        # if this is the first ping from all channels,
                        # create metadata particle and store the frequency / bin_size
                        if not power_data_dict:

                            # Initialize each channel to defaultdict
                            for channel in power_temp_dict:
                                first_ping_metadata[channel] = defaultdict(list)
                                angle_data_dict[channel] = []

                            # Fill in metadata for each channel
                            for channel, sample_data in sample_temp_dict.items():
                                self.append_metadata(first_ping_metadata[channel], channel, sample_data)

                        # Save data and metadata from each ping to *_data_dict
                        data_times.append(next_time)
                        motion.append(np.array([(sample_temp_dict[1]['heave'],  # all channels have the same motion
                                                 sample_temp_dict[1]['pitch'],
                                                 sample_temp_dict[1]['roll'])],
                                               dtype=[('heave', 'f4'), ('pitch', 'f4'), ('roll', 'f4')]))
                        for channel in power_temp_dict:
                            power_data_dict[channel].append(power_temp_dict[channel])
                            if any(angle_temp_dict[channel]):   # if split-beam data
                                angle_data_dict[channel].append(angle_temp_dict[channel])
                            tr = np.array([(sample_temp_dict[channel]['frequency'],
                                            sample_temp_dict[channel]['transmit_power'],
                                            sample_temp_dict[channel]['pulse_length'],
                                            sample_temp_dict[channel]['bandwidth'],
                                            sample_temp_dict[channel]['sample_interval'])],
                                          dtype=[('frequency', 'f4'), ('transmit_power', 'f4'),
                                                 ('pulse_length', 'f4'), ('bandwidth', 'f4'),
                                                 ('sample_interval', 'f4')])
                            tr_data_dict[channel].append(tr)

                    # except InvalidTransducer:
                    #   pass

                else:
                    input_file.seek(position + self.BLOCK_SIZE - 4)

                # Need current position in file to increment for next regex search offset
                position = input_file.tell()
                # Read the next block for regex search
                raw = input_file.read(self.BLOCK_SIZE)

            data_times = np.array(data_times)
            # Convert to numpy array and decompress power data to dB
            for channel in power_data_dict:
                power_data_dict[channel] = np.array(power_data_dict[channel]) * 10. * np.log10(2) / 256.
                if angle_data_dict[channel]:  # if split-beam data
                    angle_data_dict[channel] = np.array(angle_data_dict[channel])
                else:  # if single-beam data
                    angle_data_dict[channel] = []
                tr_data_dict[channel] = np.array(tr_data_dict[channel])

            self.first_ping_metadata = first_ping_metadata
            self.data_times = data_times
            self.motion = motion
            self.power_data_dict = power_data_dict
            self.angle_data_dict = angle_data_dict
            self.tr_data_dict = tr_data_dict

    def raw2nc(self):
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
            out_dict['date_created'] = dt.strptime(fm.group('date') + '-' + fm.group('time'),
                                                   '%Y%m%d-%H%M%S').isoformat() + 'Z'
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
                tx_pos['transducer_offset_x'][c_seq] = c['pos_x']
                tx_pos['transducer_offset_y'][c_seq] = c['pos_y']
                tx_pos['transducer_offset_z'][c_seq] = c['pos_z'] + self.ping_data_dict[c_seq+1]['transducer_depth'][0]
                beam_dict['equivalent_beam_angle'][c_seq] = c['equivalent_beam_angle']
                beam_dict['gain_correction'][c_seq] = c['gain']
                beam_dict['gpt_software_version'].append(c['gpt_software_version'])
                beam_dict['channel_id'].append(c['channel_id'])

            beam_dict['beam_width'] = bm_width
            beam_dict['beam_direction'] = bm_dir
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

        # Get nc filename
        filename = os.path.splitext(os.path.basename(self.filename))[0]
        self.nc_path = os.path.join(os.path.split(self.filename)[0], filename + '.nc')
        fm = self.FILENAME_MATCHER.match(self.filename)

        # Check if nc file already exists
        # ... if yes, abort conversion and issue warning
        # ... if not, continue with conversion
        if os.path.exists(self.nc_path):
            print('          ... this file has already been converted to .nc, conversion not executed.')
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
                                    for x in self.config_datagram['transceivers'].keys()],
                                   dtype='float32')
                ss_val = np.array([self.ping_data_dict[x]['sound_velocity'][0]
                                   for x in self.config_datagram['transceivers'].keys()],
                                  dtype='float32')
            # --- if NOT identical for all pings, save as array of dimension [frequency x ping_time]
            else:
                abs_val = np.array([self.ping_data_dict[x]['absorption_coefficient']
                                    for x in self.config_datagram['transceivers'].keys()],
                                   dtype='float32')
                ss_val = np.array([self.ping_data_dict[x]['sound_velocity']
                                   for x in self.config_datagram['transceivers'].keys()],
                                  dtype='float32')

            # Create SetGroups object
            grp = SetGroups(file_path=self.nc_path, echo_type='EK60')
            grp.set_toplevel(_set_toplevel_dict())  # top-level group
            grp.set_env(_set_env_dict())            # environment group
            grp.set_provenance(os.path.basename(self.filename),
                               _set_prov_dict())    # provenance group
            grp.set_platform(_set_platform_dict())  # platform group
            grp.set_nmea(_set_nmea_dict())          # platform/NMEA group
            grp.set_sonar(_set_sonar_dict())        # sonar group
            grp.set_beam(_set_beam_dict())          # beam group
