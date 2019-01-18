"""Functions to unpack Simrad EK60 .raw and save to .nc.

Pieces for unpacking power data part came from:
https://github.com/oceanobservatories/mi-instrument (authors: Ronald Ronquillo & Richard Han)
Specific modifications include:
- python 3.6 compatibility
- strip off dependency on other mi-instrument functions
- unpack split-beam angle data
- unpack various additional variables needed for calibration
"""


import re, os
from collections import defaultdict
from struct import unpack_from, unpack
import numpy as np
from datetime import datetime as dt
from matplotlib.dates import date2num
import pytz
from .set_nc_groups import SetGroups
from echopype.version import VERSION as ECHOPYPE_VERSION


# Set constants for unpacking .raw files
BLOCK_SIZE = 1024*4             # Block size read in from binary file to search for token
LENGTH_SIZE = 4
DATAGRAM_HEADER_SIZE = 12
CONFIG_HEADER_SIZE = 516
CONFIG_TRANSDUCER_SIZE = 320

# set global regex expressions to find all sample, annotation and NMEA sentences
SAMPLE_REGEX = b'RAW\d{1}'
SAMPLE_MATCHER = re.compile(SAMPLE_REGEX, re.DOTALL)
FILENAME_REGEX = r'(?P<prefix>\S*)-D(?P<date>\d{1,})-T(?P<time>\d{1,})'
FILENAME_MATCHER = re.compile(FILENAME_REGEX, re.DOTALL)

# Reference time "seconds since 1900-01-01 00:00:00"
REF_TIME = date2num(dt(1900, 1, 1, 0, 0, 0))

WINDOWS_EPOCH = dt(1601, 1, 1)
NTP_EPOCH = dt(1900, 1, 1)
NTP_WINDOWS_DELTA = (NTP_EPOCH - WINDOWS_EPOCH).total_seconds()

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
sample_dtype = sample_dtype.newbyteorder('<')

power_dtype = np.dtype([('power_data', '<i2')])     # 2 byte int (short)

angle_dtype = np.dtype([('athwartship', '<i1'), ('alongship', '<i1')])     # 1 byte ints


def read_config_header(chunk):
    """
    Reads the EK60 raw data file configuration header information
    from the byte string passed in as a chunk.

    :param chunk: data chunk to read the config header from
    :return:      configuration header
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
    config_header = dict(zip(field_names, values))
    return config_header


def read_config_transducer(chunk):
    """
    Reads the EK60 raw data file configuration transducer information
    from the byte string passed in as a chunk.

    :param chunk: data chunk to read the configuration transducer information from
    :return:      configuration transducer information
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


def read_header(filehandle):
    """Read header and transducer config from EK60 raw data file."""

    # Read binary file a block at a time
    raw = filehandle.read(BLOCK_SIZE)

    # Read the configuration datagram, output at the beginning of the file
    length1, = unpack_from('<l', raw)
    byte_cnt = LENGTH_SIZE

    # Configuration datagram header
    byte_cnt += DATAGRAM_HEADER_SIZE

    # Configuration: header
    config_header = read_config_header(raw[byte_cnt:byte_cnt+CONFIG_HEADER_SIZE])
    byte_cnt += CONFIG_HEADER_SIZE
    config_transducer = []
    for num_transducer in range(config_header['transducer_count']):
        config_transducer.append(read_config_transducer(raw[byte_cnt:byte_cnt+CONFIG_TRANSDUCER_SIZE]))
        byte_cnt += CONFIG_TRANSDUCER_SIZE

    # Compare length1 (from beginning of datagram) to length2 (from the end of datagram) to
    # the actual number of bytes read. A mismatch can indicate an invalid, corrupt, misaligned,
    # or missing configuration datagram or a reverse byte order binary data file.
    # A bad/missing configuration datagram header is a significant error.
    length2, = unpack_from('<l', raw, byte_cnt)
    if not (length1 == length2 == byte_cnt-LENGTH_SIZE):
        print('Possible file corruption or format incompatibility.')
#         raise InstrumentDataException(
#             "Length of configuration datagram and number of bytes read do not match: length1: %s"
#             ", length2: %s, byte_cnt: %s. Possible file corruption or format incompatibility." %
#             (length1, length2, byte_cnt+LENGTH_SIZE))
    byte_cnt += LENGTH_SIZE
    filehandle.seek(byte_cnt)
    return config_header, config_transducer


def windows_to_ntp(windows_time):
    """
    Convert a windows file timestamp into Network Time Protocol.

    :param windows_time:  100ns since Windows time epoch
    :return:
    """
    return windows_time / 1e7 - NTP_WINDOWS_DELTA


def build_windows_time(high_word, low_word):
    """
    Generate Windows time value from high and low date times.

    :param high_word:  high word portion of the Windows datetime
    :param low_word:   low word portion of the Windows datetime
    :return:  time in 100ns since 1601/01/01 00:00:00 UTC
    """
    return (high_word << 32) + low_word


def process_sample(input_file, transducer_count):
    """
    Processing one sample at a time from input_file.

    :param input_file:        EK60 raw data file name
    :param transducer_count:  number of transducers
    :return:  data contained in each sample, in the following sequence:
              channel, ntp_time, sample_data, power_data, angle_data
    """
    # log.trace('Processing one sample from input_file: %r', input_file)
    # print('Processing one sample from input_file')

    # Read and unpack the Sample Datagram into numpy array
    sample_data = np.fromfile(input_file, dtype=sample_dtype, count=1)
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
    windows_time = build_windows_time(sample_data['high_date_time'][0], sample_data['low_date_time'][0])
    ntp_time = windows_to_ntp(windows_time)

    count = sample_data['count'][0]

    # Extract array of power data
    power_data = np.fromfile(input_file, dtype=power_dtype, count=count).astype('f8')

    # Read the athwartship and alongship angle measurements
    if sample_data['mode'][0] > 1:
        angle_data = np.fromfile(input_file, dtype=angle_dtype, count=count)
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
        Possible file corruption or format incompatibility.' % (sample_data['length1'][0], length2_data['length2'][0]))

    return channel, ntp_time, sample_data, power_data, angle_data


def append_metadata(metadata, channel, sample_data):
    """
    Store metadata when reading the first ping of all channels.

    :param metadata:     first_ping_metadata[channel] to be saved to
    :param channel:      channel from which metadata is being read
    :param sample_data:  unpacked sample data from process_sample()
    :return:
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


def load_ek60_raw(raw_filename):
    """
    Parse the *.raw file.

    :param raw_filename:  absolute path/name to file to be parsed
    """
    print('%s  unpacking file: %s' % (dt.now().strftime('%H:%M:%S'), raw_filename))

    with open(raw_filename, 'rb') as input_file:  # read ('r') input file using binary mode ('b')

        config_header, config_transducer = read_header(input_file)
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
        raw = input_file.read(BLOCK_SIZE)

        # Flag used to check if data are from the same ping
        last_time = None

        while len(raw) > 4:
            # We only care for the Sample datagrams, skip over all the other datagrams
            match = SAMPLE_MATCHER.search(raw)

            if match:
                # Offset by size of length value
                match_start = match.start() - LENGTH_SIZE

                # Seek to the position of the length data before the token to read into numpy array
                input_file.seek(position + match_start)

                # try:
                next_channel, next_time, next_sample, next_power, next_angle = \
                    process_sample(input_file, config_header['transducer_count'])  # read each sample

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
                        len(angle_temp_dict) == config_header['transducer_count']:

                    # if this is the first ping from all channels,
                    # create metadata particle and store the frequency / bin_size
                    if not power_data_dict:

                        # Initialize each channel to defaultdict
                        for channel in power_temp_dict:
                            first_ping_metadata[channel] = defaultdict(list)
                            angle_data_dict[channel] = []

                        # Fill in metadata for each channel
                        for channel, sample_data in sample_temp_dict.items():
                            append_metadata(first_ping_metadata[channel], channel, sample_data)

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
                input_file.seek(position + BLOCK_SIZE - 4)

            # Need current position in file to increment for next regex search offset
            position = input_file.tell()
            # Read the next block for regex search
            raw = input_file.read(BLOCK_SIZE)

        data_times = np.array(data_times)
        # Convert to numpy array and decompress power data to dB
        for channel in power_data_dict:
            power_data_dict[channel] = np.array(power_data_dict[channel]) * 10. * np.log10(2) / 256.
            if angle_data_dict[channel]:  # if split-beam data
                angle_data_dict[channel] = np.array(angle_data_dict[channel])
            else:  # if single-beam data
                angle_data_dict[channel] = []
            tr_data_dict[channel] = np.array(tr_data_dict[channel])

        return first_ping_metadata, data_times, motion, \
            power_data_dict, angle_data_dict, tr_data_dict, \
            config_header, config_transducer


def save_raw_to_nc(raw_filename):
    """ Save data from RAW to netCDF format.

    :param raw_filename:  file name of EK60 raw data file
    """
    # Load data from RAW file
    first_ping_metadata, data_times, motion, \
        power_data_dict, angle_data_dict, tr_data_dict, \
        config_header, config_transducer = load_ek60_raw(raw_filename)

    # Get nc filename
    filename = os.path.splitext(os.path.basename(raw_filename))[0]
    nc_path = os.path.join(os.path.split(raw_filename)[0], filename + '.nc')
    fm = FILENAME_MATCHER.match(filename)

    # Retrieve variables
    tx_num = config_header['transducer_count']
    freq = np.array([x['frequency'][0] for x in first_ping_metadata.values()], dtype='float32')
    abs_val = np.array([x['absorption_coeff'][0] for x in first_ping_metadata.values()], dtype='float32')
    ss_val = np.array([x['sound_velocity'][0] for x in first_ping_metadata.values()], dtype='float32')

    # Subfunctions to set various dictionaries
    def _set_toplevel_dict():
        attrs = ('Conventions', 'keywords',
                 'sonar_convention_authority', 'sonar_convention_name',
                 'sonar_convention_version', 'summary', 'title')
        vals = ('CF-1.7, SONAR-netCDF4, '
                'ACDD-1.3', 'EK60',
                'ICES', 'SONAR-netCDF4', '1.7',
                '', '')
        out_dict = dict(zip(attrs, vals))
        out_dict['date_created'] = dt.strptime(fm.group('date') + '-' + fm.group('time'),
                                               '%Y%m%d-%H%M%S').isoformat() + 'Z'
        return out_dict

    def _set_env_dict():
        attrs = ('frequency', 'absorption_coeff', 'sound_speed')
        vals = (freq, abs_val, ss_val)
        return dict(zip(attrs, vals))

    def _set_prov_dict():
        attrs = ('conversion_software_name', 'conversion_software_version', 'conversion_time')
        vals = ('echopype', ECHOPYPE_VERSION, dt.now(tz=pytz.utc).isoformat(timespec='seconds'))  # use UTC time
        return dict(zip(attrs, vals))

    def _set_sonar_dict():
        attrs = ('sonar_manufacturer', 'sonar_model', 'sonar_serial_number',
                       'sonar_software_name', 'sonar_software_version', 'sonar_type')
        vals = ('Simrad', config_header['sounder_name'].decode('utf-8'), '',
                      '', config_header['version'].decode('utf-8'), 'echosounder')
        return dict(zip(attrs, vals))

    def _set_platform_dict():
        out_dict = dict()
        out_dict['platform_name'] = config_header['survey_name'].decode('utf-8')
        if re.search('OOI', out_dict['platform_name']):
            out_dict['platform_type'] = 'subsurface mooring'  # if OOI
        else:
            out_dict['platform_type'] = 'ship'  # default to ship
        out_dict['time'] = data_times  # [seconds since 1900-01-01] for xarray.to_netcdf conversion
        out_dict['pitch'] = np.array([x['pitch'] for x in motion.__iter__()], dtype='float32').squeeze()
        out_dict['roll'] = np.array([x['roll'] for x in motion.__iter__()], dtype='float32').squeeze()
        out_dict['heave'] = np.array([x['heave'] for x in motion.__iter__()], dtype='float32').squeeze()
        # water_level is set to 0 for EK60 since this is not separately recorded
        # and is part of transducer_depth
        out_dict['water_level'] = np.int32(0)
        return out_dict

    def _set_beam_dict():
        beam_dict = dict()
        beam_dict['beam_mode'] = 'vertical'
        beam_dict['conversion_equation_t'] = 'type_3'  # type_3 is EK60 conversion
        beam_dict['ping_time'] = data_times   # [seconds since 1900-01-01] for xarray.to_netcdf conversion
        beam_dict['backscatter_r'] = np.array([power_data_dict[x] for x in power_data_dict])
        beam_dict['frequency'] = freq                                    # added by echopype, not in convention
        beam_dict['range_bin'] = np.arange(power_data_dict[1].shape[1])  # added by echopype, not in convention

        # Loop through each transducer for variables that are the same for each file
        bm_width = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
        bm_dir = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
        tx_pos = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
        beam_dict['equivalent_beam_angle'] = np.zeros(shape=(tx_num,), dtype='float32')
        beam_dict['gain_correction'] = np.zeros(shape=(tx_num,), dtype='float32')
        beam_dict['gpt_software_version'] = []
        beam_dict['channel_id'] = []
        for c_seq, c in enumerate(config_transducer.__iter__()):
            bm_width['beamwidth_receive_major'][c_seq] = c['beam_width_alongship']
            bm_width['beamwidth_receive_minor'][c_seq] = c['beam_width_athwartship']
            bm_width['beamwidth_transmit_major'][c_seq] = c['beam_width_alongship']
            bm_width['beamwidth_transmit_minor'][c_seq] = c['beam_width_athwartship']
            bm_dir['beam_direction_x'][c_seq] = c['dir_x']
            bm_dir['beam_direction_y'][c_seq] = c['dir_y']
            bm_dir['beam_direction_z'][c_seq] = c['dir_z']
            tx_pos['transducer_offset_x'][c_seq] = c['pos_x']
            tx_pos['transducer_offset_y'][c_seq] = c['pos_y']
            tx_pos['transducer_offset_z'][c_seq] = c['pos_z'] + first_ping_metadata[c_seq+1]['transducer_depth'][0]
            beam_dict['equivalent_beam_angle'][c_seq] = c['equiv_beam_angle']
            beam_dict['gain_correction'][c_seq] = c['gain']
            beam_dict['gpt_software_version'].append(c['gpt_software_version'].decode('utf-8'))
            beam_dict['channel_id'].append(c['channel_id'].decode('utf-8'))

        # Loop through each transducer for variables that may vary at each ping
        # -- this rarely is the case for EK60 so we check first before saving
        pl_tmp = np.unique(tr_data_dict[1]['pulse_length']).size
        pw_tmp = np.unique(tr_data_dict[1]['transmit_power']).size
        bw_tmp = np.unique(tr_data_dict[1]['bandwidth']).size
        si_tmp = np.unique(tr_data_dict[1]['sample_interval']).size
        if pl_tmp==1 and pw_tmp==1 and bw_tmp==1 and si_tmp==1:
            tx_sig = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            beam_dict['sample_interval'] = np.zeros(shape=(tx_num,), dtype='float32')
            for t_seq in range(tx_num):
                tx_sig['transmit_duration_nominal'][t_seq] = tr_data_dict[t_seq + 1]['pulse_length'][0]
                tx_sig['transmit_power'][t_seq] = tr_data_dict[t_seq + 1]['transmit_power'][0]
                tx_sig['transmit_bandwidth'][t_seq] = tr_data_dict[t_seq + 1]['bandwidth'][0]
                beam_dict['sample_interval'][t_seq] = tr_data_dict[t_seq + 1]['sample_interval'][0]
        else:
            tx_sig = defaultdict(lambda: np.zeros(shape=(tx_num, data_times.size), dtype='float32'))
            beam_dict['sample_interval'] = np.zeros(shape=(tx_num, data_times.size), dtype='float32')
            for t_seq in range(tx_num):
                tx_sig['transmit_duration_nominal'][t_seq, :] = tr_data_dict[t_seq + 1]['pulse_length'].squeeze()
                tx_sig['transmit_power'][t_seq, :] = tr_data_dict[t_seq + 1]['transmit_power'].squeeze()
                tx_sig['transmit_bandwidth'][t_seq, :] = tr_data_dict[t_seq + 1]['bandwidth'].squeeze()
                beam_dict['sample_interval'][t_seq, :] = tr_data_dict[t_seq + 1]['sample_interval'].squeeze()

        # Build other parameters
        beam_dict['non_quantitative_processing'] = np.array([0, ] * freq.size, dtype='int32')
        # -- sample_time_offset is set to 2 for EK60 data, this value is NOT from sample_data['offset']
        beam_dict['sample_time_offset'] = np.array([2, ] * freq.size, dtype='int32')

        idx = [np.argwhere(tr_data_dict[x + 1]['pulse_length'][0] == config_transducer[x]['pulse_length_table']).squeeze()
               for x in range(len(config_transducer))]
        beam_dict['sa_correction'] = np.array([x['sa_correction_table'][y]
                                               for x, y in zip(config_transducer.__iter__(), np.array(idx))])
        return beam_dict

    # Create SetGroups object
    grp = SetGroups(file_path=nc_path)
    grp.set_toplevel(_set_toplevel_dict())  # top-level group
    grp.set_env(_set_env_dict())            # environment group
    grp.set_provenance(os.path.basename(raw_filename),
                       _set_prov_dict())    # provenance group
    grp.set_platform(_set_platform_dict())  # platform group
    grp.set_sonar(_set_sonar_dict())        # sonar group
    grp.set_beam(_set_beam_dict())          # beam group

