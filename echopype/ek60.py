"""
Functions to unpack Simrad EK60 .raw files
Modification from original source (cited below) included:
- python 3.6 compatibility
- strip off dependency on other mi-instrument functions
- unpack split-beam angle data
- unpack various additional variables
- support saving to netCDF file

Original source for unpacking power data part:
oceanobservatories/mi-instrument @https://github.com/oceanobservatories/mi-instrument
Authors: Ronald Ronquillo & Richard Han

"""


from collections import defaultdict
from struct import unpack_from, unpack
import numpy as np
import re
import os
from datetime import datetime as dt
from matplotlib.dates import date2num
import echopype as ep


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
    for i in range(4):
        values[i] = values[i].strip(b'\x00')

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
    Convert a windows file timestamp into Network Time Protocol
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
    # Fixed across ping
    metadata['channel'].append(channel)
    metadata['transducer_depth'].append(sample_data['transducer_depth'][0])          # [meters]
    metadata['frequency'].append(sample_data['frequency'][0])                        # [Hz]
    metadata['sound_velocity'].append(sample_data['sound_velocity'][0])              # [m/s]
    metadata['absorption_coeff'].append(sample_data['absorption_coefficient'][0])    # [dB/m]
    metadata['temperature'].append(sample_data['temperature'][0])                    # [degC]
    metadata['mode'].append(sample_data['mode'][0])             # >1: split-beam, 0: single-beam

    return metadata


def load_ek60_raw(raw_filename):
    """
    Parse the *.raw file.
    @param raw_filename absolute path/name to file to be parsed
    """
    print('%s  unpacking file: %s' % (dt.now().strftime('%H:%M:%S'), raw_filename))

    with open(raw_filename, 'rb') as input_file:  # read ('r') input file using binary mode ('b')

        config_header, config_transducer = read_header(input_file)
        transducer_count = config_header['transducer_count']

        position = input_file.tell()

        # *_data_temp_dict are for storing different channels within each ping
        # content of *_temp_dict are saved to *_data_dict whenever all channels of the same ping are unpacked
        # see below comment "Check if we have enough records to produce a new row of data"

        # Initialize output structure
        first_ping_metadata = defaultdict(list)  # metadata for each channel
        power_data_dict = defaultdict(list)      # echo power
        angle_data_dict = defaultdict(list)      # alongship and athwartship electronic angle
        motion_data_dict = defaultdict(list)     # heave, pitch, and roll motion
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
                    process_sample(input_file, transducer_count)  # read each sample

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
                        len(angle_temp_dict) == transducer_count:

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

        # convert ntp time, i.e. seconds since 1900-01-01 00:00:00 to matplotlib time
        data_times = np.array(data_times)
        data_times = (data_times / (60 * 60 * 24)) + REF_TIME

        # Convert to numpy array and decompress power data to dB
        for channel in power_data_dict:
            power_data_dict[channel] = np.array(power_data_dict[channel]) * 10. * np.log10(2) / 256.
            if angle_data_dict[channel]:  # if split-beam data
                angle_data_dict[channel] = np.array(angle_data_dict[channel])
            else:  # if single-beam data
                angle_data_dict[channel] = []
            motion_data_dict[channel] = np.array(motion_data_dict[channel])
            tr_data_dict[channel] = np.array(tr_data_dict[channel])

        return first_ping_metadata, data_times, motion, \
               power_data_dict, angle_data_dict, tr_data_dict, \
               config_header, config_transducer


def save_raw_to_nc(raw_filename):
    """
    Save data from RAW to netCDF format
    :param raw_filename:
    :return:
    """
    # Load data from RAW file
    first_ping_metadata, data_times, motion, \
    power_data_dict, angle_data_dict, tr_data_dict, \
    config_header, config_transducer = load_ek60_raw(raw_filename)

    # Get nc filename
    filename = os.path.splitext(os.path.basename(raw_filename))[0]
    nc_path = os.path.join(os.path.split(raw_filename)[0], filename + '.nc')
    fm = FILENAME_MATCHER.match(filename)

    # Create nc file by creating top-level group
    tl_attrs = ('Conventions', 'keywords',
                'sonar_convention_authority', 'sonar_convention_name', 'sonar_convention_version',
                'summary', 'title')
    tl_vals = ('CF-1.7, SONAR-netCDF4, ACDD-1.3', 'EK60',
               'ICES', 'SONAR-netCDF4', '1.7',
               '', '')
    tl_dict = dict(zip(tl_attrs, tl_vals))
    tl_dict['date_created'] = dt.strptime(fm.group('date') + '-' + fm.group('time'),
                                          '%Y%m%d-%H%M%S').isoformat() +'Z'
    ep.set_attrs_toplevel(nc_path, tl_dict)

    # Environment group
    freq_coord = np.array([x['frequency'][0] for x in first_ping_metadata.values()], dtype='float32')
    abs_val = np.array([x['absorption_coeff'][0] for x in first_ping_metadata.values()], dtype='float32')
    ss_val = np.array([x['sound_velocity'][0] for x in first_ping_metadata.values()], dtype='float32')
    env_attrs = ('frequency', 'absorption_coeff', 'sound_speed')
    env_vals = (freq_coord, abs_val, ss_val)
    env_dict = dict(zip(env_attrs, env_vals))
    ep.set_group_environment(nc_path, env_dict)

    # Provenance group
    prov_attrs = ('conversion_software_name', 'conversion_software_version', 'conversion_time')
    prov_vals = ('echopype', 'v0.1', dt.now().isoformat(timespec='seconds')+'Z')
    prov_dict = dict(zip(prov_attrs, prov_vals))
    ep.set_group_provenance(nc_path, os.path.basename(raw_filename), prov_dict)

    # Sonar group
    sonar_attrs = ('sonar_manufacturer', 'sonar_model', 'sonar_serial_number',
                   'sonar_software_name', 'sonar_software_version', 'sonar_type')
    sonar_vals = ('Simrad', config_header['sounder_name'].decode('utf-8'), '',
                  '', config_header['version'].decode('utf-8'), 'echosounder')
    sonar_dict = dict(zip(sonar_attrs, sonar_vals))
    ep.set_group_sonar(nc_path, sonar_dict)

    # Beam group
    beam_dict = dict()
    beam_dict['beam_mode'] = 'vertical'
    beam_dict['conversion_equation_t'] = 'type_3'  # type_3 is EK60 conversion
    beam_dict['ping_time'] = data_times            # here in matplotlib time
    beam_dict['backscatter_r'] = np.array([power_data_dict[x] for x in power_data_dict])
    beam_dict['beamwidth_receive_major'] = np.array([x['beam_width_alongship']
                                                     for x in config_transducer.__iter__()], dtype='float32')
    beam_dict['beamwidth_receive_minor'] = np.array([x['beam_width_athwartship']
                                                     for x in config_transducer.__iter__()], dtype='float32')
    beam_dict['beamwidth_transmit_major'] = np.array([x['beam_width_alongship']
                                                      for x in config_transducer.__iter__()], dtype='float32')
    beam_dict['beamwidth_transmit_minor'] = np.array([x['beam_width_athwartship']
                                                      for x in config_transducer.__iter__()], dtype='float32')
    beam_dict['beam_direction_x'] = np.array([x['dir_x'] for x in config_transducer.__iter__()], dtype='float32')
    beam_dict['beam_direction_y'] = np.array([x['dir_y'] for x in config_transducer.__iter__()], dtype='float32')
    beam_dict['beam_direction_z'] = np.array([x['dir_z'] for x in config_transducer.__iter__()], dtype='float32')
    beam_dict['equivalent_beam_angle'] = np.array([x['equiv_beam_angle']
                                                   for x in config_transducer.__iter__()], dtype='float32')
    beam_dict['gain_correction'] = np.array([x['gain'] for x in config_transducer.__iter__()], dtype='float32')
    beam_dict['non_quantitative_processing'] = np.array([0, ]*5, dtype='int32')
    beam_dict['sample_interval'] = np.array([x['sample_interval'] for x in tr_data_dict.values()],
                                            dtype='float32').squeeze()  # dimension frequency
    beam_dict['sample_time_offset'] = np.array([2, ]*5, dtype='int32')  # set to 2 for EK60 data, NOT from sample_data['offset']
    beam_dict['transmit_duration_nominal'] = np.array([x['pulse_length']
                                                       for x in tr_data_dict.values()], dtype='float32').squeeze()
    beam_dict['transmit_power'] = np.array([x['transmit_power']
                                            for x in tr_data_dict.values()], dtype='float32').squeeze()
    beam_dict['transmit_bandwidth'] = np.array([x['bandwidth']
                                                for x in tr_data_dict.values()], dtype='float32').squeeze()
    # Below not in convention
    beam_dict['frequency'] = freq_coord
    beam_dict['range_bin'] = np.arange(power_data_dict[1].shape[1])
    beam_dict['channel_id'] = [x['channel_id'].decode('utf-8') for x in config_transducer.__iter__()]
    beam_dict['gpt_software_version'] = [x['gpt_software_version'].decode('utf-8')
                                         for x in config_transducer.__iter__()]
    idx = [np.argwhere(tr_data_dict[x + 1]['pulse_length'][0] == config_transducer[x]['pulse_length_table']).squeeze()
           for x in range(len(config_transducer))]
    beam_dict['sa_correction'] = np.array([x['sa_correction_table'][y]
                                           for x, y in zip(config_transducer.__iter__(), np.array(idx))])
    ep.set_group_beam(nc_path, beam_dict)

    # Platform group
    platform_dict = dict()
    platform_dict['platform_name'] = config_header['survey_name'].decode('utf-8')
    platform_dict['time'] = data_times          # here in matplotlib time
    platform_dict['frequency'] = freq_coord     # this is not in convention
    platform_dict['pitch'] = np.array([x['pitch'] for x in motion.__iter__()], dtype='float32').squeeze()
    platform_dict['roll'] = np.array([x['roll'] for x in motion.__iter__()], dtype='float32').squeeze()
    platform_dict['heave'] = np.array([x['heave'] for x in motion.__iter__()], dtype='float32').squeeze()
    platform_dict['transducer_offset_x'] = np.array([x['pos_x'] for x in config_transducer.__iter__()], dtype='float32')
    platform_dict['transducer_offset_y'] = np.array([x['pos_y'] for x in config_transducer.__iter__()], dtype='float32')
    platform_dict['transducer_offset_z'] = np.array([x['pos_z'] for x in config_transducer.__iter__()],
                                                    dtype='float32') + \
                                           np.array([x['transducer_depth'][0] for x in first_ping_metadata.values()],
                                                    dtype='float32')
    platform_dict['water_level'] = np.int32(0)  # set to 0 for EK60 since this is not separately recorded
                                                # and is part of transducer_depth
    ep.set_group_platform(nc_path, platform_dict)