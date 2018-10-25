"""
Functions to unpack Simrad EK60 .raw files
Modification from original source (cited below) included:
- python 3.6 compatibility
- stripped off mi-instrument dependency to make this standalone

To be added:
- need a generic .raw filename parser
- restore logging function
- restore exception handler

Original source of parser code was from:
oceanobservatories/mi-instrument @https://github.com/oceanobservatories/mi-instrument
Author Ronald Ronquillo & Richard Han

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

m = re.match(r'(?P<prefix>\S*)-D(?P<date>\d{1,})-T(?P<time>\d{1,})', "OOI-D20180211-T164025")


# Reference time "seconds since 1900-01-01 00:00:00"
REF_TIME = date2num(dt(1900, 1, 1, 0, 0, 0))


# ---------- NEED A GENERIC FILENAME PARSER -------------
# Common EK60 *.raw filename format
# EK60_RAW_NAME_REGEX = r'(?P<Refdes>\S*)_*OOI-D(?P<Date>\d{8})-T(?P<Time>\d{6})\.raw'
# EK60_RAW_NAME_MATCHER = re.compile(EK60_RAW_NAME_REGEX)

# # Regex to extract the timestamp from the *.raw filename (path/to/OOI-DYYYYmmdd-THHMMSS.raw)
# FILE_NAME_REGEX = r'(?P<Refdes>\S*)_*OOI-D(?P<Date>\d{8})-T(?P<Time>\d{6})\.raw'
# FILE_NAME_MATCHER = re.compile(FILE_NAME_REGEX)

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

    # metadata['depth_bin_size'].append(sample_data['sound_velocity'][0] *
    #                                   sample_data['sample_interval'][0] / 2)         # [meters]

    return metadata


def load_ek60_raw(input_file_path):
    """
    Parse the *.raw file.
    @param input_file_path absolute path/name to file to be parsed
    """
    print('%s  unpacking file: %s' % (dt.now().strftime('%H:%M:%S'), input_file_path))

    with open(input_file_path, 'rb') as input_file:  # read ('r') input file using binary mode ('b')

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
        transmit_data_dict = defaultdict(list)   # transmit signal metadata
        sample_interval = []                     # sampling interval [sec]
        data_times = []                          # ping time

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
                            motion_data_dict[channel] = []

                        # Fill in metadata for each channel
                        for channel, sample_data in sample_temp_dict.items():
                            append_metadata(first_ping_metadata[channel], channel, sample_data)

                    # Save data and metadata from each ping to *_data_dict
                    sample_interval.append(next_sample['sample_interval'])
                    data_times.append(next_time)
                    for channel in power_temp_dict:
                        power_data_dict[channel].append(power_temp_dict[channel])
                        if any(angle_temp_dict[channel]):   # if split-beam data
                            angle_data_dict[channel].append(angle_temp_dict[channel])
                        motion = np.array([(sample_temp_dict[channel]['heave'],
                                            sample_temp_dict[channel]['pitch'],
                                            sample_temp_dict[channel]['roll'])],
                                          dtype=[('heave', 'f4'), ('pitch', 'f4'), ('roll', 'f4')])
                        motion_data_dict[channel].append(motion)
                        transmit = np.array([(sample_temp_dict[channel]['frequency'],
                                              sample_temp_dict[channel]['transmit_power'],
                                              sample_temp_dict[channel]['pulse_length'],
                                              sample_temp_dict[channel]['bandwidth'])],
                                            dtype=[('frequency', 'f4'), ('transmit_power', 'f4'),
                                                   ('pulse_length', 'f4'), ('bandwidth', 'f4')])
                        transmit_data_dict[channel].append(transmit)

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
        sample_interval = np.array(sample_interval).squeeze()
        for channel in power_data_dict:
            power_data_dict[channel] = np.array(power_data_dict[channel]) * 10. * np.log10(2) / 256.
            if angle_data_dict[channel]:  # if split-beam data
                angle_data_dict[channel] = np.array(angle_data_dict[channel])
            else:  # if single-beam data
                angle_data_dict[channel] = []
            motion_data_dict[channel] = np.array(motion_data_dict[channel])
            transmit_data_dict[channel] = np.array(transmit_data_dict[channel])

        return first_ping_metadata, data_times, sample_interval, \
               power_data_dict, angle_data_dict, motion_data_dict, transmit_data_dict, \
               config_header, config_transducer


def save_raw_to_nc(input_file_path):
    """
    Save data from RAW to netCDF format
    :param input_file_path:
    :return:
    """
    # Load data from RAW file
    first_ping_metadata, data_times, sample_interval, \
    power_data_dict, angle_data_dict, motion_data_dict, transmit_data_dict, \
    config_header, config_transducer = load_ek60_raw(input_file_path)

    # Get nc filename
    filename = os.path.splitext(os.path.basename(input_file_path))[0]
    nc_path = os.path.join(os.path.split(input_file_path)[0], filename+'.nc')
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
    ep.set_group_provenance(nc_path, os.path.basename(input_file_path), prov_dict)

    # Sonar group
    sonar_attrs = ('sonar_manufacturer', 'sonar_model', 'sonar_serial_number',
                   'sonar_software_name', 'sonar_software_version', 'sonar_type')
    sonar_vals = ('Simrad', 'EK60', '',
                  '', config_header['version'].decode('utf-8'), 'echosounder')
    sonar_dict = dict(zip(sonar_attrs, sonar_vals))
    ep.set_group_sonar(nc_path, sonar_dict)

    # Beam group
    beam_dict = dict()
    beam_dict['beam_mode'] = 'vertical'
    beam_dict['conversion_equation_t'] = 'type_3'  # type_3 is EK60 conversion
    beam_dict['ping_time'] = data_times            # here in matplotlib time
    beam_dict['backscatter_r'] = power_data_dict
    beam_dict['beamwidth_receive_major'] = np.array([x['beam_width_alongship'] for x in config_transducer.__iter__()])
    beam_dict['beamwidth_receive_minor'] = np.array([x['beam_width_athwartship'] for x in config_transducer.__iter__()])
    beam_dict['beamwidth_transmit_major'] = np.array([x['beam_width_alongship'] for x in config_transducer.__iter__()])
    beam_dict['beamwidth_transmit_minor'] = np.array([x['beam_width_athwartship'] for x in config_transducer.__iter__()])
    beam_dict['beam_direction_x'] = np.array([x['dir_x'] for x in config_transducer.__iter__()])
    beam_dict['beam_direction_y'] = np.array([x['dir_y'] for x in config_transducer.__iter__()])
    beam_dict['beam_direction_z'] = np.array([x['dir_z'] for x in config_transducer.__iter__()])
    beam_dict['equivalent_beam_angle'] = np.array([x['equiv_beam_angle'] for x in config_transducer.__iter__()])
    beam_dict['gain_correction'] = np.array([x['gain'] for x in config_transducer.__iter__()])
    beam_dict['non_quantitative_processing'] = 0
    beam_dict['sample_interval'] = sample_interval   # dimension ping_time
    beam_dict['sample_time_offset'] = 2              # set to 2 for EK60 data, NOT from sample_data['offset']
    beam_dict['transmit_duration_nominal'] = np.array([x['pulse_length']
                                                       for x in transmit_data_dict.values()]).squeeze()
    beam_dict['transmit_power'] = np.array([x['transmit_power']
                                            for x in transmit_data_dict.values()]).squeeze()
    beam_dict['transmit_bandwidth'] = np.array([x['bandwidth']
                                                for x in transmit_data_dict.values()]).squeeze()

# def raw2hdf5_initiate(raw_file_path,h5_file_path):
#     """
#     Unpack EK60 .raw files and save to an hdf5 files
#     INPUT:
#         fname      file to be unpacked
#         h5_fname   hdf5 file to be written in to
#     """
#     # Unpack raw into memory
#     first_ping_metadata, data_times, power_data_dict, frequencies, bin_size, \
#         config_header, config_transducer = load_ek60_raw(raw_file_path)
#
#     # Check if input dimension makes sense, if not abort
#     sz_power_data = np.empty(shape=(len(frequencies),2),dtype=int)
#     for cnt,f in zip(range(len(frequencies)),frequencies.keys()):
#         f_str = str(frequencies[f])
#         sz_power_data[cnt,:] = power_data_dict[f_str].shape
#     if np.unique(sz_power_data).shape[0]!=2:
#         print('Raw file has mismatched number of pings across channels')
#         # break
#
#     # Open new hdf5 file
#     h5_file = h5py.File(h5_file_path,'x')  # create file, fail if exists
#
#     # Store data
#     # -- ping time: resizable
#     h5_file.create_dataset('ping_time', (sz_power_data[0,1],), \
#                     maxshape=(None,), data=data_times, chunks=True)
#
#     # -- power data: resizable
#     for f in frequencies.values():
#         h5_file.create_dataset('power_data/%s' % str(f), sz_power_data[0,:], \
#                     maxshape=(sz_power_data[0,0],None), data=power_data_dict[str(f)], chunks=True)
#
#     # -- metadata: fixed sized
#     h5_file.create_dataset('metadata/bin_size', data=bin_size)
#     for m,mval in first_ping_metadata.items():
#         save_metadata(mval,'metadata',m,h5_file)
#
#     # -- header: fixed sized
#     for m,mval in config_header.items():
#         save_metadata(mval,'header',m,h5_file)
#
#     # -- transducer: fixed sized
#     for tx in range(len(config_transducer)):
#         for m,mval in config_transducer[tx].items():
#             save_metadata(mval,['transducer',tx],m,h5_file)
#
#     # Close hdf5 file
#     h5_file.close()
#
#
# def raw2hdf5_concat(raw_file_path,h5_file_path):
#     """
#     Unpack EK60 .raw files and concatenate to an existing hdf5 files
#     INPUT:
#         fname      file to be unpacked
#         h5_fname   hdf5 file to be concatenated to
#     """
#     # Unpack raw into memory
#     first_ping_metadata, data_times, power_data_dict, frequencies, bin_size, \
#         config_header, config_transducer = load_ek60_raw(raw_file_path)
#
#     # Check if input dimension makes sense, if not abort
#     sz_power_data = np.empty(shape=(len(frequencies),2),dtype=int)
#     for cnt,f in zip(range(len(frequencies)),frequencies.keys()):
#         f_str = str(frequencies[f])
#         sz_power_data[cnt,:] = power_data_dict[f_str].shape
#     if np.unique(sz_power_data).shape[0]!=2:
#         print('Raw file has mismatched number of pings across channels')
#         # break
#
#     # Open existing files
#     fh = h5py.File(h5_file_path, 'r+')
#
#     # Check if all metadata field matches, if not, print info and abort
#     flag = check_metadata('header',config_header,fh) and \
#            check_metadata('metadata',first_ping_metadata,fh) and \
#            check_metadata('transducer00',config_transducer[0],fh) and \
#            check_metadata('transducer01',config_transducer[1],fh) and \
#            check_metadata('transducer02',config_transducer[2],fh)
#
#     # Concatenating newly unpacked data into HDF5 file
#     for f in fh['power_data'].keys():
#         sz_exist = fh['power_data/'+f].shape  # shape of existing power_data mtx
#         fh['power_data/'+f].resize((sz_exist[0],sz_exist[1]+sz_power_data[0,1]))
#         fh['power_data/'+f][:,sz_exist[1]:] = power_data_dict[str(f)]
#     fh['ping_time'].resize((sz_exist[1]+sz_power_data[0,1],))
#     fh['ping_time'][sz_exist[1]:] = data_times
#
#     # Close file
#     fh.close()
#
#
# def check_metadata(group_name,dict_name,fh):
#     """
#     Check if all metadata matches
#
#     group_name   name of group in hdf5 file
#     dict_name    name of dictionary from unpacked .raw file
#     """
#     flag = []
#     for p in fh[group_name].keys():
#         if isinstance(fh[group_name][p][0], (str, bytes)):
#             if type(dict_name[p]) == bytes:
#                 flag.append(str(dict_name[p], 'utf-8') == fh[group_name][p][0])
#             else:
#                 flag.append(dict_name[p] == fh[group_name][p][0])
#         elif isinstance(fh[group_name][p][0], (np.generic, np.ndarray, int, float)):
#             flag.append(any(dict_name[p] == fh[group_name][p][:]))
#     return any(flag)
#
#
# def save_metadata(val,group_info,data_name,fh):
#     """
#     Check data type and save to hdf5.
#
#     val          data to be saved
#     group_info   a string (group name, e.g., header) or
#                  a list (group name and sequence number, e.g., [tranducer, 1]).
#     data_name    name of data set under group
#     fh           handle of the file to be saved to
#     """
#     # Assemble group and data set name to save to
#     if type(group_info) == str:  # no sequence in group_info
#         create_name = '%s/%s' % (group_info,data_name)
#     elif type(group_info) == list and len(group_info) == 2:  # have sequence in group_info
#         if type(group_info[1]) == str:
#             create_name = '%s/%s/%s' % (group_info[0], group_info[1], data_name)
#         else:
#             create_name = '%s%02d/%s' % (group_info[0], group_info[1], data_name)
#     # Save val
#     if type(val) == str or type(val) == bytes:    # when a string
#         fh.create_dataset(create_name, (1,), data=val, dtype=h5py.special_dtype(vlen=str))
#     elif type(val) == int or type(val) == float:  # when only 1 int or float object
#         fh.create_dataset(create_name, (1,), data=val)
#     elif isinstance(val, (np.generic, np.ndarray)):
#         if val.shape == ():   # when single element numpy array
#             fh.create_dataset(create_name, (1,), data=val)
#         else:               # when multi-element numpy array
#             fh.create_dataset(create_name, data=val)
#     else:  # everything else
#         fh.create_dataset(create_name, data=val)
