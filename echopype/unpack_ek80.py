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
from struct import unpack_from, unpack, calcsize
import numpy as np
import re
import h5py
from datetime import datetime as dt
from matplotlib.dates import date2num
# from base_def import BaseEnum


# Set contants for unpacking .raw files
BLOCK_SIZE = 1024*4             # Block size read in from binary file to search for token
LENGTH_SIZE = 4
DATAGRAM_HEADER_SIZE = 12
CONFIG_HEADER_SIZE = 516
CONFIG_TRANSDUCER_SIZE = 320

# set global regex expressions to find all sample, annotation and NMEA sentences
SAMPLE_REGEX = b'RAW\d{1}'
SAMPLE_MATCHER = re.compile(SAMPLE_REGEX, re.DOTALL)

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
                            # DatagramHeader
                            ('datagram_type', 'a4'),  # 4 byte string
                            ('low_date_time', 'u4'),  # 4 byte int (long)
                            ('high_date_time', 'u4'),  # 4 byte int (long)
                            # SampleDatagram
                            ('channel_number', 'i2'),  # 2 byte int (short)
                            ('mode', 'i2'),  # 2 byte int (short)
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
                            ('count', 'i4')])                     # 4 byte int (long)
sample_dtype = sample_dtype.newbyteorder('<')

power_dtype = np.dtype([('power_data', '<i2')])     # 2 byte int (short)

angle_dtype = np.dtype([('athwart', '<i1'), ('along', '<i1')])     # 1 byte ints


# # # # Just doing my own code here for now  # # # # #

# data link: https://drive.google.com/drive/folders/1RWsejJlvi7oyvje7S69HLMntUvCmH4K2?usp=sharing

import xml.etree.ElementTree as ET

headerlength = 12
fc=0 # filter count
nm=0 #nmea counter
nmea_data = [] # preallocate nmea array

with open('sample_EK80_Data 2/Amundsen_DFO-LabSea_Station5-DFO5-Phase0-D20180802-T010015-0.raw') as f:
    while True:

        dglength = np.fromfile(f, np.int32,1)

        # break if you are at end of file
        if dglength.size == 0:
            break

        dglength = dglength[0]
        header = np.fromfile(f, np.int8, 4)
        header = ''.join([chr(item) for item in header])
        print(header)
        dtime = np.fromfile(f, np.int32,2)
        

        
        # read XML headers
        if header=='XML0':
            chars = np.fromfile(f, np.int8, dglength-headerlength)
            chars = ''.join([chr(item) for item in chars])
            xml = ET.fromstring(chars)
            
        # read instrument files    
        if header=='FIL1':
            stage = np.fromfile(f, np.int16, 1)
            channel = np.fromfile(f, np.int16, 1)
            channelID = np.fromfile(f, np.int8, 128)
            channelID = ''.join([chr(item) for item in channelID])
            ncoeff = np.fromfile(f, np.int16, 1)[0]
            decim = np.fromfile(f, np.int16, 1)
            
            coeff = np.fromfile(f, np.int32, 2*ncoeff)
            
            real_coeff = coeff[::2]
            imag_coeff = coeff[1::2]
            
            fc = fc+1
            
        if header=='MRU0':
            heave = np.fromfile(f, np.int32, 1)[0]
            roll = np.fromfile(f, np.int32, 1)[0]
            pitch = np.fromfile(f, np.int32, 1)[0]
            heading = np.fromfile(f, np.int32,1)[0]
            
        if header=='RAW3':
            channelID = np.fromfile(f, np.int8, 128)
            channelID = ''.join([chr(item) for item in channelID])
            datatype = np.fromfile(f, np.int16, 1)
            spare = np.fromfile(f, np.int8, 2)
            spare = ''.join([chr(item) for item in spare])
            offset = np.fromfile(f, np.int32, 1)[0]
            count  = np.fromfile(f, np.int32, 1)[0]
            
            if (datatype & (1 << 0)) != 0: #equivalent to Matlab's "bitget"
                power = np.fromfile(f, np.int16, count)
                power = power*10*np.log10(2)/256
                if (datatype & (1 << 1)) != 0:
                    angle = np.fromfile(f, np.int8, 2*count)
                    angle = angle[::2] + 256*angle[1::2]
                    #? alongship 
                    #? athwartship
                    
            if (datatype & (1 << 3)) != 0:
                ncomplex = np.right_shift(datatype, 8, dtype = np.int16)[0]
                samples = np.fromfile(f, np.int32, 2*ncomplex*count)
                samples = samples.astype('int')
                # some shape conversions?
                
            if  ((datatype & (1 << 0)) == 0) & ((datatype & (1 << 3)) == 0):
                print('Unknown sample mode')
                
                
             # # # We need to do some configuring of raw data here # #
             # # But first, finish reading environment header (XML0 above) # #
            
        if header=='NME0':
            nmea = np.fromfile(f, np.int8, dglength-headerlength)
            nmea = ''.join([chr(item) for item in nmea])
            timestamp = dtime
            nmea_data.append(nmea_data, nmea)
            nm = nm+1
            
        end = np.fromfile(f, np.int32,1) #read value again at end of section

# # # # # # # # # # # # # #


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
    print('Chunk len:' + str(len(chunk)))
    print('Format size: ' + str(calcsize(fmt)))
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
    dglength = filehandle.read(1)
    print(dglength)
    dglength = unpack('>H',dglength)

    # Read the configuration datagram, output at the beginning of the file
    header = filehandle.read(12)
    bt = unpack_from('<l',header)
    print(bt, header, dglength)
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
    motion_data = np.array([(sample_data['heave'][0], sample_data['roll'][0], sample_data['pitch'][0])],
                           dtype=[('heave', 'f4'), ('pitch', 'f4'), ('roll', 'f4')])

    # Check for a valid channel number that is within the number of transducers config
    # to prevent incorrectly indexing into the dictionaries.
    # An out of bounds channel number can indicate invalid, corrupt,
    # or misaligned datagram or a reverse byte order binary data file.
    # Log warning and continue to try and process the rest of the file.
    if channel < 0 or channel > transducer_count:
        print('Invalid channel: %s for transducer count: %s. \n\
        Possible file corruption or format incompatibility.' % (channel, transducer_count))
        # log.warn("Invalid channel: %s for transducer count: %s."
        #          "Possible file corruption or format incompatibility.", channel, transducer_count)
        # raise InvalidTransducer

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
        # log.warn("Mismatching beginning and end length values in sample datagram: length1"
        #          ": %s, length2: %s. Possible file corruption or format incompatibility.",
        #          sample_data['length1'][0], length2_data['length2'][0])

    return channel, ntp_time, sample_data, power_data, angle_data, motion_data



def append_metadata(metadata, channel, sample_data):
    metadata['channel'].append(channel)
    metadata['transducer_depth'].append(sample_data['transducer_depth'][0])          # [meters]
    metadata['frequency'].append(sample_data['frequency'][0])                        # [Hz]
    metadata['transmit_power'].append(sample_data['transmit_power'][0])              # [Watts]
    metadata['pulse_length'].append(sample_data['pulse_length'][0])                  # [seconds]
    metadata['bandwidth'].append(sample_data['bandwidth'][0])                        # [Hz]
    metadata['sample_interval'].append(sample_data['sample_interval'][0])            # [seconds]
    metadata['sound_velocity'].append(sample_data['sound_velocity'][0])              # [m/s]
    metadata['absorption_coeff'].append(sample_data['absorption_coefficient'][0])    # [dB/m]
    metadata['temperature'].append(sample_data['temperature'][0])                    # [degC]
    metadata['depth_bin_size'].append(sample_data['sound_velocity'][0] * sample_data['sample_interval'][0] / 2)   # [meters]
    return metadata



def load_ek80_raw(input_file_path):
    """
    Parse the *.raw file.
    @param input_file_path absolute path/name to file to be parsed
    # @param output_file_path optional path to directory to write output
    If omitted outputs are written to path of input file
    """
    print('%s  unpacking file: %s' % (dt.now().strftime('%H:%M:%S'), input_file_path))

    with open(input_file_path, 'rb') as input_file:  # read ('r') input file using binary mode ('b')

        config_header, config_transducer = read_header(input_file)
        transducer_count = config_header['transducer_count']

        position = input_file.tell()

        last_time = None
        sample_data_temp_dict = defaultdict(list)
        power_data_temp_dict = defaultdict(list)
        angle_temp_dict = defaultdict(list)    # include alongship and athwardship angles
        motion_temp_dict = defaultdict(list)   # include heave, pitch, and roll

        # Initialize output structure
        first_ping_metadata = defaultdict(list)  # metadata for each channel
        power_data_dict = defaultdict(list)      # echo power
        angle_data_dict = defaultdict(list)      # contain alongship and athwartship electronic angle
        motion_data_dict = defaultdict(list)     # contain heave, pitch, and roll motion
        data_times = []
        temperature = []   # WJ: Used to check temperature reading in .RAW file --> all identical for OOI data

        # Read binary file a block at a time
        raw = input_file.read(BLOCK_SIZE)

        while len(raw) > 4:
            # We only care for the Sample datagrams, skip over all the other datagrams
            match = SAMPLE_MATCHER.search(raw)

            if match:
                # Offset by size of length value
                match_start = match.start() - LENGTH_SIZE

                # Seek to the position of the length data before the token to read into numpy array
                input_file.seek(position + match_start)

                # try:
                next_channel, next_time, next_sample, next_power, next_angle, next_motion = process_sample(input_file, transducer_count)

                if next_time != last_time:  # WJ: next_time=last_time when it's the same ping but different channel
                    # Clear out our temporary dictionaries and set the last time to this time
                    sample_data_temp_dict = defaultdict(list)
                    power_data_temp_dict = defaultdict(list)
                    angle_temp_dict = defaultdict(list)    # include both alongship and athwartship angle
                    motion_temp_dict = defaultdict(list)   # include heave, pitch, and roll
                    last_time = next_time

                # Store this data
                sample_data_temp_dict[next_channel] = next_sample
                power_data_temp_dict[next_channel] = next_power
                angle_temp_dict[next_channel] = next_angle
                motion_temp_dict[next_channel] = next_motion

                # Check if we have enough records to produce a new row of data
                # WJ: if yes this means that data from all transducer channels have been read for a particular ping
                # WJ: a new row of data means all channels of data from one ping
                # WJ: if only 2 channels of data were received but there are a total of 3 transducers,
                # WJ: the data are not stored in the final power_data_dict
                if len(sample_data_temp_dict) == len(power_data_temp_dict) == \
                        len(angle_temp_dict) == len(motion_temp_dict) == transducer_count:
                    # if this is our first set of data from all channels,
                    # create our metadata particle and store the frequency / bin_size
                    if not power_data_dict:

                        # Initialize each channel to defaultdict
                        for channel in power_data_temp_dict:
                            first_ping_metadata[channel] = defaultdict(list)
                            angle_data_dict[channel] = defaultdict(list)
                            motion_data_dict[channel] = defaultdict(list)

                        # Fill in metadata for each channel
                        for channel, sample_data in sample_data_temp_dict.items():
                            append_metadata(first_ping_metadata[channel], channel, sample_data)

                    # Save the time and power data for plotting
                    data_times.append(next_time)
                    for channel in power_data_temp_dict:
                        power_data_dict[channel].append(power_data_temp_dict[channel])
                        if any(angle_temp_dict[channel]):   # if split-beam data
                            angle_data_dict[channel]['along'].append(angle_temp_dict[channel]['along'])
                            angle_data_dict[channel]['athwart'].append(angle_temp_dict[channel]['athwart'])
                        motion_data_dict[channel]['heave'].append(motion_temp_dict[channel]['heave'][0])
                        motion_data_dict[channel]['pitch'].append(motion_temp_dict[channel]['pitch'][0])
                        motion_data_dict[channel]['roll'].append(motion_temp_dict[channel]['roll'][0])

                    temperature.append(next_sample['temperature'])  # WJ: check temperature values from .RAW file: all identical for OOI data

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
        # And then transpose power data
        for channel in power_data_dict:
            power_data_dict[channel] = np.array(power_data_dict[channel]) * 10. * np.log10(2) / 256.
            power_data_dict[channel] = power_data_dict[channel].transpose()
            if angle_data_dict[channel]:    # if split-beam data
                angle_data_dict[channel]['along'] = np.array(angle_data_dict[channel]['along'])
                angle_data_dict[channel]['athwart'] = np.array(angle_data_dict[channel]['athwart'])
            else:                           # if single-beam data
                angle_data_dict[channel]['along'] = []
                angle_data_dict[channel]['athwart'] = []
            motion_data_dict[channel]['heave'] = np.array(motion_data_dict[channel]['heave'])
            motion_data_dict[channel]['pitch'] = np.array(motion_data_dict[channel]['pitch'])
            motion_data_dict[channel]['roll'] = np.array(motion_data_dict[channel]['roll'])

        return first_ping_metadata, data_times, power_data_dict, angle_data_dict, motion_data_dict, config_header, config_transducer
