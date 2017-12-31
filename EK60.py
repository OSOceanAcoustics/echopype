# coding=utf-8

#     National Oceanic and Atmospheric Administration
#     Alaskan Fisheries Science Center
#     Resource Assessment and Conservation Engineering
#     Midwater Assessment and Conservation Engineering

#  THIS SOFTWARE AND ITS DOCUMENTATION ARE CONSIDERED TO BE IN THE PUBLIC DOMAIN
#  AND THUS ARE AVAILABLE FOR UNRESTRICTED PUBLIC USE. THEY ARE FURNISHED "AS IS."
#  THE AUTHORS, THE UNITED STATES GOVERNMENT, ITS INSTRUMENTALITIES, OFFICERS,
#  EMPLOYEES, AND AGENTS MAKE NO WARRANTY, EXPRESS OR IMPLIED, AS TO THE USEFULNESS
#  OF THE SOFTWARE AND DOCUMENTATION FOR ANY PURPOSE. THEY ASSUME NO RESPONSIBILITY
#  (1) FOR THE USE OF THE SOFTWARE AND DOCUMENTATION; OR (2) TO PROVIDE TECHNICAL
#  SUPPORT TO USERS.

'''

                      CLASS DESCRIPTION GOES HERE

'''

import os
import sys
import datetime
from pytz import timezone
import logging
import numpy as np
#from .util.raw_file import RawSimradFile, SimradEOF
#from .util import unit_conversion
#from ..processing import ProcessedData
from util.raw_file import RawSimradFile, SimradEOF
from util import unit_conversion
from processing import ProcessedData

log = logging.getLogger(__name__)


class EK60(object):

    def __init__(self):

        #  define the EK60's properties - these are "read-only" properties and should not
        #  be changed directly by the user

        #  start_time and end_time will define the time span of the data within the EK60 class
        self.start_time = None
        self.end_time = None

        #  start_ping and end_ping will define the ping span of the data within the EK60 class
        self.start_ping = None
        self.end_ping = None

        #  n_pings stores the total number of pings read
        self.n_pings = 0

        #  a list of frequencies that have been read.
        self.frequencies = []

        #  a list of stings identifying the channel IDs that have been read
        self.channel_ids = []

        #  channel_id_map maps the channel number to channel ID when reading raw data sources
        self.channel_id_map = {}

        #  n_channels stores the total number of channels in the object
        self.n_channels = 0

        #  create a dictionary to store the RawData objects
        self.raw_data = {}

        #  create a dictionary to store the Calibration object
        self.calibration_data = {}

        #  create a dictionary to store the Calibration object
        self.nmea_data = {}

        #  Define the class's "private" properties. These should not be generally be directly
        #  manipulated by the user.

        #  specify if we should read files incrementally or all at once
        self.read_incremental = False

        #  define an internal state variable that is set when we initiate incremental reading
        self._is_reading = False

        #  set read_angles to true to store angle data
        self.read_angles = True

        #  set read_angles to true to store power data
        self.read_power = True

        #  specify the max sample count to read. This property can be used to limit the
        #  number of samples read (and memory used) when your data of interest is
        self.read_max_sample_count = None

        #  data_array_dims contains the dimensions of the sample and angle data arrays
        #  specified as [n_pings, n_samples]. Values of -1 specify that the arrays will
        #  be resized appropriately to contain all pings and all samples read from the
        #  data source. Settings values > 0 will create arrays that are fixed to that size.
        #  Any samples beyond the number specified as n_samples will be dropped. When a
        #  ping is added and the data arrays are full, the sample data will be rolled such
        #  that the oldest data is dropped and the new ping is added.
        self._data_array_dims = [-1, -1]

        #  store the time and ping range parameters
        self.read_start_time = None
        self.read_end_time = None
        self.read_start_ping = None
        self.read_end_ping = None

        #  read_frequencies can be set to a list of floats specifying the frequencies to
        #  read. An empty list will result in all frequencies being read.
        self.read_frequencies = []

        #  read_channel_ids can be set to a list of strings specifying the channel_ids of
        #  the channels to read. An empty list will result in all channels being read.
        self.read_channel_ids = []

        #  this is the internal per file channel map. This map is only valid for the file
        #  currently being read. Do not alter or use this property.
        self._channel_map = {}


    def read_raw(self, raw_files, power=None, angles=None, max_sample_count=None, start_time=None,
            end_time=None, start_ping=None, end_ping=None, frequencies=None, channel_ids=None,
            time_format_string='%Y-%m-%d %H:%M:%S', incremental=None):
        '''
        read_raw reads one or many Simrad EK60 ES60/70 .raw files
        '''

        #  update the reading state variables
        if (start_time):
            self.read_start_time = self._convert_time_bound(start_time, format_string=time_format_string)
        if (end_time):
            self.read_end_time = self._convert_time_bound(end_time, format_string=time_format_string)
        if (start_ping):
            self.read_start_ping = start_ping
        if (end_ping):
            self.read_end_ping = end_ping
        if (power):
            self.read_power = power
        if (angles):
            self.read_angles = angles
        if (max_sample_count):
            self.read_max_sample_count = max_sample_count
        if (frequencies):
            self.read_frequencies = frequencies
        if (channel_ids):
            self.read_channel_ids = channel_ids
        if (incremental):
            self.read_incremental = incremental

        #TODO:  Implement incremental reading.
        #       This is going to take some re-org since we can't simply iterate thru the list of files.
        #       We need to be able to read a subset of data from a file and if required open the next
        #       file in the list and continue reading the subset.

        #  ensure that the raw_files argument is a list
        if isinstance(raw_files, str):
            raw_files = [raw_files]

        #  initialize a file counter
        n_files = 0

        #  iterate thru our list of .raw files to read
        for filename in raw_files:

            #  Read data from file and add to self.raw_data.
            with RawSimradFile(filename, 'r') as fid:

                #  read the configuration datagrams - the CON0 datagram will come first
                #  and if this is an ME70 .raw file the CON1 datagram will follow.

                #  read the CON0 configuration datagram
                config_datagram = fid.read(1)
                if (n_files == 0):
                    self.start_time = config_datagram['timestamp']

                #  check if we're reading an ME70 file with a CON1 datagram
                next_datagram = fid.peek()
                if (next_datagram == 'CON1'):
                    #  the next datagram is CON1 - read it
                    CON1_datagram = fid.read(1)
                else:
                    #  next datagram was something else, move along
                    CON1_datagram = None

                #  check if we need to create an RawData object for this channel
                self._channel_map = {}
                for channel in config_datagram['transceivers']:
                    #  get the channel ID
                    channel_id = config_datagram['transceivers'][channel]['channel_id']

                    #  check if we are reading this channel
                    if ((self.read_channel_ids) and (channel_id not in self.read_channel_ids)):
                        #  there are specific channel IDs specified and this is *NOT* one of them
                        #  so we just move along...
                        continue

                    #  check if we're reading this frequency
                    frequency = config_datagram['transceivers'][channel]['frequency']
                    if ((self.read_frequencies) and (frequency not in self.read_frequencies)):
                        #  there are specific frequencies specified and this is *NOT* one of them
                        #  so we just move along...
                        continue

                    #  check if an RawData object exists for this channel
                    if channel_id not in self.raw_data:
                        #  no - create it
                        self.raw_data[channel_id] = RawData(channel_id, store_power=self.read_power,
                                store_angles=self.read_angles, max_sample_number=self.read_max_sample_count)

                        # add this channel to the calibration data
                        self.calibration_data[channel_id] = CalibrationParameters()

                        #  and add it to our list of channel_ids
                        self.channel_ids.append(channel_id)

                        #  update our public channel id map
                        self.n_channels += 1
                        self.channel_id_map[self.n_channels] = channel_id


                    # add calibration data from config datagram
                    self.calibration_data[channel_id].append_calibration(config_datagram['transceivers'][channel])

                    #  update the internal mapping of channel number to channel ID used when reading
                    #  the datagrams. This mapping is only valid for the current file that is being read.
                    self._channel_map[channel] = channel_id

                    #  create a ChannelMetadata object to store this channel's configuration and rawfile metadata.
                    channel_metadata = ChannelMetadata(filename,
                                               config_datagram['transceivers'][channel],
                                               config_datagram['survey_name'],
                                               config_datagram['transect_name'],
                                               config_datagram['sounder_name'],
                                               config_datagram['version'],
                                               self.raw_data[channel_id].n_pings,
                                               config_datagram['timestamp'],
                                               extended_configuration=CON1_datagram)

                    #  update the channel_metadata property of the RawData object
                    self.raw_data[channel_id].current_metadata = channel_metadata

                #  read the rest of the datagrams.
                self._read_datagrams(fid, self.read_incremental)

                #  increment the file read counter
                n_files += 1

        #  trim excess data from arrays after reading and compress calibration arrays
        for channel_id in self.channel_ids:
            self.raw_data[channel_id].trim()
            self.calibration_data[channel_id].compress_data_arrays()


    def _read_datagrams(self, fid, incremental):
        '''
        _read_datagrams is an internal method to read all of the datagrams contained in
        '''

        #TODO: implement incremental reading
        #      The user should be able to specify incremental reading in their call to read_raw.
        #      incremental reading should read in the specified number of "pings", save the reader
        #      state, then return. Subsequent calls would pick up where the reading left off.
        #      As stated above, the exact mechanics need to be worked out since it will not work
        #      as currently implemented.


        num_sample_datagrams = 0
        num_sample_datagrams_skipped = 0
        num_unknown_datagrams_skipped = 0
        num_datagrams_parsed = 0

        #  while datagrams are available
        while True:
            #  try to read in the next datagram
            try:
                new_datagram = fid.read(1)
            except SimradEOF:
                #  nothing more to read
                break

            #  check if we should store this data based on time bounds
            if self.read_start_time is not None:
                if new_datagram['timestamp'] < self.read_start_time:
                    continue
            if self.read_end_time is not None:
                if new_datagram['timestamp'] > self.read_end_time:
                    continue

            #  increment the number parsed counter
            num_datagrams_parsed += 1

            #  update our end_time property
            self.end_time = new_datagram['timestamp']

            #  process the datagrams by type

            #  RAW datagrams store raw acoustic data for a channel
            if new_datagram['type'].startswith('RAW'):

                #  increment our ping counter
                if (new_datagram['channel'] == 1):
                    self.n_pings += 1

                #  check if we should store this data based on ping bounds
                if self.read_start_ping is not None:
                    if num_pings < self.read_start_ping:
                        continue
                if self.read_end_ping is not None:
                    if num_pings > self.read_end_ping:
                        continue

                #  check if we're supposed to store this channel
                if new_datagram['channel'] in self._channel_map:

                    #  set the first ping number we read
                    if (not self.start_ping):
                        self.start_ping = self.n_pings
                    #  and update the last ping number
                    self.end_ping = self.n_pings

                    #  get the channel id
                    channel_id = self._channel_map[new_datagram['channel']]

                    #  and call the appropriate channel's append_ping method
                    self.raw_data[channel_id].append_ping(new_datagram)

                    # append the calibration data
                    self.calibration_data[channel_id].append_calibration(new_datagram)

                    # increment the sample datagram counter
                    num_sample_datagrams += 1
                else:
                    num_sample_datagrams_skipped += 1

            #  NME datagrams store ancillary data as NMEA-0817 style ASCII data
            elif new_datagram['type'].startswith('NME'):
              timestamp = new_datagram['timestamp']
              nmea_type = new_datagram['nmea_type']
              nmea_string = new_datagram['nmea_string']

              if nmea_type not in self.nmea_data:
                self.nmea_data[nmea_type] = {}

              self.nmea_data[nmea_type][timestamp] = nmea_string

            #  TAG datagrams contain time-stamped annotations inserted via the recording software
            elif new_datagram['type'].startswith('TAG'):
                #TODO: Implement annotation reading
                pass
            else:
                #  unknown datagram type - issue a warning
                log.warning('Skipping unkown datagram type: %s @ %s', new_datagram['type'],
                        new_datagram['timestamp'])
                num_unknown_datagrams_skipped += 1

            if not (num_datagrams_parsed % 10000):
                log.debug('    Parsed %d datagrams (%d sample).', num_datagrams_parsed,
                        num_sample_datagrams)


        num_datagrams_skipped = num_unknown_datagrams_skipped + num_sample_datagrams_skipped
        log.info('  Read %d datagrams (%d skipped).', num_sample_datagrams, num_datagrams_skipped)


    def _convert_time_bound(self, time, format_string):
        '''
        internally all times are datetime objects converted to UTC timezone. This method
        converts arguments to comply.
        '''
        utc = timezone('utc')
        if (isinstance(time, str)):
            #  we have been passed a string, convert to datetime object
            time = datetime.datetime.strptime(time, format_string)

        #  convert to UTC and return
        return utc.localize(time)


    def get_rawdata(self, channel_number=1, channel_id=None):
        '''
        get_rawdata returns a reference to the specified RawData object for the
        specified channel id or channel number.
        '''

        if (channel_id):
            return self.raw_data.get(channel_id, None)
        else:
            #TODO: error handling
            return self.raw_data.get(self.channel_id_map[channel_number], None)


    def __str__(self):
        '''
        reimplemented string method that provides some basic info about the EK60
        '''

        #  print the class and address
        msg = str(self.__class__) + " at " + str(hex(id(self))) + "\n"

        #  print some more info about the EK60 instance
        if (self.channel_ids):
            n_channels = len(self.channel_ids)
            if (n_channels > 1):
                msg = msg + ("    EK60 object contains data from " + str(n_channels) + " channels:\n")
            else:
                msg = msg + ("    EK60 object contains data from 1 channel:\n")
            for channel in self.channel_id_map:
                msg = msg + ("        " + str(channel) + ":" + self.channel_id_map[channel] + "\n")
            msg = msg + ("    data start time: " + str(self.start_time)+ "\n")
            msg = msg + ("      data end time: " + str(self.end_time)+ "\n")
            msg = msg + ("    number of pings: " + str(self.end_ping - self.start_ping + 1)+ "\n")

        else:
            msg = msg + ("  EK60 object contains no data\n")

        return msg


class RawData(object):
    '''
    the RawData class contains a single channel's data extracted from a Simrad raw
    file. collected from an EK/ES60 or ES70. A RawData object is created for each
    unique channel in an EK/ES60 ES70 raw file.

    '''

    #  define some instrument specific constants

    #  Simrad recommends a TVG correction factor of 2 samples to compensate for receiver delay and TVG
    #  start time delay in EK60 and related hardware. Note that this correction factor is only applied
    #  when computing Sv/sv and not Sp/sp.
    TVG_CORRECTION = 2

    #  define constants used to specify the target resampling interval for the power and angle
    #  conversion functions. These values represent the standard sampling intervals for EK60 hardware
    #  when operated with the ER60 software as well as ES60/70 systems and the ME70.
    RESAMPLE_SHORTEST = 0
    RESAMPLE_16   = 0.000016
    RESAMPLE_32  = 0.000032
    RESAMPLE_64  = 0.000064
    RESAMPLE_128  = 0.000128
    RESAMPLE_256 = 0.000256
    RESAMPLE_512 = 0.000512
    RESAMPLE_1024 = 0.001024
    RESAMPLE_2048 = 0.002048
    RESAMPLE_LONGEST = 1


    def __init__(self, channel_id, n_pings=100, n_samples=1000, rolling=False,
            chunk_width=500, store_power=True, store_angles=True, max_sample_number=None):
        '''
        Creates a new, empty RawData object. The RawData class stores raw
        echosounder data from a single channel of an EK60 or ES60/70 system.

        NOTE: power is *always* stored in log form. If you manipulate power values
              directly, make sure they are stored in log form.

        if rolling is True, arrays of size (n_pings, n_samples) are created for power
        and angle data upon instantiation and are filled with NaNs. These arrays are
        fixed in size and if a ping is added beyond the "width" of the array the
        array is "rolled left", and the new ping is added at the end of the array. This
        feature is intended to support streaming data sources such as telegram
        broadcasts and the client/server interface.

        chunk_width specifies the number of columns to add to data arrays when they
        fill up when rolling == False.

        '''

        #  we can come up with a better name, but this specifies if we have a fixed data
        #  array size and roll it when it fills or if we expand the array when it fills
        self.rolling_array = bool(rolling)

        #  current_metadata stores a reference to the current ChannelMetadata object. The
        #  ChannelMetadata class stores rawfile and channel configuration properties
        #  contained in the .raw file header. When opening a new .raw file, this property
        #  must be updated before appending pings from the new file.
        self.current_metadata = None

        #  the channel ID is the unique identifier of the channel(s) stored in the object
        self.channel_id = [channel_id]

        #  a counter incremented when a ping is added - a value of -1 indicates that the
        #  data arrays have not been allocated yet.
        self.n_pings = -1

        #  specify the horizontal size (columns) of the array allocation size.
        self.chunk_width = chunk_width

        #  keep note if we should store the power and angle data
        self.store_power = store_power
        self.store_angles = store_angles

        #  max_sample_number can be set to an integer specifying the maximum number of samples
        #  that will be stored in the sample data arrays.
        self.max_sample_number = max_sample_number

        #  allow the user to specify a dtype for the raw sample data.
        self.sample_dtype = 'float32'

        #  _data_attributes is an internal list that contains the names of all of the class's
        #  "data" properties. This is used internally with gettattr/setattr to remove some
        #  code verbosity when interacting with this properties.
        self._data_attributes = ['ping_time',
                                 'channel_metadata',
                                 'ping_number',
                                 'transducer_depth',
                                 'frequency',
                                 'transmit_power',
                                 'pulse_length',
                                 'bandwidth',
                                 'sample_interval',
                                 'sound_velocity',
                                 'absorption_coefficient',
                                 'heave',
                                 'pitch',
                                 'roll',
                                 'temperature',
                                 'heading',
                                 'transmit_mode',
                                 'sample_offset',
                                 'sample_count',
                                 'power',
                                 'angles_alongship_e',
                                 'angles_athwartship_e']

        #  pulse length values are in seconds

        #  if we're using a fixed data array size, we can allocate the arrays now
        if (self.rolling_array):
            #  since we assume rolling arrays will be used in a visual or interactive
            #  application, we initialize the arrays so they can be displayed
            self._create_arrays(n_pings, n_samples, initialize=True)

            #  initialize the ping counter to indicate that our data arrays have been allocated
            self.n_pings = 0

        #  if we're not using fixed arrays, we will initialze them when append_ping is
        #  called for the first time. Until then, the RawData object will not contain
        #  the data properties.

        #  create a logger instance
        self.logger = logging.getLogger('RawData')


    def append_ping(self, sample_datagram):
        '''
        append_ping is called when adding a ping's worth of data to the object. It should accept
        the parsed values from the sample datagram. It will handle the details of managing
        the array sizes, resizing as needed (or rolling in the case of a fixed size). Append ping also
        updates the RawFileData object's end_ping and end_time values for the current file.

        Managing the data array sizes is the bulk of what this method does. It will either resize
        the array is rolling == false or roll the array if it is full and rolling == true.

        The data arrays will change size in 2 ways:

            Adding pings will add columns (or roll the array if all of the columns are filled and
            rolling == true.) This can easily be handled by allocating columns in chunks using
            the resize method of the numpy array and maintaining an index into
            the *next* available column (self.n_pings). Empty pings can be left uninitialized (if
            that is possible with resize) or set to NaN if it is free. If it takes additional steps to
            set to NaN, then just leave them at the default value.

            Changing the recording range or pulse length will either require adding rows (if there
            are more samples) or padding (if there are fewer. If rows are added to the array,
            existing pings will need to be padded with NaNs.

        If rolling == true, we will never resize the array. If a ping has more samples than the
        array has allocated the extra samples will be dropped. In all cases if a ping has fewer
        samples than the array has allocated it should be padded with NaNs.

        '''

        #  if using dynamic arrays, handle intialization of data arrays when the first ping is added
        if (self.n_pings == -1 and self.rolling_array == False):
            #  create the initial data arrays
            self._create_arrays(self.chunk_width, len(sample_datagram['power']))

            #  initialize the ping counter to indicate that our data arrays have been allocated
            self.n_pings  = 0

        #  determine the greatest number of existing samples and the greatest number of
        #  samples in this datagram. In theory the power and angle arrays should always be
        #  the same size but we'll check all to make sure.
        max_data_samples = max(self.power.shape[1],self.angles_alongship_e.shape[1],
                self.angles_athwartship_e.shape[1])
        max_new_samples = max(sample_datagram['angle'].shape[0], sample_datagram['power'].shape[0])

        #  check if we need to truncate the sample data
        if (self.max_sample_number) and (max_new_samples > self.max_sample_number):
            max_new_samples = self.max_sample_number
            sample_datagram['angle'] = sample_datagram['angle'][0:self.max_sample_number]
            sample_datagram['power'] = sample_datagram['power'][0:self.max_sample_number]

        #  create 2 variables to store our current array size
        ping_dims = self.ping_number.size
        sample_dims = max_data_samples

        #  check if we need to re-size or roll our data arrays
        if (self.rolling_array == False):
            #  check if we need to resize our data arrays
            ping_resize = False
            sample_resize = False

            #  check the ping dimension
            if (self.n_pings == ping_dims):
                #  need to resize the ping dimension
                ping_resize = True
                #  calculate the new ping dimension
                ping_dims = ping_dims + self.chunk_width

            #  check the samples dimension
            if (max_new_samples > max_data_samples):
                #  need to resize the samples dimension
                sample_resize = True
                #  calculate the new samples dimension
                sample_dims = max_new_samples

            #  determine if we resize
            if (ping_resize or sample_resize):
                #  resize the data arrays
                self._resize_arrays(ping_dims, sample_dims, self.ping_number.size,
                        max_data_samples)

            #  get an index into the data arrays for this ping and increment our ping counter
            this_ping = self.n_pings
            self.n_pings += 1

        else:
            #  check if we need to roll
            if (self.n_pings == ping_dims - 1):
                #  when a rolling array is "filled" we stop incrementing the ping counter
                #  and repeatedly append pings to the last ping index in the array
                this_ping = self.n_pings

                #  roll our array 1 ping
                self._roll_arrays(1)

        #  append the ChannelMetadata object reference for this ping
        self.channel_metadata.append(self.current_metadata)

        #  update the ChannelMetadata object with this ping number and time
        self.current_metadata.end_ping = self.n_pings
        self.current_metadata.end_time = sample_datagram['timestamp']

        #  append the datetime object representing the ping time to the ping_time property
        #self.ping_time.append(sample_datagram['timestamp'])

        #  now insert the data into our numpy arrays
        self.ping_time[this_ping] = sample_datagram['timestamp']
        self.ping_number[this_ping] = self.n_pings
        self.transducer_depth[this_ping] = sample_datagram['transducer_depth']
        self.frequency[this_ping] = sample_datagram['frequency']
        self.transmit_power[this_ping] = sample_datagram['transmit_power']
        self.pulse_length[this_ping] = sample_datagram['pulse_length']
        self.bandwidth[this_ping] = sample_datagram['bandwidth']
        self.sample_interval[this_ping] = sample_datagram['sample_interval']
        self.sound_velocity[this_ping] = sample_datagram['sound_velocity']
        self.absorption_coefficient[this_ping] = sample_datagram['absorption_coefficient']
        self.heave[this_ping] = sample_datagram['heave']
        self.pitch[this_ping] = sample_datagram['pitch']
        self.roll[this_ping] = sample_datagram['roll']
        self.temperature[this_ping] = sample_datagram['temperature']
        self.heading[this_ping] = sample_datagram['heading']
        self.transmit_mode[this_ping] = sample_datagram['transmit_mode']

        #TODO: implement storing only a subset of the sample data
        #      sample_offset marks the start of the vertical offset (offset from sample 1)
        #      and sample count marks the number of samples stored. offset + count
        #      would be the bottom sample.
        self.sample_count[this_ping] = sample_datagram['count']
        self.sample_offset[this_ping] = 0

        #  now store the 2d "sample" data
        #      determine what we need to store based on operational mode
        #      1 = Power only, 2 = Angle only 3 = Power & Angle

        #  check if we need to store power data
        if (sample_datagram['mode'] != 2) and (self.store_power):
            #  check if we need to pad or trim our sample data
            sample_pad = max_data_samples - sample_datagram['power'].shape[0]
            if (sample_pad > 0):
                #  the data array has more samples than this datagram - we need to pad the datagram
                self.power[this_ping,:] = np.pad(sample_datagram['power'],(0,sample_pad),
                        'constant', constant_values=np.nan)
            elif (sample_pad < 0):
                #  the data array has fewer samples than this datagram - we need to trim the datagram
                self.power[this_ping,:] = sample_datagram['power'][0:sample_pad]
            else:
                #  the array has the same number of samples
                self.power[this_ping,:] = sample_datagram['power']
        #  check if we need to store angle data
        if (sample_datagram['mode'] != 1) and (self.store_angles):
            #  first extract the alongship and athwartship angle data
            #  the low 8 bits are the athwartship values and the upper 8 bits are alongship.
            alongship_e = (sample_datagram['angle'] >> 8).astype('int8')
            athwartship_e = (sample_datagram['angle'] & 0xFF).astype('int8')

            #  check if we need to pad or trim our sample data
            sample_pad = max_data_samples - athwartship_e.shape[0]
            if (sample_pad > 0):
                #  the data array has more samples than this datagram - we need to pad the datagram
                self.angles_alongship_e[this_ping,:] = np.pad(alongship_e,(0,sample_pad),
                        'constant', constant_values=np.nan)
                self.angles_athwartship_e[this_ping,:] = np.pad(athwartship_e,(0,sample_pad),
                        'constant', constant_values=np.nan)
            elif (sample_pad < 0):
                #  the data array has fewer samples than this datagram - we need to trim the datagram
                self.angles_alongship_e[this_ping,:] = alongship_e[0:sample_pad]
                self.angles_athwartship_e[this_ping,:] = athwartship_e[0:sample_pad]
            else:
                #  the array has the same number of samples
                self.angles_alongship_e[this_ping,:] = alongship_e
                self.angles_athwartship_e[this_ping,:] = athwartship_e


    def delete_pings(self, remove=True, **kwargs):
        '''
        delete_pings deletes ping data defined by the start and end bounds.

        If remove == True, the arrays are shrunk. If remove == False, the data
        defined by the start and end are set to NaN
        '''

        #  get the horizontal start and end indicies
        h_index = self.get_indices(**kwargs)


    def get_data(self):
      for attr in vars(self):
        data = getattr(self, attr)
        if attr in self._data_attributes:
          yield (attr, data)


    def append(self, rawdata_object):
        '''
        append appends another RawData object to this one.
        '''

        #  append simply inserts at the end of our internal array.
        self.insert(rawdata_object, ping_number=self.ping_number[-1])


    def insert(self, raw_data_obj, ping_number=None, ping_time=None, insert_after=True):
        '''
        insert inserts the data from the provided RawData object into this RawData
        object. The insertion point is specified by ping number or time. After inserting
        data, the ping_number property is updated and the ping numbers from the insertion
        point onward will be re-numbered accordingly.

        Note that internally RawData objects always store data by ping number as read,
        appended, or inserted. Data is not sorted by time until the user calls a get_*
        method. Consequently if ping times are not ordered and/or repeating, specifying
        a ping time as an insetion point may not yield the desired results since the
        insertion index will be the *first* index that satisfies the condition.

        By default, the insert is *after* the provided ping number or time. Set the
        insert_after keyword to False to insert *before* the provided ping number or time.

        '''

        #  check that we have been given an insetion point
        if ping_number is None and ping_time is None:
            raise ValueError('Either ping_number or ping_time needs to be defined to specify an insertion point.')

        #  make sure that raw_data_obj is an EK60.RawData raw data object
        if (not isinstance(raw_data_obj, RawData)):
            raise TypeError('The object you are inserting/appending must be an instance of the EK60.RawData class')

        #  make sure that the frequencies match - we don't allow insrting/appending of different frequencies
        if (self.frequency[0] != raw_data_obj.frequency[0]):
            raise TypeError('The frequency of the RawData object you are inserting/appending does not match the ' +
                    'frequency of this object. Frequencies must match to append or insert.')

        #  determine the index of the insertion point
        idx = self.get_ping_indices(start_time=ping_time, end_time=ping_time,
                start_ping=ping_number, end_ping=ping_number)[0]

        #  check if we're inserting before or after the provided insert point and adjust as necessary
        if (insert_after):
            #  we're inserting *after* - increment the index by 1
            idx += 1

        #  get some info about the shape of the data we're inserting
        my_pings = self.ping_number.shape[0]
        new_pings = raw_data_obj.ping_number.shape[0]
        my_samples = self.power.shape[1]
        new_samples = raw_data_obj.power.shape[1]

        #  check if we need to vertically resize one of the arrays - we resize the smaller to
        #  the size of the larger array. It will automatically be padded with NaNs
        if (my_samples < new_samples):
            #  resize our data arrays
            #  check if the new array exceeds our max_sample_count
            if ((self.max_sample_number) and (new_samples > self.max_sample_number)):
                #  it does - we have to change our new_samples
                new_samples = self.max_sample_number
                #  and trim the array we're appending
                raw_data_obj._resize_arrays(new_pings, my_samples, new_pings, new_samples)
            #  and resize our arrays
            self._resize_arrays(my_pings, new_samples, my_pings, my_samples)
        elif (my_samples > new_samples):
            #  resize arrays of raw_data_obj
            raw_data_obj._resize_arrays(new_pings, my_samples, new_pings, new_samples)

        #  work thru our data properties inserting the new data
        for attribute, data in self.get_data():
            try:
                data_to_insert = getattr(raw_data_obj, attribute)
            except Exception as err:
                log.error('Error reading data from raw_data_obj, ', raw_data_obj, attribute, ': ',  type(err), err)
                return

            #  handle lists and numpy arrays appropriately
            if isinstance(data, list):
                #  this attribute is a list - create the new list
                new_data = data[0:idx] + data_to_insert + data[idx:]
                #  and update our property
                setattr(self, attribute, new_data)

            elif isinstance(data, np.ndarray):
                #  this attribute is a numpy array

                #  if we're not storing a data type, skip it
                if ((attribute == 'power') and (not self.store_power)):
                    continue
                elif ((attribute == 'angles_alongship_e') and (not self.store_angles)):
                    continue
                elif ((attribute == 'angles_athwartship_e') and (not self.store_angles)):
                    continue

                #  we have to handle the 2d and 1d differently
                if (data.ndim == 1):
                    #  concatenate the 1d data
                    if (attribute == 'ping_number'):
                        #  update ping_number so it is sequential
                        new_data = np.arange(data.shape[0]+data_to_insert.shape[0]) + 1
                    else:
                        new_data = np.concatenate((data[0:idx], data_to_insert, data[idx:]))
                else:
                    #  concatenate the 2d data
                    new_data = np.vstack((data[0:idx,:], data_to_insert, data[idx:,:]))

                #  update this attribute
                setattr(self, attribute, new_data)

        #  now update our global properties
        if (raw_data_obj.channel_id not in self.channel_id):
            self.channel_id += raw_data_obj.channel_id
        self.n_pings = self.ping_number.shape[0]


    def trim(self):
        '''
        trim deletes the empty portions of pre-allocated arrays. This should be called
        when you are done adding pings to a non-rolling raw_data instance.
        '''

        #  determine the number of samples we're storing
        n_samples = self.power.shape[1]

        #  calling _resize_arrays with
        self._resize_arrays(self.n_pings, n_samples, 0, n_samples)


    def get_ping_indices(self, start_ping=None, end_ping=None, start_time=None,
            end_time=None, time_order=True):
        '''
        get_ping_indices returns index arrays containing the indices contained in the range
        defined by the times and/or ping numbers provided. By default the indexes are in time
        order. If time_order is set to False, the data will be returned in the order they
        occur in the data arrays.
        '''

        #  if starts and/or ends are omitted, assume fist and last respectively
        if (start_ping == start_time == None):
            start_ping = self.ping_number[0]
        if (end_ping == end_time == None):
            end_ping = self.ping_number[-1]

        #  get the primary index
        if (time_order):
            #  return indices in time order
            primary_index = self.ping_time.argsort()
        else:
            #  return indices in ping order
            primary_index = self.ping_number - 1

        #  generate a boolean mask of the values to return
        if (start_time):
            mask = self.ping_time[primary_index] >= start_time
        elif (start_ping):
            mask = self.ping_number[primary_index] >= start_ping
        if (end_time):
            mask = np.logical_and(mask, self.ping_time[primary_index] <= end_time)
        elif (end_ping):
            mask = np.logical_and(mask, self.ping_number[primary_index] <= end_ping)

        #  and return the indices that are included in the specified range
        return primary_index[mask]


    def get_sv(self, calibration=None, linear=False, return_indices=None, **kwargs):
        '''
        get_sv returns a ProcessedData object containing Sv (or sv if linear is
        True).

        MATLAB readEKRaw eq: readEKRaw_Power2Sv.m

        The value passed to cal_parameters is a calibration parameters object.
        If cal_parameters == None, the calibration parameters will be extracted
        from the corresponding fields in the raw_data object.

        '''

        #  check if the user supplied an explicit list of indices to return
        if isinstance(return_indices, np.ndarray):
            if max(return_indices) > self.ping_number.shape[0]:
                raise ValueError("One or more of the return indices provided exceeds the " +
                        "number of pings in the RawData object")
        else:
            #  get an array of index values to return
            return_indices = self.get_ping_indices(**kwargs)

        #  get the power data - this step also resamples and arranges the raw data
        power_data = self.get_power(calibration=calibration,
                return_indices=return_indices, **kwargs)

        #  next, iterate thru the calibration attributes, calling the method to extract the values for each parameter
        calibration_dict = vars(calibration)
        sv_cal_keys = ['gain', 'sound_velocity', 'frequency', 'transmit_power', 'equivalent_beam_angle', \
                       'pulse_length', 'offset', 'count', 'sample_interval']
        cal_parms = {}
        for key in sv_cal_keys:
            cal_parms[key] = self._get_calibration_param(calibration, key, return_indices)


        #  get sound_velocity from the power data since get_power might have manipulated this value
        cal_parms['sound_velocity'] = np.empty((return_indices.shape[0]), dtype=self.sample_dtype)
        cal_parms['sound_velocity'].fill(power_data.sound_velocity)


        if sys.version_info >= (3, 0):
          indices = range(len(return_indices))
        else:
          indices = xrange(len(return_indices))

        #Calculate CSv for each item in arrays.
        CSv = []
        #FIXME get tvg_correction.
#        for idx in indices:  #FIXME this loop?
#          sv = power_to_Sv(idx, data=row, gain=gain, eba=eba, sa_correction=sa_correction, calibration=cal, \
#                           tvg_correction=tvg_correction, linear=linear, raw=True)
#
#          result[idx] = {'channel':power_data['channel_id'],'reduced_timestamp':power_data[1], data_key: sv}
#           
#
#
#
#    def power_to_Sv(self, idx, data=row, gain=gain, eba=eba, sa_correction=sa_correction, calibration=cal, \
#        tvg_correction=tvg_correction, linear=linear, raw=True):
#
#        #FIXME Change i to idx and add it to the subsequenct lines in this method?  or get rid of it.  
#        CSv.append(self._calc_CSv(cal_parms['gain'][i], cal_parms['sound_velocity'][i], cal_parms['frequency'][i], \
#                                    cal_parms['transmit_power'][i], cal_parms['equivalent_beam_angle'][i], \
#                                    cal_parms['pulse_length'][i]))
#
#       
#
#        range_ = _range_vector(cal_parms['offset'], cal_parms['count'], cal_parms['sound_velocity'], \
#                               cal_parms['sample_interval'], tvg_correction)
#        tvg = range_.copy()
#        tvg[tvg == 0] = 1
#        tvg = 20 * np.log10(tvg)
#        tvg[tvg < 0] = 0
#    
#        if raw:
#            raw_factor = 10.0 * np.log10(2.0) / 256.0
#        else:
#            raw_factor = 1
#    
#        convert = lambda x: x * raw_factor + tvg + 2 * absorption_coefficient * range_ - \
#            CSv - 2 * sa_correction
#    
#    
#        if data['power'].ndim == 1:
#            if linear:
#                Sv = 10 **(convert(data['power']) / 10.0)
#    
#            else:
#                Sv = convert(data['power'])
#    
#        elif data['power'].ndim == 2:
#            Sv = np.empty_like(data['power'], dtype=np.float)
#            for ping in range(data['power'].shape[1]):
#    
#                if linear:
#                    Sv[:, ping] = 10 **(convert(data['power'][:, ping]) / 10.0)
#    
#                else:
#                    Sv[:, ping] = convert(data['power'][:, ping])
#    
#        else:
#            raise ValueError('Expected a 1- or 2-dimensional array')
#    
#        return Sv
#    
#
#


    def _calc_CSv(self, gain, sound_velocity, frequency, transmit_power, eba, pulse_length):
        '''
        Calculates the CSv constant used in power <-> Sv conversions
        '''
        beta = self._calc_beta(gain, sound_velocity, frequency, transmit_power)
        CSv = 10 * np.log10(beta / 2.0 * sound_velocity * pulse_length * 10**(eba / 10.0))
    
        return CSv
    
    
    def _calc_beta(self, gain, sound_velocity, frequency, transmit_power):
        '''  
        Convenicne constant for calculating CSv and CSp
        '''
        wlength = sound_velocity / (1.0 * frequency)
        beta = transmit_power * (10**(gain / 10.0) * wlength)**2 / (16 * np.pi**2)
    
        return beta


    def _range_vector(offset, count, sound_velocity, sample_interval, tvg_correction=2.0):
        '''
        Calculates the the tvg-corrected range vector used in Sp and Sv conversions
        '''
        dR = sound_velocity * sample_interval / 2.0
        sample_range = (np.arange(0, count) + offset) * dR
        corrected_range = sample_range - (tvg_correction * dR)
        corrected_range[corrected_range < 0] = 0
    
        return corrected_range

    
    def get_ts(self, cal_parameters=None, linear=False, **kwargs):
        '''
        get_ts returns a ProcessedData object containing TS (or sigma_bs if linear is
        True). (in MATLAB code TS == Sp and sigma_bs == sp)

        MATLAB readEKRaw eq: readEKRaw_Power2Sp.m

        The value passed to cal_parameters is a calibration parameters object.
        If cal_parameters == None, the calibration parameters will be extracted
        from the corresponding fields in the raw_data object.

        '''

        #  get the horizontal start and end indicies
        h_index = self.get_indices(**kwargs)


    def get_physical_angles(self, **kwargs):
        '''
        get_physical_angles returns a processed data object that contains the alongship and
        athwartship angle data.

        This method would call getElectricalAngles to get a vertically aligned

        '''
        pass


    def get_electrical_angles(self, **kwargs):
        '''
        get_electrical_angles returns a processed data object that contains the unconverted
        angle data. The process is identical to the get_power method.
        '''


    def get_power(self, calibration=None, resample_interval=RESAMPLE_SHORTEST,
            tvg_correction=TVG_CORRECTION, resample_soundspeed=None,
            return_indices=None, **kwargs):
        '''
        get_power returns a processed data object that contains the power data. It performs
        all of the required transformations to place the raw power data into a rectangular
        array where all samples share the same thickness and are correctly arranged relative
        to each other.

        This process happens in 3 steps:

                Data are resampled so all samples have the same thickness
                Data are shifted vertically to account for the sample offsets
                Data are then regridded to a fixed time, range grid

        Each step is performed only when required. Calls to this method will return much
        faster if the raw data share the same sample thickness, offset and sound speed.

        If calibration is set to an instance of EK60.CalibrationParameters the values in
        that object (if set) will be used when performing the transformations required to
        return the results. If the required parameters are not set in the calibration
        object or if no object is provided, this method will extract these parameters from
        the raw file data.
        '''

        #  check if the user supplied an explicit list of indices to return
        if isinstance(return_indices, np.ndarray):
            if max(return_indices) > self.ping_number.shape[0]:
                raise ValueError("One or more of the return indices provided exceeds the " +
                        "number of pings in the RawData object")
        else:
            #  get an array of index values to return
            return_indices = self.get_ping_indices(**kwargs)

        #  create the ProcessedData object we will return
        power_data = ProcessedData.ProcessedData(self.channel_id, self.frequency[0])

        #  populate it with time and ping number
        #power_data.ping_time = [self.ping_time[i] for i in return_indices.tolist()]
        power_data.ping_time = self.ping_time[return_indices].copy()
        power_data.ping_number = self.ping_number[return_indices].copy()

        #  populate the calibration parameters required for this method. First, create a dict with key
        #  names that match the attributes names of the calibration parameters we require for this method
        cal_parms = {'sample_interval':None,
                     'sound_velocity':None,
                     'sample_offset':None,
                     'transducer_depth':None}

        #  next, iterate thru the dict, calling the method to extract the values for each parameter
        for key in cal_parms:
            cal_parms[key] = self._get_calibration_param(calibration, key, return_indices)

        #  check if we have multiple sample offset values and get the minimum
        unique_sample_offsets = np.unique(cal_parms['sample_offset'])
        min_sample_offset = min(unique_sample_offsets)

        # check if we need to resample our sample data
        unique_sample_interval = np.unique(cal_parms['sample_interval'])
        if (unique_sample_interval.shape[0] > 1):
            #  there are at least 2 different sample intervals in the data - we must resample the power data
            #  Since we're already in the neighborhood, we deal with adjusting sample offsets here too.
            (power_data.power, sample_interval) = self._resample_sample_data(self.power[return_indices],
                    cal_parms['sample_interval'], unique_sample_interval, resample_interval,
                    cal_parms['sample_offset'], min_sample_offset, is_power=True)
        else:
            #  we don't have to resample, but check if we need to shift any samples based on their sample offsets.
            if (unique_sample_offsets.shape[0] > 1):
                #  we have multiple sample offsets so we need to shift some of the samples
                power_data.power = self._shift_sample_offsets(self.power, return_indices,
                        cal_parms['sample_offset'], unique_sample_offsets, min_sample_offset)
            else:
                #  the data all have the same sample intervals and sample offsets - simply copy the data as is.
                power_data.power = self.power[return_indices].copy()

            #  and get the sample interval value to use for range conversion below
            sample_interval = unique_sample_interval[0]

        #  check if we have a fixed sound speed
        unique_sound_velocity = np.unique(cal_parms['sound_velocity'])
        if (unique_sound_velocity.shape[0] > 1):
            #  there are at least 2 different sound speeds in the data or provided calibration data.
            #  interpolate all data to the most common range (which is the most common sound speed)
            sound_velocity = None
            n = 0
            for speed in unique_sound_velocity:
            #  determine the sound speed with the most pings
                if (np.count_nonzero(cal_parms['sound_velocity'] == speed) > n):
                   sound_velocity = speed

            #  calculate the target range
            range = unit_conversion.get_range_vector(power_data.power.shape[1],
                        sample_interval, sound_velocity, min_sample_offset,
                        tvg_correction=tvg_correction)

            #  get an array of indexes in the output array to interpolate
            pings_to_interp = np.where(cal_parms['sound_velocity'] != sound_velocity)[0]

            #  iterate thru this list of pings to change - interpolating each ping
            for ping in pings_to_interp:
                #  resample using the provided sound speed - calculate the
                resample_range = unit_conversion.get_range_vector(power_data.power.shape[1],
                        sample_interval, cal_parms['sound_velocity'][ping], min_sample_offset,
                        tvg_correction=tvg_correction)

                power_data.power[ping,:] = np.interp(range, resample_range,
                    power_data.power[ping,:])

        else:
            #  we have a fixed sound speed - only need to calculate a single range vector
            sound_velocity = unique_sound_velocity[0]
            range = unit_conversion.get_range_vector(power_data.power.shape[1],
                        sample_interval, sound_velocity, min_sample_offset,
                        tvg_correction=tvg_correction)

        #  assign range and sound speed to our ProcessedData object
        power_data.range = range
        power_data.sound_velocity = sound_velocity

        #  compute sample thickness and set the sample offset
        power_data.sample_thickness = sample_interval * sound_velocity / 2.0
        power_data.sample_offset = min_sample_offset

        #  copy the transducer depth data
        power_data.transducer_depth = cal_parms['transducer_depth'].copy()

        #  return the ProcessedData object containing power and associated data
        return power_data


    def _shift_sample_offsets(self, data, processed_data, sample_offsets, unique_sample_offsets,
            min_sample_offset):
        '''
        _shift_sample_offsets adjusts the output array size and pads the top of the
        samples array to vertically shift the positions of the sample data in the output
        array. Pings with offsets greater than the minimum will be padded on the top,
        shifting them into their correct location relative to the other pings.

        The result is an output array with samples that are properly aligned vertically
        relative to each other with a sample offset that is constant and equal to the
        minimum of the original sample offsets.

        This method is only called if our data has a constant sample interval but
        varying sample offsets. If the data has multiple sample intervals the offset
        adjustment is done in _resample_sample_data.
        '''

        #  determine the new array size
        new_sample_dims = data.shape[1] + max(sample_offsets) - min_sample_offset

        #  create the new array
        shifted_data = np.empty((data.shape[0], new_sample_dims), dtype=self.sample_dtype, order='C')
        shifted_data.fill(np.nan)

        #  and fill it looping over the different sample offsets
        for offset in unique_sample_offsets:
            rows_this_offset = np.where(sample_offsets == offset)[0]
            start_index = offset - min_sample_offset
            end_index = start_index + data.shape[1]
            shifted_data[rows_this_offset, start_index:end_index] = data[rows_this_offset, 0:data.shape[1]]

        return shifted_data


    def _resample_sample_data(self, data, sample_intervals, unique_sample_intervals,
            resample_interval, sample_offsets, min_sample_offset, is_power=True):
        '''
        _resample_sample_data vertically resamples power or angle data given a target
        sample interval. This method also shifts samples vertically based on their sample
        offset so they are positioned correctly relative to each other. The first sample
        in the resulting array will have an offset that is the minimum of all offsets in
        the data.
        '''

        #  determine the number of pings in the new array
        n_pings = data.shape[0]

        # check if we need to substitute our resample_interval value
        if (resample_interval == self.RESAMPLE_SHORTEST):
            #  resample to the shortest sample interval in our data
            resample_interval = min(unique_sample_intervals)
        elif (resample_interval == self.RESAMPLE_LONGEST):
            #  resample to the longest sample interval in our data
            resample_interval = max(unique_sample_intervals)

        #  create a couple of dictionaries to store resampling parameters by sample interval
        #  they will be used again when we fill the output array with the resampled data.
        resample_factor = {}
        rows_this_interval = {}
        sample_offsets_this_interval = {}

        #  determine number of samples in the output array - to do this we must loop thru
        #  the sample intervals, determine the resampling factor, then find the maximum sample
        #  count at that sample interval (taking into account the sample's offset) and multiply
        #  by the resampling factor to determine the max number of samples for that sample interval.
        new_sample_dims = 0
        for sample_interval in unique_sample_intervals:
            #  determine the resampling factor
            if (resample_interval > sample_interval):
                #  we're reducing resolution - determine the number of samples to average
                resample_factor[sample_interval] = resample_interval / sample_interval
            else:
                #  we're increasing resolution - determine the number of samples to expand
                resample_factor[sample_interval] = sample_interval / resample_interval

            #  determine the rows in this subset with this sample interval
            rows_this_interval[sample_interval] = np.where(sample_intervals == sample_interval)[0]

            #  determine the net vertical shift for the samples with this sample interval
            sample_offsets_this_interval[sample_interval] = sample_offsets[rows_this_interval[sample_interval]] - \
                    min_sample_offset

            #  and determine the maximum number of samples for this sample interval - this has to
            #  be done on a row-by-row basis since sample number can change on the fly. We include
            #  the sample offset to ensure we have room to shift our samples vertically by the offset
            max_samples_this_sample_int = max(self.sample_count[rows_this_interval[sample_interval]] +
                    sample_offsets_this_interval[sample_interval])
            max_dim_this_sample_int = int(round(max_samples_this_sample_int * resample_factor[sample_interval]))
            if (max_dim_this_sample_int > new_sample_dims):
                    new_sample_dims = max_dim_this_sample_int

        #  now that we know the dimensions of the output array create the it and fill with NaNs
        resampled_data = np.empty((n_pings, new_sample_dims), dtype=self.sample_dtype, order='C')
        resampled_data.fill(np.nan)

        #  and fill it with data - We loop thru the sample intervals and within an interval extract slices
        #  of data that share the same number of samples (to reduce looping). We then determine if we're
        #  expanding or shrinking the number of samples. If expanding we simply replicate existing sample
        #  data to fill out the expaned array. If reducing, we take the mean of the samples. Power data is
        #  converted to linear units before the mean is computed and then transformed back.
        for sample_interval in unique_sample_intervals:
            #  determine the unique sample_counts for this sample interval
            unique_sample_counts = np.unique(self.sample_count[rows_this_interval[sample_interval]])
            for count in unique_sample_counts:
                #  determine if we're reducing, expanding, or keeping the same number of samples
                if (resample_interval > sample_interval):
                    #  we're reducing the number of samples

                    #  if we're resampling power convert power to linear units
                    if (is_power):
                        this_data = np.power(data[rows_this_interval[sample_interval]]
                                [self.sample_count[rows_this_interval[sample_interval]] == count] / 20.0, 10.0)

                    #  reduce the number of samples by taking the mean
                    this_data =  np.mean(this_data.reshape(-1, int(resample_factor[sample_interval])), axis=1)

                    if (is_power):
                        #  convert power back to log units
                        this_data = 20.0 * np.log10(this_data)

                elif (resample_interval < sample_interval):
                    #  we're increasing the number of samples

                    #  replicate the values to fill out the higher resolution array
                    this_data = np.repeat(data[rows_this_interval[sample_interval]]
                            [self.sample_count[rows_this_interval[sample_interval]] == count][:,0:count],
                            int(resample_factor[sample_interval]), axis=1)

                else:
                    #  no change in resolution for this sample interval
                    this_data = data[rows_this_interval[sample_interval]] \
                            [self.sample_count[rows_this_interval[sample_interval]] == count]


                #  generate the index array for this sample interval/sample count chunk of data
                rows_this_interval_count = rows_this_interval[sample_interval] \
                        [self.sample_count[rows_this_interval[sample_interval]] == count]

                #  assign new values to output array - at the same time we will shift the data by sample offset
                unique_sample_offsets = np.unique(sample_offsets_this_interval[sample_interval])
                for offset in unique_sample_offsets:
                    resampled_data[rows_this_interval_count,
                            offset:offset + this_data.shape[1]]  = this_data

        #  return the sample interval of the resampled data
        return (resampled_data, resample_interval)


    #FIXME If I remove this, will it break anything?
    def __iter__(self):
        for attribute in vars(self):
            yield (attribute, getattr(self, attribute))


    def _get_calibration_param(self, cal_object, param_name, return_indices, dtype='float32'):
        '''
        _get_calibration_param interrogates the provided cal_object for the provided param_name
        property and returns the parameter values based on what it finds. It handles 4 cases:

            If the user has provided a scalar calibration value, the function will return
            a 1D array the length of return_indices filled with that scalar.

            If the user has provided a 1D array the length of return_indices it will return
            that array without modification.

            If the user has provided a 1D array the length of self.ping_number, it will
            return a 1D array the length of return_indices that is the subset of this data
            defined by the return_indices index array.

            Lastly, if the user has not provided anything, this function will return a
            1D array the length of return_indices filled with data extracted from the raw
            data
        '''


        if (cal_object):
            #  try to get the parameter from the calibration object
            print('param_name',param_name)
            param = getattr(cal_object, param_name)
            print('param', param)
            print('type(param)', type(param))

            #  check if the input param is an numpy array
            if isinstance(param, np.ndarray):
                #  check if it is a single value array
                if (param.shape[0] == 1):
                    param_data = np.empty((return_indices.shape[0]), dtype=dtype)
                    param_data.fill(param)
                #  check if it is an array the same length as contained in the raw data
                elif (param.shape[0] == self.ping_number.shape[0]):
                    #  cal params provided as full length array, get the selection subset
                    param_data = param[return_indices]
                #  check if it is an array the same length as return_indices
                elif (param.shape[0] == return_indices.shape[0]):
                    #  cal params provided as a subset so no need to index with return_indices
                    param_data = param
                else:
                    #  it is an array that is the wrong shape
                    raise ValueError("The calibration parameter array " + param_name +
                            " is the wrong length.")
            #  not an array - check if it is a scalar int or float
            elif (type(param) == int or type(param) == float or type(param) == np.float64):
                    param_data = np.empty((return_indices.shape[0]), dtype=dtype)
                    param_data.fill(param)
            else:
                #  invalid type provided
                raise ValueError("The calibration parameter " + param_name +
                        " must be an ndarray or scalar float.")
        else:
            #  Parameter is not provided in the calibration object, copy it from the raw data.
            #  Calibration parameters are found directly in the RawData object and they are
            #  in the channel_metadata objects. If we don't find it directly in RawData then
            #  we need to fish it out of the channel_metadata objects.
            try:
                #  first check if this parameter is a direct property in RawData
                self_param = getattr(self, param_name)
                #  it is - return a view of the subset of data we're interested in
                param_data = self_param[return_indices]
            except:
                #  It is not a direct property so it must be in the channel_metadata object.
                #  Create the return array
                param_data = np.empty((return_indices.shape[0]), dtype=dtype)
                #  then populate with the data found in the channel_metadata objects
                for idx in return_indices:
                    param_data[idx] = getattr(self.channel_metadata[idx],param_name)

        return param_data


    def _roll_arrays(self, roll_pings):
        '''
        _roll_arrays is an internal method that rolls our data arrays when those arrays
        are fixed in size and we add a ping. This typically would be used for buffering
        data from streaming sources.
        '''

        #TODO: implement and test these inline rolling functions
        #      Need to profile this code to see which methods are faster. Currently all rolling is
        #      implemented using np.roll which makes a copy of the data.
        #TODO: verify rolling direction
        #      Verify the correct rolling direction for both the np.roll calls and the 2 inline
        #      functions. I *think* the calls to np.roll are correct and the inline functions roll
        #      the wrong way.

        def roll_1d(data):
            #  rolls a 1d *mostly* in place
            #  based on code found here:
            #    https://stackoverflow.com/questions/35916201/alternative-to-numpy-roll-without-copying-array
            #  THESE HAVE NOT BEEN TESTED
            temp_view = data[:-1]
            temp_copy = data[-1]
            data[1:] = temp_view
            data[0] = temp_copy

        def roll_2d(data):
            #  rolls a 2d *mostly* in place
            temp_view = data[:-1,:]
            temp_copy = data[-1,:]
            data[1:,:] = temp_view
            data[0,:] = temp_copy

        #  roll our two lists
        #self.ping_time.append(self.ping_time.pop(0))
        self.channel_metadata.append(self.channel_metadata.pop(0))

        #  roll the numpy arrays
        self.ping_time = np.roll(self.ping_time, roll_pings)
        self.ping_number = np.roll(self.ping_number, roll_pings)
        self.transducer_depth = np.roll(self.transducer_depth, roll_pings)
        self.frequency = np.roll(self.frequency, roll_pings)
        self.transmit_power = np.roll(self.transmit_power, roll_pings)
        self.pulse_length = np.roll(self.pulse_length, roll_pings)
        self.bandwidth = np.roll(self.bandwidth, roll_pings)
        self.sample_interval = np.roll(self.sample_interval, roll_pings)
        self.sound_velocity = np.roll(self.sound_velocity, roll_pings)
        self.absorption_coefficient = np.roll(self.absorption_coefficient, roll_pings)
        self.heave = np.roll(self.heave, roll_pings)
        self.pitch = np.roll(self.pitch, roll_pings)
        self.roll = np.roll(self.roll, roll_pings)
        self.temperature = np.roll(self.temperature, roll_pings)
        self.heading = np.roll(self.heading, roll_pings)
        self.transmit_mode = np.roll(self.transmit_mode, roll_pings)
        self.sample_offset = np.roll(self.sample_offset, roll_pings)
        self.sample_count = np.roll(self.sample_count, roll_pings)
        if (self.store_power):
            self.power = np.roll(self.power, roll_pings, axis=0)
        if (self.store_angles):
            self.angles_alongship_e = np.roll(self.angles_alongship_e, roll_pings, axis=0)
            self.angles_athwartship_e = np.roll(self.angles_athwartship_e, roll_pings, axis=0)


    def _resize_arrays(self, new_ping_dim, new_sample_dim, old_ping_dim, old_sample_dim):
        '''
        _resize_arrays is an internal method that handles resizing the data arrays
        '''

        def resize2d(data, ping_dim, sample_dim):
            '''
            resize2d returns a new array of the specified dimensions with the data from
            the provided array copied into it. This funciton is used when we need to resize
            2d arrays along the minor axis as ndarray.resize and numpy.resize don't maintain
            the order of the data in these cases.
            '''

            #  if the minor axis is changing we have to either concatenate or copy into a new resized array
            #  I'm taking the second approach for now as I don't think there are performance differences
            #  between the two approaches.

            #  create a new array
            new_array = np.empty((ping_dim, sample_dim))
            #  fill it with NaNs
            new_array.fill(np.nan)
            #  copy the data into our new array
            new_array[0:data.shape[0], 0:data.shape[1]] = data
            #  and return it
            return new_array

        #  resize the 1d arrays
        if (new_ping_dim != old_ping_dim):
            self.ping_time.resize((new_ping_dim))
            self.ping_number.resize((new_ping_dim))
            self.transducer_depth.resize((new_ping_dim))
            self.frequency.resize((new_ping_dim))
            self.transmit_power.resize((new_ping_dim))
            self.pulse_length.resize((new_ping_dim))
            self.bandwidth.resize((new_ping_dim))
            self.sample_interval.resize((new_ping_dim))
            self.sound_velocity.resize((new_ping_dim))
            self.absorption_coefficient.resize((new_ping_dim))
            self.heave.resize((new_ping_dim))
            self.pitch.resize((new_ping_dim))
            self.roll.resize((new_ping_dim))
            self.temperature.resize((new_ping_dim))
            self.heading.resize((new_ping_dim))
            self.transmit_mode.resize((new_ping_dim))
            self.sample_offset.resize((new_ping_dim))
            self.sample_count.resize((new_ping_dim))

        #  resize the 2d arrays
        #  NOTE: Initially 2d resize was implemented with ndarray.resize() which resizes in place and
        #              thus should be more efficient but its behavior with 2d arrays when the minor axes
        #              is changed causes the data to shift in the array "scrambling" the data.
        if (self.store_power):
            if (new_sample_dim == old_sample_dim):
                #  if the minor axes isn't changing we can use ndarray.resize()
                self.power.resize((new_ping_dim, new_sample_dim))
                # shouldn't need to pad NaNs in these cases since we'll either finll the pings or trim them.
            else:
                #  if the minor axes is changing we need to use our resize2d function
                self.power = resize2d(self.power, new_ping_dim, new_sample_dim)
        if (self.store_angles):
            #  same constraints apply to the angle data
            if (new_sample_dim == old_sample_dim):
                self.angles_alongship_e.resize((new_ping_dim, new_sample_dim))
                self.angles_athwartship_e.resize((new_ping_dim, new_sample_dim))
            else:
                self.angles_alongship_e = resize2d(self.angles_alongship_e, new_ping_dim, new_sample_dim)
                self.angles_athwartship_e = resize2d(self.angles_athwartship_e, new_ping_dim, new_sample_dim)


    def _create_arrays(self, n_pings, n_samples, initialize=False):
        '''
        _create_arrays is an internal method that initializes the RawData data arrays.
        '''

        #  ping_time and channel_metadata are lists
        #self.ping_time = []
        self.channel_metadata = []

        #  all other data properties are numpy arrays

        #  first, create uninitialized arrays
        self.ping_time = np.empty((n_pings), dtype='datetime64[s]')
        self.ping_number = np.empty((n_pings), np.int32)
        self.transducer_depth = np.empty((n_pings), np.float32)
        self.frequency = np.empty((n_pings), np.float32)
        self.transmit_power = np.empty((n_pings), np.float32)
        self.pulse_length = np.empty((n_pings), np.float32)
        self.bandwidth = np.empty((n_pings), np.float32)
        self.sample_interval = np.empty((n_pings), np.float32)
        self.sound_velocity = np.empty((n_pings), np.float32)
        self.absorption_coefficient = np.empty((n_pings), np.float32)
        self.heave = np.empty((n_pings), np.float32)
        self.pitch = np.empty((n_pings), np.float32)
        self.roll = np.empty((n_pings), np.float32)
        self.temperature = np.empty((n_pings), np.float32)
        self.heading = np.empty((n_pings), np.float32)
        self.transmit_mode = np.empty((n_pings), np.uint8)
        self.sample_offset =  np.empty((n_pings), np.uint32)
        self.sample_count = np.empty((n_pings), np.uint32)
        if (self.store_power):
            self.power = np.empty((n_pings, n_samples), dtype=self.sample_dtype, order='C')
        else:
            #  create an empty array as a place holder
            self.power = np.empty((0,0), dtype=self.sample_dtype, order='C')
        if (self.store_angles):
            self.angles_alongship_e = np.empty((n_pings, n_samples), dtype=self.sample_dtype, order='C')
            self.angles_athwartship_e = np.empty((n_pings, n_samples), dtype=self.sample_dtype, order='C')
        else:
            #  create an empty arrays as place holders
            self.angles_alongship_e = np.empty((0, 0), dtype=self.sample_dtype, order='C')
            self.angles_athwartship_e = np.empty((0, 0), dtype=self.sample_dtype, order='C')

        #  check if we should initialize them
        if (initialize):
            self.ping_time.fill(datetime.datetime('1970','1','1'))
            self.ping_number.fill(0)
            self.transducer_depth.fill(np.nan)
            self.frequency.fill(np.nan)
            self.transmit_power.fill(np.nan)
            self.pulse_length.fill(np.nan)
            self.bandwidth.fill(np.nan)
            self.sample_interval.fill(np.nan)
            self.sound_velocity.fill(np.nan)
            self.absorption_coefficient.fill(np.nan)
            self.heave.fill(np.nan)
            self.pitch.fill(np.nan)
            self.roll.fill(np.nan)
            self.temperature.fill(np.nan)
            self.heading.fill(np.nan)
            self.transmit_mode.fill(0)
            self.sample_offset.fill(0)
            self.sample_count.fill(0)
            if (self.store_power):
                self.power.fill(np.nan)
            if (self.store_angles):
                self.angles_alongship_e.fill(np.nan)
                self.angles_athwartship_e.fill(np.nan)


    def __str__(self):
        '''
        reimplemented string method that provides some basic info about the RawData object
        '''

        #  print the class and address
        msg = str(self.__class__) + " at " + str(hex(id(self))) + "\n"

        #  print some more info about the EK60 instance
        n_pings = len(self.ping_time)
        if (n_pings > 0):
            msg = msg + "                channel(s): ["
            for channel in self.channel_id:
                msg = msg + channel + ", "
            msg = msg[0:-2] + "]\n"
            msg = msg + "    frequency (first ping): " + str(self.frequency[0])+ "\n"
            msg = msg + " pulse length (first ping): " + str(self.pulse_length[0])+ "\n"
            msg = msg + "           data start time: " + str(self.ping_time[0])+ "\n"
            msg = msg + "             data end time: " + str(self.ping_time[n_pings-1])+ "\n"
            msg = msg + "           number of pings: " + str(n_pings)+ "\n"
            if (self.store_power):
                n_pings,n_samples = self.power.shape
                msg = msg + ("    power array dimensions: (" + str(n_pings)+ "," +
                        str(n_samples) +")\n")
            if (self.store_angles):
                n_pings,n_samples = self.angles_alongship_e.shape
                msg = msg + ("    angle array dimensions: (" + str(n_pings)+ "," +
                        str(n_samples) +")\n")
        else:
            msg = msg + ("  RawData object contains no data\n")

        return msg



class ChannelMetadata(object):
    '''
    The ChannelMetadata class stores the channel configuration data as well as
    some metadata about the file. One of these is created for each channel for
    every .raw file read.

    References to instances of these objects are stored in RawfileData

    '''

    def __init__(self, file, config_datagram, survey_name, transect_name, sounder_name, version,
                start_ping, start_time, extended_configuration=None):

        #  split the filename
        file = os.path.normpath(file).split(os.path.sep)

        #  store the base filename and path separately
        self.data_file = file[-1]
        self.data_file_path = os.path.sep.join(file[0:-1])

        #  define some basic metadata properties
        self.start_ping = start_ping
        self.end_ping = 0
        self.start_time = start_time
        self.end_time = None

        #  we will replicate the ConfigurationHeader struct here since there
        #  is no better place to store it
        self.survey_name = ''
        self.transect_name = ''
        self.sounder_name = ''
        self.version = ''

        #  store the ME70 extended configuration XML string
        self.extended_configuration = extended_configuration

        #  the GPT firmware version used when recording this data
        self.gpt_firmware_version = config_datagram['gpt_software_version']

        #  the beam type for this channel - split or single
        self.beam_type = config_datagram['beam_type']

        #  the channel frequency in Hz
        self.frequency_hz = config_datagram['frequency']

        #  the system gain when the file was recorded
        self.gain = config_datagram['gain']

        #  beam calibration properties
        self.equivalent_beam_angle = config_datagram['equivalent_beam_angle']
        self.beamwidth_alongship = config_datagram['beamwidth_alongship']
        self.beamwidth_athwartship = config_datagram['beamwidth_athwartship']
        self.angle_sensitivity_alongship = config_datagram['angle_sensitivity_alongship']
        self.angle_sensitivity_athwartship = config_datagram['angle_sensitivity_athwartship']
        self.angle_offset_alongship = config_datagram['angle_offset_alongship']
        self.angle_offset_athwartship = config_datagram['angle_offset_athwartship']

        #  transducer installation/orientation parameters
        self.pos_x = config_datagram['pos_x']
        self.pos_y = config_datagram['pos_y']
        self.pos_z = config_datagram['pos_z']
        self.dir_x = config_datagram['dir_x']
        self.dir_y = config_datagram['dir_y']
        self.dir_z = config_datagram['dir_z']

        #  the possile pulse lengths for the recording system
        self.pulse_length_table = config_datagram['pulse_length_table']
        self.spare2 = config_datagram['spare2']
        #  the gains set for each of the system pulse lengths
        self.gain_table = config_datagram['gain_table']
        self.spare3 = config_datagram['spare3']
        #  the sa correction values set for each pulse length
        self.sa_correction_table = config_datagram['sa_correction_table']
        self.spare4 = config_datagram['spare4']


class CalibrationParameters(object):
    '''
    The CalibrationParameters class contains parameters required for transforming
    power and electrical angle data to Sv/sv TS/SigmaBS and physical angles.
    '''

    def __init__(self):

        self.channel_id = []
        self.count = []
        self.sample_count = []
        self.frequency = []
        self.sound_velocity = []
        self.sample_interval = []
        self.absorption_coefficient = []
        self.gain = []
        self.equivalent_beam_angle = []
        self.beamwidth_alongship = []
        self.beamwidth_athwartship = []
        self.pulse_length_table = []
        self.gain_table  = []
        self.sa_correction_table = []
        self.transmit_power = []
        self.pulse_length = []
        self.angle_sensitivity_alongship = []
        self.angle_sensitivity_athwartship = []
        self.angle_offset_alongship = []
        self.angle_offset_athwartship = []
        self.transducer_depth = []

        self.sounder_name = [] #From matlab calib params but it's being stored here in the channel_metadata. Remove?
        self.sample_offset = [] #Should this come from the "offset" field in the datagrams?
        self.offset = [] 

    def append_calibration(self, datagram):
      #TODO Add code to ensure alignment with raw data arrays.  Use n_pings.
      for attribute in vars(self):
        if attribute in datagram:
          self._append_data(attribute, datagram[attribute])
      self.sample_offset = self.offset #FIXME Is this right?
      self.sample_count = self.count #FIXME Is this right?


    def _append_data(self, attribute, data):
      attr_data = getattr(self, attribute)
      if isinstance(data, np.ndarray):
        datagram_data = self.get_table_value(data, attribute)
      else:
        datagram_data = data
      attr_data.append(datagram_data)
      setattr(self, attribute, attr_data)


    def get_table_value(self, data, attribute):
      #TODO Ask Rick which value to use.
      return data[0]


    def compress_data_arrays(self):
        '''
        If any of the data arrays in this object have values that are all the same,
        replace the array with a scalar value.
        '''
        for attr in vars(self):
          data = getattr(self, attr)

          if len(set(data)) == 1:
            data = data[0]
            setattr(self, attr, data)


    def from_raw_data(self, raw_data, raw_file_idx=0):
        '''
        from_raw_data populated the CalibrationParameters object's properties given
        a reference to a RawData object.

        This would query the RawFileData object specified by raw_file_idx in the
        provided RawData object (by default, using the first).
        '''
        #TODO Ask, do we want to add calibration data from the config datagram to the raw data object
        #     in order to capture them here?
        #TODO Since calibration data is indexed the same as data now, do we still want to use raw_file_idx here?
        for attr in vars(self):
          if attr in vars(raw_data):
            data = getattr(raw_data, attr)
            self_data = getattr(self, attr)
            if not isinstance(self_data, list):
              self_data = [self_data]
              setattr(self, attr, self_data)
            self._append_data(attr, data)


    def read_ecs_file(self, ecs_file, channel):
        '''
        read_ecs_file reads an echoview ecs file and parses out the
        parameters for a given channel.
        '''
        pass

