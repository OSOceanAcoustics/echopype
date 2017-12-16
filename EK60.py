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
import datetime
from pytz import timezone
import logging
import numpy as np
from util.raw_file import RawSimradFile, SimradEOF
from util import unit_conversion
from collections import defaultdict

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

        #  a list of frequencies that have been read.
        self.frequencies = []

        #  a list of stings identifying the channel IDs that have been read
        self.channel_ids = []

        #  channel_id_map maps the channel number to channel ID when reading raw data sources
        self.channel_id_map = {}

        #  create a dictionary to store the EK60RawData objects
        self.raw_data = {}


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

        #  iterate thru our list of .raw files to read
        for filename in raw_files:

            #  Read data from file and add to self.raw_data.
            with RawSimradFile(filename, 'r') as fid:

                #  read the "CON" configuration datagrams
                #
                #    "CON0" is the channel configuration datagram for EK/ES60 and ES70 as well as ME70
                #    "CON1" is the extended channel configuration XML string for ME70
                #
                config_datagrams= {}
                datagram = fid.read(1)
                self.start_time = datagram['timestamp']

                while datagram['type'].startswith('CON'):
                    config_datagrams[datagram['type']] = datagram
                    datagram = fid.read(1)

                #  check if we need to create an EK60RawData object for this channel
                self.channel_id_map = {}
                for channel in config_datagrams['CON0']['transceivers']:
                    #  get the channel ID
                    channel_id = config_datagrams['CON0']['transceivers'][channel]['channel_id']

                    #  check if we are reading this channel
                    if ((self.read_channel_ids) and (channel_id not in self.read_channel_ids)):
                        #  there are specific channel IDs specified and this is *NOT* one of them
                        #  so we just move along...
                        continue

                    #  check if we're reading this frequency
                    frequency = config_datagrams['CON0']['transceivers'][channel]['frequency']
                    if ((self.read_frequencies) and (frequency not in self.read_frequencies)):
                        #  there are specific frequencies specified and this is *NOT* one of them
                        #  so we just move along...
                        continue

                    #  check if an EK60RawData object exists for this channel
                    if channel_id not in self.raw_data:
                        #  no - create it
                        self.raw_data[channel_id] = EK60RawData(channel_id, store_power=self.read_power,
                                store_angles=self.read_angles, max_sample_number=self.read_max_sample_count)

                        #  and add it to our list of channel_ids
                        self.channel_ids.append(channel_id)


                    #  update the mapping of channel number to channel ID used when reading the datagrams.
                    #  this mapping must be updated for each .raw file since it is possible that the
                    #  transceiver installation can change between files.
                    self.channel_id_map[channel] = channel_id

                    #  create a EK60ChannelMetadata object to store this channel's configuration and rawfile metadata.
                    channel_metadata = EK60ChannelMetadata(filename,
                                               config_datagrams['CON0']['transceivers'][channel],
                                               config_datagrams['CON0']['survey_name'],
                                               config_datagrams['CON0']['transect_name'],
                                               config_datagrams['CON0']['sounder_name'],
                                               config_datagrams['CON0']['version'],
                                               self.raw_data[channel_id].n_pings,
                                               config_datagrams['CON0']['timestamp'])

                    #  update the channel_metadata property of the RawData object
                    self.raw_data[channel_id].current_metadata = channel_metadata

                #  read the rest of the datagrams.
                self._read_datagrams(fid, self.read_incremental)

        #  trim excess data from arrays after reading
        for channel_id in self.channel_ids:
            self.raw_data[channel_id].trim()


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
        num_pings = 0

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
                    num_pings += 1

                #  check if we should store this data based on ping bounds
                if self.read_start_ping is not None:
                    if num_pings < self.read_start_ping:
                        continue
                if self.read_end_ping is not None:
                    if num_pings > self.read_end_ping:
                        continue

                #  check if we're supposed to store this channel
                if new_datagram['channel'] in self.channel_id_map:

                    #  set the first ping number we read
                    if (not self.start_ping):
                        self.start_ping = num_pings
                    #  and update the last ping number
                    self.end_ping = num_pings

                    #  get the channel id
                    channel_id = self.channel_id_map[new_datagram['channel']]

                    #  and call the appropriate channel's append_ping method
                    self.raw_data[channel_id].append_ping(new_datagram)

                    # increment the sample datagram counter
                    num_sample_datagrams += 1
                else:
                    num_sample_datagrams_skipped += 1

            #  NME datagrams store ancillary data as NMEA-0817 style ASCII data
            elif new_datagram['type'].startswith('NME'):
                #TODO:  Implement NMEA reading
                pass

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
        get_rawdata returns a reference to the specified EK60RawData object for the
        specified channel id or channel number.
        '''

        if (channel_id):
            return self.raw_data.get(channel_id, None)
        else:
            return self.raw_data.get(self.channel_ids[channel_number], None)


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
            msg = msg + ("    number of pings: " + str(self.end_ping - self.start_ping)+ "\n")

        else:
            msg = msg + ("  EK60 object contains no data\n")

        return msg


class EK60RawData(object):
    '''
    the EK60RawData class contains a single channel's data extracted from a Simrad raw
    file. collected from an EK/ES60 or ES70. A EK60RawData object is created for each
    unique channel in an EK/ES60 ES70 raw file.

    '''

    #TODO: reimplement __str__ similar to how it is done in EK60

    #  define some instrument specific constants
    SAMPLES_PER_PULSE = 4

    #FIXME Values?
    RESAMPLE_LOWEST = 64
    RESAMPLE_64 = 64
    RESAMPLE_128 = 128
    RESAMPLE_256 = 256
    RESAMPLE_512 = 512
    RESAMPLE_1024 = 1024
    RESAMPLE_2048 = 2048
    RESAMPLE_4096 = 4096
    RESAMPLE_HIGHEST = 4096

    to_shortest = 0
    to_longest = 1


    def __init__(self, channel_id, n_pings=100, n_samples=1000, rolling=False,
            chunk_width=500, store_power=True, store_angles=True, max_sample_number=None):
        '''
        Creates a new, empty EK60RawData object. The EK60RawData class stores raw
        echosounder data from a single channel of an EK60 or ES60/70 system.

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

        #  current_metadata stores a reference to the current EK60ChannelMetadata object. The
        #  EK60ChannelMetadata class stores rawfile and channel configuration properties
        #  contained in the .raw file header. When opening a new .raw file, this property
        #  must be updated before appending pings from the new file.
        self.current_metadata = None

        #  the channel ID is the unique identifier
        self.channel_id = channel_id

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
        self.logger = logging.getLogger('EK60RawData')


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

        #  append the EK60ChannelMetadata object reference for this ping
        self.channel_metadata.append(self.current_metadata)

        #  update the EK60ChannelMetadata object with this ping number and time
        self.current_metadata.end_ping = self.n_pings
        self.current_metadata.end_time = sample_datagram['timestamp']

        #  append the datetime object representing the ping time to the ping_time property
        self.ping_time.append(sample_datagram['timestamp'])

        #  now insert the data into our numpy arrays
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

        #  now store the 2d "sample" data
        #      determine what we need to store based on operational mode
        #      1 = Power only, 2 = Angle only 3 = Power & Angle

        #  check if we need to store power data
        if (sample_datagram['mode'] != 2):
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
        if (sample_datagram['mode'] != 1):
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



    '''
    I admit that I didn't really get what append_data was doing and wasn't sure why it was
    being called after append ping. After a RAW datagram is read from the disk, we only
    need to call append_ping where we insert the datagram's data into our RawData data arrays
    and deal with all of the book keeping related to that.

    '''

    def append_data(self, datagram):
        # FIXME define these attributes in one place to be used here and in parsers.py
        # FIXME or loop through checking key each time
        # or throw error if attr is missing

        try:
            ping_time = self.ping_time.copy()
            ping_time.append(datagram['ping_time'])
            transducer_depth = self.transducer_depth.copy()
            transducer_depth.append(datagram['transducer_depth'])
            frequency = self.frequency.copy()
            frequency.append(datagram['frequency'])
            transmit_power = self.transmit_power.copy()
            transmit_power.append(datagram['transmit_power'])
            pulse_length = self.pulse_length.copy()
            pulse_length.append(datagram['pulse_length'])
            bandwidth = self.bandwidth.copy()
            bandwidth.append(datagram['bandwidth'])
            sample_interval = self.sample_interval
            sample_interval.append(datagram['sample_interval'])
            sound_velocity = self.sound_velocity
            sound_velocity.append(datagram['sound_velocity'])
            absorption_coefficient = self.absorption_coefficient
            absorption_coefficient.append(datagram['absorption_coefficient'])
            heave = self.heave
            heave.append(datagram['heave'])
            pitch = self.pitch
            pitch.append(datagram['pitch'])
            roll = self.roll
            roll.append(datagram['roll'])
            temperature = self.temperature
            temperature.append(datagram['temperature'])
            heading = self.heading
            heading.append(datagram['heading'])
            transmit_mode = self.transmit_mode
            transmit_mode.append(datagram['transmit_mode'])
            #self.sample_offset.append(datagram['sample_offset'])
            #self.sample_count.append(datagram['sample_count'])
        except KeyError as err:
            #TODO Add filename and ping num in file.
            log.warning("The key, %s, wasn't found in the sample datagram.  This datagram will not be included.", err)
        else:
            self.ping_number.append(self.n_pings + 1)
            self.n_pings += 1
            self.ping_time = ping_time.copy()
            self.transducer_depth = transducer_depth.copy()
            self.frequency = frequency.copy()
            self.transmit_power = transmit_power.copy()
            self.pulse_length = pulse_length.copy()
            self.bandwidth = bandwidth.copy()
            self.sample_interval = sample_interval.copy()
            self.sound_velocity = sound_velocity.copy()
            self.absorption_coefficient = absorption_coefficient.copy()
            self.heave = heave.copy()
            self.pitch = pitch.copy()
            self.roll = roll.copy()
            self.temperature = temperature.copy()
            self.heading = heading.copy()
            self.transmit_mode = transmit_mode.copy()




    def append_raw_data(self, rawdata_object):
        '''
        append_raw_data would append another RawData object to this one. This would call
        insert_raw_data specifying the end ping number
        '''

        pass


    def delete_pings(self, remove=True, **kwargs):
        '''
        delete_pings deletes ping data defined by the start and end bounds.

        If remove == True, the arrays are shrunk. If remove == False, the data
        defined by the start and end are set to NaN
        '''

        #  get the horizontal start and end indicies
        h_index = self.get_indices(**kwargs)


    def get_data(self):
        for attribute, data in self:
            if attribute in self.data_attributes:
                yield (attribute, data)


    def insert_raw_data(self, raw_data_obj, ping_number=None, ping_time=None):
        '''
        insert_raw_data would insert the contents of another RawData object at the specified location
        into this one

        the location should be specified by either ping_number or ping_time

        '''
        if ping_number is None and ping_time is None:
            raise ValueError('Either ping_number or ping_time needs to be defined.')

        idx = self.get_index(time=ping_time, ping=ping_number)
        if idx <= self.n_pings - 1:
            for attribute, data in self.get_data():
                #TODO Do we want to make sure all data arrays are the same length?
                #TODO Do we want to wait to add the data until we've made sure all the data is good to avoid misalignment?
                try:
                    data_to_insert = getattr(raw_data_obj, attribute)
                except Exception as err:
                    log.error('Error reading data from raw_data_obj, ', raw_data_obj, attribute, ': ',  type(err), err)
                    return
                data_before_insert = data[0:idx] #Data up to index before idx.
                data_after_insert = data[idx:]


                if isinstance(data, list):
                    new_data = data_before_insert + data_to_insert + data_after_insert
                    setattr(self, attribute, new_data)
                elif isinstance(data, np.ndarray):
                    new_data = np.concatenate((data_before_insert, data_to_insert, data_after_insert))
                    setattr(self, attribute, new_data)


    def trim(self):
        '''
        trim deletes the empty portions of pre-allocated arrays. This should be called
        when you are done adding pings to a non-rolling raw_data instance.
        '''
        n_samples = self.power.shape[1]
        self._resize_arrays(self.n_pings, n_samples, self.n_pings, n_samples)


    def get_index(self, time=None, ping=None):

        def nearest_idx(list, value):
            '''
            return the index of the nearest value in a list.
            Adapted from: https://stackoverflow.com/questions/32237862/find-the-closest-date-to-a-given-date
            '''
            return list.index(min(list, key=lambda x: abs(x - value)))

        #  check if we have an start time defined and determine index
        if (ping == None and time == None):
            index = 0
        elif (ping == None):
            #  start must be defined by time
            #  make sure we've been passed a datetime object defining the start time
            if (not type(time) is datetime.datetime):
                raise TypeError('time must be a datetime object.')

            #  and find the index of the closest ping_time
            index = nearest_idx(self.ping_time, time)
        else:
            #  ping must have been provided
            #  make sure we've been passed an integer defining the start ping
            ping = int(ping)
            if (not type(ping) is int):
                raise TypeError('ping must be an Integer.')

            #  and find the index of the closest ping_number
            index = nearest_idx(self.ping_number, ping)

        return (index)


    def get_indices(self, start_ping=None, end_ping=None, start_time=None, end_time=None):
        '''
        get_indices maps ping number and/or ping time to an index into the acoustic
        data arrays.

        This should be extended to handle sample_start/sample_end, range_start/range_end
        but this would require calculating range if range was provided. Not a big deal,
        but there will need to be some mechanics to determine if range has been calculated
        and if it is still valid (i.e. no data has changed that would null the cached range
        data)

        '''

        def nearest_idx(list, value):
            '''
            return the index of the nearest value in a list.
            Adapted from: https://stackoverflow.com/questions/32237862/find-the-closest-date-to-a-given-date
            '''
            return list.index(min(list, key=lambda x: abs(x - value)))

        #  check if we have an start time defined and determine index
        if (start_ping == None and start_time == None):
            start_index = 0
        elif (start_ping == None):
            #  start must be defined by start_time
            #  make sure we've been passed a datetime object defining the start time
            if (not type(start_time) is datetime.datetime):
                raise TypeError('start_time must be a datetime object.')

            #  and find the index of the closest ping_time
            start_index = nearest_idx(self.ping_time, start_time)
        else:
            #  start_ping must have been provided
            #  make sure we've been passed an integer defining the start ping
            start_ping = int(start_ping)
            if (not type(end_ping) is int):
                raise TypeError('start_ping must be an Integer.')

            #  and find the index of the closest ping_number
            start_index = nearest_idx(self.ping_number, start_ping)

        #  check if we have an end time defined and determine index
        if (end_ping == None and end_time == None):
            end_index = -1
        elif (end_ping == None):
            #  start must be defined by end_time
            #  make sure we've been passed a datetime object defining the end time
            if (not type(end_time) is datetime.datetime):
                raise TypeError('end_time must be a datetime object.')

            #  and find the index of the closest ping_time
            end_index = nearest_idx(self.ping_time, end_time)
        else:
            #  end_ping must have been provided
            #  make sure we've been passed an integer defining the end ping
            end_ping = int(end_ping)
            if (not type(end_ping) is int):
                raise TypeError('end_ping must be an Integer.')

            #  and find the index of the closest ping_number
            end_index = nearest_idx(self.ping_number, end_ping)

        #  make sure the indices are sane
        if (start_index > end_index):
            raise ValueError('The end_ping or end_time provided comes before ' +
                    'the start_ping or start_time.')

        return (start_index, end_index)


    def get_sv(self, cal_parameters=None, linear=False, **kwargs):
        '''
        get_sv returns a ProcessedData object containing Sv (or sv if linear is
        True).

        MATLAB readEKRaw eq: readEKRaw_Power2Sv.m

        The value passed to cal_parameters is a calibration parameters object.
        If cal_parameters == None, the calibration parameters will be extracted
        from the corresponding fields in the raw_data object.

        '''

        #  get the horizontal start and end indicies
        h_index = self.get_indices(**kwargs)


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


    def get_power(self, **kwargs):
        '''
        get_power returns a processed data object that contains the power data.

        This method will vertically resample the raw power data according to the keyword inputs.
        By default we will resample to the highest resolution (shortest pulse length) in the object.

        resample = RawData.to_shortest

        '''


    def get_electrical_angles(self, **kwargs):
        '''
        '''


    def __iter__(self):
        for attribute in vars(self).keys():
            yield (attribute, getattr(self, attribute))


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
        self.ping_time.append(self.ping_time.pop(0))
        self.channel_metadata.append(self.channel_metadata.pop(0))

        #  roll the numpy arrays
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

        #  resize the arrays
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
        if (self.store_power):
            self.power.resize((new_ping_dim, new_sample_dim))
        if (self.store_angles):
            self.angles_alongship_e.resize((new_ping_dim, new_sample_dim))
            self.angles_athwartship_e.resize((new_ping_dim, new_sample_dim))

        #  check if we need to pad the existing data - if the new sample dimension is greater than
        #  the old, we must pad the new array values for all of the old samples with NaNs
        if (new_sample_dim > old_sample_dim):
            #  pad the samples of the existing pings setting them to NaN
            if (self.store_power):
                self.power[0:old_ping_dim, old_sample_dim:new_sample_dim] = np.nan
            if (self.store_angles):
                self.angles_alongship_e[0:old_ping_dim, old_sample_dim:new_sample_dim] = np.nan
                self.angles_athwartship_e[0:old_ping_dim, old_sample_dim:new_sample_dim] = np.nan


    def _create_arrays(self, n_pings, n_samples, initialize=False):
        '''
        _create_arrays is an internal method that initializes the RawData data arrays.
        '''

        #  ping_time and channel_metadata are lists
        self.ping_time = []
        self.channel_metadata = []

        #  all other data properties are numpy arrays

        #  first, create uninitialized arrays
        self.ping_number = np.empty((n_pings), np.int32)
        self.transducer_depth = np.empty((n_pings), np.float32)
        self.frequency = np.empty((n_pings), np.float32)
        self.transmit_power = np.empty((n_pings), np.float32)
        self.pulse_length = np.empty((n_pings), np.uint16)
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
            self.power = np.empty((n_pings, n_samples), np.int16)
        else:
            #  create an empty array as a place holder
            self.power = np.empty((0,0), np.int16)
        if (self.store_angles):
            self.angles_alongship_e = np.empty((n_pings, n_samples), np.int8)
            self.angles_athwartship_e = np.empty((n_pings, n_samples), np.int8)
        else:
            #  create an empty arrays as place holders
            self.angles_alongship_e = np.empty((0, 0), np.int8)
            self.angles_athwartship_e = np.empty((0, 0), np.int8)

        #  check if we should initialize them (fill with NaNs)
        #  note that int types will be filled with the smallest possible value for the type
        if (initialize):

            self.ping_number.fill(np.nan)
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
            self.transmit_mode.fill(np.nan)
            self.sample_offset.fill(np.nan)
            self.sample_count.fill(np.nan)
            if (self.store_power):
                self.power.fill(np.nan)
            if (self.store_angles):
                self.angles_alongship_e.fill(np.nan)
                self.angles_athwartship_e.fill(np.nan)


    def _resample_data(self, data, pulse_length, target_pulse_length, is_power=False):
        '''
        _resample_data returns a resamples the power or angle data based on it's pulse length
        and the provided target pulse length. If is_power is True, we log transform the
        data to average in linear units (if needed).

        The funtion returns the resized array.
        '''

        #  first make sure we need to do something
        if (pulse_length == target_pulse_length):
            #  nothing to do, just return the data unchanged
            return data

        if (target_pulse_length > pulse_length):
            #  we're reducing resolution - determine the number of samples to average
            sample_reduction = int(target_pulse_length / pulse_length)

            if (is_power):
                #  convert *power* to linear units
                data = np.power(data/20.0, 10.0)

            # reduce
            data = np.mean(data.reshape(-1, sample_reduction), axis=1)

            if (is_power):
                #  convert *power* back to log units
                data = 20.0 * np.log10(data)

        else:
            #  we're increasing resolution - determine the number of samples to expand
            sample_expansion = int(pulse_length / target_pulse_length)

            #  replicate the values to fill out the higher resolution array
            data = np.repeat(data, sample_expansion)


        #  update the pulse length and sample interval values
        data['pulse_length'] = target_pulse_length
        data['sample_interval'] = target_pulse_length / self.SAMPLES_PER_PULSE

        return data



class EK60ChannelMetadata(object):
    '''
    The EK60ChannelMetadata class stores the channel configuration data as well as
    some metadata about the file. One of these is created for each channel for
    every .raw file read.

    References to instances of these objects are stored in RawfileData

    '''

    def __init__(self, file, config_datagram, survey_name, transect_name, sounder_name, version,
                start_ping, start_time):

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

    def __init__(self, file):

        self.channel_id = ''
        self.frequency = 0
        self.sound_velocity = 0.0
        self.sample_interval = 0
        self.absorption_coefficient = 0.0
        self.gain = 0.0
        self.equivalent_beam_angle = 0.0
        self.beamwidth_alongship = 0.0
        self.beamwidth_athwartship = 0.0
        self.pulse_length_table = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.gain_table  = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.sa_correction_table = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.transmit_power = 0.0
        self.pulse_length = 0.0
        self.angle_sensitivity_alongship = 0.0
        self.angle_sensitivity_athwartship = 0.0
        self.angle_offset_alongship = 0.0
        self.angle_offset_athwartship = 0.0
        self.transducer_depth = 0.0


    def from_raw_data(self, raw_data, raw_file_idx=0):
        '''
        from_raw_data populated the CalibrationParameters object's properties given
        a reference to a RawData object.

        This would query the RawFileData object specified by raw_file_idx in the
        provided RawData object (by default, using the first).
        '''

        pass


    def read_ecs_file(self, ecs_file, channel):
        '''
        read_ecs_file reads an echoview ecs file and parses out the
        parameters for a given channel.
        '''
        pass



#        #  handle intialization on our first ping
#        if (self.n_pings == 0):
#            #  assume the initial array size doesn't involve resizing
#            data_dims = [ len(sample_datagram['power']), self.chunk_width]
#
#            #  determine the target pulse length and if we need to resize our data right off the bat
#            #  note that we use self.target_pulse_length to store the pulse length of all data in the
#            #  array
#            if (self.target_pulse_length == None):
#                #  set the target_pulse_length to the pulse length of this initial ping
#                self.target_pulse_length = sample_datagram['pulse_length']
#            else:
#                #  the vertical resolution has been explicitly specified - check if we need to resize right off the bat
#                if (self.target_pulse_length != sample_datagram['pulse_length']):
#                    #  we have to resize - determine the new initial array size
#                    if (self.target_pulse_length > sample_datagram['pulse_length']):
#                        #  we're reducing resolution
#                        data_dims[0] = data_dims[0]  / int(self.target_pulse_length /
#                                sample_datagram['pulse_length'])
#                    else:
#                        #  we're increasing resolution
#                        data_dims[0] = data_dims[0]  * int(sample_datagram['pulse_length'] /
#                                self.target_pulse_length)
#
#            #  allocate the initial arrays if we're *not* using a rolling buffer
#            if (self.rolling == False):
#                #  create acoustic data arrays - no need to fill with NaNs at this point
#                self.power = np.empty(data_dims)
#                self.angles_alongship_e = np.empty(data_dims)
#                self.angles_alongship_e = np.empty(data_dims)
#
#        #  if we're not allowing pulse_length to change, make sure it hasn't
#        if (not self.allow_pulse_length_change) and (sample_datagram['pulse_length'] != self.target_pulse_length):
#            self.logger.warning('append_ping failed: pulse_length does not match existing data and ' +
#                    'allow_pulse_length_change == False')
#            return False
#
#
#        #  check if any resizing needs to be done. The data arrays can be full (all columns filled) and would then
#        #  need to be expanded horizontally. The new sample_data vector could be longer with the same pulse_length
#        #  meaning the recording range has changed so we need to expand vertically and set the new data for exitsing
#        #  pings to NaN.
#
#        #  it is also possible that the incoming power or angle array needs to be padded with NaNs if earlier pings
#        #  were recorded to a longer range.
#
#        #  lastly, it is possible that the incoming power/angle arrays need to be trimmed if we're using a rolling
#        #  buffer where the user has set a hard limit on the vertical extent of the array.
#
#
#        #  check if our pulse length has changed
#        if (self.n_pings > 0) and (sample_datagram['pulse_length'] != self.pulse_length[self.n_pings-1]):
#            if (self.allow_pulse_length_change):
#                #  here we need to change the vertical resolution of either the incoming data or the data
#                if (sample_datagram['power']):
#                    sample_datagram['power'] = resample_data(sample_datagram['power'],
#                            sample_datagram['pulse_length'], self.target_pulse_length, is_power=True)
#                if (sample_datagram['angle_alongship_e']):
#                    sample_datagram['angle_alongship_e'] = resample_data(sample_datagram['angle_alongship_e'],
#                            sample_datagram['pulse_length'], self.target_pulse_length)
#                if (sample_datagram['angle_athwartship_e']):
#                    sample_datagram['angle_athwartship_e'] = resample_data(sample_datagram['angle_athwartship_e'],
#                            sample_datagram['pulse_length'], self.target_pulse_length)

'''

Just stashing some text down here for now





                    add rows. This will present itself as a datagram that
            has the same pulse_length but more samples. In this case we resize the array
            vertically. Same goes in terms of resizing in the most efficient way with one important
            difference: empty array elements of *existing pings* must be set to NaN.


            The last way the arrays will (possibly) change sizes is if the pulse_length changes.
            pulse_length directly effects the vertical "resolution". Since vetical resolution must be
            fixed within the 2d data arrays, we will deal with this in a couple of ways:

                if self.allow_pulse_length_change == False we will simply issue a warning and return
                False.

                if self.allow_pulse_length_change == True and self.target_pulse_length == None we
                will resample the data to the resolution of the first ping in our data arrays.

                if self.allow_pulse_length_change == True and self.target_pulse_length != None we
                will resample *all* of the data to the resolution specified by self.target_pulse_length.
                The value specified by target_pulse_length must be a valid pulse length.

                EK/ES60 ES70 pulse lengths in us: [256, 512, 1024, 2048, 4096]
                there are always 4 samples per pulse in time
                sample resolution in us by pulse length [64, 128, 256, 512, 1024]


                   ####  pulse_length units are seconds in the raw data ####
'''
