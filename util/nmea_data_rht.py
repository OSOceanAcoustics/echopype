# coding=utf-8

#     National Oceanic and Atmospheric Administration (NOAA)
#     Alaskan Fisheries Science Center (AFSC)
#     Resource Assessment and Conservation Engineering (RACE)
#     Midwater Assessment and Conservation Engineering (MACE)

#  THIS SOFTWARE AND ITS DOCUMENTATION ARE CONSIDERED TO BE IN THE PUBLIC DOMAIN
#  AND THUS ARE AVAILABLE FOR UNRESTRICTED PUBLIC USE. THEY ARE FURNISHED "AS IS."
#  THE AUTHORS, THE UNITED STATES GOVERNMENT, ITS INSTRUMENTALITIES, OFFICERS,
#  EMPLOYEES, AND AGENTS MAKE NO WARRANTY, EXPRESS OR IMPLIED, AS TO THE USEFULNESS
#  OF THE SOFTWARE AND DOCUMENTATION FOR ANY PURPOSE. THEY ASSUME NO RESPONSIBILITY
#  (1) FOR THE USE OF THE SOFTWARE AND DOCUMENTATION; OR (2) TO PROVIDE TECHNICAL
#  SUPPORT TO USERS.





import numpy as np
from .pynmea2 import NMEASentence


class nmea_data(object):
    '''
    The nmea_data class provides storage for and parsing of NMEA data commonly
    collected along with sonar data.

    '''

    CHUNK_SIZE = 500


    def __init__(self):

        #  store the raw NMEA datagrams by time to facilitate easier writing
        self.raw_datagrams = np.empty(CHUNK_SIZE, dtype=object)


        #  we'll store the message time, talker ID, and message ID
        self.nmea_times = np.empty(CHUNK_SIZE, dtype='datetime64[s]')
        self.talker_id = np.empty(CHUNK_SIZE, dtype=bytes)
        self.message_id = np.empty(CHUNK_SIZE, dtype=bytes)

        self.n_raw = 0

        self.nmea_definitions = {}

        self.nmea_definitions['GGA'] = {'message':['GGA'],
                                        'fields': {'latitude':'latitude',
                                                   'longitude':'longitude'}}

        self.nmea_definitions['GLL'] = {'message':['GLL'],
                                        'fields': {'latitude':'latitude',
                                                   'longitude':'longitude'}}

        self.nmea_definitions['RMC'] = {'message':['RMC'],
                                        'fields': {'latitude':'latitude',
                                                   'longitude':'longitude'}}

        self.nmea_definitions['position'] = {'message':['GGA','GLL','RMC'],
                                             'fields': {'latitude':'latitude',
                                                        'longitude':'longitude'}}

        self.nmea_definitions['HDT'] = {'message':['HDT'],
                                        'fields': {'heading_true':'heading_true'}}


    def _resize_arrays(self, new_size):

        self.nmea_times = np.resize(self.nmea_times,(new_size))
        self.raw_datagrams = np.resize(self.raw_datagrams,(new_size))
        self.talkers = np.resize(self.talkers,(new_size))
        self.messages = np.resize(self.messages,(new_size))



    def add_datagram(self, time, text):
        '''
        add_datagram adds a NMEA datagram to the class. It adds it to the raw_datagram
        list as well as parsing the header and adding the talker+mesage ID to the
        type_index dict.

        time is a datetime object
        text is a string containing the NMEA text

        Like I said in my email, I would modify this to use numpy arrays. It will be
        faster and easier to code. You will have to add code to manage the numpy array
        sizes, resizing when needed and a "trim" method that is called when reading is
        complete. You can follow the pattern in EK60.raw_data for this.


        '''
        header = str(text[1:6].upper())

        #  make sure we have a plausible header
        if header.isalpha() and len(header) == 5:

            self.n_raw += 1

            #  check if we need to resize our arrays
            if (self.n_raw > self.nmea_times.shape[0]):
                self._resize_arrays(self.nmea_times.shape[0] + CHUNK_SIZE)

            self.raw_datagrams[self.n_raw-1] = text
            self.nmea_times[self.n_raw-1] = time
            self.talkers[self.n_raw-1] = header[0:2]
            self.messages[self.n_raw-1] = header[2:6]


    def get_datagrams(self, message_types, start_time=None, end_time=None, talker_id=None,
            return_raw=False):
        '''
        get_datagrams returns a dictionary keyed by the requested datagram type(s) containing the
        raw or parsed NMEA datagrams and their receive times. By default the datagrams will be
        parsed using the pynema2 library. If raw == True the raw datagram text will be returned.


        '''

        datagrams = {}

        #  make sure the message_type is a list
        if (isinstance(message_types, basestring)):
            message_types = [message_types]

        for type in message_types:

            #  make sure the type is upper case
            type = type.upper()

            #  get an index for all datagrams within the time span
            return_idxs = self._get_indices(start_time, end_time, time_order=True)

            #  build a mask based on the message type and talker ID
            keep_mask = self.messages[return_idxs] == type
            if (talker_id):
                keep_mask &= self.talkers[return_idxs] == talker_id

            #  apply the mask
            return_idxs = return_idxs[keep_mask]

            n_messages = return_idxs.shape[0]

            if (return_raw):

                if (n_messages > 0):
                    datagrams[type] = {'times':self.nmea_times[return_idxs],
                                'raw_strings':self.raw_datagrams[return_idxs].copy()}
                else:
                    datagrams[type] = {'times':None, 'raw_string':None}
            else:
                if (n_messages > 0):

                    #  create an array to return the NMEA text data
                    msg_data = np.empty(n_messages, dtype=object)

                    #  parse the NMEA datagrams we're returning
                    for idx in range(n_messages):
                        try:
                            msg_data[idx] = pynmea2.parse(self.raw_datagrams[return_idxs[idx]],
                                    check=False)
                        except:
                            pass

                    datagrams[type] = {'times':self.nmea_times[return_idxs],
                                      'nmea_objects':msg_data}
                else:
                    datagrams[type] = {'times':None, 'nmea_objects':None}

        #  return the dictionary
        return datagrams


    def interpolate(self, p_data, nmea_type):






    def _get_indices(self, start_time, end_time, time_order=True):
        """
        _get_indices returns an index array containing the indices contained in the range
        defined by the times provided. By default the indexes are in time order.
        """

        primary_index = self.nmea_times.argsort()
        mask = self.ping_time[primary_index] >= start_time
        mask = np.logical_and(mask, self.ping_time[primary_index] <= end_time)

        #  and return the indices that are included in the specified range
        return primary_index[mask]










    def get_interpolate(self, data_object, nmea_data_type, nmea_talker_idx_name=None, nmea_type_idx_name=None, start_time=None, end_time=None):
        '''
        params:
            data_object: data object that inherits data container, i.e., raw_data, processed_data
            nmea_data_type: nmea data type to be interpolated. i.e., lat, lon
            nmea_talker_idx_name_idx_name: nmea talker index name
            nmea_type_idx_name_idx_name: nmea type index name
            start_time: start of data to interpolate, i.e., start_time=numpy.datetime64('2010-01-10T09:00:00.000000-0700')
            end_time: end of data to interpolate, i.e., end_time=numpy.datetime64('2020-01-10T09:00:00.000000-0700')


        '''

        #TODO Add prioritization of location data based on type.
        #TODO Get this from Chuck
        #DONE Add ability to get the data by time.
        #DONE Add a param to specify, lat, lon or something else.
        #DONE? Add code to handle outliers in lat/lon values.  Use max/min values from conf file? Add percent threshold?
        #DONE make this work with both raw and processedata objects.
        #TODO Create an array with success or fail for each.
        #TODO Add a flag to the output.
        #TODO Add an alert based on threshold based on data type, lat, lon. Add values to config file.
        #TODO if this data was munged, what gets returned?  Do we want to generated a warning? something else?
        #     Generate a warning.  If over 60%.
        #DONE Add call to get_interpolate in run_checks.py


        #Get index.
#        if start_time is not None and end_time is not None:
#            start_time_in_seconds = (start_time - datetime.fromtimestamp(0)).total_seconds()
#            end_time_in_seconds = (end_time - datetime.fromtimestamp(0)).total_seconds()
#            time_index_keys = np.sort(self.time_index.keys())[start_time_in_seconds:end_time_in_seconds]
#            time_index_values = self.time_index[time_index_keys].values()
#            index = reduce(np.intersect1d(time_index_values))
#


        if nmea_talker_idx_name is not None and nmea_type_idx_name is not None:
            index = np.intersect1d(self.nmea_talker_index[nmea_talker_idx_name], \
                                   self.nmea_type_index[nmea_type_idx_name])

        elif nmea_talker_idx_name is not None:
            index = self.nmea_talker_index[nmea_talker_idx_name]
        elif nmea_type_idx_name is not None:
            index = self.nmea_type_index[nmea_type_idx_name]
        else:
            index = range(len(self.raw_datagrams))


        nmea_time = []
        nmea_data = np.empty(0, dtype='float32')

        min_threshold, max_threshold = self.get_threshold_values(nmea_data_type)

        #Create array of interpolated data.
        for record in self.raw_datagrams[index]:
            if 'text' in record and isinstance(record['text'], str):
                sentence_data = NMEASentence.parse(record['text'])
                if 'time' in record:

                    if start_time is not None and record['time'] < start_time:
                        continue
                    if end_time is not None and record['time'] > end_time:
                        continue

                    if hasattr(sentence_data, nmea_data_type):
                        update = 1

                        try:
                            nmea_data_val = np.float32(getattr(sentence_data, nmea_data_type))
                        except ValueError as e:
                            log.warning("Skipping non-numeric value in " + \
                                    str(getattr(sentence_data, nmea_data_type)) + "." + str(e))
                            update = 0
                            continue


                        if min_threshold is not None and nmea_data_val < min_threshold:
                            update = 0
                        elif max_threshold is not None and nmea_data_val > max_threshold:
                            update = 0

                        if update:
                            nmea_data = np.append(nmea_data, nmea_data_val)
                            nmea_time.append(record['time'])



        ##Get ping timestamps.
        ping_time = data_object.ping_time


        #Convert timestamps to seconds since 1970 epoch.
        ping_time_seconds = [self.timestamp_to_float(timestamp) for timestamp in ping_time]
        nmea_time_seconds = [self.timestamp_to_float(timestamp) for timestamp in nmea_time]

        #Interpolate the data.
        #FIXME  What size should then nmea data array be to run the interpolation?
        if len(nmea_data) > 0:
            interpolated_nmea_data = np.interp(ping_time_seconds, nmea_time_seconds, nmea_data)

            #Add data to input data object.
            setattr(data_object, nmea_data_type, interpolated_nmea_data)
        else:
            log.warning("No nmea data was found in the specified parameters.")

        return self.time_index
    #FIXME uncomment    return data_object


    def read_configs(self):
        #TODO Move this method.
        config = ConfigParser.ConfigParser()
        for dirpath, dirs, files in os.walk(os.curdir, os.path.expanduser("~")):
            if self.config_file in files:
                try:
                    with open(os.path.join(dirpath,self.config_file)) as source:
                        if (sys.version_info.major > 2):
                            config.read_file( source )
                        else:
                            config.readfp( source )
                except IOError:
                    config = None
                    log.warning("Could not read config file.  No nmea data thresholds have been set.")
        return config


    def get_threshold_values(self, nmea_data_type):
        min_threshold = None
        max_threshold = None

        config = self.read_configs()
        if config is not None:
            try:
                min_threshold = config.get('nmea', 'min_' + nmea_data_type)
                min_threshold = np.float32(min_threshold)
            except:
                min_threshold = None

            try:
                max_threshold = config.get('nmea', 'max_' + nmea_data_type)
                max_threshold = np.float32(max_threshold)
            except:
                max_threshold = None

        return min_threshold, max_threshold


    def timestamp_to_float(self, timestamp):
        timestamp_datetime = timestamp.astype(datetime)
        return (timestamp_datetime - datetime.fromtimestamp(0)).total_seconds()
