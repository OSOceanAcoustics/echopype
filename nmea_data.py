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
import pynmea2

class nmea_data(object):
    '''
    The nmea_data class provides storage for and parsing of NMEA data commonly
    collected along with sonar data.

    '''

    CHUNK_SIZE = 500


    def __init__(self):

        #  store the raw NMEA datagrams by time to facilitate easier writing
        self.raw_datagrams = np.empty(nmea_data.CHUNK_SIZE, dtype=object)


        #  we'll store the message time, talker ID, and message ID
        self.nmea_times = np.empty(nmea_data.CHUNK_SIZE, dtype='datetime64[s]')
        self.talkers = np.empty(nmea_data.CHUNK_SIZE, dtype='S2')
        self.messages = np.empty(nmea_data.CHUNK_SIZE, dtype='S3')

        self.n_raw = 0

        #  nmea_definitions define the NMEA message(s) and pynmea2.NMEASentence
        #  attributes of those messages that the NMEA interpolation routine will
        #  process. These definitions can also be used to define meta-types which
        #  are data types that can be contained within multiple message types and
        #  allow the user to request the meta-type without knowing ahead of time
        #  if specific messages are contained within the data.
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
        #  define the "position" meta-type. This meta-type covers all messages that
        #  contain latitude and longitude data.
        self.nmea_definitions['position'] = {'message':['GGA','GLL','RMC'],
                                             'fields': {'latitude':'latitude',
                                                        'longitude':'longitude'}}
        self.nmea_definitions['HDT'] = {'message':['HDT'],
                                        'fields': {'heading_true':'heading_true'}}


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
                self._resize_arrays(self.nmea_times.shape[0] + nmea_data.CHUNK_SIZE)

            self.raw_datagrams[self.n_raw-1] = text
            self.nmea_times[self.n_raw-1] = time
            self.talkers[self.n_raw-1] = header[0:2]
            self.messages[self.n_raw-1] = header[2:6]


    def get_datagrams(self, message_types, start_time=None, end_time=None, talker_id=None,
            return_raw=False, return_fields=None):
        '''
        get_datagrams returns a dictionary keyed by the requested datagram type(s) containing the
        raw or parsed NMEA datagrams and their receive times. By default the datagrams will be
        parsed using the pynema2 library. If raw == True the raw datagram text will be returned.


        '''

        #  create the return dict
        datagrams = {}

        #  if we're provided a talker ID - ensure it is uppercase
        if (talker_id):
            talker_id = talker_id.upper()

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

            #  determine the number of items we're returning
            n_messages = return_idxs.shape[0]

            #  create the return dict
            if (return_raw):
                #  we're returing raw data - do not parse
                if (n_messages > 0):
                    datagrams[type] = {'times':self.nmea_times[return_idxs],
                                       'raw_strings':self.raw_datagrams[return_idxs].copy()}
                else:
                    #  nothing to return
                    datagrams[type] = {'times':None, 'raw_string':None}
            else:
                #  we're returning parsed data
                if (n_messages > 0):
                    if (return_fields):
                        #  we're asked to return specific fields from the parsed nmea data

                        #  first build the return dict
                        datagrams[type] = {'times':self.nmea_times[return_idxs]}
                        for field in return_fields:
                            datagrams[type][field] = np.empty(n_messages)

                        #  then parse the datagrams
                        for idx in range(n_messages):
                            try:
                                #  parse this datagram
                                msg_data = pynmea2.parse(self.raw_datagrams[return_idxs[idx]],
                                        check=False)
                                #  and extract the requested fields
                                for field in return_fields:
                                    try:
                                        datagrams[type][field][idx] = getattr(msg_data, field)
                                    except:
                                        #  unknown field - return NaN for this field
                                        datagrams[type][field][idx] = np.nan
                            except:
                                #  unable to parse datagram - return NaNs for all fields
                                for field in return_fields:
                                    datagrams[type][field][idx] = np.nan

                    else:
                        #  create an array to return the NMEA text data
                        msg_data = np.empty(n_messages, dtype=object)

                        #  parse the NMEA datagrams we're returning
                        for idx in range(n_messages):
                            try:
                                msg_data[idx] = pynmea2.parse(self.raw_datagrams[return_idxs[idx]],
                                        check=False)
                            except:
                                #  return None for bad datagrams
                                msg_data[idx] = None

                        datagrams[type] = {'times':self.nmea_times[return_idxs],
                                          'nmea_objects':msg_data}
                else:
                    #  nothing to return
                    datagrams[type] = {'times':None, 'nmea_objects':None}

        #  return the dictionary containing the requested message types
        return datagrams


    def interpolate(self, p_data, message_type):
        """
        interpolate returns the requested nmea data interpolated to the ping times
        that are present in the provided processed_data object.
        """

        #  make sure the message_type is NOT a list
        if (isinstance(message_type, list)):
            raise TypeError("The NMEA message type must be a string, not a list")

        if (message_type in self.nmea_definitions.keys()):
            #  we know how to handle this NMEA message type
            pass


        else:
            raise ValueError("The provided NMEA message type " + str(message_type) +
                    " is unknown to the interpolation method.")

        pass




    def _get_indices(self, start_time, end_time, time_order=True):
        """
        _get_indices returns an index array containing the indices contained in the range
        defined by the times provided. By default the indexes are in time order.
        """

        #  ensure that we have times to work with
        if (start_time is None):
            start_time = np.min(self.nmea_times)
        if (end_time is None):
            end_time = np.max(self.nmea_times)

        #  determine the indices of the data that fall within the time span provided
        primary_index = self.nmea_times.argsort()
        mask = self.nmea_times[primary_index] >= start_time
        mask = np.logical_and(mask, self.nmea_times[primary_index] <= end_time)

        #  and return the indices that are included in the specified range
        return primary_index[mask]


    def _resize_arrays(self, new_size):
        """
        _resize_arrays expands our data arrays and is called when said arrays
        are filled with data.
        """

        self.nmea_times = np.resize(self.nmea_times,(new_size))
        self.raw_datagrams = np.resize(self.raw_datagrams,(new_size))
        self.talkers = np.resize(self.talkers,(new_size))
        self.messages = np.resize(self.messages,(new_size))


    def _trim(self):
        """
        _trim_arrays is called when one is done adding data to the object. It
        removes empty elements of the data arrays.
        """

        self.nmea_times = np.resize(self.nmea_times,(self.n_raw))
        self.raw_datagrams = np.resize(self.raw_datagrams,(self.n_raw))
        self.talkers = np.resize(self.talkers,(self.n_raw))
        self.messages = np.resize(self.messages,(self.n_raw))
