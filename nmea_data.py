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
#  NOTE: echolab2 uses a modified version of pynmea2 that includes some
#  minor bug fixes and differences of opinion in terms of the data types
#  returned when parsing certain datagrams.
from . import pynmea2

class nmea_data(object):
    '''
    The nmea_data class provides storage for and parsing of NMEA data commonly
    collected along with sonar data.
    '''

    CHUNK_SIZE = 500

    def __init__(self):

        # Create a counter to keep track of the number of datagrams, This is
        # used to inform the array sizes.
        self.n_raw = 0

        # Create arrays to store raw NMEA data as well as times, talkers,
        # and message IDs.
        self.raw_datagrams = np.empty(nmea_data.CHUNK_SIZE, dtype=object)
        self.nmea_times = np.empty(nmea_data.CHUNK_SIZE, dtype='datetime64[ms]')
        self.talkers = np.empty(nmea_data.CHUNK_SIZE, dtype='U2')
        self.messages = np.empty(nmea_data.CHUNK_SIZE, dtype='U3')

        # Create a couple of lists to store the unique talkers and message IDs.
        self.talker_ids = []
        self.message_ids = []

        # nmea_definitions define the NMEA message(s) and pynmea2.NMEASentence
        # attributes of those messages that the NMEA interpolation routine
        # will process. These definitions can also be used to define meta-types
        # which are data types that can be contained within multiple message
        # types and allow the user to request the meta-type without knowing
        # ahead of time if specific messages are contained within the data.
        self.nmea_definitions = {}

        self.nmea_definitions['GGA'] = {'message': ['GGA'],
                                        'fields': ['latitude', 'longitude']}
        self.nmea_definitions['GLL'] = {'message': ['GLL'],
                                        'fields': ['latitude', 'longitude']}
        self.nmea_definitions['RMC'] = {'message': ['RMC'],
                                        'fields': ['latitude', 'longitude']}
        self.nmea_definitions['HDT'] = {'message': ['HDT'],
                                        'fields': ['heading_true']}
        self.nmea_definitions['VTG'] = {'message': ['VTG'],
                                        'fields': ['true_track',
                                                   'spd_over_grnd_kts']}
        # Meta-types
        #
        #  Define a "position" meta-type that extracts lat/lon data from GGA, RMC
        #  and/or GLL datagrams if they are present.
        self.nmea_definitions['position'] = {'message': ['GGA','RMC', 'GLL'],
                                             'fields': ['latitude',
                                                        'longitude']}
        #  for these meta-types we don't have multiple messages and are simply
        #  using the meta-type to define an easy to remember label like "speed"
        #  or "distance" that you can use instead of 'VTG' or 'VLW'.
        self.nmea_definitions['speed']    = {'message': ['VTG'],
                                             'fields': ['spd_over_grnd_kts']}
        self.nmea_definitions['attitude'] = {'message': ['SHR'],
                                             'fields': ['heave', 'pitch',
                                                        'roll']}
        self.nmea_definitions['distance'] = {'message':['VLW'],
                                             'fields': ['trip_distance_nmi']}


    def add_datagram(self, time, text, allow_duplicates=False):
        """
        Add NMEA datagrams to this object.

        add_datagram adds a NMEA datagram to the class. It adds it to the
        raw_datagram list as well as parsing the header and adding the
        talker + mesage ID to the type_index dictionary.
        Args:
            time (datetime64): Timestamp of NMEA datagram (this is likely
                different than the timestamp within the NMEA string itself).
            text (str): The raw NMEA string.
            allow_duplicates (bool): When False, NMEA datagrams that share
                the same timestamp, talker ID, and message ID with an
                existing datagram will be discarded.

        """
        # Parse the NMEA message header
        header = str(text[1:6].upper())

        # Verify we have a plausible header and then process.
        if header.isalpha() and len(header) == 5:

            #  check if we're allowing duplicates and if this is one. We need
            #  to do this since .out files can contain duplicate NMEA data.
            if (not allow_duplicates):
                dup_idx = (self.nmea_times == time)
                if (np.any(dup_idx)):
                    #  We have a time match - check the talker and message id
                    my_talker = self.talkers[dup_idx]
                    my_message = self.messages[dup_idx]
                    if ((header[0:2] in my_talker) and (header[2:6] in my_message)):
                        #  this is the same - discard it
                        return

            # Increment datagram counter.
            self.n_raw += 1

            # Check if we need to resize our arrays. If so, resize arrays.
            if self.n_raw > self.nmea_times.shape[0]:
                self._resize_arrays(self.nmea_times.shape[0] +
                                    nmea_data.CHUNK_SIZE)

            # Add this datagram and associated data to our data arrays and
            # then Add the talker and message ID to our list of unique talkers
            # and messages.
            self.raw_datagrams[self.n_raw-1] = text
            self.nmea_times[self.n_raw-1] = time
            self.talkers[self.n_raw-1] = header[0:2]
            self.messages[self.n_raw-1] = header[2:6]

            if not header[0:2] in self.talker_ids:
                self.talker_ids.append(header[0:2])
            if not header[2:5] in self.message_ids:
                self.message_ids.append(header[2:5])


    def get_datagrams(self, message_types, start_time=None, end_time=None,
                      talker_id=None, return_raw=False, return_fields=None):
        """
        Get a dictionary of raw or parsed NMEA data keyed by requested message
        type(s).

        get_datagrams returns a dictionary keyed by the requested datagram
        type(s) containing the raw or parsed NMEA datagrams and their receive
        times. By default the datagrams will be parsed using the pynema2
        library. If raw == True the raw datagram text will be returned.

        Args:
            message_types (list): List of NMEA-0183 message types (e.g.
                'GGA', 'GLL', 'RMC', 'HDT').
            start_time (datetime or datetime64): Define the starting time of
                the data to return. If None, data are returned starting with
                first time.
            end_time (datetime or datetime64): Define the ending time of
                the data to return. If None, return through last time.
            talker_id (str): Set to a specific prefix to limit data to a
                specific talker. For example, you could set it to "IN" to
                only get data from a POS-MV system. When set to None,
                the talker ID is ignored.
            return_raw (bool): Set to True to return raw strings, un-parsed by
                the pynema2 package. If false, data is returned parsed.
            return_fields (list): List of attribute names to extract from the
                parsed NMEA data. It is primarily intended to be used
                internally, but could be useful externally in certain
                circumstances. If this keyword is set, the return dictionary
                will contain data from those attributes as a numpy array
                keyed by the field name. For example, if return_fields = [
                'latitude', 'longitude'] then the dict returned by this
                method would be in the form:
                        {times:[numpy datetime64 array of NMEA datagram times],
                        latitude:[numpy float array of latitude values],
                        longitude:[numpy float array of longitude values]}

                The one major limitation with this is that currently it only
                returns numerical types. This can be extended to handle all
                types by parsing one of the datagrams and checking the types
                of the requested fields.

        Returns: Dictionary or raw or parsed NMEA data containing one or more
            attributes depending on specified parameters.

        """
        # Create the return dict.
        datagrams = {}

        # If we're provided a talker ID - ensure it is uppercase.
        if talker_id:
            talker_id = talker_id.upper()

        #  Make sure the message_type is a list.
        if isinstance(message_types, str):
            message_types = [message_types]

        for msg_type in message_types:
            msg_type = msg_type.upper()

            # Get the index for all datagrams within the time span.
            return_idxs = self._get_indices(start_time, end_time,
                    time_order=True)

            # Build a mask based on the message type and talker ID.
            keep_mask = self.messages[return_idxs] == msg_type
            if talker_id:
                keep_mask &= self.talkers[return_idxs] == talker_id

            # Apply the mask.
            return_idxs = return_idxs[keep_mask]

            # Determine the number of items we're returning.
            n_messages = return_idxs.shape[0]

            #  Create the return dict
            if return_raw:
                # Add raw data - do not parse.
                if n_messages > 0:
                    datagrams[msg_type] = {'time':self.nmea_times[return_idxs],
                                           'raw_string': (self.raw_datagrams[
                                            return_idxs].copy())}
                else:
                    # No messages to return.
                    datagrams[msg_type] = {'time':None, 'raw_string':None}
            else:
                #  Parse data and add to dictionary.
                if n_messages > 0:
                    if return_fields:
                        # We are only returning the fields specified in
                        # return_fields. First build the return dictionary by
                        # adding time values.
                        datagrams[msg_type] = {
                                           'time': self.nmea_times[return_idxs]}

                        #  TODO: Work out setting the numpy type based on the
                        #  data type of the parsed field. Parse one message
                        # so we can get the data types of the fields
                        # msg_data = pynmea2.parse(self.raw_datagrams[
                        #                          return_idxs[0]], check=False)
                        #  Need an if block here to set the numpy type based on
                        # the python type.

                        # Set up fields to hold data.
                        for field in return_fields:
                            datagrams[msg_type][field] = np.empty(n_messages)
                                                                  # , dtype=??)

                        # Then parse the datagrams.
                        for idx in range(n_messages):
                            try:
                                # Parse this datagram...
                                msg_data = pynmea2.parse(self.raw_datagrams
                                                         [return_idxs[idx]],
                                                         check=False)
                                # and extract the requested fields.
                                for field in return_fields:
                                    try:
                                        datagrams[msg_type][field][idx] = (
                                                       getattr(msg_data, field))
                                    except:
                                        # Unknown field - return NaN for
                                        # this field.
                                        datagrams[msg_type][field][idx] = np.nan
                            except:
                                # Unable to parse datagram - return NaNs for
                                # all fields.
                                for field in return_fields:
                                    datagrams[msg_type][field][idx] = np.nan

                    else:
                        # We are returning all of the fields. Create an array
                        # to return the NMEA text data
                        msg_data = np.empty(n_messages, dtype=object)

                        # Parse the NMEA datagrams.
                        for idx in range(n_messages):
                            try:
                                msg_data[idx] = pynmea2.parse(
                                           self.raw_datagrams[return_idxs[idx]],
                                           check=False)
                            except:
                                # Return None for bad datagrams that can not
                                # be parsed.
                                msg_data[idx] = None

                        datagrams[msg_type] = {
                                'time':self.nmea_times[return_idxs],
                                'data':msg_data}
                else:
                    #  Nothing to return for this message msg_type but
                    # return_fields are specified so create keys for the
                    # return fields and set to None
                    if return_fields:
                        #
                        datagrams[msg_type] = {'time':None}
                        for field in return_fields:
                            datagrams[msg_type][field] = None
                    else:
                        datagrams[msg_type] = {'time':None, 'data':None}

        # Return the dictionary containing the requested message types.
        return datagrams


    def interpolate(self, p_data, message_type, start_time=None,
                    end_time=None, talker_id=None, interp_fields=None):
        """
        interpolate returns the requested nmea data interpolated to the ping times
        that are present in the provided processed_data object.

            p_data is a processed data object that contains the ping_time vector
                to interpolate to.
            message_type is a string containing the NMEA-0183 message type
                e.g. 'GGA', 'GLL', 'RMC', 'HDT')
            start_time is a datetime or datetime64 object defining the starting time of the data
                to return. If None, the start time is the earliest time.
            end_time is a datetime or datetime64 object defining the ending time of the data
                    to return. If None, the end time is the latest time.
            talker_id can be set to a specfic prefix to limit data to a specific talker. For
                    example, you could set it to "IN" to only get data from a POS-MV system.
                    When set to None, the talker ID is ignored.
            interp_fields can be set to a list which defines the pynema2 attributes that will
                    be interpolated. This is only required if a message_type is requested that
                    isn't pre-defined in our internal nmea_definitions dictionary.

        """

        #  Make sure the message_type is NOT a list.
        if isinstance(message_type, list):
            raise TypeError("The NMEA message type must be a string, not a list")

        #  Check if we know how to interpolate this type.
        if message_type in self.nmea_definitions.keys():
            # We know how to handle this NMEA message type - get the messages
            # and fields we're operating on.
            interp_messages = self.nmea_definitions[message_type]['message']
            interp_fields = self.nmea_definitions[message_type]['fields']
        else:
            # This message type is not pre-defined, check if we've been told
            # what fields to interpolate.
            if interp_fields:
                interp_messages = [message_type]
            else:
                # We don't know what to do this this datagram type.
                raise ValueError("The provided NMEA message type {0} is unknown"
                        " to the interpolation method.".format(message_type))

        # Extract the NMEA data we need to interpolate.
        message_data = self.get_datagrams(interp_messages,
                                          start_time=start_time,
                                          end_time=end_time,
                                          talker_id=talker_id,
                                          return_fields=interp_fields)

        # Create the dictionary to return
        out_data = {}
        for field in interp_fields:
            # Set the interpolated fields to NaNs
            out_data[field] = np.empty(p_data.ping_time.shape[0])
            out_data[field][:] = np.nan


        # Determine the message types.
        message_types = message_data.keys()

        # If we find the GLL message type we are going to move it to the end
        # of the list since it stores position data with less precision
        # compared to GGA and RMC. Putting it at the end of the list will
        # ensure that data from GLL datagrams will only be used as a last
        # resort when no other data is available.
        try:
            message_types.append(message_types.pop(message_types.index('GLL')))
        except:
            # GLL datagram not in message types.
            pass

        # Work through the message types. If a meta-type is specified we will
        # have multiple types that may or may not contain data. If a specific
        #  message type was specified we should only have one.
        for msg_type in message_types:
            # Check if we have any data for this message type and if so,
            # work through the fields we're interpolating and interpolate.
            if message_data[msg_type]['time'] is not None:
                for field in interp_fields:
                    i_field = np.interp(p_data.ping_time.astype('d'),
                            message_data[msg_type]['time'].astype('d'),
                            message_data[msg_type][field],
                            left=np.nan, right=np.nan)

                    # Since it is possible that one type does not cover the
                    # entire output range, we determine where data is missing
                    #  and attempt to fill it. First we find what we're missing
                    out_nans = np.isnan(out_data[field])

                    # ...and then we determine what data we have from this type.
                    this_nans = np.isfinite(i_field)

                    # Logical_and these to determine what to fill in the output.
                    insert_idx = np.logical_and(out_nans, this_nans)

                    # ...and fill the missing fields.
                    out_data[field][insert_idx] = i_field[insert_idx]

        # Add the processed data object ping_time array to the out_data array
        # and return.
        out_data['ping_time'] = p_data.ping_time.copy()
        return out_data

        #  Old code used to add fields to the processed data object.
        # Keeping just in case.
        #
        # Insert or update the interpolated fields as attributes in the
        # processed data object.
        # for field in interp_fields:
        #     p_data.add_attribute(field, out_data[field])


    def _get_indices(self, start_time, end_time, time_order=True):
        """
        Return index of data contained in speciofied time range.

        _get_indices returns an index array containing the indices contained
        in the range defined by the times provided. By default the indexes
        are in time order.

        Args:
            start_time is a datetime or datetime64 object defining the starting
                time of the data to return. If None, the start time is the
                earliest time.
            end_time is a datetime or datetime64 object defining the ending time
                of the data to return. If None, the end time is the latest time.
            time_order (bool): Control whether if indexes are returned in time
                order (True) or not.

        Returns: Index array containing indices of data to return.

        """

        #  Ensure that we have times to work with.
        if start_time is None:
            start_time = np.min(self.nmea_times)
        if end_time is None:
            end_time = np.max(self.nmea_times)

        # Sort time index if returning time ordered indexes.
        if time_order:
            primary_index = self.nmea_times.argsort()
        else:
            primary_index = self.nmea_times

        # Determine the indices of the data that fall within the time span
        # provided.
        mask = self.nmea_times[primary_index] >= start_time
        mask = np.logical_and(mask, self.nmea_times[primary_index] <= end_time)

        #  and return the indices that are included in the specified range
        return primary_index[mask]


    def _resize_arrays(self, new_size):
        """
        Resize arrays if needed to hold more data.

        _resize_arrays expands our data arrays and is called when said arrays
        are filled with data and more data need to be added.

        Args:
            new_size (int): New size for arrays, Since these are all 1d
            arrays the value is simply an integer.

        """

        self.nmea_times = np.resize(self.nmea_times,(new_size))
        self.raw_datagrams = np.resize(self.raw_datagrams,(new_size))
        self.talkers = np.resize(self.talkers,(new_size))
        self.messages = np.resize(self.messages,(new_size))


    def trim(self):
        """
        Trim arrays to proper size after all data are added.

        trim is called when one is done adding data to the object. It
        removes empty elements of the data arrays.
        """

        self._resize_arrays(self.n_raw)


    def __str__(self):
        """
        Reimplemented string method that provides some basic info about the
        nmea_data object.

        """

        #  print the class and address
        msg = str(self.__class__) + " at " + str(hex(id(self))) + "\n"

        #  print some more info about the nmea_data instance
        if (self.n_raw > 0):
            msg = "{0}      NMEA data start time: {1}\n".format(
                                                       msg, self.nmea_times[0])
            msg = "{0}        NMEA data end time: {1}\n".format(
                                              msg,self.nmea_times[self.n_raw-1])
            msg = "{0}  number of NMEA datagrams: {1}\n".format(msg, self.n_raw)
            msg = "{0}         unique talker IDs: {1}\n".format(
                                               msg, (','.join(self.talker_ids)))
            msg = "{0}        unique message IDs: {1}\n".format(
                                              msg, (','.join(self.message_ids)))
            #  TODO: add reporting of numbers of individual message IDs
        else:
            msg = msg + ("  nmea_data object contains no data\n")

        return msg
