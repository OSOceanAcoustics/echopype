"""
Code originally developed for pyEcholab
(https://github.com/CI-CMG/pyEcholab) by NOAA AFSC.
Contains class ``NMEAData`` for storing and manipulating NMEA data.
Called by class ConvertEK60 in ``echopype/convert/ek60.py``.

| Developed by:  Zac Berkowitz <zac.berkowitz@gmail.com> under contract for
| National Oceanic and Atmospheric Administration (NOAA)
| Alaska Fisheries Science Center (AFSC)
| Midwater Assesment and Conservation Engineering Group (MACE)

TODO: fix docstring
"""


import numpy as np

class NMEAData(object):
    """The nmea_data class provides storage for and parsing of NMEA data commonly
    collected along with sonar data.
    """

    def __init__(self):

        self.CHUNK_SIZE = 500

        # Create a counter to keep track of the number of datagrams, This is
        # used to inform the array sizes.
        self.n_raw = 0

        # Create arrays to store raw NMEA data as well as times, talkers,
        # and message IDs.
        self.raw_datagrams = np.empty(self.CHUNK_SIZE, dtype=object)
        self.nmea_times = np.empty(self.CHUNK_SIZE, dtype='datetime64[ms]')
        self.talkers = np.empty(self.CHUNK_SIZE, dtype='U2')
        self.messages = np.empty(self.CHUNK_SIZE, dtype='U3')

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
                                    self.CHUNK_SIZE)

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
