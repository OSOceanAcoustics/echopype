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



    #import pynmea2 FIXME Uncomment once this is available.


from collections import defaultdict
from datetime import datetime
import numpy as np

from instruments.util.pynmea2 import NMEASentence

class NMEAData(object):
    '''
    The nmea_data class provides storage for and parsing of NMEA data commonly
    collected along with sonar data.

    Potential library to use for NMEA parsing.
        https://github.com/Knio/pynmea2

        We can just pull something like that into this project. It doesn't have
        to be this one, and we could roll our own if needed. Just throwing it
        out there.
    '''


    def __init__(self):

        #  store the raw NMEA datagrams by time to facilitate easier writing
        #  raw_datagrams is a list of dicts in the form {'time':0, 'text':''}
        #  where time is the datagram time and text is the unparsed NMEA text.
        self.raw_datagrams = np.empty(0, dtype=object)
        self.n_raw = 0

        #  type_data is a dict keyed by datagram talker+message. Each element of
        #  the dict is a list of integers that are an index into the raw_datagrams
        #  list for that talker+message. This allows easy access to the datagrams
        #  by type.  time_index is designed in the same way.
        self.type_index = defaultdict(list)
        self.time_index = defaultdict(list)
        self.nmea_talker_index = defaultdict(list)
        self.nmea_type_index = defaultdict(list)

        #  self types is a list of the unique talker+message NMEA types received.
        self.types = []
        self.nmea_talkers = []
        self.nmea_types = []


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
        header = text[1:6].upper()

        #  make sure we have a plausible header
        if header.isalpha():

            #  add the raw NMEA datagram
            self.raw_datagrams = np.append(self.raw_datagrams, {'time':np.datetime64(time), 'text':str(text)}) #pynmea2 needed string to parse.
            cur_index = len(self.raw_datagrams) - 1
            cur_index = np.dtype([(str(cur_index), np.string_)]) 
    
            self.time_index[time].append(cur_index)
    
            nmea_talker = str(text[1:3].upper())
            nmea_talker = np.dtype([(str(nmea_talker), np.string_, 'S2')]) 
            self.nmea_talker_index[nmea_talker].append(cur_index)
            self.nmea_talkers = self.nmea_talker_index.keys()
    
            nmea_type = str(text[3:6].upper())
            nmea_type = np.dtype([(str(nmea_type), np.string_, 'S3')])
            self.nmea_type_index[nmea_type].append(cur_index)
            self.nmea_types = self.nmea_type_index.keys()
    
            header = np.dtype([(str(header), np.string_, 'S3')])
            self.type_index[header].append(cur_index)
            self.types = self.type_index.keys()

        else:
            #  inform the user of a bad NMEA datagram
            self.logger.info('Malformed or missing NMEA header: ' + text)

        #  increment the index counter
        self.n_raw = self.n_raw + 1


    def get_datagrams(self, type, raw=False):
        '''
        get_datagrams returns a list of the requested datagram type. By default the
        datagram will be parsed. If raw == True the raw datagram text will be returned.
        '''

        #  make sure the type is upper case
        type = type.upper()

        #  create the return dict depending on if were returning raw or parsed
        if (raw):
            datagrams = {'type':type, 'times':[], 'text':[]}
        else:
            datagrams = {'type':type, 'times':[], 'datagram':[]}


        if (type in self.types):
            #  append the time
            datagrams['times'].append(self.raw_datagrams[type]['time'])
            if (raw):
                for dg in self.type_index[type]:
                    #  just append the raw text
                    datagrams['text'].append(self.raw_datagrams[type]['text'])
            else:
                for dg in self.type_index[type]:
                    #  parse the NMEA string using pynmea2
                    nmea_obj = pynmea2.parse(self.raw_datagrams[type]['text'])
                    datagrams['datagram'].append(nmea_obj)

        #  return the dictionary
        return datagrams


    def get_interpolate(self, processed_data, nmea_talker=None, nmea_type=None):
        if nmea_talker is not None and nmea_type is not None:
            index = np.intersect1d(self.nmea_talker_index[nmea_talker], \
                                   self.nmea_type_index[nmea_type])

        elif nmea_talker is not None:
            index = self.nmea_talker_index[nmea_talker]
        elif nmea_type is not None:
            index = self.nmea_type_index[nmea_type]
        else:
            index = range(len(self.raw_datagrams))


        #Create time, lat and lon arrays.
        lat_time = np.empty(0,  dtype='datetime64[s]')
        lon_time = np.empty(0,  dtype='datetime64[s]')
        #TODO Set data types to work with pynmea2 and interp.
        lat = np.empty(1, dtype='float32')
        lon = np.empty(1, dtype='float32')
        for record in self.raw_datagrams[index]:
            if 'text' in record and isinstance(record['text'], str):
                sentence_data = NMEASentence.parse(record['text'])
                if 'time' in record: 
                    if hasattr(sentence_data, 'lat'):
                        #lat = np.append(lat, sentence_data.lat)
                        lat = np.append(lat,  np.fromstring(sentence_data.lat, dtype=float, sep=' '))
                        lat_time = np.append(lat_time, record['time'])
                    if hasattr(sentence_data, 'lon'):
                        lon = np.append(lon, sentence_data.lon)
                        lon_time = np.append(lon_time, record['time'])


        ##Get interpolated data.
        ping_times = processed_data.ping_time

        print("type(lat)", type(lat[0]))
        print("type(ping_times[0]", type(ping_times[0]))
            
        
        interpolated_lat = np.empty(1, dtype='float32')
        #for timestamp in ping_times:
        #    np.datetime64(timestamp).astype(datetime)
        #    f = np.interp(self.to_float(timestamp), map(to_float, lat_time), lat)
        #    interpolated_lat.append(f)
        #print("interpolated_lat", interpolated_lat)


    def to_float(self, d, epoch=None):
        return (d - epoch).total_seconds()
