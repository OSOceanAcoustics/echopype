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



    #import pynmea2 TODO Uncomment once this is available.


from collections import defaultdict
from datetime import datetime
import numpy as np

import os
import sys
import logging
import ConfigParser
from instruments.util.pynmea2 import NMEASentence

log = logging.getLogger(__name__)

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

    #TODO Add functions that allow user to get all raw nmea data based on 
    #TODO time and based on types return nmea data object. all by default.
    #TODO Add interpolate_pings
    #TODO 


    def __init__(self):

        self.config_file = 'echolab.conf'

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
            log.info('Malformed or missing NMEA header: ' + text)

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

                        nmea_data_val = np.fromstring(getattr(sentence_data, nmea_data_type), dtype=float, sep=' ')

                        update = 1
                        if len(nmea_data_val) < 1:
                            update = 0
                        elif min_threshold is not None and nmea_data_val < min_threshold:
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

        return data_object


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
                min_threshold = np.fromstring(min_threshold, dtype=float, sep=' ')
            except:  
                min_threshold = None
                            
            try:
                max_threshold = config.get('nmea', 'max_' + nmea_data_type)
                max_threshold = np.fromstring(max_threshold, dtype=float, sep=' ')
            except: 
                max_threshold = None

        return min_threshold, max_threshold


    def timestamp_to_float(self, timestamp):
        timestamp_datetime = timestamp.astype(datetime)
        return (timestamp_datetime - datetime.fromtimestamp(0)).total_seconds()
