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


from .pynmea2 import pynmea2

class nmea_data(object):
    '''
    The nmea_data class provides storage for and parsing of NMEA data commonly
    collected along with sonar data.

    Potential library to use for NMEA parsing.
        https://github.com/Knio/pynmea2

        We can just pull something like that into this project. It doesn't have
        to be this one, and we could roll our own if needed. Just throwing it
        out there.
    '''


    def __init__(self, file):

        #  store the raw NMEA datagrams by time to facilitate easier writing
        #  raw_datagrams is a list of dicts in the form {'time':0, 'text':''}
        #  where time is the datagram time and text is the unparsed NMEA text.
        self.raw_datagrams = []
        self.n_raw = 0

        #  type_data is a dict keyed by datagram talker+message. Each element of
        #  the dict is a list of integers that are an index into the raw_datagrams
        #  list for that talker+message. This allows easy access to the datagrams
        #  by type.
        self.type_index = {}

        #  self types is a list of the unique talker+message NMEA types received.
        self.types = []


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

        #  add the raw NMEA datagram
        self.raw_datagrams.append({'time':time, 'text':text})

        #  extract the header
        header = text[1:6].upper()

        #  make sure we have a plausible header
        if (header.isalpha() and len(header) == 5):
            #  check if we already have this header
            if (header not in self.types):
                #  nope - add it
                self.types.append(header)
                self.type_index[header] = []
            #  update the type_index
            self.type_index[header].append(self.n_raw)
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


