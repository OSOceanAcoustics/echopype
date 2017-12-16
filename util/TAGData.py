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


class TAGData(object):
    '''
    The TAGData class provides storage for the TAG0, aka annotations datagrams
    in Simrad .raw files.
    '''

    def __init__(self, file):

        #  store the annotation text as  a list of dicts in the form {'time':0, 'text':''}
        self.annotations = []



    def add_datagram(self, time, text):
        '''
        add_datagram adds a TAG0 datagram to the

        time is a datetime object
        text is a string containing the annotation text
        '''

        #  add the raw NMEA datagram
        self.annotations.append({'time':time, 'text':text})



