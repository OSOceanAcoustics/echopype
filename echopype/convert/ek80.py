# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 10:22:56 2019

@author: Sven Gastauer
@licence: GPLv3 or higher
@credits: Wu-Jung Lee, Yoann Ladroit, Dezhang Chu, Martin J. Cox
@maintainer: Sven Gastauer
@status: development

TO DO:
    PULSE COMPRESSION
    ANGULAR POSITION
    WRITE TO NC
"""
import re
import os
from collections import defaultdict
from struct import unpack_from, unpack
import numpy as np
import pandas as pd
from datetime import datetime as dt
from lxml import etree
from itertools import groupby
from matplotlib.dates import date2num
from datetime import datetime, timedelta
import pytz
from set_nc_groups import SetGroups

#import other functions
from ocean_metrics import Ocean_Metrics
from echopype._version import get_versions
ECHOPYPE_VERSION = get_versions()['version']
del get_versions




class ConvertEK80(object):
    """Class for converting EK60 .raw files."""

    def __init__(self, _filename=""):
        
        self.filename = _filename #path to EK80 raw file
        
        #Constants for binary data unpacking
        self.BLOCK_SIZE = 1024 * 40 #Block size for search radius
        self.LENGTH_SIZE = 4
        self.DATAGRAM_HEADER_SIZE = 12
        
        #regex expressions to create index and read XML structure 
        self.SAMPLE_REGEX =  b'NME0|XML0|RAW3|TAG0|MRU0|FIL1'
        #self.SAMPLE_REGEX =  b'RAW3'
        self.SAMPLE_MATCHER = re.compile(self.SAMPLE_REGEX, re.DOTALL)
        
        self.FILENAME_REGEX = r'(?P<prefix>\S*)-D(?P<date>\d{1,})-T(?P<time>\d{1,})'
        self.FILENAME_MATCHER = re.compile(self.FILENAME_REGEX, re.DOTALL)

        
        #time and date conversions
        self.REF_TIME = date2num(dt(1900, 1, 1, 0, 0, 0))
        self.WINDOWS_EPOCH = dt(1601, 1, 1)
        self.NTP_EPOCH = dt(1900, 1, 1)
        self.NTP_WINDOWS_DELTA = (self.NTP_EPOCH - self.WINDOWS_EPOCH).total_seconds()
        
        # Initialize other params that will be unpacked from data
        self.config_header = None
        self.index_df = None
        self.environment = None
        self.filters = None
        self.parameters = None
        self.config_transducer = None
        #self.first_ping_metadata = None
        #self.data_times = None
        self.config_transceiver = None
        self.motion = None
        self.ping_data_dict = None
        self.tr_data_dict = None
        self.nc_path = None
        
    
    @property
    def filename(self):
        return self._filename
    
    @filename.setter
    def filename(self, p):
        pp = os.path.basename(p)
        _, ext = os.path.splitext(pp)
        if ext != '.raw':
            raise ValueError('Please specify a .raw file.')
            # print('Data file in manufacturer format, please convert first.')
            # print('To convert data, follow the steps below:')
        else:
            self._filename = p
    def read_header(self, raw):
        h_dtype = np.dtype([('lowDate','i4'),
                            ('highDate','i4'),
                            ('chID', 'S128'),
                            ('dataType','I'),
                            ('Offset','I'),
                            ('SampleCount','I')])
        dg_head = np.frombuffer(raw, dtype=h_dtype, count=1, offset=4)
        dg_head
        ntSecs = (dg_head[0][1] * 2 ** 32 + dg_head[0][0]) / 10000000
        dgTime = datetime(1601, 1, 1, 0, 0, 0) + timedelta(seconds = ntSecs)
        dg_head = list(dg_head[0])
        dg_head.append(dgTime)
        keys = ['lowDateTime','highDateTime','channelID','dataType','Offset','SampleCount','datetime']
        return dict(zip(keys,dg_head))
    
    def _windows_to_ntp(self, windows_time):
        """
        Convert a windows file timestamp into Network Time Protocol.

        :param windows_time: 100ns since Windows time epoch
        :return: timestamp into Network Time Protocol (NTP).
        """
        return windows_time / 1e7 - self.NTP_WINDOWS_DELTA

    @staticmethod
    def _build_windows_time(high_word, low_word):
        """
        Generate Windows time value from high and low date times.

        :param high_word: high word portion of the Windows datetime
        :param low_word: low word portion of the Windows datetime
        :return: time in 100ns since 1601/01/01 00:00:00 UTC
        """
        return (high_word << 32) + low_word
    
    def cmpPwrEK80(self,ping, config, ztrd = 75):
        '''
        Compute power from complex signal
        This function assumes that a config was read previously and information such as impedance and number of transducers is available
        @params ping: ping dictionary that will be updated
        @params ztrd: Integer - Nominal Impedance, set to 75 Ohms by default
        @units ztrd: Ohm
        
        
        '''
        cid = ping['cid']
        impedance = int(config[0][cid]['Impedance'])
        y = ping['comp_sig']
        nb_elem = len(y)
        y = pd.DataFrame(y)
        y = y.sum(axis = 1) / nb_elem
        power = nb_elem * ( abs(y) / (2 * np.sqrt(2) )) **2 * (( int(impedance) + ztrd ) / int(impedance) )**2 / ztrd
        ping.update({'y':y})
        ping.update({'power':power})
    
    def index_ek80(self, file_input):
        '''
        create index of datagrams
        Currently this is not used, as the file is read in blocks
        @param file_input: open file
        @returns: Dataframe with datagram - datagram type and start - starting byte
        '''
        ind = []
        for match in re.finditer(self.SAMPLE_REGEX,file_input):
            ind.append((match.group(), match.span()[0]))
        return(pd.DataFrame(ind, columns=['datagram','start']))
            
    
    def xml2d(self,e):
        
        """
        Convert an etree into a dict structure
    
        @type  e: etree.Element
        @param e: the root of the tree
        @returns: The dictionary representation of the XML tree
        """
        
        def _xml2d(e):
            kids = dict(e.attrib)
            if e.text:
                kids['__text__'] = e.text
            if e.tail:
                kids['__tail__'] = e.tail
            for k, g in groupby(e, lambda x: x.tag):
                g = [ _xml2d(x) for x in g ] 
                kids[k]=  g
            return kids
        return { e.tag : _xml2d(e) }
    
    def parse_XML(self, xml_string):
        parser = etree.XMLParser(recover=True)
        xml_info = etree.fromstring(xml_string, parser=parser)
        
        return self.xml2d(xml_info)
    
    def _get_config_transducer(self):
        self.config_transducer = defaultdict(list)
        for c in range(self.n_trans):
            self.config_transducer[c] = self.config_transceiver[0][c]
            
            if 'Version' in self.config_transducer[c]:
                del self.config_transducer[c]['Version']
            if 'Channels' in self.config_transducer[c]:
                self.config_transducer[c].update(self.config_transducer[c]['Channels'][0]['Channel'][0])
                if 'Transducer' in self.config_transducer[c]:
                    self.config_transducer[c].update(self.config_transducer[c]['Transducer'][0])
                    del self.config_transducer[c]['Transducer']
                    del self.config_transducer[c]['Channels']
                    del self.config_transducer[c]['__text__']
                    del self.config_transducer[c]['__tail__']
        


    
    def load_ek80_raw(self):
        """
        Parse EK80 raw file
        """
        #create temporary dictionaries for datagrams
        config_transducer_tmp_dict = defaultdict(list)
        config_transceiver_tmp_dict = defaultdict(list)
        config_header_tmp_dict = defaultdict(list)
        ping_tmp_dict = defaultdict(list)
        data_times = defaultdict(list)
        NMEA_tmp_dict = defaultdict(list)
        environment_tmp_dict = defaultdict(list)
        parameters_tmp_dict = defaultdict(list)
        filters_tmp_dict = defaultdict(list)
        mru_tmp_dict = defaultdict(list)
        #get index of file for ease of reading, if slow, increase block size
        #for future maybe multiprocess or gpu usage for iterative processes
        #alternatively the index could be created using index_ek80 function, which is equivalent, without block reading
        
        ind2 = []
        print("Indexing file...")
        with open(self.filename,'rb') as bin_file:
            #bin_file.seek(7)
            position = bin_file.tell()
            
            raw = bin_file.read(self.BLOCK_SIZE)
            while len(raw) > 4:
        
                for match in re.finditer(b'NME0|XML0|RAW3|TAG0|MRU0|FIL1', raw):
                    
                    if match:
                        if match.span()[0] >= 4:
                           l =  unpack('i', raw[match.span()[0]-self.LENGTH_SIZE : match.span()[0]])[0]
                           ind2.append([match.group(), position + match.span()[0],l])
                    else:
                        bin_file.seek(position + self.BLOCK_SIZE - 4)
                position = bin_file.tell()
                # Read the next block for regex search
                bin_file.seek(position - 4)
                position = position - 4
                raw = bin_file.read(self.BLOCK_SIZE)
        idx = pd.DataFrame(ind2, columns=['datagram','start','length'])
        
        #create count objects for different data types
        nmea_count = 0
        parameters_count = 0
        environment_count = 0
        ping_count = 0
        filter_count = 0
        config_count = 0
        mru_count =0 
        
        #reopen file and look fo information at the detected indices
        with open(self.filename, "rb") as bin_file:
            print("Reading datagrams...")
            for i in range(len(idx)):
                #load the part of interest into memory
                bin_file.seek(idx['start'][i])
                raw = bin_file.read(idx['length'][i])
                
                ################################################
                ##### NME0 Reader
                ################################################
                if idx['datagram'][i] == b'NME0':
                    nmea_string = raw[self.DATAGRAM_HEADER_SIZE:]
                    if len(nmea_string) > 6:
                        nmea_type = nmea_string[3:6] 
                        nmea_ori = nmea_string[1:3] #SD for sounder, Depth
                    NMEA_tmp_dict[nmea_count] = dict({'string': nmea_string,'type':nmea_type,'ori':nmea_ori})
                    nmea_count += 1
                    
                ###########
                # FIL1 - Filter data
                ###########
                elif idx['datagram'][i] == b'FIL1':
                     ncoeff = np.frombuffer(raw, dtype='u2', count=1, offset= self.DATAGRAM_HEADER_SIZE + 128)[0]
                     #print(ncoeff)
                     FIL1_dtype = np.dtype([('chID', 'S128'),('NCoeff','i2'),('DecFac','i2'),('Coeff', str(2 * ncoeff) + 'f4')])
                     f1 = np.frombuffer(raw, dtype = FIL1_dtype, count=1, offset = 4 + self.DATAGRAM_HEADER_SIZE)
                     filters_tmp_dict[filter_count] = dict({'NCoeff':ncoeff,'DecFac':f1['DecFac'][0], 'Coeff': f1['Coeff'][0]})
                     filter_count += 1
                
                ###########
                # MRU - Motion
                ###########
                elif idx['datagram'][i] == b'MRU0':
                     MRU_dtype = np.dtype([('heave','f4'),('roll','f4'),('pitch','f4'),('heading','f4')])
                     var = ['heave','roll','pitch','heading']
                     mru = np.frombuffer(raw, dtype=MRU_dtype, count=1,offset=4)
                     mru_tmp_dict[mru_count] = dict(zip(var,mru[0]))
                     mru_count += 1
                     
                #########
                # XML0 - Configs
                #########
                elif idx['datagram'][i] == b'XML0':
                    xml_string = np.frombuffer(raw, dtype = 'a'+str(idx['length'][i] - self.DATAGRAM_HEADER_SIZE), 
                                               count=1, 
                                               offset = self.DATAGRAM_HEADER_SIZE)[0]
                    xml0_dict = self.parse_XML(xml_string)
                    xmltype = list(xml0_dict.keys())[0]
                    
                    #########
                    # Environment
                    #########
                    
                    if xmltype == 'Environment':
                        environment_tmp_dict[environment_count] = xml0_dict['Environment']
                        environment_count += 1
                    
                    #########
                    # Ping parameters
                    #########
                    if xmltype == 'Parameter':
                        parameters_tmp_dict[parameters_count] = xml0_dict['Parameter']['Channel'][0]
                        parameters_count += 1
                    
                    #########
                    # Transceiver and transducer configs
                    #########   
                    if xmltype == 'Configuration':
                        config_header_tmp_dict[config_count] = xml0_dict['Configuration']['Header']
                        config_transceiver_tmp_dict[config_count] = xml0_dict['Configuration']['Transceivers'][0]['Transceiver']
                        config_transducer_tmp_dict[config_count] = xml0_dict['Configuration']['Transducers'][0]['Transducer']
                        config_count += 1
                        self.n_trans = len(xml0_dict['Configuration']['Transceivers'][0]['Transceiver'])
                        ping_count = np.repeat(0,self.n_trans)
                        self.CID = []
                        for c in range(self.n_trans):
                            self.CID.append(xml0_dict['Configuration']['Transceivers'][0]['Transceiver'][c]['Channels'][0]['Channel'][0]['ChannelID'])
                            ping_tmp_dict[c] = defaultdict(list)
                            data_times[c] = defaultdict(list)
                        
                        
                ########
                # RAW 3
                ########
                elif idx['datagram'][i] == b'RAW3':
                    ping_tmp = {}
                    #get ping header information into temporary ping dictionary
                    ping_tmp = self.read_header(raw)
                    windows_time = self._build_windows_time(ping_tmp['highDateTime'], ping_tmp['lowDateTime'])
                    ntp_time = self._windows_to_ntp(windows_time)
                    #get channel ID
                    cid = self.CID.index(ping_tmp['channelID'].decode('utf-8'))
                    ping_tmp.update({'cid':cid})
                    #get current ping number
                    ping_n = ping_count[cid]
                    ping_count[cid] += 1
                    #get sample count
                    sampleCount = ping_tmp['SampleCount']
                     
                    #convert datatype to binary and reverse
                    dTbin = format(ping_tmp['dataType'],'b')[::-1]
                    
                    #check file format for complex, and angular information to compute power
                    if dTbin[0:10][0] == format(1,'b'):
                        if dTbin[0:10][1] == format(1, 'b'):
                            #power and angular information
                            if sampleCount * 4 == idx['length'][i] - self.DATAGRAM_HEADER_SIZE - 12 - 128:
                                ping_tmp.update({'power': np.frombuffer(raw, 
                                                      dtype = str(int(sampleCount)) + 'i2', 
                                                      count=1,
                                                      offset=140)[0] * 10. * np.log10(2) / 256.})
                                angles = np.frombuffer(raw, 
                                                      dtype = str(int(2*sampleCount)) + 'i1', 
                                                      count=1,
                                                      offset=140 + 2 * sampleCount)[0].reshape(2,sampleCount)
                                ping_tmp.update({'acrossPhi': angles[0]})
                                ping_tmp.ip({'alongPhi': angles[1]})
                            else:
                                #only power information
                                 ping_tmp.update({'power': np.frombuffer(raw, 
                                                                        dtype = str(int(sampleCount)) + 'i2', 
                                                                        count=1,
                                                                        offset=140)[0] * 10. * np.log10(2) / 256. })
                    #...complex samples are available
                    else: 
                    
                        nb_cplx_per_samples =int(dTbin[7:][::-1],2)
                         
                        if dTbin[3] == format(1, 'b'):
                             fmt = 'f' #'float32'
                             ds = 4 #4 bytes
                        elif dTbin[2] == format(1, 'b'):
                             fmt = 'e' #'float16'
                             ds = 2 #2 bytes
                        if sampleCount > 0:
                            temp = np.frombuffer(raw, 
                                               dtype=np.dtype([('cplx_samples', fmt + str(ds))]), 
                                               count=int(nb_cplx_per_samples) * int(sampleCount), 
                                               offset=140)
                            temp =np.array([i[0] for i in temp]).T
                            temp = temp.reshape(int(sampleCount),-1)
                            if temp.size % nb_cplx_per_samples != 0:
                                sampleCount = 0
                            else:
                                sampleCount = temp.size / nb_cplx_per_samples
                            comp_sig = {}
                            for isig in range(int(nb_cplx_per_samples/2)):
                                comp_sig.update([('comp_sig_%i'%int(isig), 
                                                  temp[ 1 + 2 * (isig) - 1, : ] + complex(0,1) * temp[2+2*(isig)-1,:])  ]  )
                            ping_tmp.update({'comp_sig':comp_sig})
                            self.cmpPwrEK80(ping = ping_tmp, config= config_transceiver_tmp_dict)
                    
                    ping_tmp_dict[cid][ping_n] = ping_tmp  
                    data_times[cid][ping_n] = ntp_time
                    
                    #get power, y and complex np arrays in shape 
                    
                    power_data  = [defaultdict] *  self.n_trans
                    y_data  = [defaultdict] *  self.n_trans
                    comp_sig_data  = [defaultdict] *  self.n_trans
                    for c in range(self.n_trans):
                        if 'power' in ping_tmp_dict[c][0]:
                            power_data[c] = np.array([ping_tmp_dict[c][x]['power'] for x in ping_tmp_dict[c]]).squeeze()
                        if 'y' in ping_tmp_dict[c][0]:                    
                            y_data[c] = np.array([ping_tmp_dict[c][x]['y'] for x in ping_tmp_dict[c]]).squeeze()
                        if 'comp_sig' in ping_tmp_dict[c][0]:
                            comp_sig_data[c] = np.array([np.transpose(np.asarray([(v) for v in ping_tmp_dict[c][x]['comp_sig'].values()])) for x in ping_tmp_dict[c]]).squeeze()

                # Initialize other params that will be unpacked from data
        self.ping_count = ping_count
        self.config_trans = config_transducer_tmp_dict
        self.config_transceiver = config_transceiver_tmp_dict
        self._get_config_transducer()
        self.config_header = config_header_tmp_dict
        self.index_df = idx
        self.environment = environment_tmp_dict
        self.filters = filters_tmp_dict
        self.parameters = parameters_tmp_dict
        self.mru_dict =  mru_tmp_dict
        self.data_times = pd.DataFrame.from_dict(data_times)
        #self.first_ping_metadata = None
        #self.data_times = None
        self.motion = None
        self.power_data_dict = power_data
        self.y_data_dict = y_data
        self.comp_sig_data_dict = comp_sig_data
        
        self.tr_data_dict = config_transceiver_tmp_dict
        self.nc_path = None
        

    def raw2nc(self):
        
        """
        Save data from RAW to netCDF format.
        """

        # Subfunctions to set various dictionaries
        def _set_toplevel_dict():
            attrs = ('Conventions', 'keywords',
                     'sonar_convention_authority', 'sonar_convention_name',
                     'sonar_convention_version', 'summary', 'title')
            vals = ('CF-1.7, SONAR-netCDF4, ACDD-1.3', 'EK60',
                    'ICES', 'SONAR-netCDF4', '1.7',
                    '', '')
            out_dict = dict(zip(attrs, vals))
            out_dict['date_created'] = dt.strptime(re.search("D([0-9]{8})-T([0-9]{6})", self.filename).group(),'D%Y%m%d-T%H%M%S').isoformat() + 'Z'

            return out_dict

        def _set_env_dict():
            attrs = ('frequency', 'absorption_coeff', 'sound_speed','salinity','acidity','temperature')
            vals = (freq, abs_val, ss_val, sal_val, ac_val, temp_val)
            return dict(zip(attrs, vals))

        def _set_prov_dict():
            attrs = ('conversion_software_name', 'conversion_software_version', 'conversion_time')
            vals = ('echopype', ECHOPYPE_VERSION, dt.now(tz=pytz.utc).isoformat(timespec='seconds'))  # use UTC time
            return dict(zip(attrs, vals))

        def _set_sonar_dict():
            attrs = ('sonar_manufacturer', 'sonar_model', 'sonar_serial_number',
                           'sonar_software_name', 'sonar_software_version', 'sonar_type')
            vals = (self.config_header[0][0]['Copyright'], 
                    [self.config_transducer[i]['TransducerName'] for i in range(self.n_trans)], 
                    [self.config_transducer[i]['SerialNumber'] for i in range(self.n_trans)], 
                    self.config_header[0][0]['ApplicationName'], 
                    self.config_header[0][0]['Version'], 
                    'echosounder')
            return dict(zip(attrs, vals))

        def _set_platform_dict():
            out_dict = dict()
            out_dict['platform_name'] = 'platform'
            '''
            if re.search('OOI', out_dict['platform_name']):
                out_dict['platform_type'] = 'subsurface mooring'  # if OOI
            else:
            '''
            out_dict['platform_type'] = 'ship'  # default to ship
            out_dict['time'] = self.data_times  # [seconds since 1900-01-01] for xarray.to_netcdf conversion
            out_dict['pitch'] = np.array([(v['pitch']) for k, v in self.mru_dict.items()], dtype='float32').squeeze()
            out_dict['roll'] = np.array([(v['roll']) for k, v in self.mru_dict.items()], dtype='float32').squeeze()
            out_dict['heave'] = np.array([(v['heave']) for k, v in self.mru_dict.items()], dtype='float32').squeeze()
            # water_level is set to 0 for EK60 since this is not separately recorded
            # and is part of transducer_depth
            out_dict['water_level'] = np.int32(0)
            return out_dict

        def _set_beam_dict():
            beam_dict = dict()
            beam_dict['beam_mode'] = [self.config_trans[0][k]['TransducerOrientation'] for k in range(self.n_trans)]
            beam_dict['conversion_equation_t'] = 'type_3'  # type_3 is EK60 conversion
            beam_dict['ping_time'] = self.data_times   # [seconds since 1900-01-01] for xarray.to_netcdf conversion
            beam_dict['backscatter_r'] = np.array([self.power_data_dict[x] for x in self.power_data_dict])
            beam_dict['backscatter_y'] = np.array([self.y_data_dict[x] for x in self.y_data_dict])
            beam_dict['frequency'] = freq                                    # added by echopype, not in convention
            beam_dict['range_bin'] = np.arange(self.power_data_dict[1].shape[1])  # added by echopype, not in convention

            # Loop through each transducer for variables that are the same for each file
            bm_width = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            bm_dir = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            tx_pos = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            beam_dict['equivalent_beam_angle'] = np.zeros(shape=(tx_num,), dtype='float32')
            beam_dict['gain_correction'] = np.zeros(shape=(tx_num,), dtype='float32')
            beam_dict['gpt_software_version'] = [self.config_transducer[k]['TransceiverSoftwareVersion'] for k in range(self.n_trans)]
            beam_dict['channel_id'] = []
            
            #for c_seq, c in enumerate(sim.config_transducer.items()):
            #    print(c[1]['Frequency'])
            bm_width['beamwidth_receive_major'] = [self.config_transducer[k]['BeamWidthAlongship'] for k in range(self.n_trans)]
            bm_width['beamwidth_receive_minor'] = [self.config_transducer[k]['BeamWidthAthwartship'] for k in range(self.n_trans)]
            bm_width['beamwidth_transmit_major'] = [self.config_transducer[k]['BeamWidthAlongship'] for k in range(self.n_trans)]
            bm_width['beamwidth_transmit_minor'] = [self.config_transducer[k]['BeamWidthAthwartship'] for k in range(self.n_trans)]
            bm_dir['beam_direction_x'] = [self.config_trans[0][k]['TransducerAlphaX'] for k in range(self.n_trans)]
            bm_dir['beam_direction_y'] = [self.config_trans[0][k]['TransducerAlphaY'] for k in range(self.n_trans)]
            bm_dir['beam_direction_z'] = [self.config_trans[0][k]['TransducerAlphaZ'] for k in range(self.n_trans)]
            tx_pos['transducer_offset_x'] = [self.config_trans[0][k]['TransducerOffsetX'] for k in range(self.n_trans)]
            tx_pos['transducer_offset_y'] = [self.config_trans[0][k]['TransducerOffsetY'] for k in range(self.n_trans)]
            tx_pos['transducer_offset_z'] = [self.config_trans[0][k]['TransducerOffsetZ'] for k in range(self.n_trans)]
            beam_dict['equivalent_beam_angle'] = [self.config_transducer[k]['EquivalentBeamAngle'] for k in range(self.n_trans)]
            beam_dict['gain_correction'] = [self.config_transducer[k]['Gain'] for k in range(self.n_trans)]
            beam_dict['gpt_software_version'] = [self.config_transducer[k]['TransceiverSoftwareVersion'] for k in range(self.n_trans)]
            beam_dict['channel_id'] = [self.config_transducer[k]['ChannelID'] for k in range(self.n_trans)]
            
            # Loop through each transducer for variables that may vary at each ping
            # -- this rarely is the case for EK60 so we check first before saving
            param_df  = pd.DataFrame.from_dict(self.parameters, orient='index')
            
            pl_tmp = sum(param_df['PulseDuration'].groupby(param_df['ChannelID']).nunique())/self.n_trans
            pw_tmp = sum(param_df['TransmitPower'].groupby(param_df['ChannelID']).nunique())/self.n_trans
            bw_tmp = sum(param_df['PulseForm'].groupby(param_df['ChannelID']).nunique())/self.n_trans
            si_tmp = sum(param_df['SampleInterval'].groupby(param_df['ChannelID']).nunique())/self.n_trans
            if pl_tmp==1 and pw_tmp==1 and bw_tmp==1 and si_tmp==1:
                tx_sig = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
                beam_dict['sample_interval'] = np.zeros(shape=(tx_num,), dtype='float32')
                tx_sig['transmit_duration_nominal'] = np.hstack(param_df['PulseDuration'].groupby(param_df['ChannelID']).unique().tolist())
                tx_sig['transmit_power'] = np.hstack(param_df['TransmitPower'].groupby(param_df['ChannelID']).unique().tolist())
                tx_sig['transmit_bandwidth'] = -9999
                beam_dict['sample_interval'] = np.hstack(param_df['SampleInterval'].groupby(param_df['ChannelID']).unique().tolist())
            else:
                tx_sig = defaultdict(lambda: np.zeros(shape=(tx_num, ping_num), dtype='float32'))
                beam_dict['sample_interval'] = np.zeros(shape=(tx_num, ping_num), dtype='float32')
                for t_seq in range(tx_num):
                    cids = param_df['ChannelID'].unique()[t_seq]
                    tx_sig['transmit_duration_nominal'][t_seq, :] = param_df['PulseDuration'][param_df['ChannelID']==cids[t_seq]].squeeze()
                    tx_sig['transmit_power'][t_seq, :] = param_df['TransmitPower'][param_df['ChannelID']==cids[t_seq]].squeeze()
                    tx_sig['transmit_bandwidth'][t_seq, :] = 0
                    beam_dict['sample_interval'][t_seq, :] = param_df['SampleInterval'][param_df['ChannelID']==cids[t_seq]].squeeze()

            # Build other parameters
            beam_dict['non_quantitative_processing'] = np.array([0, ] * freq.size, dtype='int32')
            # -- sample_time_offset is set to 2 for EK60 data, this value is NOT from sample_data['offset']
            beam_dict['sample_time_offset'] = np.array([2, ] * freq.size, dtype='int32')

            idx = [np.argwhere(self.tr_data_dict[x + 1]['pulse_length'][0] ==
                               self.config_transducer[x]['pulse_length_table']).squeeze()
                   for x in range(len(self.config_transducer))]
            beam_dict['sa_correction'] = np.array([x['sa_correction_table'][y]
                                                   for x, y in zip(self.config_transducer.__iter__(), np.array(idx))])
            return beam_dict, bm_width, bm_dir, tx_pos, tx_sig

        # Load data from RAW file
        self.load_ek80_raw()

        # Get nc filename
        filename = os.path.splitext(os.path.basename(self.filename))[0]
        self.nc_path = os.path.join(os.path.split(self.filename)[0], filename + '.nc')
        fm = self.FILENAME_MATCHER.match(self.filename)

        # Check if nc file already exists
        # ... if yes, abort conversion and issue warning
        # ... if not, continue with conversion
        if os.path.exists(self.nc_path):
            print('          ... this file has already been converted to .nc, conversion not executed.')
        else:
            # Retrieve variables
            #make dataframes for ease of use
            trans_df  = pd.DataFrame.from_dict(self.config_transducer, orient='index')
            env_df = pd.DataFrame.from_dict(self.environment[0])

            tx_num = self.n_trans
            ping_num = self.ping_count #n pings for each trnadcuer
            freq = trans_df['Frequency'] #list of center frequencies
            #get absorption for all center frequencies
            abs_val = [Ocean_Metrics().seawater_absorption(f=f,
                       S=float(env_df['Salinity']), 
                       T = float(env_df['Temperature']),
                       D = float(env_df['Depth']), 
                       pH = float(env_df['Acidity'])) for f in trans_df['Frequency'].astype('float')/1000]
            ss_val = self.environment['SoundSpeed']
            sal_val = self.environment['Salinity']
            ac_val = self.environment['Acidity']
            temp_val = self.environment['Temperature']

            # Create SetGroups object
            grp = SetGroups(file_path=self.nc_path)
            grp.set_toplevel(_set_toplevel_dict())  # top-level group
            grp.set_env(_set_env_dict())            # environment group
            grp.set_provenance(os.path.basename(self.filename),
                               _set_prov_dict())    # provenance group
            grp.set_platform(_set_platform_dict())  # platform group
            grp.set_sonar(_set_sonar_dict())        # sonar group
            grp.set_beam(*_set_beam_dict())         # beam group
        
        