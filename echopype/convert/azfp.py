import os
import numpy as np
import xarray as xr
import xml.dom.minidom
import math
from datetime import datetime as dt
from datetime import timezone
from .azfp_set_groups import SetAZFPGroups
from struct import unpack
from echopype._version import get_versions
ECHOPYPE_VERSION = get_versions()['version']
del get_versions


class ConvertAZFP:
    """Class for converting AZFP `.01A` files """

    def __init__(self, _path='', _xml_path=''):
        self.path = _path
        self.xml_path = _xml_path
        self.file_name = os.path.basename(self.path)
        self.FILE_TYPE = 64770
        self.HEADER_SIZE = 124
        self.HEADER_FORMAT = ">HHHHIHHHHHHHHHHHHHHHHHHHHHHHHHHHHHBBBBHBBBBBBBBHHHHHHHHHHHHHHHHHHHH"
        self.parameters = {
            # FILE LOADING AND AVERAGING:
            'proc_dir': 1,          # 1 will prompt for an entire directory to process
                                    # 0 will prompt to load individual files in a directory
            'data_file_name': "12022316.01A",   # "" will prompt for hourly AZFP files to load
            # "" will prompt for XML filename if no XML file exists in the directory
            'xml_file_name': "12022310.XML",
            'platform_name': "",    # Name of the platform. Fill in with actual value
            'platform_type': "subsurface mooring",   # Type of platform. Defaults to "subsurface mooring"
            'salinity': 29.6,       # Salinity in psu
            'bins_to_avg': 1,       # Number of range bins to average
            'time_to_avg': 40,      # number of time values to average
            'pressure': 60,         # in dbars (~ depth of instument in meters)
                                    # can be approximate. Used in soundspeed and absorption calc
            'hourly_avg_temp': 18,  # Default value if no AZFP temperature is found.
                                    # Used to calculate sound-speed and range
            # PLOTTING
            'plot': 1,              # Show an echogram plot for each channel
            'channel': 1,           # freq to plot #1-4, Default = 1
            'value_2_plot': 2,      # 1,2,3,4 = Counts, Sv, TS, Temperature/Tilts, default 2
            # for Sv and Ts plotting only, values with counts < NoiseFloor will set to -150,
            # can use individual values for each frequency, ex. "noise_floor: [10000,11000,11000,11500]"
            'noise_floor': 10000,   # Default = 10000
            # Instrument on the bottom looking up (range bins), 1 at surface looking down (depth bins).
            # This changes the ydir on the echogram plots only.
            'orientation': 1,       # Default = 1
            # Use tilt corrected ranges for the echogram plots
            # Will give a warning if the tilt magnitudes are unreasonable (>20 deg)
            'use_tilt_corr': 0      # Default = 0
        }
        # Adds to self.parameters the contents of the xml file
        self.loadAZFPxml()

    def loadAZFPxml(self):
        ''' Parses the AZFP  XML file '''
        def get_value_by_tag_name(tag_name, element=0):
            """Returns the value in an XML tag given the tag name and the number of occurances"""
            return px.getElementsByTagName(tag_name)[element].childNodes[0].data

        px = xml.dom.minidom.parse(self.xml_path)
        self.parameters['num_freq'] = int(get_value_by_tag_name('NumFreq'))
        self.parameters['serial_number'] = int(get_value_by_tag_name('SerialNumber'))
        self.parameters['burst_interval'] = float(get_value_by_tag_name('BurstInterval'))
        self.parameters['pings_per_burst'] = int(get_value_by_tag_name('PingsPerBurst'))
        self.parameters['average_burst_pings'] = int(get_value_by_tag_name('AverageBurstPings'))

        # Temperature coeff
        self.parameters['ka'] = float(get_value_by_tag_name('ka'))
        self.parameters['kb'] = float(get_value_by_tag_name('kb'))
        self.parameters['kc'] = float(get_value_by_tag_name('kc'))
        self.parameters['A'] = float(get_value_by_tag_name('A'))
        self.parameters['B'] = float(get_value_by_tag_name('B'))
        self.parameters['C'] = float(get_value_by_tag_name('C'))

        # tilts
        self.parameters['X_a'] = float(get_value_by_tag_name('X_a'))
        self.parameters['X_b'] = float(get_value_by_tag_name('X_b'))
        self.parameters['X_c'] = float(get_value_by_tag_name('X_c'))
        self.parameters['X_d'] = float(get_value_by_tag_name('X_d'))
        self.parameters['Y_a'] = float(get_value_by_tag_name('Y_a'))
        self.parameters['Y_b'] = float(get_value_by_tag_name('Y_b'))
        self.parameters['Y_c'] = float(get_value_by_tag_name('Y_c'))
        self.parameters['Y_d'] = float(get_value_by_tag_name('Y_d'))

        # Initializing fields for each tranducer frequency
        self.parameters['dig_rate'] = []
        self.parameters['lock_out_index'] = []
        self.parameters['gain'] = []
        self.parameters['pulse_length'] = []
        self.parameters['DS'] = []
        self.parameters['EL'] = []
        self.parameters['TVR'] = []
        self.parameters['VTX'] = []
        self.parameters['BP'] = []
        self.parameters['range_samples'] = []
        self.parameters['range_averaging_samples'] = []
        # Get parameters for each transducer frequency
        for jj in range(self.parameters['num_freq']):
            self.parameters['range_samples'].append(int(get_value_by_tag_name('RangeSamples', jj)))
            self.parameters['range_averaging_samples'].append(int(get_value_by_tag_name('RangeAveragingSamples', jj)))
            self.parameters['dig_rate'].append(float(get_value_by_tag_name('DigRate', jj)))
            self.parameters['lock_out_index'].append(float(get_value_by_tag_name('LockOutIndex', jj)))
            self.parameters['gain'].append(float(get_value_by_tag_name('Gain', jj)))
            self.parameters['pulse_length'].append(float(get_value_by_tag_name('PulseLen', jj)))
            self.parameters['DS'].append(float(get_value_by_tag_name('DS', jj)))
            self.parameters['EL'].append(float(get_value_by_tag_name('EL', jj)))
            self.parameters['TVR'].append(float(get_value_by_tag_name('TVR', jj)))
            self.parameters['VTX'].append(float(get_value_by_tag_name('VTX0', jj)))
            self.parameters['BP'].append(float(get_value_by_tag_name('BP', jj)))
        self.parameters['sensors_flag'] = float(get_value_by_tag_name('SensorsFlag'))

    @staticmethod
    def get_fields():
        '''Returns the fields contained in each header of the raw file'''
        _fields = (
            ('profile_flag', 'u2'),
            ('profile_number', 'u2'),
            ('serial_number', 'u2'),
            ('ping_status', 'u2'),
            ('burst_int', 'u4'),
            ('year', 'u2'),                 # Year
            ('month', 'u2'),                # Month
            ('day', 'u2'),                  # Day
            ('hour', 'u2'),                 # Hour
            ('minute', 'u2'),               # Minute
            ('second', 'u2'),               # Second
            ('hundredths', 'u2'),           # Hundreths of a second
            ('dig_rate', 'u2', 4),          # Digitalization rate for each channel
            ('lockout_index', 'u2', 4),     # Lockout index for each channel
            ('num_bins', 'u2', 4),          # Number of bins for each channel
            ('range_samples', 'u2', 4),     # Range ramples per bin for each channel
            ('ping_per_profile', 'u2'),     # Number of pings per profile
            ('avg_pings', 'u2'),            # Flag indicating whether the pings average in time
            ('num_acq_pings', 'u2'),        # Pings aquired in the burst
            ('ping_period', 'u2'),          # Ping period in seconds
            ('first_ping', 'u2'),
            ('last_ping', 'u2'),
            ('data_type', "u1", 4),         # Datatype for each channel 1=Avg Data (5bytes), 0=raw (2bytes)
            ('data_error', 'u2'),           # Error number is an error occurred
            ('phase', 'u1'),                # Plase number used to aquire this profile
            ('overrun', 'u1'),              # 1 if an overrun occurred
            ('num_chan', 'u1'),             # 1, 2, 3, or 4
            ('gain', 'u1', 4),              # gain channel 1-4
            ('spare_chan', 'u1'),           # spare channel
            ('pulse_length', 'u2', 4),      # Pulse length chan 1-4 uS
            ('board_num', 'u2', 4),         # The board the data came from channel 1-4
            ('frequency', 'u2', 4),         # frequency for channel 1-4 in hz
            ('sensor_flag', 'u2'),          # Flag indicating if pressure sensor or temperature sensor is availible
            ('ancillary', 'u2', 5),         # Tilt-X, Y, Battery, Pressure, Temperature
            ('ad', 'u2', 2)                 # AD channel 6 and 7
        )
        return _fields

    def _split_header(self, raw, header_unpacked, ii, Data):
        """Splits the header information into a dictionary

        Parameters
        ----------
        raw
            open binary file
        header_unpacked
            output of struct unpack of raw file
        ii
            ping number
        Data
            current unpacked data

        Returns
        -------
            True or False depending on whether the unpack succeeded
        """
        Flag = header_unpacked[0]
        if Flag != self.FILE_TYPE:
            check_eof = raw.read(1)
            if check_eof:
                print("Error: Unknown file type")
                return False
        fields = self.get_fields()
        i = 0
        for field in fields:
            if len(field) == 3:
                arr = []
                for _ in range(field[2]):
                    arr.append(header_unpacked[i])
                    i += 1
                Data[ii][field[0]] = arr
            else:
                Data[ii][field[0]] = header_unpacked[i]
                i += 1
        return True

    def _add_counts(self, raw, ii, Data):
        """Unpacks the echosounder raw data. Modifies Data in place

        Parameters
        ----------
        raw
            open binary file
        ii
            ping number
        Data
            current unpacked data.
        """
        for jj in range(Data[ii]['num_chan']):
            counts_byte_size = Data[ii]['num_bins'][jj]
            if Data[ii]['data_type'][jj]:
                if Data[ii]['avg_pings']:
                    divisor = Data[ii]['ping_per_profile'] * Data[ii]['range_samples'][jj]
                else:
                    divisor = Data[ii]['range_samples'][jj]
                ls = unpack(">" + "I" * counts_byte_size, raw.read(counts_byte_size * 4))     # Linear sum
                lso = unpack(">" + "B" * counts_byte_size, raw.read(counts_byte_size * 1))    # linear sum overflow
                v = (np.array(ls) + np.array(lso) * 4294967295) / divisor
                v = (np.log10(v) - 2.5) * (8 * 65535) * self.parameters['DS'][jj]
                v[np.isinf(v)] = 0
                Data[ii]['counts'].append(v)
            else:
                counts_chunk = raw.read(counts_byte_size * 2)
                counts_unpacked = unpack(">" + "H" * counts_byte_size, counts_chunk)
                Data[ii]['counts'].append(counts_unpacked)

    def _print_status(self, path, Data):
        """Prints message to console giving information about the raw file being parsed

        Parameters
        ----------
        path
            path to the 01A file
        Data
            current unpacked data
        """
        filename = os.path.basename(path)
        timestamp = dt(Data[0]['year'], Data[0]['month'], Data[0]['day'], Data[0]['hour'],
                       Data[0]['minute'], int(Data[0]['second'] + Data[0]['hundredths'] / 100))
        timestr = timestamp.strftime("%d-%b-%Y %H:%M:%S")
        (pathstr, name) = os.path.split(self.parameters['xml_file_name'])
        print("File: {} - Loading Profile #{} {} with xml={} Bins2Avg={} Time2Avg={} Salinity={:.2f} Pressure={:.1f}\n"
              .format(filename, Data[0]['profile_number'], timestr, name, self.parameters['bins_to_avg'],
                      self.parameters['time_to_avg'], self.parameters['salinity'], self.parameters['pressure']))

    def parse_raw(self):
        """Parses a raw AZFP file of the 01A file format"""

        """Start of computation subfunctions"""
        def compute_temp(counts):
            """Returns the temperature in celsius given from xml data and the counts from ancillary"""
            v_in = 2.5 * (counts / 65535)
            R = (self.parameters['ka'] + self.parameters['kb'] * v_in) / (self.parameters['kc'] - v_in)
            T = 1 / (self.parameters['A'] + self.parameters['B'] * (math.log(R)) +
                     self.parameters['C'] * (math.log(R) ** 3)) - 273
            return T

        def compute_avg_temp(Data, hourly_avg_temp):
            """Input the data with temperature values and averages all the temperatures

            Parameters
            ----------
            Data
                current unpacked data
            hourly_avg_temp
                xml parameter

            Returns
            -------
                the average temperature
            """
            sum = 0
            total = 0
            for ii in range(len(Data)):
                val = Data[ii]['temperature']
                if not math.isnan(val):
                    total += 1
                    sum += val
            if total == 0:
                print("**** No AZFP temperature found. Using default of {:.2f} "
                      "degC to calculate sound-speed and range\n"
                      .format(hourly_avg_temp))
                return hourly_avg_temp    # default value
            else:
                return sum / total

        def compute_tilt(N, a, b, c, d):
            return a + b * (N) + c * (N)**2 + d * (N)**3

        def compute_avg_tilt_mag(Data):
            """Calculates the average of the cosine tilt magnitudes

            Parameters
            ----------
            Data
                current unpacked data

            Returns
            -------
                number of the average of the cosine tilt magnitudes
            """
            sum = 0
            for ii in range(len(Data)):
                sum += Data[ii]['cos_tilt_mag']
            return sum / len(Data)

        def compute_ss(T, P, S):
            """Computes the sound speed

            Parameters
            ----------
            T
                Temperature
            P
                Pressure
            S
                Salinity

            Returns
            -------
                The sound speed in m/s
            """
            z = T / 10
            return (1449.05 + z * (45.7 + z * ((-5.21) + 0.23 * z)) + (1.333 + z * ((-0.126) + z * 0.009)) *
                    (S - 35.0) + (P / 1000) * (16.3 + 0.18 * (P / 1000)))

        def compute_sea_abs(T, F, P, S):
            """Computes the absorption coefficient

            Parameters
            ----------
            T
                Temperature
            F: Numpy array
                Frequency
            P
                Pressure
            S
                Salinity

            Returns
            -------
                Numpy array containing the sea absorption for each frequency in dB/m
            """

            T_k = T + 273.0
            f1 = 1320.0 * T_k * math.exp(-1700 / T_k)
            f2 = (1.55e7) * T_k * math.exp(-3052 / T_k)

            # Coefficients for absorption calculations
            k = 1 + P / 10.0
            a = (8.95e-8) * (1 + T * ((2.29e-2) - (5.08e-4) * T))
            b = (S / 35.0) * (4.88e-7) * (1 + 0.0134 * T) * (1 - 0.00103 * k + (3.7e-7) * (k * k))
            c = ((4.86e-13) * (1 + T * ((-0.042) + T * ((8.53e-4) - T * 6.23e-6))) *
                 (1 + k * (-(3.84e-4) + k * 7.57e-8)))
            F_k = F * 1000
            if S == 0:
                return c * F_k ** 2
            else:
                return ((a * f1 * (F_k ** 2)) / ((f1 * f1) + (F_k ** 2)) +
                        (b * f2 * (F_k ** 2)) / ((f2 * f2) + (F_k ** 2)) + c * (F_k ** 2))

        """ End of computation subfunctions"""

        with open(self.path, 'rb') as raw:
            ii = 0
            Data = []
            eof = False
            while not eof:
                header_chunk = raw.read(self.HEADER_SIZE)
                if header_chunk:
                    header_unpacked = unpack(self.HEADER_FORMAT, header_chunk)
                    test_dict = {}
                    Data.append(test_dict)
                    # Reading will stop if the file contains an unexpected flag
                    if self._split_header(raw, header_unpacked, ii, Data):
                        Data[ii]['counts'] = []
                        # Appends the actual 'data values' to Data
                        self._add_counts(raw, ii, Data)
                        # Preallocate array if data averaging to #values in the hourly file x number
                        # if ii == 0 and (self.parameters['bins_to_avg'] > 1 or
                        #                 self.parameters['time_to_avg'] > 1):
                        #     pavg_arr = self.get_pavg_arr(ii, Data)
                        if ii == 0:
                            # Display information about the file that was loaded in
                            self._print_status(self.file_name, Data)
                        # Compute temperature from Data[ii]['ancillary][4]
                        Data[ii]['temperature'] = compute_temp(Data[ii]['ancillary'][4])
                        # compute x tilt from Data[ii]['ancillary][0]
                        Data[ii]['tilt_x'] = compute_tilt(Data[ii]['ancillary'][0],
                                                          self.parameters['X_a'], self.parameters['X_b'],
                                                          self.parameters['X_c'], self.parameters['X_d'])
                        # Compute y tilt from Data[ii]['ancillary][1]
                        Data[ii]['tilt_y'] = compute_tilt(Data[ii]['ancillary'][1],
                                                          self.parameters['Y_a'], self.parameters['Y_b'],
                                                          self.parameters['Y_c'], self.parameters['Y_d'])
                        # Compute cos tilt magnitude from tilt x and y values
                        Data[ii]['cos_tilt_mag'] = math.cos((math.sqrt(
                                                            Data[ii]['tilt_x'] ** 2 +
                                                            Data[ii]['tilt_y'] ** 2)) * math.pi / 180)

                    else:
                        break
                else:
                    # End of file
                    eof = True
                ii += 1

        # Compute hourly average temperature for sound speed calculation
        Data[0]['hourly_avg_temp'] = compute_avg_temp(Data, self.parameters['hourly_avg_temp'])
        Data[0]['sound_speed'] = compute_ss(Data[0]['hourly_avg_temp'], self.parameters['pressure'],
                                            self.parameters['salinity'])
        Data[0]['hourly_avg_cos'] = compute_avg_tilt_mag(Data)

        # Sampling volume for bin m from eqn. 11 pg. 86 of the AZFP Operator's Manual
        m = []
        for jj in range(Data[0]['num_chan']):
            m.append(np.arange(1, len(Data[0]['counts'][jj]) - self.parameters['bins_to_avg'] + 2,
                     self.parameters['bins_to_avg']))
        m = xr.DataArray(m, coords=[('frequency', Data[0]['frequency']), ('range_bin', list(range(len(m[0]))))])
        # Create DataArrays for broadcasting on dimension frequency
        frequency = np.array(Data[0]['frequency'], dtype=np.int64)
        lockout_index = xr.DataArray(Data[0]['lockout_index'], coords=[('frequency', frequency)])
        range_samples = xr.DataArray(Data[0]['range_samples'], coords=[('frequency', frequency)])
        pulse_length = xr.DataArray(Data[0]['pulse_length'], coords=[('frequency', frequency)])
        dig_rate = xr.DataArray(Data[0]['dig_rate'], coords=[('frequency', frequency)])

        # Calculate range from soundspeed for each frequency
        Data[0]['range'] = (Data[0]['sound_speed'] * lockout_index /
                            (2 * dig_rate) + Data[0]['sound_speed'] / 4 *
                            (((2 * m - 1) * range_samples * self.parameters['bins_to_avg'] - 1) /
                            dig_rate + pulse_length / 1e6))
        # Compute absorption for each frequency
        Data[0]['sea_abs'] = compute_sea_abs(Data[0]['hourly_avg_temp'], frequency,
                                             self.parameters['pressure'], self.parameters['salinity'])

        # The max number of values that can be averaged is the number of pings
        if self.parameters['time_to_avg'] > len(Data):
            self.parameters['time_to_avg'] = len(Data)
        self.data = Data

    def get_ping_time(self):
        """Returns the ping times"""

        try:
            self.data
        except AttributeError:
            self.parse_raw()

        ping_time = [dt(d['year'], d['month'], d['day'], d['hour'], d['minute'],
                     int(d['second'] + d['hundredths'] / 100)).replace(tzinfo=timezone.utc).timestamp()
                     for d in self.data]
        return ping_time

    def raw2nc(self):
        """Save data from raw 01A format to netCDF4 .nc format
        """

        """Subfunctions to set various dictionaries"""
        def _set_toplevel_dict():
            attrs = ('Conventions', 'keywords',
                     'sonar_convention_authority', 'sonar_convention_name',
                     'sonar_convention_version', 'summary', 'title')
            vals = ('CF-1.7, SONAR-netCDF4-1.0, ACDD-1.3', 'AZFP',
                    'ICES', 'SONAR-netCDF4', '1.0',
                    '', '')
            out_dict = dict(zip(attrs, vals))
            # Date is acquired from time of first ping
            date_created = dt.utcfromtimestamp(ping_time[0]).isoformat(timespec='seconds') + 'Z'
            out_dict['date_created'] = date_created
            return out_dict

        def _set_env_dict():
            temps = [d['temperature'] for d in self.data]
            abs_val = self.data[0]['sea_abs']
            ss_val = [self.data[0]['sound_speed']] * 4           # Sound speed independent of frequency
            salinity = [self.parameters['salinity']] * 4    # Salinity independent of frequency
            pressure = [self.parameters['pressure']] * 4    # Pressure independent of frequency

            attrs = ('frequency', 'absorption_coeff', 'sound_speed', 'salinity', 'temperature', 'pressure')
            vals = (freq, abs_val, ss_val, salinity, temps, pressure)
            return dict(zip(attrs, vals))

        def _set_platform_dict():
            out_dict = dict()
            out_dict['platform_name'] = self.parameters['platform_name']
            out_dict['platform_type'] = self.parameters['platform_type']
            # water_level is set to 0 for AZFP since this is not recorded
            out_dict['water_level'] = 0
            return out_dict

        def _set_prov_dict():
            attrs = ('conversion_software_name', 'conversion_software_version', 'conversion_time')
            vals = ('echopype', ECHOPYPE_VERSION, dt.utcnow().isoformat(timespec='seconds') + 'Z')  # use UTC time
            return dict(zip(attrs, vals))

        def _set_sonar_dict():
            attrs = ('sonar_manufacturer', 'sonar_model', 'sonar_serial_number',
                     'sonar_software_name', 'sonar_software_version', 'sonar_type')
            vals = ('ASL Environmental Sciences', 'Acoustic Zooplankton Fish Profiler', self.data[0]['serial_number'],
                    'Based on AZFP Matlab Toolbox', '1.4', 'echosounder')
            return dict(zip(attrs, vals))

        def _set_beam_dict():
            def calc_sv_offset(freq, pulse_length):
                """Calculate a compensation for the effects of finite response
                times of both the recieving and transmitting parts of the transducer.
                The correction magnitude depends on the length of the transmitted pulse
                and the response time (transmission and reception) of the transducer.

                Parameters
                ----------
                freq
                    frequency in Hz
                pulse_length
                    pulse length in ms
                """
                if freq > 38000:
                    if pulse_length == 300:
                        return 1.1
                    elif pulse_length == 500:
                        return 0.8
                    elif pulse_length == 700:
                        return 0.5
                    elif pulse_length == 900:
                        return 0.3
                    elif pulse_length == 1000:
                        return 0.3
                else:
                    if pulse_length == 500:
                        return 1.1
                    elif pulse_length == 1000:
                        return 0.7

            cos_tilt_mag = [d['cos_tilt_mag'] for d in self.data]
            tilt_x_counts = [d['ancillary'][0] for d in self.data]
            tilt_y_counts = [d['ancillary'][1] for d in self.data]
            # range_samples taken from xml data
            # range_samples = [d['range_samples'] for d in self.data]
            # dig_rate of 1st ping is used
            # dig_rate = [d['dig_rate'] for d in self.data]
            dig_rate = np.array(self.data[0]['dig_rate'])
            temp_counts = [d['ancillary'][4] for d in self.data]
            tilt_x = [d['tilt_x'] for d in self.data]
            tily_y = [d['tilt_y'] for d in self.data]
            ping_time = [dt(d['year'], d['month'], d['day'], d['hour'], d['minute'],
                         int(d['second'] + d['hundredths'] / 100)).replace(tzinfo=timezone.utc).timestamp()
                         for d in self.data]

            # Initialize variables in the output xarray Dataset
            N = []
            Sv_offset = []
            # Loop over each frequency "jj"
            for jj in range(self.data[0]['num_chan']):
                # Loop over all pings for each frequency
                N.append(np.array([d['counts'][jj] for d in self.data]))
                # Calculate correction to Sv due to a non square tramsit pulse
                Sv_offset.append(calc_sv_offset(freq[jj], self.data[0]['pulse_length'][jj]))

            tdn = np.array(self.parameters['pulse_length']) / 1e6  # Convert microseconds to seconds
            range_samples = np.array(self.parameters['range_samples'])

            # Check if dig_rate and range_samples is unique within each frequency
            # TODO: develop handling for alternative (which should be very rare, if it happens at all)
            # if np.unique(dig_rate, axis=0).shape[0] == 1 & np.unique(range_samples, axis=0).shape[0] == 1:
            #     sample_int = np.unique(range_samples, axis=0) / \
            #                  np.unique(dig_rate, axis=0)  # sample interval for every ping for each channel
            sample_int = np.array(range_samples) / np.array(dig_rate)
            range_bin = np.arange(np.size(N, 2))
            ping_bin = np.arange(np.size(N, 1))
            range_averaging_samples = self.parameters['range_averaging_samples']
            # range_out = xr.DataArray(self.data[0]['range'].data, coords=[('frequency', freq), ('range_bin', range_bin)])
            # TODO: the range_out variable is from line 432-436, which implements eqn(11) from p.86 of AZFP manual
            # Let's remove this from the parsing routine, because this is not part of the convention and should go to
            # the plotting routine.
            # Try to understand eqn(11) from p.86 is doing and recreate that in your notebook based on parsed outputs.
            # Hint: this requires: sound speed, range_bin, sample_int, pulse_length, dig_rate.

            tilt_corr_range = self.data[0]['range'] * self.data[0]['hourly_avg_cos']

            beam_dict = dict()
            beam_dict['backscatter_r'] = N
            beam_dict['EBA'] = self.parameters['BP']
            beam_dict['gain_correction'] = self.parameters['gain']
            beam_dict['sample_interval'] = sample_int
            beam_dict['transmit_duration_nominal'] = tdn
            beam_dict['range'] = np.array(self.data[0]['range'].data)
            beam_dict['tilt_corr_range'] = tilt_corr_range
            beam_dict['temperature_counts'] = temp_counts
            beam_dict['tilt_x_count'] = tilt_x_counts
            beam_dict['tilt_y_count'] = tilt_y_counts
            beam_dict['tilt_x'] = tilt_x
            beam_dict['tilt_y'] = tily_y
            beam_dict['cos_tilt_mag'] = cos_tilt_mag
            beam_dict['DS'] = self.parameters['DS']
            beam_dict['EL'] = self.parameters['EL']
            beam_dict['TVR'] = self.parameters['TVR']
            beam_dict['VTX'] = self.parameters['VTX']
            beam_dict['Sv_offset'] = Sv_offset
            beam_dict['range_samples'] = range_samples
            beam_dict['range_averaging_samples'] = range_averaging_samples
            beam_dict['sea_abs'] = self.data[0]['sea_abs']
            beam_dict['frequency'] = freq
            beam_dict['ping_time'] = ping_time
            beam_dict['range_bin'] = range_bin
            beam_dict['ping_bin'] = ping_bin
            beam_dict['number_of_frequency'] = self.parameters['num_freq']
            beam_dict['number_of_pings_per_burst'] = self.parameters['pings_per_burst']
            beam_dict['average_burst_pings_flag'] = self.parameters['average_burst_pings']
            # Temperature coefficients
            beam_dict['temperature_ka'] = self.parameters['ka']
            beam_dict['temperature_kb'] = self.parameters['kb']
            beam_dict['temperature_kc'] = self.parameters['kc']
            beam_dict['temperature_A'] = self.parameters['A']
            beam_dict['temperature_B'] = self.parameters['B']
            beam_dict['temperature_C'] = self.parameters['C']
            # Tilt coefficients
            beam_dict['tilt_X_a'] = self.parameters['X_a']
            beam_dict['tilt_X_b'] = self.parameters['X_b']
            beam_dict['tilt_X_c'] = self.parameters['X_c']
            beam_dict['tilt_X_d'] = self.parameters['X_d']
            beam_dict['tilt_Y_a'] = self.parameters['Y_a']
            beam_dict['tilt_Y_b'] = self.parameters['Y_b']
            beam_dict['tilt_Y_c'] = self.parameters['Y_c']
            beam_dict['tilt_Y_d'] = self.parameters['Y_d']
            # Data averaging
            beam_dict['time_to_avg'] = self.parameters['time_to_avg']
            beam_dict['bins_to_avg'] = self.parameters['bins_to_avg']
            return beam_dict

        def _set_vendor_specific_dict():
            out_dict = {
                'ping_time': ping_time,
                'frequency': freq,
                'profile_flag': [d['profile_flag'] for d in self.data],
                'profile_number': [d['profile_number'] for d in self.data],
                'ping_status': [d['ping_status'] for d in self.data],
                'burst_interval': [d['burst_int'] for d in self.data],
                'digitization_rate': [d['dig_rate'] for d in self.data],     # Dim: frequency
                'lock_out_index': [d['lockout_index'] for d in self.data],   # Dim: frequency
                'num_bins': [d['num_bins'] for d in self.data],              # Dim: frequency
                'range_samples': [d['range_samples'] for d in self.data],    # Dim: frequency
                'ping_per_profile': [d['ping_per_profile'] for d in self.data],
                'average_pings_flag': [d['avg_pings'] for d in self.data],
                'number_of_acquired_pings': [d['num_acq_pings'] for d in self.data],
                'ping_period': [d['ping_period'] for d in self.data],
                'first_ping': [d['first_ping'] for d in self.data],
                'last_ping': [d['last_ping'] for d in self.data],
                'data_type': [d['data_type'] for d in self.data],
                'data_error': [d['data_error'] for d in self.data],
                'phase': [d['phase'] for d in self.data],
                'number_of_channels': [d['num_chan'] for d in self.data],
                'spare_channel': [d['spare_chan'] for d in self.data],
                'board_number': [d['board_num'] for d in self.data],         # Dim: frequency
                'sensor_flag': [d['sensor_flag'] for d in self.data],
                'ancillary': [d['ancillary'] for d in self.data],            # 5 values
                'ad_channels': [d['ad'] for d in self.data]                  # 2 values
            }
            ancillary_len = list(range(len(out_dict['ancillary'][0])))
            ad_len = list(range(len(out_dict['ad_channels'][0])))
            out_dict['ancillary_len'] = ancillary_len
            out_dict['ad_len'] = ad_len
            return out_dict

        try:
            self.data
        except AttributeError:
            self.parse_raw()

        filename = os.path.splitext(os.path.basename(self.path))[0]
        self.nc_path = os.path.join(os.path.split(self.path)[0], filename + '.nc')

        freq = np.array(self.data[0]['frequency']) * 1000    # Frequency in Hz
        ping_time = self.get_ping_time()

        if os.path.exists(self.nc_path):    # USED FOR TESTING
            os.remove(self.nc_path)

        if os.path.exists(self.nc_path):
            print('          ... this file has already been converted to .nc, conversion not executed.')
        else:
            vendor = 'AZFP'
            # Create SetGroups object
            grp = SetAZFPGroups(file_path=self.nc_path)
            grp.set_toplevel(_set_toplevel_dict())      # top-level group
            grp.set_env(_set_env_dict())        # environment group
            grp.set_provenance(os.path.basename(self.file_name), _set_prov_dict())    # provenance group
            grp.set_platform(_set_platform_dict())      # platform group
            grp.set_sonar(_set_sonar_dict())                    # sonar group
            grp.set_beam(_set_beam_dict())                      # beam group
            grp.set_vendor_specific(_set_vendor_specific_dict())    # AZFP Vendor specific group
