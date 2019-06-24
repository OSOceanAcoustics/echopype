import os
import re
import numpy as np
import xarray as xr
import xml.dom.minidom
import math
from datetime import datetime as dt
from datetime import timezone
from set_nc_groups import SetGroups
from struct import unpack
from echopype._version import get_versions
ECHOPYPE_VERSION = get_versions()['version']
del get_versions


path = "D:\\Documents\\Projects\\echopype\\toolbox\\12022316.01A"
xml_path = "D:\\Documents\\Projects\\echopype\\toolbox\\12022310.XML"


class ConvertAZFP:
    """Class for converting raw .01A AZFP files """

    def __init__(self, _path=""):
        self.path = _path
        self.file_name = os.path.basename(path)
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
            """Input a minidom parsed xml file, a tag name, and the number of the tag's occurances
               returns the value in the tag"""
            return px.getElementsByTagName(tag_name)[element].childNodes[0].data

        px = xml.dom.minidom.parse(xml_path)
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

    def split_header(self, raw, header_unpacked, ii, Data):
        ''' Input open raw file, header, current bin, and the Data
            adds to Data the values from header_unpacked
            returns True if successful, False otherwise'''
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

    def add_counts(self, raw, ii, Data):
        """Input open raw file, the ping number, and Data
            adds "counts" to Data for given bin. Represents most of the measured data
            Assumed to be valid past header"""
        for jj in range(Data[ii]['num_chan']):
            if Data[ii]['data_type'][jj]:
                if Data[ii]['avg_pings']:
                    divisor = Data[ii]['ping_per_profile'] * Data[ii]['range_samples'][jj]
                else:
                    divisor = Data[ii]['range_samples'][jj]
                # ls and lso unimplemented
            else:
                counts_byte_size = Data[ii]['num_bins'][jj]
                counts_chunk = raw.read(counts_byte_size * 2)
                counts_unpacked = unpack(">" + "H" * counts_byte_size, counts_chunk)
                Data[ii]['counts'].append(counts_unpacked)

        return True

    def get_pavg_arr(self, ii, Data):
        pavg_arr = []
        for jj in range(Data[ii]['num_chan']):
            pavg_arr.append([])
        return pavg_arr

    def print_status(self, path, Data):
        """Prints message to console giving information about the raw file being parsed"""
        filename = os.path.basename(path)
        timestamp = dt(Data[0]['year'], Data[0]['month'], Data[0]['day'], Data[0]['hour'],
                                Data[0]['minute'], int(Data[0]['second'] + Data[0]['hundredths'] / 100))
        timestr = timestamp.strftime("%d-%b-%Y %H:%M:%S")
        (pathstr, name) = os.path.split(self.parameters['xml_file_name'])
        print("File: {} - Loading Profile #{} {} with xml={} Bins2Avg={} Time2Avg={} Salinity={:.2f} Pressure={:.1f}\n"
              .format(filename, Data[0]['profile_number'], timestr, name, self.parameters['bins_to_avg'],
                      self.parameters['time_to_avg'], self.parameters['salinity'], self.parameters['pressure']))

    def parse_raw(self):
        """Parses a RAW AZFP file and returns the result"""

        """Start of computation subfunctions"""
        def compute_temp(counts):
            """Returns the temperature in celsius given from xml data and the counts from ancillary"""
            v_in = 2.5 * (counts / 65535)
            R = (self.parameters['ka'] + self.parameters['kb'] * v_in) / (self.parameters['kc'] - v_in)
            T = 1 / (self.parameters['A'] + self.parameters['B'] * (math.log(R)) +
                     self.parameters['C'] * (math.log(R) ** 3)) - 273
            return T

        def compute_avg_temp(Data, hourly_avg_temp):
            """Input the data with temperature values and averages all the temperatures""" 
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
            """Input the data and calculates the average of the cosine tilt magnitudes"""
            sum = 0
            for ii in range(len(Data)):
                sum += Data[ii]['cos_tilt_mag']
            return sum / len(Data)

        def compute_ss(T, P, S):
            z = T / 10
            return (1449.05 + z * (45.7 + z * ((-5.21) + 0.23 * z)) + (1.333 + z * ((-0.126) + z * 0.009)) *
                    (S - 35.0) + (P / 1000) * (16.3 + 0.18 * (P / 1000)))

        def compute_sea_abs(Data, jj, P, S):
            """Input Data, frequency number, pressure and salinity to calculate the absorption coefficient using
            the hourly average temperature, pressure, salinity, and transducer frequency"""
            T = Data[0]['hourly_avg_temp']
            F = Data[0]['frequency'][jj]

            # Calculate relaxation frequencies
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
                return c * F_k ^ 2
            else:
                # Update later: f_k might be a list
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
                    if self.split_header(raw, header_unpacked, ii, Data):
                        Data[ii]['counts'] = []
                        # Appends the actual 'data values' to Data
                        self.add_counts(raw, ii, Data)
                        # Preallocate array if data averaging to #values in the hourly file x number
                        # if ii == 0 and (self.parameters['bins_to_avg'] > 1 or
                        #                 self.parameters['time_to_avg'] > 1):
                        #     pavg_arr = self.get_pavg_arr(ii, Data)
                        if ii == 0:
                            # Display information about the file that was loaded in
                            self.print_status(path, Data)
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
                        # Compute power if the data is being averaged
                        if self.parameters['bins_to_avg'] > 1 or self.parameters['time_to_avg'] > 1:
                            pavg_arr = self.get_pavg_arr(ii, Data)
                            Data[ii]['pavg'] = []
                            for jj in range(Data[ii]['num_chan']):
                                x = self.parameters['EL'][jj] - 2.5 / self.parameters['DS'][jj]
                                # el = x + np.array(Data[ii]['counts'][jj]) / (26214 * self.parameters['DS'][jj])
                                # P = 10 ** (el / 10)
                                # if P.any:
                                #     Data[ii]['pavg'].append(P)
                                #     # Used to simplify time averaging
                                #     pavg_arr[jj].append([P])
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
        Data[0]['range'] = []       # Initializing range
        Data[0]['sea_abs'] = []     # Initializing sea absorption
        for jj in range(Data[0]['num_chan']):
            # Sampling volume for bin m from eqn. 11 pg. 86 of the AZFP Operator's Manual
            m = np.arange(1, len(Data[0]['counts'][jj]) - self.parameters['bins_to_avg'] + 2,
                          self.parameters['bins_to_avg'])
            # Calculate range from soundspeed for each frequency
            Data[0]['range'].append(Data[0]['sound_speed'] * Data[0]['lockout_index'][jj] /
                                    (2 * Data[0]['dig_rate'][jj]) + Data[0]['sound_speed'] / 4 *
                                    (((2 * m - 1) * Data[0]['range_samples'][jj] * self.parameters['bins_to_avg'] - 1) /
                                    Data[0]['dig_rate'][jj] + Data[0]['pulse_length'][jj] / 1e6))
            # Compute absorption for each frequency
            Data[0]['sea_abs'].append(compute_sea_abs(Data, jj, self.parameters["pressure"],
                                                      self.parameters["salinity"]))
            if self.parameters['time_to_avg'] > len(Data):
                self.parameters['time_to_avg'] = len(Data)

            # Bin average ----INCOMPLETE----
            if self.parameters['bins_to_avg'] > 1:
                for jj in Data[0]['num_chan']:
                    bins_to_avg = self.parameters['bins_to_avg']
                    num_bins = Data[0]['counts'][jj]
                    num_bins = len(np.arange(0, len(num_bins) - bins_to_avg + 1, bins_to_avg))

            # Time average  ----INCOMPLETE----
            if self.parameters['time_to_avg'] > 1:
                # num_time = math.floor(len(Data) / self.parameters['time_to_avg'])
                for kk in range(Data[0]['num_chan']):
                    # el_avg = 10 * math.log10(1)
                    pass
            else:  # no time averaging but may still have range averaging
                if self.parameters['bins_to_avg'] > 1:
                    pass  # INCOMPLETE
                else:
                    pass

        return Data

    def create_output(self, Data):
        """Requires a parsed raw AZFP file as input
        This method returns an xarray dataset containing N, Sv, TS, 
        range, tilt corrected range, temperature, tilt x, tilt y,
        sea absorption, frequency, ping times, number of channels,
        burst interval, hourly average temperature, and number of acquired pings"""
        def calc_sv_offset(freq, pulse_length):
            """Input the frequency in kHz and the pulse length for the
            function to calculate a compensation for the effects of finite response
            times of both the recieving and transmitting parts of the transducer.
            The correction magnitude depends on the length of the transmitted pulse
            and the response time (transmission and reception) of the transducer."""
            if freq > 38:
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

        cos_tilt_mag = [d['cos_tilt_mag'] for d in Data]
        tilt_x_counts = [d['ancillary'][0] for d in Data]
        tilt_y_counts = [d['ancillary'][1] for d in Data]
        range_samples = [d['range_samples'] for d in Data]
        dig_rate = [d['dig_rate'] for d in Data]
        temp_counts = [d['ancillary'][4] for d in Data]
        tilt_x = [d['tilt_x'] for d in Data]
        tily_y = [d['tilt_y'] for d in Data]
        date_out = [dt(d['year'], d['month'], d['day'], d['hour'], d['minute'],
                       int(d['second'] + d['hundredths'] / 100)).replace(tzinfo=timezone.utc).timestamp()
                    for d in Data]

        # Initialize variables in the output xarray Dataset
        N = []
        ts = []
        sv = []
        sv_offset = 0
        sv_offset = []
        freq = Data[0]['frequency']
        # Loop over each frequency "jj"
        for jj in range(Data[0]['num_chan']):
            # Loop over all pings for each frequency
            N.append(np.array([d['counts'][jj] for d in Data]))
            # Calculate correction to Sv due to a non square tramsit pulse
            sv_offset.append(calc_sv_offset(freq[jj], Data[0]['pulse_length'][jj]))
            ts_calc = (self.parameters['EL'][jj] - 2.5 / self.parameters['DS'][jj] +
                       N[jj] / (26214 * self.parameters['DS'][jj]) - self.parameters['TVR'][jj] -
                       20 * np.log10(self.parameters['VTX'][jj]) +
                       40 * np.log10(np.tile(Data[0]['range'][jj], (np.size(N[jj], 0), 1))) +
                       2 * Data[0]['sea_abs'][jj] * np.tile(Data[0]['range'][jj], (np.size(N[jj], 0), 1))
                       )
            sv_calc = list(self.parameters['EL'][jj] - 2.5 / self.parameters['DS'][jj] +
                           N[jj] / (26214 * self.parameters['DS'][jj]) - self.parameters['TVR'][jj] -
                           20 * np.log10(self.parameters['VTX'][jj]) +
                           20 * np.log10(np.tile(Data[0]['range'][jj], (np.size(N[jj], 0), 1))) +
                           2 * Data[0]['sea_abs'][jj] * np.tile(Data[0]['range'][jj], (np.size(N[jj], 0), 1)) -
                           10 * np.log10(0.5 * Data[0]['sound_speed'] * self.parameters['pulse_length'][jj] / 1e6 *
                           self.parameters['BP'][jj]) + sv_offset[jj]
                           )
            ts.append(ts_calc)
            sv.append(sv_calc)

        freq = np.array(freq) * 1000    # Convert to Hz from kHz
        tdn = np.array(self.parameters['pulse_length']) * 1000  # Convert microseconds to seconds
        sample_int = np.transpose(np.array(range_samples) / np.array(dig_rate))
        range_bin = list(range(np.size(N, 2)))
        ras = self.parameters['range_averaging_samples']
        rs = self.parameters['range_samples']
        range_out = xr.DataArray(np.stack(Data[0]['range']), coords=[('frequency', freq), ('range_bin', range_bin)])
        tilt_corr_range = range_out * Data[0]['hourly_avg_cos']
        output = xr.Dataset({'backscatter_r': (['frequency', 'ping_time', 'range_bin'], np.stack(N)),
                             'Sv': (['frequency', 'ping_time', 'range_bin'], np.stack(sv)),
                             'TS': (['frequency', 'ping_time', 'range_bin'], np.stack(ts)),
                             'equivalent_beam_angle': (['frequency'], self.parameters['BP']),
                             'gain_correction': (['frequency'], self.parameters['gain']),
                             'sample_interval': (['frequency', 'ping_time'], sample_int, {'units': 'seconds'}),
                             'transmit_duration_nominal': (['frequency'], tdn, {'units': 'seconds'}),
                             'range': range_out,
                             'tilt_corr_range': tilt_corr_range,
                             'temperature_counts': (['ping_time'], temp_counts),
                             'tilt_x_count': (['ping_time'], tilt_x_counts),
                             'tilt_y_count': (['ping_time'], tilt_y_counts),
                             'tilt_x': (['ping_time'], tilt_x),
                             'tilt_y': (['ping_time'], tily_y),
                             'cos_tilt_mag': (['ping_time'], cos_tilt_mag),
                             'DS': (['frequency'], self.parameters['DS']),
                             'EL': (['frequency'], self.parameters['EL']),
                             'TVR': (['frequency'], self.parameters['TVR']),
                             'VTX': (['frequency'], self.parameters['VTX']),
                             'Sv_offset': (['frequency'], sv_offset),
                             'number_of_samples_digitized_per_pings': (['frequency'], rs),
                             'number_of_digitized_samples_averaged_per_pings': (['frequency'], ras),
                             'sea_abs': (['frequency'], Data[0]['sea_abs'])},
                            coords={'frequency': (['frequency'], freq,
                                                  {'units': 'Hz',
                                                   'valid_min': 0.0}),
                                    'ping_time': (['ping_time'], date_out,
                                                  {'axis': 'T',
                                                   'calendar': 'gregorian',
                                                   'long_name': 'Timestamp of each ping',
                                                   'standard_name': 'time',
                                                   'units': 'seconds since 1970-01-01'}),
                                    'range_bin': (['range_bin'], range_bin)},
                            attrs={'beam_mode': '',
                                   'conversion_equation_t': 'type_4',
                                   'number_of_frequency': self.parameters['num_freq'],
                                   'number_of_pings_per_burst': self.parameters['pings_per_burst'],
                                   'average_burst_pings_flag': self.parameters['average_burst_pings'],
                                   # Temperature coefficients
                                   'temperature_ka': self.parameters['ka'],
                                   'temperature_kb': self.parameters['kb'],
                                   'temperature_kc': self.parameters['kc'],
                                   'temperature_A': self.parameters['A'],
                                   'temperature_B': self.parameters['B'],
                                   'temperature_C': self.parameters['C'],
                                   # Tilt coefficients
                                   'tilt_X_a': self.parameters['X_a'],
                                   'tilt_X_b': self.parameters['X_b'],
                                   'tilt_X_c': self.parameters['X_c'],
                                   'tilt_X_d': self.parameters['X_d'],
                                   'tilt_Y_a': self.parameters['Y_a'],
                                   'tilt_Y_b': self.parameters['Y_b'],
                                   'tilt_Y_c': self.parameters['Y_c'],
                                   'tilt_Y_d': self.parameters['Y_d']})
        return output

    def convert_to_nc(self):
        def _set_toplevel_dict():
            attrs = ('Conventions', 'keywords',
                     'sonar_convention_authority', 'sonar_convention_name',
                     'sonar_convention_version', 'summary', 'title')
            vals = ('CF-1.7, SONAR-netCDF4-1.0, ACDD-1.3', 'AZFP',
                    'ICES', 'SONAR-netCDF4', '1.0',
                    '', '')
            out_dict = dict(zip(attrs, vals))
            # Date is acquired from time of first ping
            date_created = dt.utcfromtimestamp(Output.ping_time.values[0]).isoformat(timespec='seconds') + 'Z'
            out_dict['date_created'] = date_created
            return out_dict

        def _set_env_dict():
            temps = [d['temperature'] for d in Data]
            freq = np.array(Data[0]['frequency']) * 1000    # Frequency in Hz
            abs_val = Data[0]['sea_abs']
            ss_val = [Data[0]['sound_speed']] * 4           # Sound speed independent of frequency

            attrs = ('frequency', 'absorption_coeff', 'sound_speed', 'salinity', 'temperature', 'pressure')
            vals = (freq, abs_val, ss_val, self.parameters['salinity'], temps, self.parameters['pressure'])
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
            vals = ('ASL Environmental Sciences', 'Acoustic Zooplankton Fish Profiler', Data[0]['serial_number'],
                    'Based on AZFP Matlab Toolbox', '1.4', 'echosounder')
            return dict(zip(attrs, vals))

        def _set_beam_dict():
            return None, None, None, None, None, Output

        def _set_vendor_specific_dict():
            out_dict = {
                'ping_time': Output.ping_time.values,
                'frequency': Output.frequency.values,
                'profile_flag': [d['profile_flag'] for d in Data],
                'profile_number': [d['profile_number'] for d in Data],
                'ping_status': [d['ping_status'] for d in Data],
                'burst_interval': [d['burst_int'] for d in Data],
                'digitization_rate': [d['dig_rate'] for d in Data],     # Dim: frequency
                'lock_out_index': [d['lockout_index'] for d in Data],   # Dim: frequency
                'num_bins': [d['num_bins'] for d in Data],              # Dim: frequency
                'range_samples': [d['range_samples'] for d in Data],    # Dim: frequency
                'ping_per_profile': [d['ping_per_profile'] for d in Data],
                'average_pings_flag': [d['avg_pings'] for d in Data],
                'number_of_acquired_pings': [d['num_acq_pings'] for d in Data],
                'ping_period': [d['ping_period'] for d in Data],
                'first_ping': [d['first_ping'] for d in Data],
                'last_ping': [d['last_ping'] for d in Data],
                'data_type': [d['data_type'] for d in Data],
                'data_error': [d['data_error'] for d in Data],
                'phase': [d['phase'] for d in Data],
                'number_of_channels': [d['num_chan'] for d in Data],
                'spare_channel': [d['spare_chan'] for d in Data],
                'board_number': [d['board_num'] for d in Data],         # Dim: frequency
                'sensor_flag': [d['sensor_flag'] for d in Data],
                'ancillary': [d['ancillary'] for d in Data],            # 5 values
                'ad_channels': [d['ad'] for d in Data]                  # 2 values
            }
            ancillary_len = list(range(len(out_dict['ancillary'][0])))
            ad_len = list(range(len(out_dict['ad_channels'][0])))
            out_dict['ancillary_len'] = ancillary_len
            out_dict['ad_len'] = ad_len
            return out_dict

        Data = self.parse_raw()
        Output = self.create_output(Data)
        filename = os.path.splitext(os.path.basename(self.path))[0]
        self.nc_path = os.path.join(os.path.split(self.path)[0], filename + '.nc')

        if os.path.exists(self.nc_path):    # USED FOR TESTING
            os.remove(self.nc_path)

        if os.path.exists(self.nc_path):
            print('          ... this file has already been converted to .nc, conversion not executed.')
        else:
            vendor = 'AZFP'
            # Create SetGroups object
            grp = SetGroups(file_path=self.nc_path)
            grp.set_toplevel(_set_toplevel_dict())      # top-level group
            grp.set_env(_set_env_dict(), vendor)        # environment group
            grp.set_provenance(os.path.basename(self.file_name), _set_prov_dict())    # provenance group
            grp.set_platform(_set_platform_dict(), vendor)      # platform group
            grp.set_sonar(_set_sonar_dict())                    # sonar group
            grp.set_beam(*_set_beam_dict(), vendor)             # beam group
            grp.set_vendor_specific(_set_vendor_specific_dict(), vendor)    # AZFP Vendor specific group


file1 = ConvertAZFP(path)
file1.convert_to_nc()
pass
