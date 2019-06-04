import os
import numpy as np
import xml.dom.minidom
import datetime
import math
from struct import unpack

path = "D:\\Documents\\Projects\\echopype\\toolbox\\12022316.01A"
xml_path = "D:\\Documents\\Projects\\echopype\\toolbox\\12022310.XML"


class ConvertAZFP:
    """Class for converting raw .01A AZFP files """

    def __init__(self, _path=""):
        self.path = _path
        self.file_name = os.path.split(path)[1]
        self.FILE_TYPE = 64770
        self.HEADER_SIZE = 124
        self.FORMAT = ">HHHHIHHHHHHHHHHHHHHHHHHHHHHHHHHHHHBBBBHBBBBBBBBHHHHHHHHHHHHHHHHHHHH"
        self.parameters = {
            # FILE LOADING AND AVERAGING:
            'proc_dir': 1,          # 1 will prompt for an entire directory to process
                                    # 0 will prompt to load individual files in a directory
            'data_file_name': "12022316.01A",   # "" will prompt for hourly AZFP files to load
            # "" will prompt for XML filename if no XML file exists in the directory
            'xml_file_name': "12022310.XML",

            'salinity': 29.6,       # Salinity in psu
            'bins_to_avg': 1,        # Number of range bins to average
            'time_to_avg': 40,       # number of time values to average
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

    def parse_raw(self):
        with open(self.path, 'rb') as raw:
            ii = 0
            Data = []
            eof = False
            while not eof:
                header_chunk = raw.read(self.HEADER_SIZE)
                if header_chunk:
                    header_unpacked = unpack(self.FORMAT, header_chunk)
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
                        # Compute temperature from Data[ii]['ancillary][5]
                        Data[ii]['temperature'] = self.compute_temp(Data[ii]['ancillary'][4])
                        # compute x tilt from Data[ii]['ancillary][0]
                        Data[ii]['tilt_x'] = self.compute_tilt(Data[ii]['ancillary'][0],
                                                               self.parameters['X_a'], self.parameters['X_b'],
                                                               self.parameters['X_c'], self.parameters['X_d'])
                        # Compute y tilt from Data[ii]['ancillary][1]
                        Data[ii]['tilt_y'] = self.compute_tilt(Data[ii]['ancillary'][1],
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
        Data[0]['hourly_avg_temp'] = self.compute_avg_temp(Data)
        Data[0]['sound_speed'] = self.compute_ss(Data[0]['hourly_avg_temp'], self.parameters['pressure'],
                                                 self.parameters['salinity'])
        Data[0]['hourly_avg_cos'] = self.compute_avg_tilt_mag(Data)
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
            Data[0]['sea_abs'].append(self.compute_sea_abs(Data, jj))
            if self.parameters['time_to_avg'] > len(Data):
                self.parameters['time_to_avg'] = len(Data)

            # Bin average ----INCOMPLETE----
            if self.parameters['bins_to_avg'] > 1:
                for jj in Data[0]['num_chan']:
                    bins_to_avg = self.parameters['bins_to_avg']
                    num_bins = Data[0]['counts'][jj]
                    num_bins = len(np.arange(0, len(num_bins) - bins_to_avg + 1, bins_to_avg))

            # Time average
            if self.parameters['time_to_avg'] > 1:
                num_time = math.floor(length(Data)/self.parameters['time_to_avg'])
                for kk in range(Data[0]['num_chan']):
                    el_avg = 10 * math.log10(1)
                pass
            
        return Data

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
        """Input open raw file, the bin number, and Data
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
        filename = os.path.split(path)[1]
        timestamp = datetime.datetime(Data[0]['year'], Data[0]['month'], Data[0]['day'], Data[0]['hour'],
                                      Data[0]['minute'], int(Data[0]['second'] + Data[0]['hundredths'] / 100))
        timestr = timestamp.strftime("%d-%b-%Y %H:%M:%S")
        (pathstr, name) = os.path.split(self.parameters['xml_file_name'])
        print("File: {} - Loading Profile #{} {} with xml={} Bins2Avg={} Time2Avg={} Salinity={:.2f} Pressure={:.1f}\n"
              .format(filename, Data[0]['profile_number'], timestr, name, self.parameters['bins_to_avg'],
                      self.parameters['time_to_avg'], self.parameters['salinity'], self.parameters['pressure']))

    def compute_temp(self, counts):
        """Returns the temperature in celsius given from xml data and the counts from ancillary"""
        v_in = 2.5 * (counts / 65535)
        R = (self.parameters['ka'] + self.parameters['kb'] * v_in) / (self.parameters['kc'] - v_in)
        T = 1 / (self.parameters['A'] + self.parameters['B'] * (math.log(R)) +
                 self.parameters['C'] * (math.log(R) ** 3)) - 273
        return T

    def compute_avg_temp(self, Data):
        """Input the data with temperature values and averages all the temperatures""" 
        sum = 0
        total = 0
        for ii in range(len(Data)):
            val = Data[ii]['temperature']
            if not math.isnan(val):
                total += 1
                sum += val
        if total == 0:
            print("**** No AZFP temperature found. Using default of {:.2f} degC to calculate sound-speed and range\n"
                  .format(self.parameters['hourly_avg_temp']))
            return self.parameters['hourly_avg_temp']    # default value
        else:
            return sum / total

    def compute_avg_tilt_mag(self, Data):
        """Input the data and calculates the average of the cosine tilt magnitudes"""
        sum = 0
        for ii in range(len(Data)):
            sum += Data[ii]['cos_tilt_mag']
        return sum / len(Data)

    def compute_ss(self, T, P, S):
        z = T / 10
        return (1449.05 + z * (45.7 + z * ((-5.21) + 0.23 * z)) + (1.333 + z * ((-0.126) + z * 0.009)) *
                (S - 35.0) + (P / 1000) * (16.3 + 0.18 * (P / 1000)))

    def compute_tilt(self, N, a, b, c, d):
        return a + b * (N) + c * (N)**2 + d * (N)**3

    def compute_sea_abs(self, Data, jj):
        """Input Data and the frequency number to calculate the absorption coefficient using
        the hourly average temperature, pressure, salinity, and transducer frequency"""
        T = Data[0]['hourly_avg_temp']
        P = self.parameters['pressure']
        S = self.parameters['salinity']
        F = Data[0]['frequency'][jj]

        # Calculate relaxation frequencies
        T_k = T + 273.0
        f1 = 1320.0 * T_k * math.exp(-1700 / T_k)
        f2 = (1.55e7) * T_k * math.exp(-3052 / T_k)

        # Coefficients for absorption calculations
        k = 1 + P / 10.0
        a = (8.95e-8) * (1 + T * ((2.29e-2) - (5.08e-4) * T))
        b = (S / 35.0) * (4.88e-7) * (1 + 0.0134 * T) * (1 - 0.00103 * k + (3.7e-7) * (k * k))
        c = (4.86e-13) * (1 + T * ((-0.042) + T * ((8.53e-4) - T * 6.23e-6))) * (1 + k * (-(3.84e-4) + k * 7.57e-8))
        F_k = F * 1000
        if S == 0:
            return c * F_k ^ 2
        else:
            # Update later: f_k might be a list
            return ((a * f1 * (F_k ** 2)) / ((f1 * f1) + (F_k ** 2)) +
                    (b * f2 * (F_k ** 2)) / ((f2 * f2) + (F_k ** 2)) + c * (F_k ** 2))

    def loadAZFPxml(self):

        def get_value_by_tag_name(px, tag_name, element=0):
            """Input a minidom parsed xml file, a tag name, and the number of the tag's occurances
               returns the value in the tag"""
            return px.getElementsByTagName(tag_name)[element].childNodes[0].data

        px = xml.dom.minidom.parse(xml_path)
        self.parameters['num_freq'] = int(get_value_by_tag_name(px, 'NumFreq'))
        self.parameters['serial_number'] = int(get_value_by_tag_name(px, 'SerialNumber'))
        self.parameters['burst_interval'] = float(get_value_by_tag_name(px, 'BurstInterval'))
        self.parameters['pings_per_burst'] = int(get_value_by_tag_name(px, 'PingsPerBurst'))
        self.parameters['average_burst_pings'] = int(get_value_by_tag_name(px, 'AverageBurstPings'))

        # Temperature coeff
        self.parameters['ka'] = float(get_value_by_tag_name(px, 'ka'))
        self.parameters['kb'] = float(get_value_by_tag_name(px, 'kb'))
        self.parameters['kc'] = float(get_value_by_tag_name(px, 'kc'))
        self.parameters['A'] = float(get_value_by_tag_name(px, 'A'))
        self.parameters['B'] = float(get_value_by_tag_name(px, 'B'))
        self.parameters['C'] = float(get_value_by_tag_name(px, 'C'))

        # tilts
        self.parameters['X_a'] = float(get_value_by_tag_name(px, 'X_a'))
        self.parameters['X_b'] = float(get_value_by_tag_name(px, 'X_b'))
        self.parameters['X_c'] = float(get_value_by_tag_name(px, 'X_c'))
        self.parameters['X_d'] = float(get_value_by_tag_name(px, 'X_d'))
        self.parameters['Y_a'] = float(get_value_by_tag_name(px, 'Y_a'))
        self.parameters['Y_b'] = float(get_value_by_tag_name(px, 'Y_b'))
        self.parameters['Y_c'] = float(get_value_by_tag_name(px, 'Y_c'))
        self.parameters['Y_d'] = float(get_value_by_tag_name(px, 'Y_d'))

        # Initializing fields for each tranducer frequency
        self.parameters['dig_rate'] = []
        self.parameters['lock_out_index'] = []
        self.parameters['gain'] = []
        self.parameters['pulse_len'] = []
        self.parameters['DS'] = []
        self.parameters['EL'] = []
        self.parameters['TVR'] = []
        self.parameters['VTX'] = []
        self.parameters['BP'] = []

        # Get parameters for each transducer frequency
        for jj in range(self.parameters['num_freq']):
            self.parameters['dig_rate'].append(float(get_value_by_tag_name(px, 'DigRate', jj)))
            self.parameters['lock_out_index'].append(float(get_value_by_tag_name(px, 'LockOutIndex', jj)))
            self.parameters['gain'].append(float(get_value_by_tag_name(px, 'Gain', jj)))
            self.parameters['pulse_len'].append(float(get_value_by_tag_name(px, 'PulseLen', jj)))
            self.parameters['DS'].append(float(get_value_by_tag_name(px, 'DS', jj)))
            self.parameters['EL'].append(float(get_value_by_tag_name(px, 'EL')))
            self.parameters['TVR'].append(float(get_value_by_tag_name(px, 'TVR', jj)))
            self.parameters['VTX'].append(float(get_value_by_tag_name(px, 'VTX0', jj)))
            self.parameters['BP'].append(float(get_value_by_tag_name(px, 'BP', jj)))
        self.parameters['sensors_flag'] = float(get_value_by_tag_name(px, 'SensorsFlag'))

    @staticmethod
    def get_fields():
        '''Returns the fields contained in each header of the raw file'''
        _fields = (
            ('profile_flag', 'u2'),
            ('profile_number', 'u2'),
            ('serial_number', 'u2'),
            ('ping_status', 'u2'),
            ('burst_int', 'u4'),
            ('year', 'u2'),                 # 012 - Year
            ('month', 'u2'),                # 014 - Month
            ('day', 'u2'),                  # 016 - Day
            ('hour', 'u2'),                 # 018 - Hour
            ('minute', 'u2'),               # 020 - Minute
            ('second', 'u2'),               # 022 - Second
            ('hundredths', 'u2'),           # 024 - Hundreths of a second
            ('dig_rate', 'u2', 4),
            ('lockout_index', 'u2', 4),
            ('num_bins', 'u2', 4),
            ('range_samples', 'u2', 4),
            ('ping_per_profile', 'u2'),
            ('avg_pings', 'u2'),
            ('num_acq_pings', 'u2'),
            ('ping_period', 'u2'),
            ('first_ping', 'u2'),
            ('last_ping', 'u2'),
            ('data_type', "u1", 4),
            ('data_error', 'u2'),
            ('phase', 'u1'),
            ('over_run', 'u1'),
            ('num_chan', 'u1'),
            ('gain', 'u1', 4),
            ('spare_chan', 'u1'),
            ('pulse_length', 'u2', 4),
            ('board_num', 'u2', 4),
            ('frequency', 'u2', 4),
            ('sensor_flag', 'u2'),
            ('ancillary', 'u2', 5),   # Tilt-X, Y, Battery, Pressure, Temperature
            ('ad', 'u2', 2)            # AD channel 6 and 7
        )

        return _fields


file1 = ConvertAZFP(path)
d = file1.parse_raw()
