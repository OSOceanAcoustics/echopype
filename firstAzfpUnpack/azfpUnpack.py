"""
@package mi.dataset.driver.zplsc_c
@file mi/dataset/driver/zplsc_c/zplsc_c_echogram.py
@author Craig Risien/Rene Gelinas
@brief ZPLSC Echogram generation for the ooicore

Release notes:

This class supports the generation of ZPLSC-C echograms.
"""

# from collections import defaultdict
# from struct import unpack_from, unpack
import struct
from ctypes import *
import os
# import re
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date
from datetime import datetime
import ntplib
import time
import json
# import numpy as np
# import mi.dataset.driver.zplsc_c.zplsc_functions as zf

PROFILE_DATA_DELIMITER = b'\xfd\x02'  # Byte Offset 0 and 1


class ZplscCParameters(object):
    # TODO: This class should be replaced by methods to get the CCs from the system.
    # Configuration Parameters
    Salinity = 32   # Salinity in psu
    Pressure = 150  # in dbars (~ depth of instrument in meters).
    Bins2Avg = 1    # number of range bins to average - 1 is no averaging


class ZplscCCalibrationCoefficients(object):
    # TODO: This class should be replaced by methods to get the CCs from the system.
    ka = 464.3636
    kb = 3000.0
    kc = 1.893
    A = 0.001466
    B = 0.0002388
    C = 0.000000100335

    TVR = []
    VTX = []
    BP = []
    EL = []
    DS = []

    # Freq 38kHz
    TVR.append(1.691999969482e2)
    VTX.append(1.533999938965e2)
    BP.append(8.609999902546e-3)
    EL.append(1.623000030518e2)
    DS.append(2.280000038445e-2)

    # Freq 125kHz
    TVR.append(1.668999938965e2)
    VTX.append(5.8e+01)
    BP.append(1.530999969691e-2)
    EL.append(1.376999969482e2)
    DS.append(2.280000038445e-2)

    # Freq 200kHz
    TVR.append(1.688999938965e2)
    VTX.append(9.619999694824e1)
    BP.append(1.530999969691e-2)
    EL.append(1.456000061035e2)
    DS.append(2.250000089407e-2)

    # Freq 455kHz
    TVR.append(1.696000061035e2)
    VTX.append(1.301000061035e2)
    BP.append(8.609999902546e-3)
    EL.append(1.491999969482e2)
    DS.append(2.300000004470e-2)


class ZPLSCCEchogram(object):
    def __init__(self):
        self.cc = ZplscCCalibrationCoefficients()
        self.params = ZplscCParameters()

    def compute_backscatter(self, profile_hdr, chan_data, sound_speed, depth_range, sea_absorb):
        """
        Compute the backscatter volumes values for one zplsc_c profile data record.
        This code was borrowed from ASL MatLab code that reads in zplsc-c raw data
        and performs calculations in order to compute the backscatter volume in db.

        :param profile_hdr: Raw profile header with metadata from the zplsc-c instrument.
        :param chan_data: Raw frequency data from the zplsc-c instrument.
        :param sound_speed: Speed of sound at based on speed of sound, pressure and salinity.
        :param depth_range: Range of the depth of the measurements
        :param sea_absorb: Seawater absorption coefficient for each frequency
        :return: sv: Volume backscatter in db
        """

        _N = []
        if self.params.Bins2Avg > 1:
            for chan in range(profile_hdr.num_channels):
                el = self.cc.EL[chan] - 2.5/self.cc.DS[chan] + np.array(chan_data[chan])/(26214*self.cc.DS[chan])
                power = 10**(el/10)

                # Perform bin averaging
                num_bins = len(chan_data[chan])/self.params.Bins2Avg
                pwr_avg = []
                for _bin in range(num_bins):
                    pwr_avg.append(np.mean(power[_bin*self.params.Bins2Avg:(_bin+1)*self.params.Bins2Avg]))

                el_avg = 10*np.log10(pwr_avg)
                _N.append(np.round(26214*self.cc.DS[chan]*(el_avg - self.cc.EL[chan] + 2.5/self.cc.DS[chan])))

        else:
            for chan in range(profile_hdr.num_channels):
                _N.append(np.array(chan_data[chan]))

        sv = []
        for chan in range(profile_hdr.num_channels):
            # Calculate correction to Sv due to non square transmit pulse
            sv_offset = compute_sv_offset(profile_hdr.frequency[chan], profile_hdr.pulse_length[chan])
            sv.append(self.cc.EL[chan]-2.5/self.cc.DS[chan] + _N[chan]/(26214*self.cc.DS[chan]) - self.cc.TVR[chan] -
                      20*np.log10(self.cc.VTX[chan]) + 20*np.log10(depth_range[chan]) +
                      2*sea_absorb[chan]*depth_range[chan] -
                      10*np.log10(0.5*sound_speed*profile_hdr.pulse_length[chan]/1e6*self.cc.BP[chan]) +
                      sv_offset)

        return sv

    def compute_echogram_metadata(self, profile_hdr):
        """
        Compute the metadata parameters needed to compute the zplsc-c volume backscatter values.

        :param  profile_hdr: Raw profile header with metadata from the zplsc-c instrument.
        :return: sound_speed : Speed of sound based on temperature, pressure and salinity.
                 depth_range : Range of depth values of the zplsc-c data.
                 sea_absorb : Sea absorption based on temperature, pressure, salinity and frequency.
        """

        # If the temperature sensor is available, compute the temperature from the counts.
        temperature = 0
        if profile_hdr.is_sensor_available:
            temperature = zplsc_c_temperature(profile_hdr.temperature, self.cc.ka, self.cc.kb, self.cc.kc,
                                                 self.cc.A, self.cc.B, self.cc.C)

        sound_speed = zplsc_c_ss(temperature, self.params.Pressure, self.params.Salinity)

        _m = []
        depth_range = []
        for chan in range(profile_hdr.num_channels):
            _m.append(np.array([x for x in range(1, int((profile_hdr.num_bins[chan]/self.params.Bins2Avg)+1))]))
            depth_range.append(sound_speed*profile_hdr.lockout_index[0]/(2*profile_hdr.digitization_rate[0]) +
                               (sound_speed/4)*(((2*_m[chan]-1)*profile_hdr.range_samples[0]*self.params.Bins2Avg-1) /
                                                float(profile_hdr.digitization_rate[0]) +
                                                profile_hdr.pulse_length[0]/1e6))

        sea_absorb = []
        for chan in range(profile_hdr.num_channels):
            # Calculate absorption coefficient for each frequency.
            sea_absorb.append(zplsc_c_absorbtion(temperature, self.params.Pressure, self.params.Salinity,
                                                    profile_hdr.frequency[chan]))

        return sound_speed, depth_range, sea_absorb


class DataParticleKey():
    PKT_FORMAT_ID = "pkt_format_id"
    PKT_VERSION = "pkt_version"
    STREAM_NAME = "stream_name"
    INTERNAL_TIMESTAMP = "internal_timestamp"
    PORT_TIMESTAMP = "port_timestamp"
    DRIVER_TIMESTAMP = "driver_timestamp"
    PREFERRED_TIMESTAMP = "preferred_timestamp"
    QUALITY_FLAG = "quality_flag"
    VALUES = "values"
    VALUE_ID = "value_id"
    VALUE = "value"
    BINARY = "binary"
    NEW_SEQUENCE = "new_sequence"


class Parser(object):
    """ abstract class to show API needed for plugin poller objects """

    def get_records(self, max_count):
        """
        Returns a list of particles (following the instrument driver structure).
        """
        # raise NotImplementedException("get_records() not overridden!")
        raise Exception("NotImplementedException")

    def _publish_sample(self, samples):
        """
        Publish the samples with the given publishing callback.
        @param samples The list of data particle to publish up to the system
        """
        if isinstance(samples, list):
            self._publish_callback(samples)
        else:
            self._publish_callback([samples])

    def _extract_sample(self, particle_class, regex, raw_data, port_timestamp=None, internal_timestamp=None,
                        preferred_ts=DataParticleKey.INTERNAL_TIMESTAMP):
        """
        Extract sample from a response line if present and publish
        parsed particle

        @param particle_class The class to instantiate for this specific
            data particle. Parameterizing this allows for simple, standard
            behavior from this routine
        @param regex The regular expression that matches a data sample if regex
                     is none then process every line
        @param raw_data data to input into this particle.
        @param port_timestamp the port_timestamp (default: None)
        @param internal_timestamp the internal_timestamp (default: None)
        @param preferred_ts the preferred timestamp (default: INTERNAL_TIMESTAMP)
        @retval return a raw particle if a sample was found, else None
        """

        particle = None

        try:
            if regex is None or regex.match(raw_data):
                particle = particle_class(raw_data, port_timestamp=port_timestamp,
                                          internal_timestamp=internal_timestamp,
                                          preferred_timestamp=preferred_ts)

                # need to actually parse the particle fields to find out of there are errors
                particle.generate_dict()
                encoding_errors = particle.get_encoding_errors()
                if encoding_errors:
                    # log.warn("Failed to encode: %s", encoding_errors)
                    # raise SampleEncodingException("Failed to encode: %s" % encoding_errors)
                    raise Exception("Failed to encode")

        # except (RecoverableSampleException, SampleEncodingException) as e:
        except:
            raise Exception("Sample Exception")
            # log.error("Sample exception detected: %s raw data: %s", e, raw_data)
            # if self._exception_callback:
            #     self._exception_callback(e)
            # else:
            #     raise e

        return particle

class SimpleParser(Parser):

    def __init__(self, config, stream_handle, exception_callback):
        """
        Initialize the simple parser, which does not use state or the chunker
        and sieve functions.
        @param config: The parser configuration dictionary
        @param stream_handle: The stream handle of the file to parse
        @param exception_callback: The callback to use when an exception occurs
        """

        # the record buffer which will store all parsed particles
        self._record_buffer = []
        # a flag indicating if the file has been parsed or not
        self._file_parsed = False
        self._stream_handle = stream_handle

    def parse_file(self):
        """
        This method must be overridden.  This method should open and read the file and parser the data within, and at
        the end of this method self._record_buffer will be filled with all the particles in the file.
        """
        # raise NotImplementedException("parse_file() not overridden!")
        raise Exception("NotImplementedException")

    def get_records(self, number_requested=1):
        """
        Initiate parsing the file if it has not been done already, and pop particles off the record buffer to
        return as many as requested if they are available in the buffer.
        @param number_requested the number of records requested to be returned
        @return an array of particles, with a length of the number requested or less
        """
        particles_to_return = []

        if number_requested > 0:
            if self._file_parsed is False:
                self.parse_file()
                self._file_parsed = True

        while len(particles_to_return) < number_requested and len(self._record_buffer) > 0:
            particles_to_return.append(self._record_buffer.pop(0))

        return particles_to_return


class ZplscCParticleKey():
    """
    Class that defines fields that need to be extracted for the data particle.
    """
    TRANS_TIMESTAMP = "zplsc_c_transmission_timestamp"
    SERIAL_NUMBER = "serial_number"
    PHASE = "zplsc_c_phase"
    BURST_NUMBER = "burst_number"
    TILT_X = "zplsc_c_tilt_x_counts"
    TILT_Y = "zplsc_c_tilt_y_counts"
    BATTERY_VOLTAGE = "zplsc_c_battery_voltage_counts"
    TEMPERATURE = "zplsc_c_temperature_counts"
    PRESSURE = "zplsc_c_pressure_counts"
    IS_AVERAGED_DATA = "zplsc_c_is_averaged_data"
    FREQ_CHAN_1 = "zplsc_frequency_channel_1"
    VALS_CHAN_1 = "zplsc_values_channel_1"
    DEPTH_CHAN_1 = "zplsc_depth_range_channel_1"
    FREQ_CHAN_2 = "zplsc_frequency_channel_2"
    VALS_CHAN_2 = "zplsc_values_channel_2"
    DEPTH_CHAN_2 = "zplsc_depth_range_channel_2"
    FREQ_CHAN_3 = "zplsc_frequency_channel_3"
    VALS_CHAN_3 = "zplsc_values_channel_3"
    DEPTH_CHAN_3 = "zplsc_depth_range_channel_3"
    FREQ_CHAN_4 = "zplsc_frequency_channel_4"
    VALS_CHAN_4 = "zplsc_values_channel_4"
    DEPTH_CHAN_4 = "zplsc_depth_range_channel_4"

class AzfpProfileHeader(BigEndianStructure):
    _pack_ = 1                              # 124 bytes in the header (includes the 2 byte delimiter)
    _fields_ = [                            # V Byte Offset (from delimiter)
        ('burst_num', c_ushort),            # 002 - Burst number
        ('serial_num', c_ushort),           # 004 - Instrument Serial number
        ('ping_status', c_ushort),          # 006 - Ping Status
        ('burst_interval', c_uint),         # 008 - Burst Interval (seconds)
        ('year', c_ushort),                 # 012 - Year
        ('month', c_ushort),                # 014 - Month
        ('day', c_ushort),                  # 016 - Day
        ('hour', c_ushort),                 # 018 - Hour
        ('minute', c_ushort),               # 020 - Minute
        ('second', c_ushort),               # 022 - Second
        ('hundredths', c_ushort),           # 024 - Hundreths of a second
        ('digitization_rate', c_ushort*4),  # 026 - Digitization Rate (channels 1-4) (64000, 40000 or 20000)
        ('lockout_index', c_ushort*4),      # 034 - The sample number of samples skipped at start of ping (channels 1-4)
        ('num_bins', c_ushort*4),           # 042 - Number of bins (channels 1-4)
        ('range_samples', c_ushort*4),      # 050 - Range samples per bin (channels 1-4)
        ('num_pings_profile', c_ushort),    # 058 - Number of pings per profile
        ('is_averaged_pings', c_ushort),    # 060 - Indicates if pings are averaged in time
        ('num_pings_burst', c_ushort),      # 062 - Number of pings that have been acquired in this burst
        ('ping_period', c_ushort),          # 064 - Ping period in seconds
        ('first_ping', c_ushort),           # 066 - First ping number (if averaged, first averaged ping number)
        ('second_ping', c_ushort),          # 068 - Last ping number (if averaged, last averaged ping number)
        ('is_averaged_data', c_ubyte*4),    # 070 - 1 = averaged data (5 bytes), 0 = not averaged (2 bytes)
        ('error_num', c_ushort),            # 074 - Error number if an error occurred
        ('phase', c_ubyte),                 # 076 - Phase used to acquire this profile
        ('is_overrun', c_ubyte),            # 077 - 1 if an over run occurred
        ('num_channels', c_ubyte),          # 078 - Number of channels (1, 2, 3 or 4)
        ('gain', c_ubyte*4),                # 079 - Gain (channels 1-4) 0, 1, 2, 3 (Obsolete)
        ('spare', c_ubyte),                 # 083 - Spare
        ('pulse_length', c_ushort*4),       # 084 - Pulse length (channels 1-4) (uS)
        ('board_num', c_ushort*4),          # 092 - Board number of the data (channels 1-4)
        ('frequency', c_ushort*4),          # 100 - Board frequency (channels 1-4)
        ('is_sensor_available', c_ushort),  # 108 - Indicate if pressure/temperature sensor is available
        ('tilt_x', c_ushort),               # 110 - Tilt X (counts)
        ('tilt_y', c_ushort),               # 112 - Tilt Y (counts)
        ('battery_voltage', c_ushort),      # 114 - Battery voltage (counts)
        ('pressure', c_ushort),             # 116 - Pressure (counts)
        ('temperature', c_ushort),          # 118 - Temperature (counts)
        ('ad_channel_6', c_ushort),         # 120 - AD channel 6
        ('ad_channel_7', c_ushort)          # 122 - AD channel 7
        ]

def generate_image_file_path(filepath, output_path=None):
    # Extract the file time from the file name
    absolute_path = os.path.abspath(filepath)
    filename = os.path.basename(absolute_path).upper()
    directory_name = os.path.dirname(absolute_path)

    output_path = directory_name if output_path is None else output_path
    image_file = filename.replace('.01A', '.png')
    return os.path.join(output_path, image_file)

# class ZplscCCalibrationCoefficients(object):
#     # TODO: This class should be replaced by methods to get the CCs from the system.
#     DS = list()
#
#     # Freq 38kHz
#     DS.append(2.280000038445e-2)
#
#     # Freq 125kHz
#     DS.append(2.280000038445e-2)
#
#     # Freq 200kHz
#     DS.append(2.250000089407e-2)
#
#     # Freq 455kHz
#     DS.append(2.300000004470e-2)


class ZplscCParser(SimpleParser):
    def __init__(self, config, stream_handle, exception_callback):
        super(ZplscCParser, self).__init__(config, stream_handle, exception_callback)
        self._particle_type = None
        self._gen = None
        self.ph = None  # The profile header of the current record being processed.
        self.cc = ZplscCCalibrationCoefficients()
        self.is_first_record = True
        self.hourly_avg_temp = 0
        self.zplsc_echogram = ZPLSCCEchogram()

    def find_next_record(self):
        good_delimiter = True
        delimiter = self._stream_handle.read(2)
        while delimiter not in [PROFILE_DATA_DELIMITER, b'']:
            good_delimiter = False
            # try:
            #     prtStr = struct.unpack('>H',delimiter)
            # except:
            #     input()
            delimiter = delimiter[1:2]
            delimiter += self._stream_handle.read(1)

        if not good_delimiter:
            self._exception_callback('Invalid record delimiter found.\n')

    def parse_record(self):
        """
        Parse one profile data record of the zplsc-c data file.
        """
        chan_values = [[], [], [], []]
        overflow_values = [[], [], [], []]

        # Parse the data values portion of the record.
        for chan in range(self.ph.num_channels):
            num_bins = self.ph.num_bins[chan]

            # Set the data structure format for the scientific data, based on whether
            # the data is averaged or not. Construct the data structure and read the
            # data bytes for the current channel. Unpack the data based on the structure.
            if self.ph.is_averaged_data[chan]:
                data_struct_format = '>' + str(num_bins) + 'I'
            else:
                data_struct_format = '>' + str(num_bins) + 'H'
            data_struct = struct.Struct(data_struct_format)
            data = self._stream_handle.read(data_struct.size)
            chan_values[chan] = data_struct.unpack(data)

            # If the data type is for averaged data, calculate the averaged data taking the
            # the linear sum channel values and overflow values and using calculations from
            # ASL MatLab code.
            if self.ph.is_averaged_data[chan]:
                overflow_struct_format = '>' + str(num_bins) + 'B'
                overflow_struct = struct.Struct(overflow_struct_format)
                overflow_data = self._stream_handle.read(num_bins)
                overflow_values[chan] = overflow_struct.unpack(overflow_data)

                if self.ph.is_averaged_pings:
                    divisor = self.ph.num_pings_profile * self.ph.range_samples[chan]
                else:
                    divisor = self.ph.range_samples[chan]

                linear_sum_values = np.array(chan_values[chan])
                linear_overflow_values = np.array(overflow_values[chan])

                values = (linear_sum_values + (linear_overflow_values * 0xFFFFFFFF)) / divisor
                values = (np.log10(values) - 2.5) * (8 * 0xFFFF) * self.cc.DS[chan]
                values[np.isinf(values)] = 0
                chan_values[chan] = values

        # Convert the date and time parameters to a epoch time from 01-01-1900.
        timestamp = (datetime(self.ph.year, self.ph.month, self.ph.day,
                              self.ph.hour, self.ph.minute, self.ph.second,
                              (self.ph.hundredths * 10000)) - datetime(1900, 1, 1)).total_seconds()

        sound_speed, depth_range, sea_absorb = self.zplsc_echogram.compute_echogram_metadata(self.ph)

        chan_values = self.zplsc_echogram.compute_backscatter(self.ph, chan_values, sound_speed, depth_range,
                                                              sea_absorb)

        zplsc_particle_data = {
            ZplscCParticleKey.TRANS_TIMESTAMP: timestamp,
            ZplscCParticleKey.SERIAL_NUMBER: str(self.ph.serial_num),
            ZplscCParticleKey.PHASE: self.ph.phase,
            ZplscCParticleKey.BURST_NUMBER: self.ph.burst_num,
            ZplscCParticleKey.TILT_X: self.ph.tilt_x,
            ZplscCParticleKey.TILT_Y: self.ph.tilt_y,
            ZplscCParticleKey.BATTERY_VOLTAGE: self.ph.battery_voltage,
            ZplscCParticleKey.PRESSURE: self.ph.pressure,
            ZplscCParticleKey.TEMPERATURE: self.ph.temperature,
            ZplscCParticleKey.IS_AVERAGED_DATA: list(self.ph.is_averaged_data),
            ZplscCParticleKey.FREQ_CHAN_1: float(self.ph.frequency[0]),
            ZplscCParticleKey.VALS_CHAN_1: list(chan_values[0]),
            ZplscCParticleKey.DEPTH_CHAN_1: list(depth_range[0]),
            ZplscCParticleKey.FREQ_CHAN_2: float(self.ph.frequency[1]),
            ZplscCParticleKey.VALS_CHAN_2: list(chan_values[1]),
            ZplscCParticleKey.DEPTH_CHAN_2: list(depth_range[1]),
            ZplscCParticleKey.FREQ_CHAN_3: float(self.ph.frequency[2]),
            ZplscCParticleKey.VALS_CHAN_3: list(chan_values[2]),
            ZplscCParticleKey.DEPTH_CHAN_3: list(depth_range[2]),
            ZplscCParticleKey.FREQ_CHAN_4: float(self.ph.frequency[3]),
            ZplscCParticleKey.VALS_CHAN_4: list(chan_values[3]),
            ZplscCParticleKey.DEPTH_CHAN_4: list(depth_range[3])
        }

        return zplsc_particle_data, timestamp, chan_values, depth_range

    def parse_file(self):
        self.ph = AzfpProfileHeader()
        self.find_next_record()
        while self._stream_handle.readinto(self.ph):
            try:
                # Parse the current record
                zplsc_particle_data, timestamp, _, _ = self.parse_record()

                # Create the data particle
                particle = self._extract_sample(ZplscCRecoveredDataParticle, None, zplsc_particle_data, timestamp,
                                                timestamp, DataParticleKey.PORT_TIMESTAMP)
                if particle is not None:
                    # log.trace('Parsed particle: %s' % particle.generate_dict())
                    print('Parsed particle: %s' % particle.generate_dict())
                    self._record_buffer.append(particle)

            except:
                Exception("timestamp has invalid format")
            # except (IOError, OSError) as ex:
            #     self._exception_callback('Reading stream handle: %s: %s\n' % (self._stream_handle.name, ex.message))
            #     return
            # except struct.error as ex:
            #     self._exception_callback('Unpacking the data from the data structure: %s' % ex.message)
            # except exceptions.ValueError as ex:
            #     self._exception_callback('Transition timestamp has invalid format: %s' % ex.message)
            # except (SampleException, RecoverableSampleException) as ex:
            #     self._exception_callback('Creating data particle: %s' % ex.message)

            # Clear the profile header data structure and find the next record.
            self.ph = AzfpProfileHeader()
            self.find_next_record()

    def create_echogram(self, echogram_file_path=None):
        """
        Parse the *.O1A zplsc_c data file and create the echogram from this data.
        :param echogram_file_path: Path to store the echogram locally.
        :return:
        """
        import logging
        sv_dict = {}
        data_times = []
        frequencies = {}
        depth_range = []

        input_file_path = self._stream_handle.name  # None
        # logging.info('Begin processing echogram data: %r', input_file_path)
        image_path = generate_image_file_path(input_file_path, echogram_file_path)

        self.ph = AzfpProfileHeader()
        self.find_next_record()
        while self._stream_handle.readinto(self.ph):
            try:
                _, timestamp, chan_data, depth_range = self.parse_record()

                if not sv_dict:
                    range_chan_data = range(1, len(chan_data) + 1)
                    sv_dict = {channel: [] for channel in range_chan_data}
                    frequencies = {channel: float(self.ph.frequency[channel - 1]) for channel in range_chan_data}

                for channel in sv_dict:
                    sv_dict[channel].append(chan_data[channel - 1])

                data_times.append(timestamp)

            except:
                raise Exception("Something didn't work in create_echogram")
            # except (IOError, OSError) as ex:
            #     self._exception_callback(ex)
            #     return
            # except struct.error as ex:
            #     self._exception_callback(ex)
            # except exceptions.ValueError as ex:
            #     self._exception_callback(ex)
            # except (SampleException, RecoverableSampleException) as ex:
            #     self._exception_callback(ex)

            # Clear the profile header data structure and find the next record.
            self.ph = AzfpProfileHeader()
            self.find_next_record()

        # log.info('Completed processing all data: %r', input_file_path)
        print('Completed processing all data: %r' % input_file_path)

        data_times = np.array(data_times)

        for channel in sv_dict:
            sv_dict[channel] = np.array(sv_dict[channel])

        logging.info('Begin generating echogram: %r', image_path)

        plot = ZPLSPlot(data_times, sv_dict, frequencies, depth_range[0][-1], depth_range[0][0])
        plot.generate_plots()
        plot.write_image(image_path)

        # log.info('Completed generating echogram: %r', image_path)
        print('Completed generating echogram: %r' % image_path)

    def rec_exception_callback(exception):
        """
        Callback function to log exceptions and continue.

        @param exception - Exception that occurred
        """

        # log.info("Exception occurred: %s", exception.message)
        print("Exception occured: %s" % exception.message)

# class ParticleDataHandler(object):
#     def __init__(self): particle_data_handler
#
#     def addParticleSample(self, sample_type, sample):
#         log.debug("Sample type: %s, Sample data: %s", sample_type, sample)
#         self._samples.setdefault(sample_type, []).append(sample)
#
#     def setParticleDataCaptureFailure(self):
#         log.debug("Particle data capture failed")
#         self._failure = True


class DataSetDriverConfigKeys():
    PARTICLE_MODULE = "particle_module"
    PARTICLE_CLASS = "particle_class"
    PARTICLE_CLASSES_DICT = "particle_classes_dict"
    DIRECTORY = "directory"
    STORAGE_DIRECTORY = "storage_directory"
    PATTERN = "pattern"
    FREQUENCY = "frequency"
    FILE_MOD_WAIT_TIME = "file_mod_wait_time"
    HARVESTER = "harvester"
    PARSER = "parser"
    MODULE = "module"
    CLASS = "class"
    URI = "uri"
    CLASS_ARGS = "class_args"


def compute_sv_offset(frequency, pulse_length):
    """
    A correction must be made to compensate for the effects of the finite response
    times of both the receiving and transmitting parts of the instrument. The magnitude
    of the correction will depend on the length of the transmitted pulse, and the response
    time (on both transmission and reception) of the instrument.

    :param frequency: Frequency in KHz
    :param pulse_length: Pulse length in uSecs
    :return:
    """

    sv_offset = 0

    if frequency > 38:  # 125,200,455,769 kHz
        if pulse_length == 300:
            sv_offset = 1.1
        elif pulse_length == 500:
            sv_offset = 0.8
        elif pulse_length == 700:
            sv_offset = 0.5
        elif pulse_length == 900:
            sv_offset = 0.3
        elif pulse_length == 1000:
            sv_offset = 0.3
    else:  # 38 kHz
        if pulse_length == 500:
            sv_offset = 1.1
        elif pulse_length == 1000:
            sv_offset = 0.7

    return sv_offset

#!/usr/bin/env python
"""
@package mi.dataset.driver.zplsc_c.zplsc_functions
@file mi.dataset.driver.zplsc_c/zplsc_functions.py
@author Rene Gelinas
@brief Module containing ZPLSC related data calculations.
"""

import numpy as np


def zplsc_b_decompress(power):
    """
    Description:

        Convert a list of compressed power values to numpy array of
        decompressed power.  This code was from the zplsc_b.py parser,
        when it only produced an echogram.

    Implemented by:

        2017-06-27: Rene Gelinas. Initial code.

    :param power: List of zplsc B series power values in a compressed format.
    :return: Decompressed numpy array of power values.
    """

    decompress_power = np.array(power) * 10. * np.log10(2) / 256.

    return decompress_power


def zplsc_c_temperature(counts, ka, kb, kc, a, b, c):
    """
    Description:

        Compute the temperature from the counts passed in.
        This Code was lifted from the ASL MatLab code LoadAZFP.m

    Implemented by:

        2017-06-23: Rene Gelinas. Initial code.

    :param counts: Raw data temperature counts from the zplsc-c raw data file.
    :param ka:
    :param kb:
    :param kc:
    :param a:
    :param b:
    :param c:
    :return: temperature
    """

    vin = 2.5 * (counts / 65535)
    r = (ka + kb*vin) / (kc - vin)
    temperature = 1 / (a + b * (np.log(r)) + c * (np.log(r)**3)) - 273

    return temperature


def zplsc_c_tilt(counts, a, b, c, d):
    """
    Description:

        Compute the tilt from the counts passed in from the zplsc A series.
        This Code was from the ASL MatLab code LoadAZFP.m

    Implemented by:

        2017-06-23: Rene Gelinas. Initial code.

    :param counts:
    :param a:
    :param b:
    :param c:
    :param d:
    :return: tilt value
    """

    tilt = a + (b * counts) + (c * counts**2) + (d * counts**3)

    return tilt


def zplsc_c_ss(t, p, s):
    """
    Description:

        Compute the ss from the counts passed in.
        This Code was from the ASL MatLab code LoadAZFP.m

    Implemented by:

        2017-06-23: Rene Gelinas. Initial code.

    :param t:
    :param p:
    :param s:
    :return:
    """

    z = t/10
    sea_c = 1449.05 + (z * (45.7 + z*((-5.21) + 0.23*z))) + ((1.333 + z*((-0.126) + z*0.009)) * (s-35.0)) + \
        (p/1000)*(16.3+0.18*(p/1000))

    return sea_c


def zplsc_c_absorbtion(t, p, s, freq):
    """
    Description:

        Calculate Absorption coeff using Temperature, Pressure and Salinity and transducer frequency.
        This Code was from the ASL MatLab code LoadAZFP.m

    Implemented by:

        2017-06-23: Rene Gelinas. Initial code.


    :param t:
    :param p:
    :param s:
    :param freq:  Frequency in KHz
    :return: sea_abs
    """

    # Calculate relaxation frequencies
    t_k = t + 273.0
    f1 = 1320.0*t_k * np.exp(-1700/t_k)
    f2 = 1.55e7*t_k * np.exp(-3052/t_k)

    # Coefficients for absorption equations
    k = 1 + p/10.0
    a = 8.95e-8 * (1 + t*(2.29e-2 - 5.08e-4*t))
    b = (s/35.0)*4.88e-7*(1+0.0134*t)*(1-0.00103*k + 3.7e-7*(k*k))
    c = 4.86e-13*(1+t*((-0.042)+t*(8.53e-4-t*6.23e-6)))*(1+k*(-3.84e-4+k*7.57e-8))
    freqk = freq*1000
    sea_abs = (a*f1*(freqk**2))/((f1*f1)+(freqk**2))+(b*f2*(freqk**2))/((f2*f2)+(freqk**2))+c*(freqk**2)

    return sea_abs


def compute_sv_offset(frequency, pulse_length):
    """
    A correction must be made to compensate for the effects of the finite response
    times of both the receiving and transmitting parts of the instrument. The magnitude
    of the correction will depend on the length of the transmitted pulse, and the response
    time (on both transmission and reception) of the instrument.

    :param frequency: Frequency in KHz
    :param pulse_length: Pulse length in uSecs
    :return:
    """

    sv_offset = 0

    if frequency > 38:  # 125,200,455,769 kHz
        if pulse_length == 300:
            sv_offset = 1.1
        elif pulse_length == 500:
            sv_offset = 0.8
        elif pulse_length == 700:
            sv_offset = 0.5
        elif pulse_length == 900:
            sv_offset = 0.3
        elif pulse_length == 1000:
            sv_offset = 0.3
    else:  # 38 kHz
        if pulse_length == 500:
            sv_offset = 1.1
        elif pulse_length == 1000:
            sv_offset = 0.7

    return sv_offset


class BaseEnum(object):
    """Base class for enums.

    Used to code agent and instrument states, events, commands and errors.
    To use, derive a class from this subclass and set values equal to it
    such as:
    @code
    class FooEnum(BaseEnum):
       VALUE1 = "Value 1"
       VALUE2 = "Value 2"
    @endcode
    and address the values as FooEnum.VALUE1 after you import the
    class/package.

    Enumerations are part of the code in the MI modules since they are tightly
    coupled with what the drivers can do. By putting the values here, they
    are quicker to execute and more compartmentalized so that code can be
    re-used more easily outside of a capability container as needed.
    """

    @classmethod
    def list(cls):
        """List the values of this enum."""
        return [getattr(cls, attr) for attr in dir(cls) if \
                not callable(getattr(cls, attr)) and not attr.startswith('__')]

    @classmethod
    def dict(cls):
        """Return a dict representation of this enum."""
        result = {}
        for attr in dir(cls):
            if not callable(getattr(cls, attr)) and not attr.startswith('__'):
                result[attr] = getattr(cls, attr)
        return result

    @classmethod
    def has(cls, item):
        """Is the object defined in the class?

        Use this function to test
        a variable for enum membership. For example,
        @code
        if not FooEnum.has(possible_value)
        @endcode
        @param item The attribute value to test for.
        @retval True if one of the class attributes has value item, false
        otherwise.
        """
        return item in cls.list()


class DataParticleValue(BaseEnum):
    JSON_DATA = "JSON_Data"
    ENG = "eng"
    OK = "ok"
    CHECKSUM_FAILED = "checksum_failed"
    OUT_OF_RANGE = "out_of_range"
    INVALID = "invalid"
    QUESTIONABLE = "questionable"


class DataParticle(object):
    """
    This class is responsible for storing and ultimately generating data
    particles in the designated format from the associated inputs. It
    fills in fields as necessary, and is a valid Data Particle
    that can be sent up to the InstrumentAgent.

    It is the intent that this class is subclassed as needed if an instrument must
    modify fields in the outgoing packet. The hope is to have most of the superclass
    code be called by the child class with just values overridden as needed.
    """

    # data particle type is intended to be defined in each derived data particle class.  This value should be unique
    # for all data particles.  Best practice is to access this variable using the accessor method:
    # data_particle_type()
    _data_particle_type = None

    def __init__(self, raw_data,
                 port_timestamp=None,
                 internal_timestamp=None,
                 preferred_timestamp=None,
                 quality_flag=DataParticleValue.OK,
                 new_sequence=None):
        """ Build a particle seeded with appropriate information

        @param raw_data The raw data used in the particle
        """
        if new_sequence is not None and not isinstance(new_sequence, bool):
            raise TypeError("new_sequence is not a bool")

        self.contents = {
            DataParticleKey.PKT_FORMAT_ID: DataParticleValue.JSON_DATA,
            DataParticleKey.PKT_VERSION: 1,
            DataParticleKey.PORT_TIMESTAMP: port_timestamp,
            DataParticleKey.INTERNAL_TIMESTAMP: internal_timestamp,
            DataParticleKey.DRIVER_TIMESTAMP: ntplib.system_to_ntp_time(time.time()),
            DataParticleKey.PREFERRED_TIMESTAMP: preferred_timestamp,
            DataParticleKey.QUALITY_FLAG: quality_flag,
        }
        self._encoding_errors = []
        if new_sequence is not None:
            self.contents[DataParticleKey.NEW_SEQUENCE] = new_sequence

        self.raw_data = raw_data
        self._values = None

    def __eq__(self, arg):
        """
        Quick equality check for testing purposes. If they have the same raw
        data, timestamp, they are the same enough for this particle
        """
        allowed_diff = .000001
        if self._data_particle_type != arg._data_particle_type:
            # log.debug('Data particle type does not match: %s %s', self._data_particle_type, arg._data_particle_type)
            print('Data particle type does not match: %s %s' % (self._data_particle_type, arg._data_particle_type))
            return False

        if self.raw_data != arg.raw_data:
            # log.debug('Raw data does not match')
            print('Raw data does not match')
            return False

        t1 = self.contents[DataParticleKey.INTERNAL_TIMESTAMP]
        t2 = arg.contents[DataParticleKey.INTERNAL_TIMESTAMP]

        if (t1 is None) or (t2 is None):
            tdiff = allowed_diff
        else:
            tdiff = abs(t1 - t2)

        if tdiff > allowed_diff:
            # log.debug('Timestamp %s does not match %s', t1, t2)
            print('Timestamp %s does not match %s' % (t1,t2))
            return False

        generated1 = json.loads(self.generate())
        generated2 = json.loads(arg.generate())
        missing, differing = self._compare(generated1, generated2, ignore_keys=[DataParticleKey.DRIVER_TIMESTAMP,
                                                                                DataParticleKey.PREFERRED_TIMESTAMP])
        if missing:
            # log.error('Key mismatch between particle dictionaries: %r', missing)
            print('Key mismatch between particle dictionaries: %r' % missing)
            return False

        if differing:
            # log.error('Value mismatch between particle dictionaries: %r', differing)
            print('Value mismatch between particle dictionaries: %r' % differing)

        return True

    @staticmethod
    def _compare(d1, d2, ignore_keys=None):
        ignore_keys = ignore_keys if ignore_keys else []
        missing = set(d1).symmetric_difference(d2)
        differing = {}
        for k in d1:
            if k in ignore_keys or k in missing:
                continue
            if d1[k] != d2[k]:
                differing[k] = (d1[k], d2[k])

        return missing, differing

    def set_internal_timestamp(self, timestamp=None, unix_time=None):
        """
        Set the internal timestamp
        @param timestamp: NTP timestamp to set
        @param unit_time: Unix time as returned from time.time()
        @raise InstrumentParameterException if timestamp or unix_time not supplied
        """
        if timestamp is None and unix_time is None:
            # raise InstrumentParameterException("timestamp or unix_time required")
            raise Exception("timestamp or unix_time required")

        if unix_time is not None:
            timestamp = ntplib.system_to_ntp_time(unix_time)

        # Do we want this to happen here or in down stream processes?
        # if(not self._check_timestamp(timestamp)):
        #    raise InstrumentParameterException("invalid timestamp")

        self.contents[DataParticleKey.INTERNAL_TIMESTAMP] = float(timestamp)

    def set_port_timestamp(self, timestamp=None, unix_time=None):
        """
        Set the port timestamp
        @param timestamp: NTP timestamp to set
        @param unix_time: Unix time as returned from time.time()
        @raise InstrumentParameterException if timestamp or unix_time not supplied
        """
        if timestamp is None and unix_time is None:
            # raise InstrumentParameterException("timestamp or unix_time required")
            raise Exception("timestamp or unix_time required")

        if unix_time is not None:
            timestamp = ntplib.system_to_ntp_time(unix_time)

        # Do we want this to happen here or in down stream processes?
        if not self._check_timestamp(timestamp):
            # raise InstrumentParameterException("invalid timestamp")
            raise Exception("invalid timestamp")

        self.contents[DataParticleKey.PORT_TIMESTAMP] = float(timestamp)

    def set_value(self, id, value):
        """
        Set a content value, restricted as necessary

        @param id The ID of the value to set, should be from DataParticleKey
        @param value The value to set
        @raises ReadOnlyException If the parameter cannot be set
        """
        if (id == DataParticleKey.INTERNAL_TIMESTAMP) and (self._check_timestamp(value)):
            self.contents[DataParticleKey.INTERNAL_TIMESTAMP] = value
        else:
            # raise ReadOnlyException("Parameter %s not able to be set to %s after object creation!" %
            #                         (id, value))
            raise Exception("Parameter %s not able to be set to %s after object creation!" % (id,value))

    def get_value(self, id):
        """ Return a stored value

        @param id The ID (from DataParticleKey) for the parameter to return
        @raises NotImplementedException If there is an invalid id
        """
        if DataParticleKey.has(id):
            return self.contents[id]
        else:
            # raise NotImplementedException("Value %s not available in particle!", id)
            raise Exception("Value %s not available in particle!" % id)

    def data_particle_type(self):
        """
        Return the data particle type (aka stream name)
        @raise: NotImplementedException if _data_particle_type is not set
        """
        if self._data_particle_type is None:
            # raise NotImplementedException("_data_particle_type not initialized")
            raise Exception("_data_particle_type not initialized")

        return self._data_particle_type

    def generate_dict(self):
        """
        Generate a simple dictionary of sensor data and timestamps, without
        going to JSON. This is useful for the times when JSON is not needed to
        go across an interface. There are times when particles are used
        internally to a component/process/module/etc.
        @retval A python dictionary with the proper timestamps and data values
        @throws InstrumentDriverException if there is a problem wtih the inputs
        """
        # verify preferred timestamp exists in the structure...
        if not self._check_preferred_timestamps():
            # raise SampleException("Preferred timestamp not in particle!")
            raise Exception("Preferred timestamp not in particle!")

        # build response structure
        self._encoding_errors = []
        if self._values is None:
            self._values = self._build_parsed_values()
        result = self._build_base_structure()
        result[DataParticleKey.STREAM_NAME] = self.data_particle_type()
        result[DataParticleKey.VALUES] = self._values

        return result

    def generate(self, sorted=False):
        """
        Generates a JSON_parsed packet from a sample dictionary of sensor data and
        associates a timestamp with it

        @param sorted Returned sorted json dict, useful for testing, but slow,
           so dont do it unless it is important
        @return A JSON_raw string, properly structured with port agent time stamp
           and driver timestamp
        @throws InstrumentDriverException If there is a problem with the inputs
        """
        json_result = json.dumps(self.generate_dict(), sort_keys=sorted)
        return json_result

    def _build_parsed_values(self):
        """
        Build values of a parsed structure. Just the values are built so
        so that a child class can override this class, but call it with
        super() to get the base structure before modification

        @return the values tag for this data structure ready to JSONify
        @raises SampleException when parsed values can not be properly returned
        """
        # raise SampleException("Parsed values block not overridden")
        raise Exception("Parsed values block not overriden")

    def _build_base_structure(self):
        """
        Build the base/header information for an output structure.
        Follow on methods can then modify it by adding or editing values.

        @return A fresh copy of a core structure to be exported
        """
        result = dict(self.contents)
        # clean out optional fields that were missing
        if not self.contents[DataParticleKey.PORT_TIMESTAMP]:
            del result[DataParticleKey.PORT_TIMESTAMP]
        if not self.contents[DataParticleKey.INTERNAL_TIMESTAMP]:
            del result[DataParticleKey.INTERNAL_TIMESTAMP]
        return result

    def _check_timestamp(self, timestamp):
        """
        Check to make sure the timestamp is reasonable

        @param timestamp An NTP4 formatted timestamp (64bit)
        @return True if timestamp is okay or None, False otherwise
        """
        if timestamp is None:
            return True
        if not isinstance(timestamp, float):
            return False

        # is it sufficiently in the future to be unreasonable?
        if timestamp > ntplib.system_to_ntp_time(time.time() + (86400 * 365)):
            return False
        else:
            return True

    def _check_preferred_timestamps(self):
        """
        Check to make sure the preferred timestamp indicated in the
        particle is actually listed, possibly adjusting to 2nd best
        if not there.

        @throws SampleException When there is a problem with the preferred
            timestamp in the sample.
        """
        if self.contents[DataParticleKey.PREFERRED_TIMESTAMP] is None:
            # raise SampleException("Missing preferred timestamp, %s, in particle" %
            #                       self.contents[DataParticleKey.PREFERRED_TIMESTAMP])
            raise Exception("Missing preferred timestamp, %s, in particle" %
                                   self.contents[DataParticleKey.PREFERRED_TIMESTAMP])

        # This should be handled downstream.  Don't want to not publish data because
        # the port agent stopped putting out timestamps
        # if self.contents[self.contents[DataParticleKey.PREFERRED_TIMESTAMP]] == None:
        #    raise SampleException("Preferred timestamp, %s, is not defined" %
        #                          self.contents[DataParticleKey.PREFERRED_TIMESTAMP])

        return True

    def _encode_value(self, name, value, encoding_function):
        """
        Encode a value using the encoding function, if it fails store the error in a queue
        """
        encoded_val = None

        try:
            encoded_val = encoding_function(value)
        except Exception as e:
            # log.error("Data particle error encoding. Name:%s Value:%s", name, value)
            print("Data particle error encoding. Name: %s Value: %s" % (name,value))
            self._encoding_errors.append({name: value})
        return {DataParticleKey.VALUE_ID: name,
                DataParticleKey.VALUE: encoded_val}

    def get_encoding_errors(self):
        """
        Return the encoding errors list
        """
        return self._encoding_errors

# def get_logging_metaclass(log_level='trace'):
#     class_map = {
#         'trace': LoggingMetaClass,
#         'debug': DebugLoggingMetaClass,
#         'info': InfoLoggingMetaClass,
#         'warn': WarnLoggingMetaClass,
#         'error': ErrorLoggingMetaClass,
#     }
#
#     return class_map.get(log_level, LoggingMetaClass)

# METACLASS = get_logging_metaclass('trace')


class CommonDataParticleType(BaseEnum):
    """
    This enum defines all the common particle types defined in the modules.  Currently there is only one, but by
    using an enum here we have the opportunity to define more common data particles.
    """
    RAW = "raw"


class DataParticleType(BaseEnum):
    """
    Data particle types produced by this driver
    """
    RAW = CommonDataParticleType.RAW

class ZplscCRecoveredDataParticle(DataParticle):
    # __metaclass__ = METACLASS

    def __init__(self, *args, **kwargs):
        super(ZplscCRecoveredDataParticle, self).__init__(*args, **kwargs)
        self._data_particle_type = DataParticleType.ZPLSC_C_PARTICLE_TYPE

    def _build_parsed_values(self):
        """
        Build parsed values for Instrument Data Particle.
        @return: list containing type encoded "particle value id:value" dictionary pairs
        """
        # Particle Mapping table, where each entry is a tuple containing the particle
        # field name, count(or count reference) and a function to use for data conversion.

        port_timestamp = self.raw_data[ZplscCParticleKey.TRANS_TIMESTAMP]
        self.contents[DataParticleKey.PORT_TIMESTAMP] = port_timestamp

        return [{DataParticleKey.VALUE_ID: name, DataParticleKey.VALUE: None}
                if self.raw_data[name] is None else
                {DataParticleKey.VALUE_ID: name, DataParticleKey.VALUE: value}
                for name, value in self.raw_data.iteritems()]



REF_TIME = date2num(datetime(1900, 1, 1, 0, 0, 0))

class ZPLSPlot(object):
    font_size_small = 14
    font_size_large = 18
    num_xticks = 25
    num_yticks = 7
    interplot_spacing = 0.1
    lower_percentile = 5
    upper_percentile = 95

    def __init__(self, data_times, channel_data_dict, frequency_dict, min_y, max_y, _min_db=None, _max_db=None):
        self.fig = None
        self.power_data_dict = self._transpose_and_flip(channel_data_dict)

        if (_min_db is None) or (_max_db is None):
            self.min_db, self.max_db = self._get_power_range(channel_data_dict)
        else:
            self.min_db = _min_db
            self.max_db = _max_db

        self.frequency_dict = frequency_dict

        # convert ntp time, i.e. seconds since 1900-01-01 00:00:00 to matplotlib time
        self.data_times = (data_times / (60 * 60 * 24)) + REF_TIME
        bin_size, _ = self.power_data_dict[1].shape
        self._setup_plot(min_y, max_y, bin_size)

    def generate_plots(self):
        """
        Generate plots for all channels in data set
        """
        # freq_to_channel = {v: k for k, v in self.frequency_dict.iteritems()}
        freq_to_channel = {v: k for k, v in self.frequency_dict.items()}  # Changed to Python3 syntax
        data_axes = []
        for index, frequency in enumerate(sorted(freq_to_channel)):
            channel = freq_to_channel[frequency]
            td_f = self.frequency_dict[channel]
            title = 'Volume Backscatter (Sv) :Channel #%d: Frequency: %.1f kHz' % (channel, td_f)
            data_axes.append(self._generate_plot(self.ax[index], self.power_data_dict[channel], title,
                                                 self.min_db[channel], self.max_db[channel]))

        if data_axes:
            self._display_x_labels(self.ax[-1], self.data_times)
            self.fig.tight_layout(rect=[0, 0.0, 0.97, 1.0])
            for index in range(len(data_axes)):
                self._display_colorbar(self.fig, data_axes[index], index)

    def write_image(self, filename):
        self.fig.savefig(filename)
        plt.close(self.fig)
        self.fig = None

    def _setup_plot(self, min_y, max_y, bin_size):
        # subset the yticks so that we don't plot every one
        yticks = np.linspace(0, bin_size, self.num_yticks)

        # create range vector (depth in meters)
        yticklabels = np.round(np.linspace(min_y, max_y, self.num_yticks)).astype(int)

        self.fig, self.ax = plt.subplots(len(self.frequency_dict), sharex='all', sharey='all')
        self.fig.subplots_adjust(hspace=self.interplot_spacing)
        self.fig.set_size_inches(40, 19)

        if not isinstance(self.ax, np.ndarray):
            self.ax = [self.ax]

        for axes in self.ax:
            axes.grid(False)
            axes.set_ylabel('depth (m)', fontsize=self.font_size_small)
            axes.set_yticks(yticks)
            axes.set_yticklabels(yticklabels, fontsize=self.font_size_small)
            axes.tick_params(axis="both", labelcolor="k", pad=4, direction='out', length=5, width=2)
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.spines['bottom'].set_visible(False)
            axes.spines['left'].set_visible(False)

    def _display_colorbar(self, fig, data_axes, order):
        # Add a colorbar to the specified figure using the data from the given axes
        num_freqs = len(self.frequency_dict)
        plot_bottom = 0.086
        verticle_space = 0.03
        height_factor = 0.0525

        # Calculate the position of the colorbar
        width = 0.01
        height = (1.0/num_freqs) - height_factor
        left = 0.965
        bottom = plot_bottom + ((num_freqs-order-1) * (verticle_space+height))

        ax = fig.add_axes([left, bottom, width, height])
        cb = fig.colorbar(data_axes, cax=ax, use_gridspec=True)
        cb.set_label('dB', fontsize=ZPLSPlot.font_size_large)
        cb.ax.tick_params(labelsize=ZPLSPlot.font_size_small)

    @staticmethod
    def _get_power_range(power_dict):
        # Calculate the power data range across each channel
        max_db = {}
        min_db = {}
        # for channel, channel_data in power_dict.iteritems():
        for channel, channel_data in power_dict.items():  # Changed to Python3 syntax
            all_power_data = np.concatenate(channel_data)
            max_db[channel] = np.nanpercentile(all_power_data, ZPLSPlot.upper_percentile)
            min_db[channel] = np.nanpercentile(all_power_data, ZPLSPlot.lower_percentile)

        return min_db, max_db

    @staticmethod
    def _transpose_and_flip(power_dict):
        for channel in power_dict:
            # Transpose array data so we have time on the x-axis and depth on the y-axis
            power_dict[channel] = power_dict[channel].transpose()
            # reverse the Y axis (so depth is measured from the surface (at the top) to the ZPLS (at the bottom)
            power_dict[channel] = power_dict[channel][::-1]
        return power_dict

    @staticmethod
    def _generate_plot(ax, power_data, title, min_db, max_db):
        """
        Generate a ZPLS plot for an individual channel
        :param ax:  matplotlib axis to receive the plot image
        :param power_data:  Transducer data array
        :param title:  plot title
        :param min_db: minimum power level
        :param max_db: maximum power level
        """
        # only generate plots for the transducers that have data
        if power_data.size <= 0:
            return

        ax.set_title(title, fontsize=ZPLSPlot.font_size_large)
        # return plt.imshow(ax, power_data, interpolation='none', aspect='auto', cmap='jet', vmin=min_db, vmax=max_db)
        # return plt.imshow(power_data, interpolation='none', aspect='auto', cmap='jet', vmin=min_db, vmax=max_db)  # Complained about ax as arg
        return ax.imshow(power_data, interpolation='none', aspect='auto', cmap='jet', vmin=min_db, vmax=max_db)

    @staticmethod
    def _display_x_labels(ax, data_times):
        time_format = '%Y-%m-%d\n%H:%M:%S'
        time_length = data_times.size
        # X axis label
        # subset the xticks so that we don't plot every one
        if time_length < ZPLSPlot.num_xticks:
            ZPLSPlot.num_xticks = time_length
        xticks = np.linspace(0, time_length, ZPLSPlot.num_xticks)
        xstep = int(round(xticks[1]))
        # format trans_array_time array so that it can be used to label the x-axis
        xticklabels = [i for i in num2date(data_times[::xstep])] + [num2date(data_times[-1])]
        xticklabels = [i.strftime(time_format) for i in xticklabels]

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        ax.set_xlabel('time (UTC)', fontsize=ZPLSPlot.font_size_small)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, horizontalalignment='center', fontsize=ZPLSPlot.font_size_small)
        ax.set_xlim(0, time_length)




# # de fichero  C:\Oceanhackweek\proyecto\AFZP_matlab\mi_instrument\mi\dataset\driver\zplsc_c\zplsc_functions.py
# decompress_power = np.array(power) * 10. * np.log10(2) / 256.
# vin = 2.5 * (counts / 65535)
# r = (ka + kb*vin) / (kc - vin)
# temperature = 1 / (a + b * (np.log(r)) + c * (np.log(r)**3)) - 273
# tilt = a + (b * counts) + (c * counts**2) + (d * counts**3)
# z = t/10
# sea_c = 1449.05 + (z * (45.7 + z*((-5.21) + 0.23*z))) + ((1.333 + z*((-0.126) + z*0.009)) * (s-35.0)) + \(p/1000)*(16.3+0.18*(p/1000))
# # Calculate relaxation frequencies
# t_k = t + 273.0
# f1 = 1320.0*t_k * np.exp(-1700/t_k)
# f2 = 1.55e7*t_k * np.exp(-3052/t_k)
#
# # Coefficients for absorption equations
# k = 1 + p/10.0
# a = 8.95e-8 * (1 + t*(2.29e-2 - 5.08e-4*t))
# b = (s/35.0)*4.88e-7*(1+0.0134*t)*(1-0.00103*k + 3.7e-7*(k*k))
# c = 4.86e-13*(1+t*((-0.042)+t*(8.53e-4-t*6.23e-6)))*(1+k*(-3.84e-4+k*7.57e-8))
# freqk = freq*1000
# sea_abs = (a*f1*(freqk**2))/((f1*f1)+(freqk**2))+(b*f2*(freqk**2))/((f2*f2)+(freqk**2))+c*(freqk**2)
#
# #aplicar compute_sv_offset(frequency, pulse_length)

