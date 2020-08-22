from collections import defaultdict
from .utils.nmea_data import NMEAData

FILENAME_DATETIME_EK60 = '(?P<survey>.+)?-?D(?P<date>\w{1,8})-T(?P<time>\w{1,6})-?(?P<postfix>\w+)?.raw'
FILENAME_DATETIME_AZFP = '\w+.raw'
NMEA_GPS_SENTECE = 'GGA'


class ParseBase:
    """Parent class for all convert classes.
    """
    def __init__(self, files, params, compress=True, overwrite=False):
        self.source_file = files
        self.ui_param = params
        self.compress = compress
        self.overwrite = overwrite
        self.timestamp_pattern = None  # regex pattern used to grab datetime embedded in filename
        self.nmea_gps_sentence = None  # select GPS datagram in _set_platform_dict()

    def __str__(self):
        """Overload the print function to allow user to print basic properties of this object.

        Print out should include: source file name, source folder location, echosounder model.
        """

    def _print_status(self):
        """Prints message to console giving information about the raw file being parsed.
        """
        # TODO: this currently is only in convert/azfp.py and
        #  embedded in other methods for EK60 and EK80.
        #  Let's consolidate the same operation in one method.


class ParseEK(ParseBase):
    """Class for converting data from Simrad echosounders.
    """
    def __init__(self, files, params):
        super().__init__(files, params)

        # Parent class attributes
        self.timestamp_pattern = FILENAME_DATETIME_EK60  # regex pattern used to grab datetime embedded in filename
        self.nmea_gps_sentence = NMEA_GPS_SENTECE  # select GPS datagram in _set_platform_dict()

        # Class attributes
        self.config_datagram = None
        self.nmea_data = NMEAData()  # object for NMEA data
        self.ping_data_dict = {}  # dictionary to store metadata
        self.power_dict = {}     # dictionary to store power data
        self.angle_dict = {}     # dictionary to store angle data
        self.ping_time = []      # list to store ping time
        self.num_range_bin_groups = None  # number of range_bin groups

    def parse_raw(self):
        """This method calls private functions to parse the raw data file.
        """

    def _check_env_param_uniqueness(self):
        """Check if env parameters are unique throughout the file.
        """
        # Right now we don't handle changes in the environmental parameters and
        # just store the first set of unpacked env parameters.
        # We should have this function to check this formally, and save the env
        # parameters along the ping_time dimension if they change.
        # NOTE: discuss this in light of what AZFP has.

        # TODO:
        #  EK80: there is only 1 XML-environment datagram for each file
        #        so no need to check for uniqueness for the variables.
        #  EK60: the environment variables are wrapped in RAW0 datagrams
        #        for each ping -- see docstring _read_datagrams()
        #        so will have to check for uniqueness.
        #        variables to check are:
        #           - sound_velocity
        #        	- absorption_coefficient
        # 	        - temperature

    def _check_tx_param_uniqueness(self):
        """Check if transmit parameters are unique throughout the file.
        """
        # Right now we don't handle the case if any transmit parameters change in the middle of the file.
        # We also haven't run into any file that has that happening?
        # At minimum we should spit out an error message.
        # For EK60 the params are: pulse_length, transmit_power, bandwidth, sample_interval
        # For EK80 the params will include frequency-related params

        # TODO:
        #  EK80: each RAW3 datagrams is preceded by an XML-parameter datagram,
        #        which contains tx parameters for the following channel data
        #        -- see docstring _read_datagrams() for detail
        #        we will have to check for uniqueness for these variables.
        #  EK60: the environment variables are wrapped in RAW0 datagrams
        #        for each ping -- see docstring _read_datagrams()
        #        we will have to check for uniqueness for these parameters
        #        variables to check are:
        #           - transmit_power
        #           - pulse_length
        #           - bandwidth
        #           - sample_interval
        #           - transmit_mod(  # 0 = Active, 1 = Passive, 2 = Test, -1 = Unknown)
        #           - offset

    def _read_datagrams(self):
        """Read all datagrams.

        A sample EK60 RAW0 datagram:
            {'type': 'RAW0',
            'low_date': 71406392,
            'high_date': 30647127,
            'channel': 1,
            'mode': 3,
            'transducer_depth': 9.149999618530273,
            'frequency': 18000.0,
            'transmit_power': 2000.0,
            'pulse_length': 0.0010239999974146485,
            'bandwidth': 1573.66552734375,
            'sample_interval': 0.00025599999935366213,
            'sound_velocity': 1466.0,
            'absorption_coefficient': 0.0030043544247746468,
            'heave': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'temperature': 4.0,
            'heading': 0.0,
            'transmit_mode': 1,
            'spare0': '\x00\x00\x00\x00\x00\x00',
            'offset': 0,
            'count': 1386,
            'timestamp': numpy.datetime64('2018-02-11T16:40:25.276'),
            'bytes_read': 5648,
            'power': array([ -6876,  -8726, -11086, ..., -11913, -12522, -11799], dtype=int16),
            'angle': array([[ 110,   13],
                [   3,   -4],
                [ -54,  -65],
                ...,
                [ -92, -107],
                [-104, -122],
                [  82,   74]], dtype=int8)}

        A sample EK80 XML-parameter datagram:
            {'channel_id': 'WBT 545612-15 ES200-7C',
             'channel_mode': 0,
             'pulse_form': 1,
             'frequency_start': '160000',
             'frequency_end': '260000',
             'pulse_duration': 0.001024,
             'sample_interval': 5.33333333333333e-06,
             'transmit_power': 15.0,
             'slope': 0.01220703125}

        A sample EK80 XML-environment datagram:
            {'type': 'XML0',
             'low_date': 3137819385,
             'high_date': 30616609,
             'timestamp': numpy.datetime64('2017-09-12T23:49:10.723'),
             'bytes_read': 448,
             'subtype': 'environment',
             'environment': {'depth': 240.0,
              'acidity': 8.0,
              'salinity': 33.7,
              'sound_speed': 1486.4,
              'temperature': 6.9,
              'latitude': 45.0,
              'sound_velocity_profile': [1.0, 1486.4, 1000.0, 1486.4],
              'sound_velocity_source': 'Manual',
              'drop_keel_offset': 0.0,
              'drop_keel_offset_is_manual': 0,
              'water_level_draft': 0.0,
              'water_level_draft_is_manual': 0,
              'transducer_name': 'Unknown',
              'transducer_sound_speed': 1490.0},
             'xml': '<?xml version="1.0" encoding="utf-8"?>\r\n<Environment Depth="240" ... />\r\n</Environment>'}
        """

    def _append_channel_ping_data(self):
        """Append non-backscatter data for each ping.
        """
        # This is currently implemented for EK60, but not for EK80.
        # line 94-123 in convert/ek80.py can be made into this method:
        # TODO: simplify the calls `current_parameters['channel_id']` to
        #  `ch_id = current_parameters['channel_id']`
        #  and then just use ch_id, to increase code readbility

    def _find_range_group(self):
        """Find the pings at which range_bin changes.
        """
        # Use this function to check if all pings in this file contains the same range_bin.
        # If not, find out the index of which changes happen.

    def _check_ping_channel_match(self):
        """Check if the number of RAW datagrams loaded are integer multiples of the number of channels.
        """
        # Check line 312 of convert/ek60.py

    def _clean_channel(self):
        """Remove channels that do not record any pings.
        """
        # This seems to be what line 211 of convert/ek80.py is doing?


class ParseEK60(ParseEK):
    """Class for converting data from Simrad EK60 echosounders.
    """
    def parse_raw(self):
        """Parse raw data file from Simrad EK60 echosounder.
        """


class ParseEK80(ParseEK):
    """Class for converting data from Simrad EK60 echosounders.
    """
    def __init__(self, files, params):
        super().__init__(files, params)
        self.complex_dict = {}  # dictionary to store complex data
        self.n_complex_dict = {}  # dictionary to store the number of beams in split-beam complex data
        self.environment = {}  # dictionary to store environment data
        # self.parameters = defaultdict(dict)  # Dictionary to hold parameter data --> use self.ping_data_dict
        self.mru = defaultdict(list)  # Dictionary to store MRU data (heading, pitch, roll, heave)
        self.fil_coeffs = defaultdict(dict)  # Dictionary to store PC and WBT coefficients
        self.fil_df = defaultdict(dict)  # Dictionary to store filter decimation factors
        self.ch_ids = []  # List of all channel ids
        self.recorded_ch_ids = []  # TODO: what is the difference between this and ch_ids?

        # TODO: we shouldn't need `power_dict_split`, `angle_dict_split` and `ping_time_split`,
        #  and can just get the values from the original corresponding storage values
        #  `power_dict`, `angle_dict`, `ping_time`

    def parse_raw(self):
        """Parse raw data file from Simrad EK80 echosounder.
        """
        # TODO: line 191 block can be substitute by `self.parameters[ch_id] = defaultdict(list)`

    def _sort_ch_bb_cw(self):
        """Sort which channels are broadband (BB) and continuous wave (CW).
        """


class ParseAZFP(ParseBase):
    """Class for converting data from ASL Environmental Sciences AZFP echosounder.
    """
    def __init__(self, files, params):
        super().__init__(files, params)
        # Parent class attributes
        self.timestamp_pattern = FILENAME_DATETIME_EK60  # regex pattern used to grab datetime embedded in filename

        # TODO: the following constants should just be constants on top of azfp.py and not
        #  class attributes since they do not change.
        self.FILE_TYPE = 64770
        self.HEADER_SIZE = 124
        self.HEADER_FORMAT = ">HHHHIHHHHHHHHHHHHHHHHHHHHHHHHHHHHHBBBBHBBBBBBBBHHHHHHHHHHHHHHHHHHHH"

        # Class attributes
        self.parameters = dict()
        self.unpacked_data = None

    def loadAZFPxml(self):
        """Parse XML file to get params for reading AZFP data."""

    def parse_raw(self):
        """Parse raw data file from AZFP echosounder.
        """

    def _get_fields(self):
        """Returns the fields contained in each header of the raw file.
        """

    def _print_status(self):
        """Prints message to console giving information about the raw file being parsed.
        """
        # TODO: should use self.unpacked_data instead of passing in that separately.

    def _split_header(self):
        """Splits the header information into a dictionary.
        """
        # TODO: the current _split_header(self, raw, header_unpacked, unpacked_data, fields)
        #  abuses the class structure --> just use self.unpacked_data within this method

    def _add_counts(self):
        """Unpacks the echosounder raw data. Modifies unpacked_data in place.
        """
        # TODO: similar to _split_header(), just use self.unpacked_data within this method

    def _check_uniqueness(self):
        """Check for ping-by-ping consistency of sampling parameters and reduce if identical.
        """

    def _get_ping_time(self):
        """Assemble ping time from parsed values.
        """

    @staticmethod
    def _calc_Sv_offset():
        """Calculate the compensation factor for Sv calculation.
        """
        # TODO: this method seems should be in echopype.process
