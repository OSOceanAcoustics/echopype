from collections import defaultdict
from .utils.ek_raw_io import RawSimradFile
from .utils.ek_raw_io import SimradEOF
from datetime import datetime as dt
import numpy as np
import os

FILENAME_DATETIME_EK60 = '(?P<survey>.+)?-?D(?P<date>\\w{1,8})-T(?P<time>\\w{1,6})-?(?P<postfix>\\w+)?.raw'


class ParseBase:
    """Parent class for all convert classes.
    """
    def __init__(self, file, storage_options):
        self.source_file = file
        self.timestamp_pattern = None  # regex pattern used to grab datetime embedded in filename
        self.ping_time = []            # list to store ping time
        self.storage_options = storage_options

    def _print_status(self):
        """Prints message to console giving information about the raw file being parsed.
        """


class ParseEK(ParseBase):
    """Class for converting data from Simrad echosounders.
    """
    def __init__(self, file, params, storage_options):
        super().__init__(file, storage_options)

        # Parent class attributes
        self.timestamp_pattern = FILENAME_DATETIME_EK60  # regex pattern used to grab datetime embedded in filename

        # Class attributes
        self.config_datagram = None
        self.ping_data_dict = defaultdict(lambda: defaultdict(list))
        self.ping_time = defaultdict(list)  # store ping time according to channel
        self.num_range_bin_groups = None  # number of range_bin groups
        self.ch_ids = defaultdict(list)   # Stores the channel ids for each data type (power, angle, complex)
        self.data_type = self._select_datagrams(params)

        self.nmea = defaultdict(list)   # Dictionary to store NMEA data(timestamp and string)
        self.mru = defaultdict(list)  # Dictionary to store MRU data (heading, pitch, roll, heave)
        self.fil_coeffs = defaultdict(dict)  # Dictionary to store PC and WBT coefficients
        self.fil_df = defaultdict(dict)  # Dictionary to store filter decimation factors

        self.CON1_datagram = None   # Holds the ME70 CON1 datagram

    def _print_status(self):
        time = self.config_datagram['timestamp'].astype(dt).strftime("%Y-%b-%d %H:%M:%S")
        print(f"{dt.now().strftime('%H:%M:%S')}  parsing file {os.path.basename(self.source_file)}, "
              f"time of first ping: {time}")

    def parse_raw(self):
        """Parse raw data file from Simrad EK60, EK80, and EA640 echosounders.
        """
        with RawSimradFile(self.source_file,
                           'r', storage_options=self.storage_options) as fid:
            self.config_datagram = fid.read(1)
            self.config_datagram['timestamp'] = np.datetime64(
                self.config_datagram['timestamp'].replace(tzinfo=None), '[ms]')
            if "configuration" in self.config_datagram:
                for v in self.config_datagram["configuration"].values():
                    if "pulse_duration" not in v and "pulse_length" in v:
                        # it seems like sometimes this field can appear with the name "pulse_length"
                        # and in the form of floats separated by semicolons
                        v["pulse_duration"] = [float(x) for x in v["pulse_length"].split(";")]

            # If exporting to XML file (EK80/EA640 only), print a message
            if 'print_export_msg' in self.data_type:
                if 'ENV' in self.data_type:
                    xml_type = 'environment'
                elif 'CONFIG' in self.data_type:
                    xml_type = 'configuration'
                print(f"{dt.now().strftime('%H:%M:%S')} exporting {xml_type} XML file")
                # Don't parse anything else if only the config xml is required.
                if 'CONFIG' in self.data_type:
                    return
            # If not exporting to XML, print the usual converting message
            else:
                self._print_status()

            # Check if reading an ME70 file with a CON1 datagram.
            next_datagram = fid.peek()
            if next_datagram == 'CON1':
                self.CON1_datagram = fid.read(1)
            else:
                self.CON1_datagram = None

            # IDs of the channels found in the dataset
            # self.ch_ids = list(self.config_datagram['configuration'].keys())

            # Read the rest of datagrams
            self._read_datagrams(fid)

        if 'ALL' in self.data_type:
            # Convert ping time to 1D numpy array, stored in dict indexed by channel,
            #  this will help merge data from all channels into a cube
            for ch, val in self.ping_time.items():
                self.ping_time[ch] = np.array(val)

            # Manufacturer-specific power conversion factor
            INDEX2POWER = (10.0 * np.log10(2.0) / 256.0)

            # Rectangularize all data and convert to numpy array indexed by channel
            for data_type in ['power', 'angle', 'complex']:
                for k, v in self.ping_data_dict[data_type].items():
                    if all(x is None for x in v):  # if no data in a particular channel
                        self.ping_data_dict[data_type][k] = None
                    else:
                        # Sort complex and power/angle channels
                        self.ch_ids[data_type].append(k)
                        self.ping_data_dict[data_type][k] = self.pad_shorter_ping(v)
                        if data_type == 'power':
                            self.ping_data_dict[data_type][k] = \
                                self.ping_data_dict[data_type][k].astype('float32') * INDEX2POWER

    def _read_datagrams(self, fid):
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
        num_datagrams_parsed = 0

        while True:
            try:
                # TODO: @ngkvain: what I need in the code to not PARSE the raw0/3 datagram
                #  when users only want CONFIG or ENV, but the way this is implemented
                #  the raw0/3 datagrams are still parsed, you are just not saving them
                new_datagram = fid.read(1)

            except SimradEOF:
                break

            # Convert the timestamp to a datetime64 object.
            new_datagram['timestamp'] = np.datetime64(new_datagram['timestamp'].replace(tzinfo=None), '[ms]')

            num_datagrams_parsed += 1

            # Skip any datagram that the user does not want to save

            if (not any(new_datagram['type'].startswith(dgram) for dgram in self.data_type) and
               'ALL' not in self.data_type):
                continue
            # XML datagrams store environment or instrument parameters for EK80
            if new_datagram['type'].startswith("XML"):
                if new_datagram['subtype'] == 'environment' and ('ENV' in self.data_type or 'ALL' in self.data_type):
                    self.environment = new_datagram['environment']
                    self.environment['xml'] = new_datagram['xml']
                    self.environment['timestamp'] = new_datagram['timestamp']
                    # Don't parse anything else if only the environment xml is required.
                    if 'ENV' in self.data_type:
                        break
                elif new_datagram['subtype'] == 'parameter' and ('ALL' in self.data_type):
                    current_parameters = new_datagram['parameter']

            # RAW0 datagrams store raw acoustic data for a channel for EK60
            elif new_datagram['type'].startswith('RAW0'):
                # Save channel-specific ping time. The channels are stored as 1-based indices
                self.ping_time[new_datagram['channel']].append(new_datagram['timestamp'])

                # Append ping by ping data
                self._append_channel_ping_data(new_datagram)

            # RAW3 datagrams store raw acoustic data for a channel for EK80
            elif new_datagram['type'].startswith('RAW3'):
                curr_ch_id = new_datagram['channel_id']
                # Check if the proceeding Parameter XML does not match with data in this RAW3 datagram
                if current_parameters['channel_id'] != curr_ch_id:
                    raise ValueError("Parameter ID does not match RAW")

                # Save channel-specific ping time
                self.ping_time[curr_ch_id].append(new_datagram['timestamp'])

                # Append ping by ping data
                new_datagram.update(current_parameters)
                self._append_channel_ping_data(new_datagram)

            # NME datagrams store ancillary data as NMEA-0817 style ASCII data.
            elif new_datagram['type'].startswith('NME'):
                self.nmea['timestamp'].append(new_datagram['timestamp'])
                self.nmea['nmea_string'].append(new_datagram['nmea_string'])

            # MRU datagrams contain motion data for each ping for EK80
            elif new_datagram['type'].startswith("MRU"):
                self.mru['heading'].append(new_datagram['heading'])
                self.mru['pitch'].append(new_datagram['pitch'])
                self.mru['roll'].append(new_datagram['roll'])
                self.mru['heave'].append(new_datagram['heave'])
                self.mru['timestamp'].append(new_datagram['timestamp'])

            # FIL datagrams contain filters for proccessing bascatter data for EK80
            elif new_datagram['type'].startswith("FIL"):
                self.fil_coeffs[new_datagram['channel_id']][new_datagram['stage']] = new_datagram['coefficients']
                self.fil_df[new_datagram['channel_id']][new_datagram['stage']] = new_datagram['decimation_factor']

            # TAG datagrams contain time-stamped annotations inserted via the recording software
            elif new_datagram['type'].startswith('TAG'):
                print('TAG datagram encountered.')

            # BOT datagrams contain sounder detected bottom depths from .bot files
            elif new_datagram['type'].startswith('BOT'):
                print('BOT datagram encountered.')

            # DEP datagrams contain sounder detected bottom depths from .out files
            # as well as reflectivity data
            elif new_datagram['type'].startswith('DEP'):
                print('DEP datagram encountered.')
            else:
                print("Unknown datagram type: " + str(new_datagram['type']))

    def _append_channel_ping_data(self, datagram):
        """Append ping by ping data.
        """
        # TODO: do a thorough check with the convention and processing
        # unsaved = ['channel', 'channel_id', 'low_date', 'high_date', # 'offset', 'frequency' ,
        #            'transmit_mode', 'spare0', 'bytes_read', 'type'] #, 'n_complex']
        ch_id = datagram['channel_id'] if 'channel_id' in datagram else datagram['channel']
        for k, v in datagram.items():
            # if k not in unsaved:
            self.ping_data_dict[k][ch_id].append(v)

    @staticmethod
    def pad_shorter_ping(data_list) -> np.ndarray:
        """
        Pad shorter ping with NaN: power, angle, complex samples.

        Parameters
        ----------
        data_list : list
            Power, angle, or complex samples for each channel from RAW3 datagram.
            Each ping is one entry in the list.

        Returns
        -------
        out_array : np.ndarray
            Numpy array containing samplings from all pings.
            The array is NaN-padded if some pings are of different lengths.
        """
        lens = np.array([len(item) for item in data_list])
        if np.unique(lens).size != 1:  # if some pings have different lengths along range
            if data_list[0].ndim == 2:
                # Angle data have an extra dimension for alongship and athwartship samples
                mask = lens[:, None, None] > np.array([np.arange(lens.max())] * 2).T
            else:
                mask = lens[:, None] > np.arange(lens.max())
            # Take care of problem of np.nan being implicitly "real"
            if isinstance(data_list[0][0], (np.complex, np.complex64, np.complex128)):
                out_array = np.full(mask.shape, np.nan + 0j)
            else:
                out_array = np.full(mask.shape, np.nan)

            # Fill in values
            out_array[mask] = np.concatenate(data_list).reshape(-1)  # reshape in case data > 1D
        else:
            out_array = np.array(data_list)
        return out_array

    def _select_datagrams(self, params):
        """ Translates user input into specific datagrams or ALL

        Valid use cases:
        # get GPS info only (EK60, EK80)
        # ec.to_netcdf(data_type='GPS')

        # get configuration XML only (EK80)
        # ec.to_netcdf(data_type='CONFIG')

        # get environment XML only (EK80)
        # ec.to_netcdf(data_type='ENV')
        """
