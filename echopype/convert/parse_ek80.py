from collections import defaultdict
from .utils.ek_raw_io import RawSimradFile
from datetime import datetime as dt
import numpy as np
from .parse_base import ParseEK


class ParseEK80(ParseEK):
    """Class for converting data from Simrad EK80 echosounders.
    """
    def __init__(self, file, params):
        super().__init__(file, params)
        self.n_complex_dict = {}  # dictionary to store the number of beams in split-beam complex data
        self.environment = {}  # dictionary to store environment data
        # self.parameters = defaultdict(dict)  # Dictionary to hold parameter data --> use self.ping_data_dict
        self.mru = defaultdict(list)  # Dictionary to store MRU data (heading, pitch, roll, heave)
        self.fil_coeffs = defaultdict(dict)  # Dictionary to store PC and WBT coefficients
        self.fil_df = defaultdict(dict)  # Dictionary to store filter decimation factors
        self.ch_ids = []  # List of all channel ids
        self.bb_ch_ids = []
        self.cw_ch_ids = []
        self.recorded_ch_ids = []  # Channels where power data is present. Not necessarily the same as self.ch_ids
        self.sonar_type = 'EK80'
        self.data_type = self._select_datagrams(params)

    def parse_raw(self):
        """Parse raw data file from Simrad EK80 echosounder.
        """
        with RawSimradFile(self.source_file, 'r') as fid:
            self.config_datagram = fid.read(1)
            self.config_datagram['timestamp'] = np.datetime64(self.config_datagram['timestamp'], '[ms]')
            if 'EXPORT' in self.data_type:
                xml_type = 'environment' if 'ENV' in self.data_type else 'configuration'
                print(f"{dt.now().strftime('%H:%M:%S')} exporting {xml_type} XML file")
            else:
                self._print_status()

            # IDs of the channels found in the dataset
            self.ch_ids = list(self.config_datagram[self.config_datagram['subtype']])

            # Parameters recorded for each frequency for each ping
            self.ping_data_dict = defaultdict(lambda: defaultdict(list))

            for ch_id in self.ch_ids:
                self.ping_data_dict['frequency'][ch_id].append(
                    self.config_datagram['configuration'][ch_id]['transducer_frequency'])
                self.n_complex_dict[ch_id] = -1

            # Read the rest of datagrams
            self._read_datagrams(fid)
            # Remove empty lists
            for ch_id in self.ch_ids:
                if all(x is None for x in self.ping_data_dict['power'][ch_id]):
                    self.ping_data_dict['power'][ch_id] = None
                    self.ping_data_dict['angle'][ch_id] = None
                if all(x is None for x in self.ping_data_dict['complex'][ch_id]):
                    self.ping_data_dict['complex'][ch_id] = None

        if 'ALL' in self.data_type:
            self._clean_channel()
            # Save which channel ids are bb and which are ch because rectangularize() removes channel ids
            self.bb_ch_ids, self.cw_ch_ids = self._sort_ch_bb_cw()
            self.ping_data_dict['power'], self.ping_data_dict['angle'] = self._rectangularize(
                self.ping_data_dict['power'], self.ping_data_dict['angle'])
            self.ping_data_dict['complex'], _ = self._rectangularize(self.ping_data_dict['complex'])

            self.nmea_time = np.array(self.nmea_time)
            self.raw_nmea_string = np.array(self.raw_nmea_string)

    def _sort_ch_bb_cw(self):
        """Sort which channels are broadband (BB) and continuous wave (CW).
        Returns a tuple containing a list of bb channel ids and a list of cw channel ids
        """
        bb_ch_ids = []
        cw_ch_ids = []
        for k, v in self.ping_data_dict['complex'].items():
            if v is not None:
                bb_ch_ids.append(k)
            else:
                if self.ping_data_dict['power'][k] is not None:
                    cw_ch_ids.append(k)
        return bb_ch_ids, cw_ch_ids
