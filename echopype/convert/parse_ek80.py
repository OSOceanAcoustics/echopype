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

        self.ch_ids = []  # List of all channel ids
        self.bb_ch_ids = []
        self.cw_ch_ids = []

    def _remove_unrecorded_channels(self):
        """If channels record real power data and some record complex power data, then remove
        the channels with real power data from the ping_data of the complex data and vice versa.
        This also gives the names the of the bb and cw channels."""
        # The ping_data that will have unused channels be removed
        params = ['complex', 'power', 'angle']
        for param in params:
            self.ping_data_dict[param] = {k: v for k, v in self.ping_data_dict[param].items()
                                          if v[0] is not None}
        self.bb_ch_ids = list(self.ping_data_dict['complex'].keys())
        self.cw_ch_ids = list(self.ping_data_dict['power'].keys())
        self.n_complex_dict = {k: v for k, v in self.n_complex_dict.items() if k in self.bb_ch_ids}

    def parse_raw(self):
        """Parse raw data file from Simrad EK80 echosounder.
        """
        with RawSimradFile(self.source_file, 'r') as fid:
            self.config_datagram = fid.read(1)
            self.config_datagram['timestamp'] = np.datetime64(
                self.config_datagram['timestamp'].replace(tzinfo=None), '[ms]')

            if 'print_export_msg' in self.data_type:
                if 'ENV' in self.data_type:
                    xml_type = 'environment'
                elif 'CONFIG' in self.data_type:
                    xml_type = 'configuration'
                print(f"{dt.now().strftime('%H:%M:%S')} exporting {xml_type} XML file")
            else:
                self._print_status()

            # IDs of the channels found in the dataset
            self.ch_ids = list(self.config_datagram[self.config_datagram['subtype']])

            # Parameters recorded for each frequency for each ping
            for ch_id in self.ch_ids:
                self.ping_data_dict['frequency'][ch_id].append(
                    self.config_datagram['configuration'][ch_id]['transducer_frequency'])
                self.n_complex_dict[ch_id] = None

            # Read the rest of datagrams
            self._read_datagrams(fid)

        if 'ALL' in self.data_type:
            # TODO: @ngkavin: the next 2 lines can be removed
            #  if you take the assembling a Dataset approach detailed in _rectangularize()
            self.ping_time = np.unique(self.ping_time)
            self._match_ch_ping_time()
            # Save which channel ids are bb and which are ch because rectangularize() removes channel ids
            # Also removes cw channels from bb data and bb channels from cw data
            self._remove_unrecorded_channels()

            # Create rectangular arrays from a dictionary of potentially irregular arrays.
            if self.bb_ch_ids:
                self.ping_data_dict['complex'] = self._rectangularize(self.ping_data_dict['complex'])
            if self.cw_ch_ids:
                self.ping_data_dict['power'] = self._rectangularize(self.ping_data_dict['power'])
                self.ping_data_dict['angle'] = self._rectangularize(self.ping_data_dict['angle'], is_power=False)

    def _select_datagrams(self, params):
        """ Translates user input into specific datagrams or ALL
        """
        def translate_to_dgram(s):
            if s == 'ALL':
                return ['ALL']
            elif s == 'GPS':
                return ['NME', 'MRU', 'GPS']
            elif s == 'CONFIG':
                return ['CONFIG']
            elif s == 'ENV':
                return ['ENV']
            # EXPORT_XML flag passed in only by the to_xml function
            # Used to print the export message when writing to an xml file
            elif s == 'EXPORT_XML':
                return ['print_export_msg']
            else:
                raise ValueError(f"Unknown data type", params)
        # Params is a string when user sets data_type in to_netcdf/to_zarr
        if isinstance(params, str):
            dgrams = translate_to_dgram(params)
        # Params is a list when the parse classes are called by to_xml
        else:
            dgrams = []
            for p in params:
                dgrams += translate_to_dgram(p)
        return dgrams
