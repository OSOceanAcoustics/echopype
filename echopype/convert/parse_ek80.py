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
        self.environment = {}  # dictionary to store environment data

        # List of all ch ids split into power [cw] and complex [bb]
        self.ch_ids = defaultdict(list)

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
            # self.ch_ids = list(self.config_datagram['configuration'].keys())

            # Read the rest of datagrams
            self._read_datagrams(fid)

        if 'ALL' in self.data_type:
            # Convert ping time to 1D numpy array, stored in dict indexed by channel,
            #  this will help merge data from all channels into a cube
            for ch, val in self.ping_time.items():
                self.ping_time[ch] = np.array(val)

            # Rectangularize all data and convert to numpy array indexed by channel
            for data_type in ['power', 'angle', 'complex']:
                for k, v in self.ping_data_dict[data_type].items():
                    if all(x is None for x in v):  # if no data in a particular channel
                        self.ping_data_dict[data_type][k] = None
                    else:
                        # Sort bb and cw channels
                        self.ch_ids[data_type].append(k)
                        self.ping_data_dict[data_type][k] = self.pad_shorter_ping(v)

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
