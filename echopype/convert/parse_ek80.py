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
            self.ch_ids = list(self.config_datagram['configuration'].keys())

            # TODO remove?
            # Parameters recorded for each frequency for each ping
            # for ch_id in self.ch_ids:
            #     self.ping_data_dict['frequency'][ch_id].append(
            #         self.config_datagram['configuration'][ch_id]['transducer_frequency'])
            #     self.n_complex_dict[ch_id] = None

            # Read the rest of datagrams
            self._read_datagrams(fid)

        if 'ALL' in self.data_type:
            # Convert ping time to 1D numpy array, stored in dict indexed by channel,
            #  this will help merge data from all channels into a cube
            for ch, val in self.ping_time.items():
                self.ping_time[ch] = np.array(val)

            # Rectangularize all data and convert to numpy array indexed by channel
            for data_type in ['power', 'angle', 'complex']:
                print(data_type)
                for k, v in self.ping_data_dict[data_type].items():
                    print(k)
                    if all(x is None for x in v):  # if no data in a particular channel
                        self.ping_data_dict[data_type][k] = None
                    else:
                        self.ping_data_dict[data_type][k] = self.pad_shorter_ping(v)

            # Save which channel ids are bb and which are ch because rectangularize() removes channel ids
            # Also removes cw channels from bb data and bb channels from cw data
            self._remove_unrecorded_channels()

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
