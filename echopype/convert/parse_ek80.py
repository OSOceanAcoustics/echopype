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

        # TODO: @ngkvain: you use many of the below attributes in ParseBase,
        #  this breaks the object oriented structure
        #  e.g. mru, fil_coeffs, data_type, etc.
        self.mru = defaultdict(list)  # Dictionary to store MRU data (heading, pitch, roll, heave)
        self.fil_coeffs = defaultdict(dict)  # Dictionary to store PC and WBT coefficients
        self.fil_df = defaultdict(dict)  # Dictionary to store filter decimation factors
        self.ch_ids = []  # List of all channel ids
        self.bb_ch_ids = []
        self.cw_ch_ids = []
        self.recorded_ch_ids = []  # Channels where power data is present. Not necessarily the same as self.ch_ids
        self.sonar_type = 'EK80'
        self.data_type = self._select_datagrams(params)  # TODO: @ngkvain: you use this in ParseBase

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
            self.config_datagram['timestamp'] = np.datetime64(self.config_datagram['timestamp'], '[ms]')
            if 'EXPORT' in self.data_type:
                # TODO: @ngkavin: brittle, see change i did under convert.to_xml
                xml_type = 'environment' if 'ENV' in self.data_type else 'configuration'
                print(f"{dt.now().strftime('%H:%M:%S')} exporting {xml_type} XML file")
            else:
                self._print_status()

            # IDs of the channels found in the dataset
            self.ch_ids = list(self.config_datagram[self.config_datagram['subtype']])

            # Parameters recorded for each frequency for each ping
            # TODO: @ngkavin: you are initializing attribute outside out __init__ and this should be done in ParseBase
            self.ping_data_dict = defaultdict(lambda: defaultdict(list))

            # for ch_id in self.ch_ids:
            #     self.ping_data_dict['frequency'][ch_id].append(
            #         self.config_datagram['configuration'][ch_id]['transducer_frequency'])
            #     # TODO: @ngkavin: what is this -1 for? you need to add an explicit comment if doing this
            #     self.n_complex_dict[ch_id] = -1

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
            # TODO: @ngkavin:
            #  consolidate all code to clean up empty ping_data_dict
            #  and also do the determiniation of cw and bb mode there.
            self.bb_ch_ids, self.cw_ch_ids = self._sort_ch_bb_cw()

            # TODO: @ngkavin: converting to numpy array should be done in the parent class
            #  since it's the same for both EK60 and EK80
            self.nmea_time = np.array(self.nmea_time)
            self.raw_nmea_string = np.array(self.raw_nmea_string)

    # TODO: @ngkavin: functions called should be placed before the calling function
    def _sort_ch_bb_cw(self):
        """Sort which channels are broadband (BB) and continuous wave (CW).
        Returns a tuple containing a list of bb channel ids and a list of cw channel ids
        """
        bb_ch_ids = []
        cw_ch_ids = []
        for ch in self.ch_ids:
            if self.ping_data_dict['complex'][ch] is not None:
                bb_ch_ids.append(ch)
            elif self.ping_data_dict['power'][ch] is not None:
                cw_ch_ids.append(ch)
        return bb_ch_ids, cw_ch_ids
