from collections import defaultdict
from .utils.ek_raw_io import RawSimradFile
import numpy as np
from .parse_base import ParseEK


class ParseEK60(ParseEK):
    """Class for converting data from Simrad EK60 echosounders.
    """

    def __init__(self, file, params):
        super().__init__(file, params)

        self.CON1_datagram = None

    def parse_raw(self):
        """Parse raw data file from Simrad EK60 echosounder.
        """

        with RawSimradFile(self.source_file, 'r') as fid:
            # Read the CON0 configuration datagram. Only keep 1 if multiple files
            if self.config_datagram is None:
                self.config_datagram = fid.read(1)
                self.config_datagram['timestamp'] = np.datetime64(
                    self.config_datagram['timestamp'].replace(tzinfo=None), '[ms]')
                self._print_status()

                self.ping_data_dict = defaultdict(lambda: defaultdict(list))
                for ch_num in self.config_datagram['transceivers'].keys():
                    self.ping_data_dict['frequency'][ch_num].append(
                        self.config_datagram['transceivers'][ch_num]['frequency'])

            else:
                fid.read(1)

            # Check if reading an ME70 file with a CON1 datagram.
            next_datagram = fid.peek()
            if next_datagram == 'CON1':
                self.CON1_datagram = fid.read(1)
            else:
                self.CON1_datagram = None

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
            for data_type in ['power', 'angle']:
                for k, v in self.ping_data_dict[data_type].items():
                    if all(x is None for x in v):  # if no data in a particular channel
                        self.ping_data_dict[data_type][k] = None
                    else:
                        self.ping_data_dict[data_type][k] = self.pad_shorter_ping(v)
                        if data_type == 'power':
                            self.ping_data_dict[data_type][k] = \
                                self.ping_data_dict[data_type][k].astype('float32') * INDEX2POWER

    def _select_datagrams(self, params):
        # Translates user input into specific datagrams or ALL
        if params == 'ALL':
            return ['ALL']
        elif params == 'GPS':
            return ['NME', 'GPS']
        else:
            raise ValueError("Unknown data type", params)
