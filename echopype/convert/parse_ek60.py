from collections import defaultdict
from .utils.ek_raw_io import RawSimradFile
import numpy as np
from .parse_base import ParseEK


class ParseEK60(ParseEK):
    """Class for converting data from Simrad EK60 echosounders.
    """

    def __init__(self, file, params):
        super().__init__(file, params)
        self.sonar_type = 'EK60'
        self.data_type = self._select_datagrams(params)

    def parse_raw(self):
        """Parse raw data file from Simrad EK60 echosounder.
        """

        with RawSimradFile(self.source_file, 'r') as fid:
            # Read the CON0 configuration datagram. Only keep 1 if multiple files
            if self.config_datagram is None:
                self.config_datagram = fid.read(1)
                # TODO: @ngkavin: different from EK80 code, why?
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
                self.CON1_datagram = None   # TODO: @ngkavin: not initialized

            # Read the rest of datagrams
            self._read_datagrams(fid)

        if 'ALL' in self.data_type:
            # Make a regctangular array (when there is a switch of range_bin in the middle of a file
            # or when range_bin size changes across channels)
            # TODO: @ngkavin: change after _rectangularize() is updated
            self._match_ch_ping_time()
            self.ping_data_dict['power'], self.ping_data_dict['angle'] = self._rectangularize(
                self.ping_data_dict['power'], self.ping_data_dict['angle'])

            # TODO: @ngkavin: converting to numpy array should be done in the parent class
            #  since it's the same for both EK60 and EK80
            self.nmea_time = np.array(self.nmea_time)
            self.raw_nmea_string = np.array(self.raw_nmea_string)
