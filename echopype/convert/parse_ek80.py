from .parse_base import ParseEK


class ParseEK80(ParseEK):
    """Class for converting data from Simrad EK80 echosounders.
    """
    def __init__(self, file, params, storage_options={}):
        super().__init__(file, params, storage_options)
        self.environment = {}  # dictionary to store environment data

    def _select_datagrams(self, params):
        """ Translates user input into specific datagrams or ALL
        """
        def translate_to_dgram(s):
            if s == 'ALL':
                return ['ALL']
            # The GPS flag indicates that only the NME and MRU datagrams are parsed.
            # It is kept in the list because it is used in SetGroups to flag that only the platform group is saved.
            elif s == 'GPS':
                return ['NME', 'MRU']
            # CONFIG flag indicates that only the configuration XML is parsed.
            # The XML flag is not needed because the configuration is always the first datagram parsed.
            elif s == 'CONFIG':
                return ['CONFIG']
            # XML flag indicates that XML0 datagrams should be read.
            # ENV flag indicates that of the XML datagrams, only keep the environment datagrams
            elif s == 'ENV':
                return ['XML', 'ENV']
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
