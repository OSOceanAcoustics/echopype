from .parse_base import ParseEK


class ParseEK60(ParseEK):
    """Class for converting data from Simrad EK60 echosounders.
    """

    def __init__(self, file, params, storage_options={}):
        super().__init__(file, params, storage_options)

    def _select_datagrams(self, params):
        # Translates user input into specific datagrams or ALL
        if params == 'ALL':
            return ['ALL']
        elif params == 'GPS':
            return ['NME']
        else:
            raise ValueError("Unknown data type", params)
