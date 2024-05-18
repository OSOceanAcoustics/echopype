from .parse_base import ParseEK


class ParseEK60(ParseEK):
    """Class for converting data from Simrad EK60 echosounders."""

    def __init__(self, file, file_meta, bot_file, idx_file, storage_options={}, sonar_model="EK60"):
        super().__init__(file, bot_file, idx_file, storage_options, sonar_model)
