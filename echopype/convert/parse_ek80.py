from .parse_base import ParseEK


class ParseEK80(ParseEK):
    """Class for converting data from Simrad EK80 echosounders."""

    def __init__(
        self,
        file,
        bot_file="",
        idx_file="",
        storage_options={},
        sonar_model="EK80",
        **kwargs,
    ):
        super().__init__(file, bot_file, idx_file, storage_options, sonar_model)
        self.environment = {}  # dictionary to store environment data
