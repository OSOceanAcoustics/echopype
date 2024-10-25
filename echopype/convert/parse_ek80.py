import numpy as np

from .parse_base import ParseEK
from .utils.ek_raw_io import RawSimradFile


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


def is_EK80(raw_file, storage_options):
    """Parse raw data to check if it is from Simrad EK80 echosounder."""
    with RawSimradFile(raw_file, "r", storage_options=storage_options) as fid:
        config_datagram = fid.read(1)
        config_datagram["timestamp"] = np.datetime64(
            config_datagram["timestamp"].replace(tzinfo=None), "[ns]"
        )

        # Only EK80 files have configuration in self.config_datagram
        if "configuration" in config_datagram:
            return True
        else:
            return False
