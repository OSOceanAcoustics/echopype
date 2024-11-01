import numpy as np


from .parse_base import ParseEK
from .utils.ek_raw_io import RawSimradFile



class ParseEK60(ParseEK):
    """Class for converting data from Simrad EK60 echosounders."""

    def __init__(
        self,
        file,
        bot_file="",
        idx_file="",
        storage_options={},
        sonar_model="EK60",
        **kwargs,
    ):
        super().__init__(file, bot_file, idx_file, storage_options, sonar_model)



def is_ER60(raw_file, storage_options):
    """Check if a raw data file is from Simrad EK60 echosounder."""
    with RawSimradFile(raw_file, "r", storage_options=storage_options) as fid:
        config_datagram = fid.read(1)
        config_datagram["timestamp"] = np.datetime64(
            config_datagram["timestamp"].replace(tzinfo=None), "[ns]"
        )
        # Return True if the sounder name matches "ER60"
        try:
            return config_datagram["sounder_name"] in {"ER60"}
        except KeyError:
            return False



def is_EK60(raw_file, storage_options):

    """Check if a raw data file is from Simrad EK60 echosounder."""
    with RawSimradFile(raw_file, "r", storage_options=storage_options) as fid:
        config_datagram = fid.read(1)
        config_datagram["timestamp"] = np.datetime64(
            config_datagram["timestamp"].replace(tzinfo=None), "[ns]"
        )

        try:
            # Return True if the sounder name matches "EK60"
            return config_datagram["sounder_name"] in {"EK60"}
        except KeyError:
            return False

