from datetime import datetime as dt
from typing import Dict

from _echopype_version import version as ECHOPYPE_VERSION
from typing_extensions import Literal

ProcessType = Literal["conversion", "processing"]


def echopype_prov_attrs(process_type: ProcessType, source_files: str = None) -> Dict[str, str]:
    """
    Standard echopype software attributes for provenance

    Parameters
    ----------
    process_type : ProcessType
        Echopype process function type
    source_files: str
        Source file path. A list of files is not currently supported
    """
    prov_dict = {
        f"{process_type}_software_name": "echopype",
        f"{process_type}_software_version": ECHOPYPE_VERSION,
        f"{process_type}_time": dt.utcnow().isoformat(timespec="seconds") + "Z",  # use UTC time
    }
    if source_files:
        # TODO: src_filenames will be replaced with a new variable, source_filenames
        #   Also, come to think of it, source files is not "echopype provenance" info per se
        prov_dict["src_filenames"] = source_files

    return prov_dict
