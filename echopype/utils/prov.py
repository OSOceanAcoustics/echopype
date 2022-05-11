from datetime import datetime as dt
from typing import Dict, List, Tuple, Any, Union

from _echopype_version import version as ECHOPYPE_VERSION
from typing_extensions import Literal

ProcessType = Literal["conversion", "processing"]


def echopype_prov_attrs(process_type: ProcessType) -> Dict[str, str]:
    """
    Standard echopype software attributes for provenance

    Parameters
    ----------
    process_type : ProcessType
        Echopype process function type
    """
    prov_dict = {
        f"{process_type}_software_name": "echopype",
        f"{process_type}_software_version": ECHOPYPE_VERSION,
        f"{process_type}_time": dt.utcnow().isoformat(timespec="seconds") + "Z",  # use UTC time
    }

    return prov_dict


def source_files_vars(source_paths: Union[str, List[Any]]) -> Dict[str, Tuple]:
    """
    Create source_filenames provenance variable dict to be used for creating
    xarray dataarray.

    Parameters
    ----------
    source_paths: Union[str, List[Any]]
        Source file paths as either a single path string or a list of Path-type paths

    Returns
    -------
    source_files_var: Dict[str, Tuple]
        Single-element dict containing a tuple for creating the
        source_filenames xarray dataarray with filenames dimension
    """

    # Handle both a list of pathlib paths and a plain string containing a single path
    if type(source_paths) is str:
        source_files = [source_paths]
    else:
        source_files = [str(p) for p in source_paths]

    source_files_var = {
        "source_filenames": (
            "filenames",
            source_files,
            {"long_name": "Source filenames"},
        ),
    }

    return source_files_var
