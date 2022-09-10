from datetime import datetime as dt
from pathlib import PosixPath
from typing import Any, Dict, List, Tuple, Union

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


def source_files_vars(source_paths: Union[str, List[Any]]) -> Tuple[Dict[str, Tuple], Dict[str, Tuple]]:
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
    source_files_coord: Dict[str, Tuple]
        Single-element dict containing a tuple for creating the
        filenames coordinate variable dataarray
    """

    # Handle a plain string containing a single path,
    # a single pathlib Path, or a list of strings or pathlib paths
    if type(source_paths) in (str, PosixPath):
        source_files = [str(source_paths)]
    else:
        source_files = [str(p) for p in source_paths]

    source_files_var = {
        "source_filenames": (
            "filenames",
            source_files,
            {"long_name": "Source filenames"},
        ),
    }

    source_files_coord = {
        "filenames": (
            "filenames",
            list(range(len(source_files))),
            {"long_name": "Source filenames index"},
        ),
    }
    return source_files_var, source_files_coord
