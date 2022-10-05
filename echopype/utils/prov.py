from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from _echopype_version import version as ECHOPYPE_VERSION
from typing_extensions import Literal

# TODO: It'd be cleaner to use PathHint, but it leads to a circular import error
# from ..core import PathHint

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


def source_files_vars(
    source_paths: Union[str, List[Any]], meta_source_paths: Union[str, List[Any]] = None
) -> Tuple[Dict[str, Tuple], Dict[str, Tuple], Dict[str, Tuple]]:
    """
    Create source_filenames and meta_source_filenames provenance
    variables dicts to be used for creating xarray DataArray.

    Parameters
    ----------
    source_paths : Union[str, List[Any]]
        Source file paths as either a single path string or a list of Path-type paths
    meta_source_paths : Union[str, List[Any]]
        Source file paths for metadata files (often as XML files),
        as either a single path string or a list of Path-type paths

    Returns
    -------
    source_files_var : Dict[str, Tuple]
        Single-element dict containing a tuple for creating the
        source_filenames xarray DataArray with filenames dimension
    meta_source_files_var : Dict[str, Tuple]
        Single-element dict containing a tuple for creating the
        meta_source_filenames xarray DataArray with filenames dimension
    source_files_coord : Dict[str, Tuple]
        Single-element dict containing a tuple for creating the
        filenames coordinate variable DataArray
    """

    def _source_files(paths):
        """Handle a plain string containing a single path,
        a single pathlib Path, or a list of strings or pathlib paths
        """
        if isinstance(paths, (str, Path)):
            return [str(paths)]
        else:
            return [str(p) for p in paths if isinstance(p, (str, Path))]

    source_files = _source_files(source_paths)
    source_files_var = {
        "source_filenames": (
            "filenames",
            source_files,
            {"long_name": "Source filenames"},
        ),
    }

    if meta_source_paths is not None:
        meta_source_files = _source_files(meta_source_paths)
        meta_source_files_var = {
            "meta_source_filenames": (
                "filenames",
                meta_source_files,
                {"long_name": "Metadata source filenames"},
            ),
        }
    else:
        meta_source_files_var = None

    source_files_coord = {
        "filenames": (
            "filenames",
            list(range(len(source_files))),
            {"long_name": "Index for data and metadata source filenames"},
        ),
    }

    return source_files_var, meta_source_files_var, source_files_coord
