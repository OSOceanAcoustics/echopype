from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from _echopype_version import version as ECHOPYPE_VERSION
from typing_extensions import Literal

from .log import _init_logger

# TODO: It'd be cleaner to use PathHint, but it leads to a circular import error
# from ..core import PathHint

ProcessType = Literal["conversion", "processing"]

logger = _init_logger(__name__)


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


def _source_files(paths: Union[str, Path, List[Any], Tuple[Any], np.ndarray]):
    """
    Create sanitized list of string paths from heterogeneous path inputs.

    Parameters
    ----------
    paths : Union[str, Path, List[str, Path], Tuple[str, Path], np.ndarray]
        File paths as either a single path string or pathlib Path,
        a sequence (tuple, list or np.ndarray) of strings or pathlib Paths,
        or a mixed sequence that may contain another sequence as an element.

    Returns
    -------
    paths_list : List[str]
        List of file paths. Empty list if no source path element was parsed successfully.
    """
    sequence_types = (list, tuple, np.ndarray)
    if isinstance(paths, (str, Path)):
        return [str(paths)]
    elif isinstance(paths, sequence_types):
        paths_list = []
        for p in paths:
            if isinstance(p, (str, Path)):
                paths_list.append(str(p))
            elif isinstance(p, sequence_types):
                paths_list += [str(pp) for pp in p if isinstance(pp, (str, Path))]
            else:
                logger.warning(
                    f"Unrecognized file path element type, path element will not be written to (meta)source_file provenance attribute. {p}"  # noqa
                )
        return paths_list
    else:
        logger.warning(
            f"Unrecognized file path element type, path element will not be written to (meta)source_file provenance attribute. {paths}"  # noqa
        )
        return []


def source_files_vars(
    source_paths: Union[str, Path, List[Any], Tuple[Any], np.ndarray],
    meta_source_paths: Union[None, str, Path, List[Any], Tuple[Any], np.ndarray] = None,
) -> Dict[str, Dict[str, Tuple]]:
    """
    Create source_filenames and meta_source_filenames provenance
    variables dicts to be used for creating xarray DataArray.

    Parameters
    ----------
    source_paths : Union[str, Path, List[str, Path], Tuple[str, Path], np.ndarray]
        Source file paths as either a single path string or pathlib Path,
        a sequence (tuple, list or np.ndarray) of strings or pathlib Paths,
        or a mixed sequence that may contain another sequence as an element.
    meta_source_paths : Union[None, str, Path, List[str, Path], Tuple[str, Path], np.ndarray]
        Source file paths for metadata files (often as XML files), as either a
        single path string or pathlib Path, a sequence (tuple, list or np.ndarray)
        of strings or pathlib Paths, or a mixed sequence that may contain another
        sequence as an element.

    Returns
    -------
    files_vars : Dict[str, Dict[str, Tuple]]
        Contains 3 items:
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

    source_files = _source_files(source_paths)
    files_vars = dict()

    files_vars["source_files_var"] = {
        "source_filenames": (
            "filenames",
            source_files,
            {"long_name": "Source filenames"},
        ),
    }

    if meta_source_paths is not None:
        meta_source_files = _source_files(meta_source_paths)
        files_vars["meta_source_files_var"] = {
            "meta_source_filenames": (
                "filenames",
                meta_source_files,
                {"long_name": "Metadata source filenames"},
            ),
        }
    else:
        files_vars["meta_source_files_var"] = None

    files_vars["source_files_coord"] = {
        "filenames": (
            "filenames",
            list(range(len(source_files))),
            {"long_name": "Index for data and metadata source filenames"},
        ),
    }

    return files_vars
