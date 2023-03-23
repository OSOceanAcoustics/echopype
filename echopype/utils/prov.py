import functools
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import xarray as xr
from _echopype_version import version as ECHOPYPE_VERSION
from numpy.typing import NDArray
from typing_extensions import Literal

from .log import _init_logger

ProcessType = Literal["conversion", "processing", "mask"]
# Note that this PathHint is defined differently from the one in ..core
PathHint = Union[str, Path]
PathSequenceHint = Union[List[PathHint], Tuple[PathHint], NDArray[PathHint]]

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


def _sanitize_source_files(paths: Union[PathHint, PathSequenceHint]):
    """
    Create sanitized list of string paths from heterogeneous path inputs.

    Parameters
    ----------
    paths : Union[PathHint, PathSequenceHint]
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
                    "Unrecognized file path element type, path element will not be"
                    f" written to (meta)source_file provenance attribute. {p}"
                )
        return paths_list
    else:
        logger.warning(
            "Unrecognized file path element type, path element will not be"
            f" written to (meta)source_file provenance attribute. {paths}"
        )
        return []


def source_files_vars(
    source_paths: Union[PathHint, PathSequenceHint],
    meta_source_paths: Union[PathHint, PathSequenceHint] = None,
) -> Dict[str, Dict[str, Tuple]]:
    """
    Create source_filenames and meta_source_filenames provenance
    variables dicts to be used for creating xarray DataArray.

    Parameters
    ----------
    source_paths : Union[PathHint, PathSequenceHint]
        Source file paths as either a single path string or pathlib Path,
        a sequence (tuple, list or np.ndarray) of strings or pathlib Paths,
        or a mixed sequence that may contain another sequence as an element.
    meta_source_paths : Union[PathHint, PathSequenceHint]
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
            filenames coordinate variable xarray DataArray
    """

    source_files = _sanitize_source_files(source_paths)
    files_vars = dict()

    files_vars["source_files_var"] = {
        "source_filenames": (
            "filenames",
            source_files,
            {"long_name": "Source filenames"},
        ),
    }

    if meta_source_paths is None or meta_source_paths == "":
        files_vars["meta_source_files_var"] = None
    else:
        meta_source_files = _sanitize_source_files(meta_source_paths)
        files_vars["meta_source_files_var"] = {
            "meta_source_filenames": (
                "filenames",
                meta_source_files,
                {"long_name": "Metadata source filenames"},
            ),
        }

    files_vars["source_files_coord"] = {
        "filenames": (
            "filenames",
            list(range(len(source_files))),
            {"long_name": "Index for data and metadata source filenames"},
        ),
    }

    return files_vars


# L0 is not actually used by echopype but is included for completeness
PROCESSING_LEVELS = dict(
    L0="Level 0",
    L1A="Level 1A",
    L1B="Level 1B",
    L2A="Level 2A",
    L2B="Level 2B",
    L3A="Level 3A",
    L3B="Level 3B",
    L4="Level 4",
)


def add_processing_level(processing_level_code: str, is_echodata: bool = False) -> Any:
    """
    Wraps functions or methods that return either an xr.Dataset or an echodata object

    Parameters
    ----------
    processing_level_code : str
        Data processing level code. Can be either the exact code (eg, L1A, L2B, L4)
        or using * as a wildcard for either level or sublevel (eg, L*A, L2*) where
        the wildcard value of the input is propagated to the output.
    is_echodata : bool
        Flag specifying if the decorated function returns an EchoData object (optional)

    Returns
    -------
    An xr.Dataset or EchoData object with processing level attributes
    inserted if appropriate, or unchanged otherwise.
    """

    def wrapper(func):
        # TODO: Add conventions attr, with "ACDD-1.3" entry, if not already present?
        def _attrs_dict(processing_level):
            return {
                "processing_level": processing_level,
                "processing_level_url": "https://echopype.readthedocs.io/processing_levels",
            }

        # Found the class method vs module function solution in
        # https://stackoverflow.com/a/49100736
        if len(func.__qualname__.split(".")) > 1:
            # Handle class methods
            @functools.wraps(func)
            def inner(self, *args, **kwargs):
                func(self, *args, **kwargs)
                processing_level = PROCESSING_LEVELS[processing_level_code]
                self["Top-level"] = self["Top-level"].assign_attrs(_attrs_dict(processing_level))
                return self

            return inner
        else:
            # Handle stand-alone module functions
            @functools.wraps(func)
            def inner(*args, **kwargs):
                dataobj = func(*args, **kwargs)
                if is_echodata:
                    ed = dataobj
                    if (
                        "longitude" in ed["Platform"]
                        and not ed["Platform"]["longitude"].isnull().all()
                    ):
                        # The decorator is passed the exact, final level code, with sublevel
                        processing_level = PROCESSING_LEVELS[processing_level_code]
                        ed["Top-level"] = ed["Top-level"].assign_attrs(
                            _attrs_dict(processing_level)
                        )

                    return ed
                elif isinstance(dataobj, xr.Dataset):
                    ds = dataobj
                    insert_attrs = False
                    if processing_level_code in PROCESSING_LEVELS:
                        # The decorator is passed the exact, final level code, with sublevel
                        processing_level = PROCESSING_LEVELS[processing_level_code]
                        insert_attrs = True
                    elif "*" in processing_level_code and len(processing_level_code) in (2, 3):
                        if "input_processing_level" in ds.attrs.keys():
                            if processing_level_code[-1] == "*":
                                # The decorator is passed a level code without sublevel (eg, L3*).
                                # The decorated function's "input" dataset's sublevel (A or B) will
                                # be propagated to the function's output dataset. For L2 and L3
                                sublevel = ds.attrs["input_processing_level"][-1]
                                level = processing_level_code[1]
                                processing_level = PROCESSING_LEVELS[f"L{level}{sublevel}"]
                                del ds.attrs["input_processing_level"]
                                insert_attrs = True
                            elif processing_level_code[1] == "*":
                                # The decorator is passed a sublevel code without level (eg, L*A).
                                # The decorated function's "input" dataset's level (2 or 3) will
                                # be propagated to the function's output dataset. For L2 and L3
                                sublevel = processing_level_code[-1]
                                level = ds.attrs["input_processing_level"][-2]
                                processing_level = PROCESSING_LEVELS[f"L{level}{sublevel}"]
                                del ds.attrs["input_processing_level"]
                                insert_attrs = True

                    if insert_attrs:
                        ds = ds.assign_attrs(_attrs_dict(processing_level))

                    return ds
                else:
                    return dataobj

            return inner

    return wrapper


def insert_input_processing_level(ds, input_ds):
    """
    Copy processing_level attribute from input xr.Dataset, if it exists,
    and write it out as input_processing_level

    Parameters
    ----------
    ds : xr.Dataset
        The xr.Dataset returned by the decorated function
    input_ds : xr.Dataset
        The xr.Dataset that is the "input" to the decorated function

    Returns
    -------
    ds with input_processing_level attribute inserted if appropriate,
    a renamed copy of the processing_level attribute from input_ds if present.
    """
    if "processing_level" in input_ds.attrs.keys():
        return ds.assign_attrs({"input_processing_level": input_ds.attrs["processing_level"]})
    else:
        return ds
