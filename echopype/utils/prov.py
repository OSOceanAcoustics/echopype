import functools
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Union

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


# TODO:
#   - New decorator function that inserts processing-level global attributes, if appropriate.
#     Product level attributes will be added only if lat-lon coordinates (and depth for >= L2?) are present.
#     Attributes: processing_level, processing_level_url
#   - DONE. Dict mapping a processing level code (eg, "L1A") to the longer text that will be inserted in the attr
#   - DONE. Function that each function will use to read its "input" xr.dataset processing_level (if it exists)
#     and pass it on to the output xr.dataset as input_processing_level for use by the decorator
#   - DONE. For L2 & L3, the "input" dataset will typically already have a processing_level attribute;
#     read and propagate the sub-level code (A & B).
#   - (EXTERNAL) Associate (ie, document) processing level code with each relevant function listed in
#     https://github.com/OSOceanAcoustics/echopype/pull/817#issuecomment-1474564449
#   - DONE. echodata.update_platform should result in the addition of a processing level attr.
#     But as a class method, doing this may be trickier. The challenge is how to pass "self" to the decorator
#     function/wrapper. Consider wrapping add_processing_level in another function, and using that
#     as the class-method decorator?

# DEVELOPMENT PLAN:
# 1. DONE. Create initial version of the dictionary
# 2. DONE. First create decorator that doesn't check for lat-lon coords. Start by applying it to
#    consolidate.add_laton? (but my version is the old one, until WJ approves; or merge my add_latlon branch?)
# 3. Add testing for valid lat-lon
# 4. DONE. Add compute_MVBS handler: If the input Sv ds has a processing_level, read it;
#    if processing_level is present, extract its sublevel (A or B) and propagate it

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


def add_processing_level(processing_level_code, is_echodata=False, inherit_sublevel=False):
    """
    Wraps functions or methods that return either an xr.Dataset or an echodata object

    Parameters
    ----------
    processing_level_code : str
        Data processing level code, either with or without sublevel code.
        eg, L1A, L2B, L3
    is_echodata : bool
        Flag specifying if the decorated function returns an EchoData object (optional)
    inherit_sublevel : bool
        Flag specifying if the processing sublevel will be inherited from
        the "input" dataset (optional)

    Returns
    -------
    An xr.Dataset or EchoData object with processing level attributes
    inserted if appropriate, or unchanged otherwise.
    """

    def wrapper(func):
        # TODO: ADD conventions attr, with "ACDD-1.3" entry?? Or maybe skip for now?
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

                # This hard-wiring may not be necessary, ultimately.
                # But right I'm ensuring that it's only applicable to echodata.update_platform
                if func.__qualname__.split(".")[-1] == "update_platform":
                    processing_level = PROCESSING_LEVELS[processing_level_code]
                    self["Top-level"] = self["Top-level"].assign_attrs(
                        _attrs_dict(processing_level)
                    )

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
                    ):  # noqa
                        # The decorator is passed the exact, final level code, with sublevel
                        processing_level = PROCESSING_LEVELS[processing_level_code]
                        ed["Top-level"] = ed["Top-level"].assign_attrs(
                            _attrs_dict(processing_level)
                        )

                    return ed
                elif isinstance(dataobj, xr.Dataset):
                    ds = dataobj
                    if processing_level_code in PROCESSING_LEVELS:
                        # The decorator is passed the exact, final level code, with sublevel
                        processing_level = PROCESSING_LEVELS[processing_level_code]
                    elif inherit_sublevel and "input_processing_level" in ds.attrs.keys():
                        # The decorator is passed a level code without sublevel (eg, L3).
                        # The decorated function's "input" dataset's sublevel (A or B) will
                        # be propagated to the function's output dataset. For L2 and L3
                        sublevel = ds.attrs["input_processing_level"][-1]
                        processing_level = PROCESSING_LEVELS[processing_level_code + sublevel]
                        del ds.attrs["input_processing_level"]
                    else:
                        # Processing level attributes will not be inserted
                        # TODO: Raise a warning that processing_level_code is not in PROCESSING_LEVELS
                        #   and other criteria were not met?
                        return ds

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
