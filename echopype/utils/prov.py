import datetime
import functools
import inspect
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import xarray as xr
from _echopype_version import version as ECHOPYPE_VERSION
from numpy.typing import NDArray
from typing_extensions import Literal

ProcessType = Literal["conversion", "combination", "processing", "mask"]
# Note that this PathHint is defined differently from the one in ..core
PathHint = Union[str, Path]
PathSequenceHint = Union[List[PathHint], Tuple[PathHint], NDArray[PathHint]]


def echopype_prov_attrs(process_type: ProcessType) -> Dict[str, str]:
    """
    Standard echopype software attributes for provenance

    Parameters
    ----------
    process_type : ProcessType
        Echopype process function type
    """
    if sys.version_info < (3, 11, 0):
        utc_now = datetime.datetime.utcnow().isoformat(timespec="seconds") + "+00:00"
    else:
        utc_now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    prov_dict = {
        f"{process_type}_software_name": "echopype",
        f"{process_type}_software_version": ECHOPYPE_VERSION,
        f"{process_type}_time": utc_now,
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
                warnings.warn(
                    "Unrecognized file path element type, path element will not be"
                    f" written to (meta)source_file provenance attribute. {p}",
                    category=UserWarning,
                )
        return paths_list
    else:
        warnings.warn(
            "Unrecognized file path element type, path element will not be"
            f" written to (meta)source_file provenance attribute. {paths}",
            category=UserWarning,
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


def _check_valid_latlon(ds):
    """Verify that the dataset contains valid latitude and longitude variables"""
    if (
        "longitude" in ds
        and not ds["longitude"].isnull().all()
        and "latitude" in ds
        and not ds["latitude"].isnull().all()
    ):
        return True
    else:
        return False


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
                "processing_level_url": "https://echopype.readthedocs.io/en/stable/processing-levels.html",  # noqa
            }

        if not (
            processing_level_code in PROCESSING_LEVELS
            or re.fullmatch(r"L\*[A|B]|L[1-4]\*", processing_level_code)
        ):
            raise ValueError(
                f"Decorator processing_level_code {processing_level_code} "
                f"used in {func.__qualname__} is invalid."
            )

        # Found the class method vs module function solution in
        # https://stackoverflow.com/a/49100736
        if len(func.__qualname__.split(".")) > 1:
            # Handle class methods
            @functools.wraps(func)
            def inner(self, *args, **kwargs):
                func(self, *args, **kwargs)
                if _check_valid_latlon(self["Platform"]):
                    processing_level = PROCESSING_LEVELS[processing_level_code]
                    self["Top-level"] = self["Top-level"].assign_attrs(
                        _attrs_dict(processing_level)
                    )
                else:
                    print(
                        "EchoData object (converted raw file) does not contain "
                        "valid Platform location data. Processing level attributes "
                        "will not be added."
                    )

            return inner
        else:
            # Handle stand-alone module functions
            @functools.wraps(func)
            def inner(*args, **kwargs):
                dataobj = func(*args, **kwargs)
                if is_echodata:
                    # Most functions return a single EchoData object.
                    # BI500 open_raw returns (echodata, ds_cal).
                    ed = dataobj[0] if isinstance(dataobj, tuple) else dataobj

                    if _check_valid_latlon(ed["Platform"]):
                        # The decorator is passed the exact, final level code, with sublevel
                        processing_level = PROCESSING_LEVELS[processing_level_code]
                        ed["Top-level"] = ed["Top-level"].assign_attrs(
                            _attrs_dict(processing_level)
                        )
                    else:
                        print(
                            "EchoData object (converted raw file) does not contain "
                            "valid Platform location data. Processing level attributes "
                            "will not be added."
                        )

                    # Return the original output so BI500 keeps both echodata and ds_cal.
                    return dataobj
                elif isinstance(dataobj, xr.Dataset):
                    ds = dataobj
                    if _check_valid_latlon(ds):
                        if processing_level_code in PROCESSING_LEVELS:
                            # The decorator is passed the exact, final level code, with sublevel
                            processing_level = PROCESSING_LEVELS[processing_level_code]
                        elif (
                            "*" in processing_level_code
                            and "input_processing_level" in ds.attrs.keys()
                        ):
                            if processing_level_code[-1] == "*":
                                # The decorator is passed a level code without sublevel (eg, L3*).
                                # The decorated function's "input" dataset's sublevel (A or B) will
                                # be propagated to the function's output dataset. For L2 and L3
                                sublevel = ds.attrs["input_processing_level"][-1]
                                level = processing_level_code[1]
                            elif processing_level_code[1] == "*":
                                # The decorator is passed a sublevel code without level (eg, L*A).
                                # The decorated function's "input" dataset's level (2 or 3) will
                                # be propagated to the function's output dataset. For L2 and L3
                                sublevel = processing_level_code[-1]
                                level = ds.attrs["input_processing_level"][-2]

                            processing_level = PROCESSING_LEVELS[f"L{level}{sublevel}"]
                            del ds.attrs["input_processing_level"]
                        else:
                            raise RuntimeError(
                                "Processing level attributes (processing_level_code {processing_level_code}) "  # noqa
                                f"cannot be added. Please ensure that {func.__qualname__} "
                                "uses the function insert_input_processing_level."
                            )

                        ds = ds.assign_attrs(_attrs_dict(processing_level))
                    else:
                        print(
                            "xarray Dataset does not contain valid location data. "
                            "Processing level attributes will not be added."
                        )
                        if "input_processing_level" in ds.attrs:
                            del ds.attrs["input_processing_level"]

                    return ds
                else:
                    raise RuntimeError(
                        f"{func.__qualname__}: Processing level decorator function cannot be used "
                        "with a function that does not return an xarray Dataset or EchoData object"
                    )

            return inner

    return wrapper


def _history_arg_repr(value: Any) -> str:
    """Compact string representation of an argument value for the history attribute."""
    from ..echodata import EchoData

    if isinstance(value, EchoData):
        source = value.converted_raw_path or value.source_file
        return f"EchoData({str(source)!r})" if source is not None else "EchoData()"
    if isinstance(value, Path):
        return repr(str(value))
    if isinstance(value, (xr.Dataset, xr.DataArray, np.ndarray)):
        return f"<{type(value).__name__}>"
    return repr(value)


def _is_method(func) -> bool:
    return next(iter(inspect.signature(func).parameters), None) == "self"


def _history_call_str(func, args, kwargs) -> str:
    """Render ``func(arg1=..., arg2=...)`` from the arguments explicitly passed by the caller."""
    bound = inspect.signature(func).bind(*args, **kwargs)
    arg_strs = []
    for name, value in bound.arguments.items():
        kind = bound.signature.parameters[name].kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            arg_strs += [f"{k}={_history_arg_repr(v)}" for k, v in value.items()]
        elif kind is inspect.Parameter.VAR_POSITIONAL:
            arg_strs += [_history_arg_repr(v) for v in value]
        elif name == "self" and _is_method(func):
            continue
        else:
            arg_strs.append(f"{name}={_history_arg_repr(value)}")
    func_name = func.__qualname__ if _is_method(func) else func.__name__
    return f"{func_name}({', '.join(arg_strs)})"


def _append_history(attrs: Dict[str, Any], entry: str) -> Dict[str, Any]:
    if attrs.get("history"):
        return {"history": f"{attrs['history']}\n{entry}"}
    return {"history": entry}


def add_history(func):
    """
    Wraps functions that return an xr.Dataset, xr.DataArray, or EchoData object and
    appends an entry to the ``history`` attribute of the output, in the form
    ``YYYY-MM-DDTHH:MM:SS+offset: func(arg1=..., arg2=...);``.

    For EchoData objects the attribute is stored in the Top-level group.
    EchoData methods that modify the object in place and return None are also supported.
    """

    @functools.wraps(func)
    def inner(*args, **kwargs):
        from ..echodata import EchoData

        dataobj = func(*args, **kwargs)

        timestamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        entry = f"{timestamp}: {_history_call_str(func, args, kwargs)};"

        modified_in_place = dataobj is None and _is_method(func) and isinstance(args[0], EchoData)
        if modified_in_place or isinstance(dataobj, EchoData):
            ed = args[0] if modified_in_place else dataobj
            ed["Top-level"] = ed["Top-level"].assign_attrs(
                _append_history(ed["Top-level"].attrs, entry)
            )
        elif isinstance(dataobj, (xr.Dataset, xr.DataArray)):
            dataobj = dataobj.assign_attrs(_append_history(dataobj.attrs, entry))
        else:
            raise RuntimeError(
                f"{func.__qualname__}: History decorator function cannot be used with a "
                "function that does not return an xarray Dataset, DataArray, or EchoData object"
            )
        return dataobj

    return inner


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
