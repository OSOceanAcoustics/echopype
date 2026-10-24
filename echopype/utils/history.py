import datetime
import functools
import inspect
from pathlib import Path
from typing import Any, Dict

import numpy as np
import xarray as xr


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

        # Handle methods which return more than one value.
        # BI500 `open_raw` returns (echodata, ds_cal)
        object = dataobj[0] if isinstance(dataobj, tuple) else dataobj

        timestamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        entry = f"{timestamp}: {_history_call_str(func, args, kwargs)};"

        modified_in_place = object is None and _is_method(func) and isinstance(args[0], EchoData)
        if modified_in_place or isinstance(object, EchoData):
            ed = args[0] if modified_in_place else object
            ed["Top-level"] = ed["Top-level"].assign_attrs(
                _append_history(ed["Top-level"].attrs, entry)
            )
        elif isinstance(object, (xr.Dataset, xr.DataArray)):
            object = object.assign_attrs(_append_history(object.attrs, entry))
        else:
            raise RuntimeError(
                f"{func.__qualname__}: History decorator function cannot be used with a "
                "function that does not return an xarray Dataset, DataArray, or EchoData object"
            )
        return (object, *dataobj[1:]) if isinstance(dataobj, tuple) else object

    return inner
