import operator as op
import pathlib
from typing import List, Optional, Union

import numpy as np
import xarray as xr


def apply_mask(source_ds: Union[xr.Dataset, str, pathlib.Path],
               var_name: str,
               mask: Union[Union[xr.DataArray, str, pathlib.Path],
                           List[Union[xr.DataArray, str, pathlib.Path]]],
               fill_value: Union[int, float, np.ndarray, xr.DataArray] = np.nan) -> xr.Dataset:
    """
    Applies the provided mask(s) to the variable ``var_name``
    in the provided Dataset ``source_ds``.

    Parameters
    ----------
    source_ds: xr.Dataset or str or pathlib.Path
        Points to a Dataset that contains the variable the mask should be applied to
    var_name: str
        The variable name in ``source_ds`` that the mask should be applied to
    mask: xr.DataArray or str or pathlib.Path or list of xr.DataArray or str or pathlib.Path
        The mask(s) to apply to the variable specified by ``var_name``. This input can be a
        single input or list that corresponds to a DataArray or a path to a DataArray.
    fill_value: int or float or np.ndarray or xr.DataArray, default=np.nan
        Specifies the value(s) at false indices

    Returns
    -------
    xr.Dataset
        A Dataset with the same format of ``source_ds`` with the mask(s) applied to ``var_name``

    Notes
    -----
    If the input ``mask`` is a list, then a logical AND will be used to produce the final
    mask that will be applied to ``var_name``.
    """

    # TODO: use validate_source_Sv on source_ds

    # TODO: if fill_value is an array, then make sure its dimensions match var_name variable

    print("hello")