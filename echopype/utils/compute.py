"""compute.py

Module containing various helper functions
for performing computations within echopype.
"""

from typing import Union

import dask.array
import numpy as np


def _log2lin(data: Union[dask.array.Array, np.ndarray]) -> Union[dask.array.Array, np.ndarray]:
    """Perform log to linear transform on data

    Parameters
    ----------
    data : dask.array.Array or np.ndarray
         The data to be transformed

    Returns
    -------
    dask.array.Array or np.ndarray
        The transformed data
    """
    return 10 ** (data / 10)


def _lin2log(data: Union[dask.array.Array, np.ndarray]) -> Union[dask.array.Array, np.ndarray]:
    """Perform linear to log transform on data

    Parameters
    ----------
    data : dask.array.Array or np.ndarray
         The data to be transformed

    Returns
    -------
    dask.array.Array or np.ndarray
        The transformed data
    """
    return 10 * np.log10(data)
