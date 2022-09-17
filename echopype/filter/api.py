from typing import List, Union

import numpy as np
import xarray as xr


def median(ds: xr.Dataset, winsize: int) -> xr.Dataset:
    """
    Apply median filter to the to the Sv data.

    This function adds the alongship/athwartship angle data stored in the Sonar/Beam_group1 group
    of the original data file. In cases when the angles do not exist, a warning is issued and
    no angle variables are added to the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        An Sv or MVBS dataset for which the geographical locations will be added to
    winsize : int
        Window size to apply the median filter.

    Returns
    -------
    The input dataset with the Sv data variable smoothed with a median filter.
    """
    pass


def conv(ds: xr.Dataset, kernel: Union[int, List, np.ndarray]) -> xr.Dataset:
    """
    Apply convolution filter to the Sv data.

    Parameters
    ----------
    ds : xr.Dataset
        An Sv or MVBS dataset for which the geographical locations will be added to
    kernel
        The convolutional kernel.

        - If is an integer, a square matrix of dimension `kernel` by `kernel` filled
          with 1s will be used.
        - If it is a list or an array, it must be able to be reshaped into a square matrix.
          The reshaping will follow C-like index order as default in numpy.

    Returns
    -------
    The input dataset with the Sv data filtered with the convolutional kernel.
    """
    pass
