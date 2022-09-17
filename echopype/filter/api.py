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
        An Sv dataset the filter will be applied to.
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
        An Sv dataset the filter will be applied to.
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


def remove_noise(ds: xr.Dataset, method: str, save_add_var: bool = False, **kwargs) -> xr.Dataset:
    """
    Remove noise from the Sv data.

    Parameters
    ----------
    ds : xr.Dataset
        dataset containing ``Sv`` and ``echo_range`` [m]
    method : str
        method of noise removal

        - "mean_bkg": remove noise by using estimates of background noise
            from mean calibrated power of a collection of pings.
            Proposed by De Robertis & Higginbottom 2007, see ``noise.mean_bkg`` for detail.
        - "spike": remove spikes commonly resulted from interference from other instruments.
            See ``noise.spike`` for detail.

    # TODO
    What is the best way to list parameters required for each method in the docstring?

    Returns
    -------
    The input dataset with the corrected Sv and parameters used in the noise removal process.
    If ``save_add_var`` is ``True``, variables such as the original Sv data and
    the estimated background noise are saved in the output dataset.
    """
    pass
