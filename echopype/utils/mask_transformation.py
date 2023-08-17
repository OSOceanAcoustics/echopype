"""
Algorithms for resampling data arrays.

Copyright (c) XXXX

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# import modules
import warnings
import numpy as np
from scipy.interpolate import interp1d
from geopy.distance import distance


def lin(variable):
    """
    Turn variable into the linear domain.

    Args:
        variable (float): array of elements to be transformed.

    Returns:
        float:array of elements transformed
    """

    lin = 10 ** (variable / 10)
    return lin


def log(variable):
    """
    Turn variable into the logarithmic domain. This function will return -999
    in the case of values less or equal to zero (undefined logarithm). -999 is
    the convention for empty water or vacant sample in fisheries acoustics.

    Args:
        variable (float): array of elements to be transformed.

    Returns:
        float: array of elements transformed
    """

    # convert variable to float array
    back2single = False
    back2list = False
    back2int = False
    if not isinstance(variable, np.ndarray):
        if isinstance(variable, list):
            variable = np.array(variable)
            back2list = True
        else:
            variable = np.array([variable])
            back2single = True
    if variable.dtype == 'int64':
        variable = variable * 1.0
        back2int = True

    # compute logarithmic value except for zeros values, which will be -999 dB
    mask = np.ma.masked_less_equal(variable, 0).mask
    variable[mask] = np.nan
    log = 10 * np.log10(variable)
    log[mask] = -999

    # convert back to original data format and return
    if back2int:
        log = np.int64(log)
    if back2list:
        log = log.tolist()
    if back2single:
        log = log[0]
    return log
