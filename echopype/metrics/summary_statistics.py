"""
"echo metrics" functions
Reference:
Urmy et al. 2012. Measuring the vertical distributional variability of pelagic fauna in Monterey Bay.
ICES Journal of Marine Science 69 (2): 184-196.
http://icesjms.oxfordjournals.org/cgi/content/full/fsr205?ijkey=tSU0noNUWz4bj57&keytype=ref
Implementation:
https://github.com/ElOceanografo/EchoMetrics/blob/master/echometrics/echometrics.py
"""

import numpy as np


def delta_depth(ds):
    return np.abs(ds['depth'].diff(dim='depth').mean().values)


def convert_to_linear(ds):
    return 10 ** (ds['Sv'] / 10)  # convert Sv to linear domain


def abundance(ds):
    dz = delta_depth(ds)
    sv = convert_to_linear(ds)
    sv = sv.assign_coords(depth=sv.depth.values[::-1])  # reverse depth coordinate
    return 10 * np.log10((sv * dz).sum(dim='depth'))  # integrate over depth


def center_of_mass(ds):
    dz = delta_depth(ds)
    sv = convert_to_linear(ds)
    return ((sv.depth * sv * dz).sum(dim='depth') /
            (sv * dz).sum(dim='depth'))


def inertia(ds):
    dz = delta_depth(ds)
    sv = convert_to_linear(ds)
    cm = center_of_mass(ds)
    return (((sv.depth - cm) ** 2 * sv * dz).sum(dim='depth') /
            (sv * dz).sum(dim='depth'))


def evenness(ds):
    dz = delta_depth(ds)
    sv = convert_to_linear(ds)
    return (((sv * dz).sum(dim='depth')) ** 2 /
            (sv ** 2 * dz).sum(dim='depth'))


def aggregation(ds):
    return 1/evenness(ds)


