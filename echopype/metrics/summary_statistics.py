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


def delta_z(ds, zname="range"):
    if zname not in ds:
        raise ValueError(f"{zname} not in the input Dataset!")
    dz = ds[zname].diff(
    	dim="range_bin"
    )  
    return dz.where(dz != 0, other = np.nan)


def convert_to_linear(ds, name="Sv"):
    return 10 ** (ds[name] / 10)  # convert Sv to linear domain


def abundance(ds, zname="range"):
    dz = delta_z(ds, zname=zname)
    sv = convert_to_linear(ds, "Sv")
    return (10 * np.log10(sv * dz)).sum(dim="range_bin")  # integrate over depth


def center_of_mass(ds, zname="range"):
    dz = delta_z(ds, zname=zname)
    sv = convert_to_linear(ds, "Sv")
    return ((ds[zname] * sv * dz).sum(dim="range_bin") / (sv * dz).sum(dim="range_bin"))


def inertia(ds, zname="range"):
    dz = delta_z(ds, zname=zname)
    sv = convert_to_linear(ds, "Sv")
    cm = center_of_mass(ds)
    return ((ds[zname] - cm) ** 2 * sv * dz).sum(dim="range_bin") / (sv * dz).sum(
    	dim="range_bin"
    )


def evenness(ds, zname="range"):
    dz = delta_z(ds, zname=zname)
    sv = convert_to_linear(ds, "Sv")
    return ((sv * dz).sum(dim="range_bin")) ** 2 / (sv ** 2 * dz).sum(dim="range_bin")


def aggregation(ds, zname="range"):
    return 1/evenness(ds, zname=zname)


