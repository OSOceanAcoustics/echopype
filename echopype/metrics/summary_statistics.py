"""
"echo metrics" functions
Reference:
Urmy et al. 2012. Measuring the vertical distributional
variability of pelagic fauna in Monterey Bay.
ICES Journal of Marine Science 69 (2): 184-196.
http://icesjms.oxfordjournals.org/cgi/content/full/fsr205?ijkey=tSU0noNUWz4bj57&keytype=ref
Implementation:
https://github.com/ElOceanografo/EchoMetrics/blob/master/echometrics/echometrics.py
"""

import numpy as np


def delta_z(ds, zname="range"):
    """Utility Function: Calculates widths between range bins (dz) for discretized integral

    Parameters
    ----------
    ds: Dataset
    zname: str
        pre-defined label for "range" DataArray

    Returns
    ----------
    DataArray
    """
    if zname not in ds:
        raise ValueError(f"{zname} not in the input Dataset!")
    dz = ds[zname].diff(dim="range_bin")
    return dz.where(dz != 0, other=np.nan)


def convert_to_linear(ds, name="Sv"):
    """Utility Function: Converts Sv DataArray to linear domain

    Parameters
    ----------
    ds: Dataset
    zname: str
        pre-defined label for "Sv" DataArray

    Returns
    ----------
    DataArray
    """
    return 10 ** (ds[name] / 10)


def abundance(ds, zname="range"):
    """Calculates area-backscattering strength
       (Integral of volumetric backscatter over entire water column)
       Unit: dB re 1 m^2 m^-2

    Parameters
    ----------
    ds: Dataset
    zname: str
        pre-defined label for "range" DataArray

    Returns
    ----------
    DataArray
    """
    dz = delta_z(ds, zname=zname)
    sv = convert_to_linear(ds, "Sv")
    return 10 * np.log10((sv * dz).sum(dim="range_bin"))


def center_of_mass(ds, zname="range"):
    """Calculates mean location
       (Average of all depths sampled weighted by sv values)
       Unit: M

    Parameters
    ----------
    ds: Dataset
    zname: DataArray
        pre-defined label for "range" DataArray

    Returns
    ----------
    DataArray
    """
    dz = delta_z(ds, zname=zname)
    sv = convert_to_linear(ds, "Sv")
    return (ds[zname] * sv * dz).sum(dim="range_bin") / (sv * dz).sum(dim="range_bin")


def inertia(ds, zname="range"):
    """Calculates dispersion
       (Sum of squared distances from the center of mass,
       weighted by sv at each distance and normalized by total sa)
       Unit: m^-2

    Parameters
    ----------
    ds: Dataset
    zname: DataArray
        pre-defined label for "range" DataArray

    Returns
    ----------
    DataArray
    """
    dz = delta_z(ds, zname=zname)
    sv = convert_to_linear(ds, "Sv")
    cm = center_of_mass(ds)
    return ((ds[zname] - cm) ** 2 * sv * dz).sum(dim="range_bin") / (sv * dz).sum(
        dim="range_bin"
    )


def evenness(ds, zname="range"):
    """Calculates equivalent area, or area that would be occupied
       if all datacells contained the mean density
       (Squared integral of sv over depth divided by depth integral of sv^2)
       Unit: m

    Parameters
    ----------
    ds: Dataset
    zname: DataArray
        pre-defined label for "range" DataArray

    Returns
    ----------
    DataArray
    """
    dz = delta_z(ds, zname=zname)
    sv = convert_to_linear(ds, "Sv")
    return ((sv * dz).sum(dim="range_bin")) ** 2 / (sv ** 2 * dz).sum(dim="range_bin")


def aggregation(ds, zname="range"):
    """Calculated index of aggregation, reciprocal of evenness
       Unit: m^-1

    Parameters
    ----------
    ds: Dataset
    zname: DataArray
        pre-defined label for "range" DataArray

    Returns
    ----------
    DataArray
    """
    return 1 / evenness(ds, zname=zname)
