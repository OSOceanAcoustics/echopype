"""
"echo metrics" functions
Reference:
Urmy et al. 2012. Measuring the vertical distributional
variability of pelagic fauna in Monterey Bay.
ICES Journal of Marine Science 69 (2): 184-196.
https://doi.org/10.1093/icesjms/fsr205
Original implementation:
https://github.com/ElOceanografo/EchoMetrics/blob/master/echometrics/echometrics.py
"""

import numpy as np
import xarray as xr


def delta_z(ds: xr.Dataset, range_label="range") -> xr.DataArray:
    """Helper function to calculate widths between range bins (dz) for discretized integral.

    Parameters
    ----------
    ds : xr.Dataset
    range_label : str
        Name of an xarray DataArray in `ds` containing range information.

    Returns
    -------
    xr.DataArray
    """
    if range_label not in ds:
        raise ValueError(f"{range_label} not in the input Dataset!")
    dz = ds[range_label].diff(dim="range_bin")
    return dz.where(dz != 0, other=np.nan)


def convert_to_linear(ds: xr.Dataset, Sv_label="Sv") -> xr.DataArray:
    """Helper function to convert volume backscattering strength (Sv) values to linear domain.

    Parameters
    ----------
    ds : xr.Dataset
    Sv_label : str
        Name of an xarray DataArray in `ds` containing volume backscattering strength (Sv).

    Returns
    -------
    xr.DataArray
    """
    return 10 ** (ds[Sv_label] / 10)


def abundance(ds: xr.Dataset, range_label="range") -> xr.DataArray:
    """Calculates the area-backscattering strength (Sa) [unit: dB re 1 m^2 m^-2].

    This quantity is the integral of volumetric backscatter over range.

    Parameters
    ----------
    ds : xr.Dataset
    range_label : str
        Name of an xarray DataArray in `ds` containing range information.

    Returns
    -------
    xr.DataArray
    """
    dz = delta_z(ds, range_label=range_label)
    sv = convert_to_linear(ds, "Sv")
    return 10 * np.log10((sv * dz).sum(dim="range_bin"))


def center_of_mass(ds: xr.Dataset, range_label="range") -> xr.DataArray:
    """Calculates the mean backscatter location [unit: m].

    This quantity is the weighted average of backscatter along range.

    Parameters
    ----------
    ds : xr.Dataset
    range_label : str
        Name of an xarray DataArray in `ds` containing range information.

    Returns
    -------
    xr.DataArray
    """
    dz = delta_z(ds, range_label=range_label)
    sv = convert_to_linear(ds, "Sv")
    return (ds[range_label] * sv * dz).sum(dim="range_bin") / (sv * dz).sum(
        dim="range_bin"
    )


def dispersion(ds: xr.Dataset, range_label="range") -> xr.DataArray:
    """Calculates the inertia (I) [unit: m^-2].

    This quantity measures dispersion or spread of backscatter from the center of mass.

    Parameters
    ----------
    ds : xr.Dataset
    range_label : str
        Name of an xarray DataArray in `ds` containing range information.

    Returns
    -------
    xr.DataArray
    """
    dz = delta_z(ds, range_label=range_label)
    sv = convert_to_linear(ds, "Sv")
    cm = center_of_mass(ds)
    return ((ds[range_label] - cm) ** 2 * sv * dz).sum(dim="range_bin") / (sv * dz).sum(
        dim="range_bin"
    )


def evenness(ds: xr.Dataset, range_label="range") -> xr.DataArray:
    """Calculates the equivalent area (EA) [unit: m].

    This quantity represents the area that would be occupied if all datacells
    contained the mean density.

    Parameters
    ----------
    ds : xr.Dataset
    range_label : str
        Name of an xarray DataArray in `ds` containing range information.

    Returns
    -------
    xr.DataArray
    """
    dz = delta_z(ds, range_label=range_label)
    sv = convert_to_linear(ds, "Sv")
    return ((sv * dz).sum(dim="range_bin")) ** 2 / (sv ** 2 * dz).sum(dim="range_bin")


def aggregation(ds: xr.Dataset, range_label="range") -> xr.DataArray:
    """Calculated the index of aggregation (IA) [unit: m^-1].

    This quantity is reciprocal of the equivalent area.
    IA is high when small areas are much denser than the rest of the distribution.

    Parameters
    ----------
    ds : xr.Dataset
    range_label : str
        Name of an xarray DataArray in `ds` containing range information.

    Returns
    -------
    xr.DataArray
    """
    return 1 / evenness(ds, range_label=range_label)
