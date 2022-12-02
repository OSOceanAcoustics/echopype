"""
Contains functions necessary to compute the split-beam (alongship/athwartship)
angles and add them to a Dataset.
"""
from typing import Tuple

import numpy as np
import xarray as xr


def _get_splitbeam_angle_power_CW(ds_beam: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Obtains the split-beam angle data from power encoded data with CW waveform.

    Parameters
    ----------
    ds_beam: xr.Dataset
        An ``EchoData`` beam group containing angle information needed for
        split-beam angle calculation

    Returns
    -------
    theta_fc: xr.Dataset
        The calculated split-beam alongship angle
    phi_fc: xr.Dataset
        The calculated split-beam athwartship angle

    Raises
    ------
    NotImplementedError
        If all ``beam_type`` values are not equal to 1

    Notes
    -----
    Can be used on both EK60 and EK80 data

    Computation done for ``beam_type=1``:
    ``physical_angle = ((raw_angle * 180 / 128) / sensitivity) - offset``
    """

    # raw_angle scaling constant
    conversion_const = 180 / 128

    # TODO: is this function useful or just annoying? Should we put it outside this function?
    def compute_split_beam_beamtype1(angle_type: str) -> xr.Dataset:
        """Compute a split-beam angle for ``beam_type=1`` based on ``angle_type``"""
        return (
            conversion_const
            * ds_beam[f"angle_{angle_type}"]
            / ds_beam[f"angle_sensitivity_{angle_type}"]
            - ds_beam[f"angle_offset_{angle_type}"]
        )

    # ensure that the beam_type is appropriate for calculation
    if np.all(ds_beam["beam_type"].data == 1):
        # obtain split-beam alongship angle
        theta_fc = compute_split_beam_beamtype1(angle_type="alongship")

        # obtain split-beam athwartship angle
        phi_fc = compute_split_beam_beamtype1(angle_type="athwartship")

    else:
        raise NotImplementedError("Computing split-beam angle is only available for beam_type=1!")

    return theta_fc, phi_fc


def _get_splitbeam_angle_complex_CW(ds_beam: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Obtains the split-beam angle data from complex encoded data with CW waveform.

    Parameters
    ----------
    ds_beam: xr.Dataset
        An ``EchoData`` beam group containing angle information needed for
        split-beam angle calculation

    Returns
    -------
    theta_fc: xr.Dataset
        The calculated split-beam alongship angle
    phi_fc: xr.Dataset
        The calculated split-beam athwartship angle
    """

    # TODO: ensure that the beam_type is appropriate for calculation

    return xr.Dataset(), xr.Dataset()


def _get_splitbeam_angle_complex_BB(ds_beam: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Obtains the split-beam angle data from complex encoded data with BB waveform.

    Parameters
    ----------
    ds_beam: xr.Dataset
        An ``EchoData`` beam group containing angle information needed for
        split-beam angle calculation

    Returns
    -------
    theta_fc: xr.Dataset
        The calculated split-beam alongship angle
    phi_fc: xr.Dataset
        The calculated split-beam athwartship angle
    """

    # TODO: ensure that the beam_type is appropriate for calculation

    return xr.Dataset(), xr.Dataset()


def _add_splitbeam_angle_to_ds(
    theta_fc: xr.Dataset, phi_fc: xr.Dataset, ds: xr.Dataset
) -> xr.Dataset:
    """
    Adds the split-beam angle data to the provided input ``ds``.

    Parameters
    ----------
    theta_fc: xr.Dataset
        The calculated split-beam alongship angle
    phi_fc: xr.Dataset
        The calculated split-beam athwartship angle
    ds: xr.Dataset
        The Dataset that ``theta_fc`` and ``phi_fc`` will be added to

    Returns
    -------
    xr.Dataset
        The input Dataset ``ds`` with split-beam angle data.
    """

    # TODO: do we want to add anymore attributes to these variables?
    # add appropriate attributes to theta_fc and phi_fc
    theta_fc.attrs["long_name"] = "split-beam alongship angle"
    phi_fc.attrs["long_name"] = "split-beam athwartship angle"

    # add the split-beam angles to the provided Dataset
    ds["angle_alongship"] = theta_fc
    ds["angle_athwartship"] = phi_fc

    return ds
