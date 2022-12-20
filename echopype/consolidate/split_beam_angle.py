"""
Contains functions necessary to compute the split-beam (alongship/athwartship)
angles and add them to a Dataset.
"""
from typing import Optional, Tuple

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
    conversion_const = 180.0 / 128.0

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

    # drop the beam dimension in theta_fc and phi_fc, if it exists
    if "beam" in theta_fc.dims:
        theta_fc = theta_fc.drop("beam").squeeze(dim="beam")
        phi_fc = phi_fc.drop("beam").squeeze(dim="beam")

    return theta_fc, phi_fc


def _compute_small_angle_approx_splitbeam_angle(
    backscatter: xr.DataArray, angle_sensitivity: xr.DataArray, angle_offset: xr.DataArray
) -> xr.DataArray:
    """
    Computes a split-beam angle based off of backscatter, angle sensitivity,
    and angle offset using a small angle approximation of ``sin``.

    Parameters
    ----------
    backscatter: xr.DataArray
        The backscatter alongship or athwartship
    angle_sensitivity: xr.DataArray
        The angle sensitivity alongship or athwartship
    angle_offset: xr.DataArray
        The angle offset alongship or athwartship

    Returns
    -------
    xr.DataArray
        Computed Split-beam angle values alongship or athwartship
    """
    return (
        np.arctan2(np.imag(backscatter), np.real(backscatter))
        / angle_sensitivity  # convert from electrical angle to physical angle
        / np.pi
        * 180.0  # convert from radian to degree
        - angle_offset  # correct for offset
    )


def _compute_backscatter_alongship_athwartship(
    ds_beam: xr.Dataset,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Computes the alongship or athwartship backscatter using the backscatter
    along the forward, aft, starboard, and port directions.

    Parameters
    ----------
    ds_beam: xr.Dataset
        An ``EchoData`` beam group with ``backscatter_r`` and ``backscatter_i``
        data in the forward, aft, starboard, and port directions

    Returns
    -------
    backscatter_theta: xr.DataArray
        The backscatter alongship
    backscatter_phi: xr.DataArray
        The backscatter athwartship
    """

    # get complex representation of backscatter
    backscatter = ds_beam["backscatter_r"] + 1j * ds_beam["backscatter_i"]

    # get backscatter in the forward, aft, starboard, and port directions
    backscatter_fore = 0.5 * (backscatter.isel(beam=2) + backscatter.isel(beam=3))  # forward
    backscatter_aft = 0.5 * (backscatter.isel(beam=0) + backscatter.isel(beam=1))  # aft
    backscatter_star = 0.5 * (backscatter.isel(beam=0) + backscatter.isel(beam=3))  # starboard
    backscatter_port = 0.5 * (backscatter.isel(beam=1) + backscatter.isel(beam=2))  # port

    # compute the alongship and athwartship backscatter
    backscatter_theta = backscatter_fore * np.conj(backscatter_aft)  # alongship
    backscatter_phi = backscatter_star * np.conj(backscatter_port)  # athwartship

    return backscatter_theta, backscatter_phi


def _get_splitbeam_angle_complex_CW(
    ds_beam: xr.Dataset, angle_sens_scale: Optional[xr.DataArray] = None
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Obtains the split-beam angle data from complex encoded data with CW waveform.

    Parameters
    ----------
    ds_beam: xr.Dataset
        An ``EchoData`` beam group containing angle information needed for
        split-beam angle calculation
    angle_sens_scale: xr.DataArray, optional
        A DataArray with the same ``channel`` dimension length as ``ds_beam``,
        which will be used to "scale" angle sensitivity based on frequency

    Returns
    -------
    theta_fc: xr.Dataset
        The calculated split-beam alongship angle
    phi_fc: xr.Dataset
        The calculated split-beam athwartship angle

    Notes
    -----
    Calculation is done by estimating the sphere position in the beam via split-beam
    processing. The estimate is in a band-average sense i.e. by computing the phase
    difference via the pulse compression outputs from combined transducer sectors.
    """

    # ensure that the beam_type is appropriate for calculation
    if np.all(np.logical_or(ds_beam["beam_type"].data == 1, ds_beam["beam_type"].data == 65)):

        # get backscatter alongship and athwartship
        backscatter_theta, backscatter_phi = _compute_backscatter_alongship_athwartship(ds_beam)

        # get angle sensitivity alongship and athwartship
        angle_sensitivity_alongship_fc = ds_beam["angle_sensitivity_alongship"].isel(
            ping_time=0, beam=0
        )
        angle_sensitivity_athwartship_fc = ds_beam["angle_sensitivity_athwartship"].isel(
            ping_time=0, beam=0
        )

        # "Scale" angle sensitivity based on frequency, if necessary
        if angle_sens_scale is not None:
            angle_sensitivity_alongship_fc = angle_sensitivity_alongship_fc * angle_sens_scale
            angle_sensitivity_athwartship_fc = angle_sensitivity_athwartship_fc * angle_sens_scale

        # get angle offset alongship and athwartship
        angle_offset_alongship_fc = ds_beam["angle_offset_alongship"].isel(ping_time=0, beam=0)
        angle_offset_athwartship_fc = ds_beam["angle_offset_athwartship"].isel(ping_time=0, beam=0)

        # compute split-beam angle alongship for beam_type=1
        theta_fc = _compute_small_angle_approx_splitbeam_angle(
            backscatter=backscatter_theta,
            angle_sensitivity=angle_sensitivity_alongship_fc,
            angle_offset=angle_offset_alongship_fc,
        )
        # compute split-beam angle athwartship for beam_type=1
        phi_fc = _compute_small_angle_approx_splitbeam_angle(
            backscatter=backscatter_phi,
            angle_sensitivity=angle_sensitivity_athwartship_fc,
            angle_offset=angle_offset_athwartship_fc,
        )

    else:
        raise NotImplementedError(
            "Computing split-beam angle is only available for beam_type=1 or 65!"
        )

    # drop the beam dimension in theta_fc and phi_fc, if it exists
    if "beam" in theta_fc.coords:
        theta_fc = theta_fc.drop_vars("beam")
        phi_fc = phi_fc.drop("beam")

    return theta_fc, phi_fc


def _get_splitbeam_angle_complex_BB_nopc(ds_beam: xr.Dataset) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Obtains the split-beam angle data from complex encoded data with BB waveform
    and without using pulse compression.

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

    # nominal frequency [Hz]
    freq_nominal = 120e3  # nominal frequency [Hz]

    # calculate center frequency
    freq_center = (
        ds_beam["frequency_start"].isel(ping_time=0, beam=0)
        + ds_beam["frequency_end"].isel(ping_time=0, beam=0)
    ) / 2.0

    # get "scale" for angle sensitivity
    ang_sense_scale = freq_center / freq_nominal

    # calculate the split-beam angle
    # TODO: should we use _get_splitbeam_angle_complex_CW or just rewrite all of the code?
    theta_fc, phi_fc = _get_splitbeam_angle_complex_CW(ds_beam, ang_sense_scale)

    return theta_fc, phi_fc


def _get_splitbeam_angle_complex_BB_pc(ds_beam: xr.Dataset) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Obtains the split-beam angle data from complex encoded data with BB waveform
    and with pulse compression.

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

    # TODO: make sure to check that the appropriate beam_type is being used

    return xr.DataArray(), xr.DataArray()


def _add_splitbeam_angle_to_ds(
    theta_fc: xr.Dataset,
    phi_fc: xr.Dataset,
    ds: xr.Dataset,
    return_dataset: bool,
    source_ds_path: Optional[str] = None,
    file_type: Optional[str] = None,
    storage_options: dict = {},
) -> Optional[xr.Dataset]:
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
    return_dataset: bool
        Whether a dataset will be returned or not
    source_ds_path: str, optional
        The path to the file corresponding to ``ds``, if it exists
    file_type: {"netcdf4", "zarr"}, optional
        The file type corresponding to ``source_ds_path``
    storage_options: dict, default={}
        Any additional parameters for the storage backend, corresponding to the
        path ``source_ds_path``

    Returns
    -------
    xr.Dataset or None
        If ``return_dataset=False``, nothing will be returned. If ``return_dataset=True``
        either the input dataset ``ds`` or a lazy-loaded Dataset (obtained from
        the path provided by ``source_ds_path``) with the split-beam angle data added
        will be returned.
    """

    # TODO: do we want to add anymore attributes to these variables?
    # add appropriate attributes to theta_fc and phi_fc
    theta_fc.attrs["long_name"] = "split-beam alongship angle"
    phi_fc.attrs["long_name"] = "split-beam athwartship angle"

    # assign names to split-beam angle data
    theta_fc.name = "angle_alongship"
    phi_fc.name = "angle_athwartship"

    if source_ds_path is not None:

        # put the variables into a Dataset so they can be written at the same time
        splitb_ds = xr.Dataset(
            data_vars={"angle_alongship": theta_fc, "angle_athwartship": phi_fc},
            coords=theta_fc.coords,
        )

        # write the split-beam angle data to the provided path
        if file_type == "netcdf4":
            splitb_ds.to_netcdf(
                path=source_ds_path, mode="a", encoding=splitb_ds.encoding, **storage_options
            )
        else:
            splitb_ds.to_zarr(
                store=source_ds_path, mode="a", encoding=splitb_ds.encoding, **storage_options
            )

    else:

        # add the split-beam angles to the provided Dataset
        ds["angle_alongship"] = theta_fc
        ds["angle_athwartship"] = phi_fc

    if return_dataset and (source_ds_path is not None):
        # open up and return Dataset in source_ds_path
        return xr.open_dataset(source_ds_path, engine=file_type, chunks="auto", **storage_options)
    elif return_dataset:
        return ds
    else:
        return None
