"""
Contains functions necessary to compute the split-beam (alongship/athwartship)
angles and add them to a Dataset.
"""
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr

from ..calibrate.ek80_complex import compress_pulse, get_transmit_signal


def _compute_angle_from_complex(
    bs: xr.DataArray, beam_type: int, sens: List[xr.DataArray], offset: List[xr.DataArray]
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Compute split-beam angles from raw data from transducer sectors.

    Can be used for data from a single channel or multiple channels,
    depending on what is in ``bs``.

    Parameters
    ----------
    bs: xr.DataArray
        Complex backscatter samples from a single channel or multiple channels
    beam_type: int
        The type of beam being considered
    sens: list of xr.DataArray
        A list of length two where the first element corresponds to the
        angle sensitivity alongship and the second corresponds to the
        angle sensitivity athwartship
    offset: list of xr.DataArray
        A list of length two where the first element corresponds to the
        angle offset alongship and the second corresponds to the
        angle offset athwartship

    Returns
    -------
    theta: xr.DataArray
        The calculated split-beam alongship angle for a specific channel
    phi: xr.DataArray
        The calculated split-beam athwartship angle for a specific channel

    Notes
    -----
    This function should only be used for data with complex backscatter.
    """

    # 4-sector transducer
    if beam_type == 1:
        bs_fore = (bs.isel(beam=2) + bs.isel(beam=3)) / 2  # forward
        bs_aft = (bs.isel(beam=0) + bs.isel(beam=1)) / 2  # aft
        bs_star = (bs.isel(beam=0) + bs.isel(beam=3)) / 2  # starboard
        bs_port = (bs.isel(beam=1) + bs.isel(beam=2)) / 2  # port

        bs_theta = bs_fore * np.conj(bs_aft)
        bs_phi = bs_star * np.conj(bs_port)
        theta = np.arctan2(np.imag(bs_theta), np.real(bs_theta)) / np.pi * 180
        phi = np.arctan2(np.imag(bs_phi), np.real(bs_phi)) / np.pi * 180

    # 3-sector transducer with or without center element
    elif beam_type in [17, 49, 65, 81]:
        # 3-sector
        if beam_type == 17:
            bs_star = bs.isel(beam=0)
            bs_port = bs.isel(beam=1)
            bs_fore = bs.isel(beam=2)
        else:
            # 3-sector + 1 center element
            bs_star = (bs.isel(beam=0) + bs.isel(beam=3)) / 2
            bs_port = (bs.isel(beam=1) + bs.isel(beam=3)) / 2
            bs_fore = (bs.isel(beam=2) + bs.isel(beam=3)) / 2

        bs_fac1 = bs_fore * np.conj(bs_star)
        bs_fac2 = bs_fore * np.conj(bs_port)
        fac1 = np.arctan2(np.imag(bs_fac1), np.real(bs_fac1)) / np.pi * 180
        fac2 = np.arctan2(np.imag(bs_fac2), np.real(bs_fac2)) / np.pi * 180

        theta = (fac1 + fac2) / np.sqrt(3)
        phi = fac2 - fac1

    # EC150–3C
    elif beam_type == 97:
        raise NotImplementedError

    else:
        raise ValueError("beam_type not recognized!")

    theta = theta / sens[0] - offset[0]
    phi = phi / sens[1] - offset[1]

    return theta, phi


def get_angle_power_samples(
    ds_beam: xr.Dataset, angle_params: dict
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Obtain split-beam angle from CW mode power samples.

    Parameters
    ----------
    ds_beam: xr.Dataset
        An ``EchoData`` Sonar/Beam_group1 group (complex samples always in Beam_group1)
    angle_params : dict
        A dictionary containing angle_offset/angle_sensitivity parameters from the calibrated dataset

    Returns
    -------
    theta: xr.Dataset
        Split-beam alongship angle
    phi: xr.Dataset
        Split-beam athwartship angle

    Notes
    -----
    Can be used on both EK60 and EK80 data

    Computation done for ``beam_type=1``:
    ``physical_angle = ((raw_angle * 180 / 128) / sensitivity) - offset``
    """

    # raw_angle scaling constant
    conversion_const = 180.0 / 128.0

    def _e2f(angle_type: str) -> xr.Dataset:
        """Convert electric angle to physical angle for split-beam data"""
        return (
            conversion_const
            * ds_beam[f"angle_{angle_type}"]
            / angle_params[f"angle_sensitivity_{angle_type}"]
            - angle_params[f"angle_offset_{angle_type}"]
        )

    # add split-beam angle if at least one channel is split-beam
    # in the case when some channels are split-beam and some single-beam
    # the single-beam channels will be all NaNs and _e2f would run through and output NaNs
    if not np.all(ds_beam["beam_type"].data == 0):
        theta = _e2f(angle_type="alongship")  # split-beam alongship angle
        phi = _e2f(angle_type="athwartship")  # split-beam athwartship angle

    else:
        raise ValueError(
            "Computing physical split-beam angle is only available for data "
            "from split-beam transducers!"
        )

    # TODO: review and remove this block
    # drop the beam dimension in theta and phi, if it exists
    if "beam" in theta.dims:
        theta = theta.drop("beam").squeeze(dim="beam")
        phi = phi.drop("beam").squeeze(dim="beam")

    return theta, phi


def get_angle_complex_samples(
    ds_beam: xr.Dataset, angle_params: dict, pc_params: dict = None
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Obtain split-beam angle from CW or BB mode complex samples.

    Parameters
    ----------
    ds_beam : xr.Dataset
        An ``EchoData`` Sonar/Beam_group1 group (complex samples always in Beam_group1)
    angle_params : dict
        A dictionary containing angle_offset/angle_sensitivity parameters from the calibrated dataset
    pc_params : dict
        Parameters needed for pulse compression
        This dict also serves as a flag for whether to apply pulse compression

    Returns
    -------
    theta : xr.Dataset
        Split-beam alongship angle
    phi : xr.Dataset
        Split-beam athwartship angle
    """

    # Get complex backscatter samples
    bs = ds_beam["backscatter_r"] + 1j * ds_beam["backscatter_i"]

    # Pulse compression if pc_params exists
    if pc_params is not None:
        tx, tx_time = get_transmit_signal(
            beam=ds_beam,
            coeff=pc_params,  # this is filter_coeff with fs added
            waveform_mode="BB",
            fs=pc_params["receiver_sampling_frequency"],  # this is the added fs
        )
        bs = compress_pulse(backscatter=bs, chirp=tx)  # has beam dim

    # Compute angles
    # unique beam_type existing in the dataset
    beam_type_all_ch = np.unique(ds_beam["beam_type"].isel(ping_time=0).data)

    if beam_type_all_ch.size == 1:
        # If beam_type is the same for all channels, process all channels at once
        theta, phi = _compute_angle_from_complex(
            bs=bs,
            beam_type=beam_type_all_ch[0],  # beam_type for all channels
            sens=[
                angle_params["angle_sensitivity_alongship"],
                angle_params["angle_sensitivity_athwartship"],
            ],
            offset=[
                angle_params["angle_offset_alongship"],
                angle_params["angle_offset_athwartship"],
            ],
        )
    else:
        # beam_type different for some channels, process each channel separately
        theta, phi = [], []
        for ch_id in bs["channel"].data:
            theta_ch, phi_ch = _compute_angle_from_complex(
                bs=bs.sel(channel=ch_id),
                # beam_type is not time-varying
                beam_type=(
                    ds_beam["beam_type"].sel(channel=ch_id).isel(ping_time=0).drop("ping_time")
                ),
                sens=[
                    angle_params["angle_sensitivity_alongship"].sel(channel=ch_id),
                    angle_params["angle_sensitivity_athwartship"].sel(channel=ch_id),
                ],
                offset=[
                    angle_params["angle_offset_alongship"].sel(channel=ch_id),
                    angle_params["angle_offset_athwartship"].sel(channel=ch_id),
                ],
            )
            theta.append(theta_ch)
            phi.append(phi_ch)

        # Combine angles from all channels
        theta = xr.DataArray(
            data=theta,
            coords={
                "channel": bs["channel"],
                "ping_time": bs["ping_time"],
                "range_sample": bs["range_sample"],
            },
        )
        phi = xr.DataArray(
            data=phi,
            coords={
                "channel": bs["channel"],
                "ping_time": bs["ping_time"],
                "range_sample": bs["range_sample"],
            },
        )

    # TODO: review and remove this block
    # drop the beam dimension in theta and phi, if it exists
    if "beam" in theta.coords:
        theta = theta.drop_vars("beam")
        phi = phi.drop("beam")

    return theta, phi


def add_angle_to_ds(
    theta: xr.Dataset,
    phi: xr.Dataset,
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
    theta: xr.Dataset
        The calculated split-beam alongship angle
    phi: xr.Dataset
        The calculated split-beam athwartship angle
    ds: xr.Dataset
        The Dataset that ``theta`` and ``phi`` will be added to
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
    # add appropriate attributes to theta and phi
    theta.attrs["long_name"] = "split-beam alongship angle"
    phi.attrs["long_name"] = "split-beam athwartship angle"

    if source_ds_path is not None:
        # put the variables into a Dataset, so they can be written at the same time
        # add ds attributes to splitb_ds since they will be overwritten by to_netcdf/zarr
        splitb_ds = xr.Dataset(
            data_vars={"angle_alongship": theta, "angle_athwartship": phi},
            coords=theta.coords,
            attrs=ds.attrs,
        )

        # release any resources linked to ds (necessary for to_netcdf)
        ds.close()

        # write the split-beam angle data to the provided path
        if file_type == "netcdf4":
            splitb_ds.to_netcdf(path=source_ds_path, mode="a", **storage_options)
        else:
            splitb_ds.to_zarr(store=source_ds_path, mode="a", **storage_options)

        if return_dataset:
            # open up and return Dataset in source_ds_path
            return xr.open_dataset(source_ds_path, engine=file_type, chunks={}, **storage_options)

    else:
        # add the split-beam angles to the provided Dataset
        ds["angle_alongship"] = theta
        ds["angle_athwartship"] = phi

        if return_dataset:
            # return input dataset with split-beam angle data
            return ds
