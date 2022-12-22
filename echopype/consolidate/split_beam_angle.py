"""
Contains functions necessary to compute the split-beam (alongship/athwartship)
angles and add them to a Dataset.
"""
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr

from ..echodata import EchoData


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
    theta: xr.Dataset
        The calculated split-beam alongship angle
    phi: xr.Dataset
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

    def _e2f(angle_type: str) -> xr.Dataset:
        """Convert electric angle to physical angle for split-beam data"""
        return (
            conversion_const
            * ds_beam[f"angle_{angle_type}"]
            / ds_beam[f"angle_sensitivity_{angle_type}"]
            - ds_beam[f"angle_offset_{angle_type}"]
        )

    # ensure that the beam_type is appropriate for calculation
    if np.all(ds_beam["beam_type"].data != 0):

        # obtain split-beam alongship angle
        theta = _e2f(angle_type="alongship")

        # obtain split-beam athwartship angle
        phi = _e2f(angle_type="athwartship")

    else:
        raise NotImplementedError(
            "Computing physical split-beam angle is only available for data "
            "from split-beam transducers!"
        )

    # drop the beam dimension in theta and phi, if it exists
    if "beam" in theta.dims:
        theta = theta.drop("beam").squeeze(dim="beam")
        phi = phi.drop("beam").squeeze(dim="beam")

    return theta, phi


def _compute_small_angle_approx_splitbeam_angle(
    backscatter: xr.DataArray, angle_sensitivity: xr.DataArray, angle_offset: xr.DataArray
) -> xr.DataArray:
    """
    Computes a split-beam angle based off of backscatter, angle sensitivity,
    and angle offset using a small angle approximation of ``arcsin``.

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


def _get_splitbeam_angle_complex_CW(ds_beam: xr.Dataset) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Obtains the split-beam angle data from complex encoded data with CW waveform.

    Parameters
    ----------
    ds_beam: xr.Dataset
        An ``EchoData`` beam group containing angle information needed for
        split-beam angle calculation

    Returns
    -------
    theta: xr.Dataset
        The calculated split-beam alongship angle
    phi: xr.Dataset
        The calculated split-beam athwartship angle
    """

    # ensure that the beam_type is appropriate for calculation
    if np.all(ds_beam["beam_type"].data == 1):

        # get backscatter alongship and athwartship
        backscatter_theta, backscatter_phi = _compute_backscatter_alongship_athwartship(ds_beam)

        # get angle sensitivity alongship and athwartship
        angle_sensitivity_alongship_fc = ds_beam["angle_sensitivity_alongship"].isel(
            ping_time=0, beam=0
        )
        angle_sensitivity_athwartship_fc = ds_beam["angle_sensitivity_athwartship"].isel(
            ping_time=0, beam=0
        )

        # get angle offset alongship and athwartship
        angle_offset_alongship_fc = ds_beam["angle_offset_alongship"].isel(ping_time=0, beam=0)
        angle_offset_athwartship_fc = ds_beam["angle_offset_athwartship"].isel(ping_time=0, beam=0)

        # compute split-beam angle alongship for beam_type=1
        theta = _compute_small_angle_approx_splitbeam_angle(
            backscatter=backscatter_theta,
            angle_sensitivity=angle_sensitivity_alongship_fc,
            angle_offset=angle_offset_alongship_fc,
        )
        # compute split-beam angle athwartship for beam_type=1
        phi = _compute_small_angle_approx_splitbeam_angle(
            backscatter=backscatter_phi,
            angle_sensitivity=angle_sensitivity_athwartship_fc,
            angle_offset=angle_offset_athwartship_fc,
        )

    else:
        raise NotImplementedError("Computing split-beam angle is only available for beam_type=1!")

    # drop the beam dimension in theta and phi, if it exists
    if "beam" in theta.coords:
        theta = theta.drop_vars("beam")
        phi = phi.drop("beam")

    return theta, phi


def get_interp_offset(
    param: str, chan_id: str, freq_center: xr.DataArray, ed: EchoData
) -> np.ndarray:
    """
    Obtains an angle offset by first interpolating the
    ``angle_offset_alongship`` or ``angle_offset_athwartship``
    data found in the ``Vendor_specific`` group and then
    selecting the offset corresponding to the frequency center
    value for ``channel=chan_id``.

    Parameters
    ----------
    param: {"angle_offset_alongship", "angle_offset_athwartship"}
        The angle offset data to select in the ``Vendor_specific`` group
    chan_id: str
        The channel used to select the frequency center value
    freq_center: xr.DataArray
        An Array filled with frequency center values with coordinate ``channel``
    ed: EchoData
        An ``EchoData`` object holding the raw data

    Returns
    -------
    np.ndarray
        Array filled with the requested angle offset values
    """

    freq_wanted = freq_center.sel(channel=chan_id)
    return (
        ed["Vendor_specific"][param].sel(cal_channel_id=chan_id).interp(cal_frequency=freq_wanted)
    ).values


def get_offset(
    ds_b: xr.Dataset, fc: xr.DataArray, freq_nominal: xr.DataArray, ed: EchoData
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Obtains the alongship and athwartship angle offsets.

    Parameters
    ----------
    ds_b: xr.Dataset
        The dataset corresponding to a beam group
    fc: xr.DataArray
        Array corresponding to the frequency center
    freq_nominal: xr.DataArray
        Array of frequency nominal values
    ed: EchoData
        An ``EchoData`` object holding the raw data

    Returns
    -------
    offset_along: xr.DataArray
        Array corresponding to the angle alongship offset
    offset_athwart: xr.DataArray
        Array corresponding to the angle athwartship offset
    """

    # initialize lists that will hold offsets
    offset_along = []
    offset_athwart = []

    # obtain the offsets for each channel
    for ch in fc["channel"].values:
        if ch in ed["Vendor_specific"]["cal_channel_id"]:

            # calculate offsets using Vendor_specific values
            offset_along.append(
                get_interp_offset(param="angle_offset_alongship", chan_id=ch, freq_center=fc, ed=ed)
            )
            offset_athwart.append(
                get_interp_offset(
                    param="angle_offset_athwartship", chan_id=ch, freq_center=fc, ed=ed
                )
            )
        else:

            # calculate offsets using data in ds_b
            offset_along.append(
                ds_b["angle_offset_alongship"].sel(channel=ch).isel(ping_time=0, beam=0)
                * fc.sel(channel=ch)
                / freq_nominal.sel(channel=ch)
            )
            offset_athwart.append(
                ds_b["angle_offset_athwartship"].sel(channel=ch).isel(ping_time=0, beam=0)
                * fc.sel(channel=ch)
                / freq_nominal.sel(channel=ch)
            )

    # construct offset DataArrays from lists
    offset_along = xr.DataArray(
        offset_along, coords={"channel": fc["channel"], "ping_time": fc["ping_time"]}
    )
    offset_athwart = xr.DataArray(
        offset_athwart, coords={"channel": fc["channel"], "ping_time": fc["ping_time"]}
    )
    return offset_along, offset_athwart


def get_splitbeam(
    bs: xr.Dataset, beam_type: int, sens: List[xr.DataArray], offset: List[xr.DataArray]
):
    """
    Obtains the split-beam angle data alongship and athwartship
    using data from a single channel.

    Parameters
    ----------
    bs: xr.Dataset
        Complex representation of backscatter
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
    theta: xr.Dataset
        The calculated split-beam alongship angle for a specific channel
    phi: xr.Dataset
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


def _get_splitbeam_angle_complex_BB_nopc(
    ds_beam: xr.Dataset, ed: EchoData
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Obtains the split-beam angle data from complex encoded data with BB waveform
    and without using pulse compression.

    Parameters
    ----------
    ds_beam: xr.Dataset
        An ``EchoData`` beam group containing angle information needed for
        split-beam angle calculation
    ed: EchoData
        An ``EchoData`` object holding the raw data

    Returns
    -------
    theta: xr.Dataset
        The calculated split-beam alongship angle
    phi: xr.Dataset
        The calculated split-beam athwartship angle
    """

    # nominal frequency [Hz]
    freq_nominal = ds_beam["frequency_nominal"]

    # calculate center frequency
    freq_center = (ds_beam["frequency_start"] + ds_beam["frequency_end"]).isel(beam=0) / 2

    # obtain the angle alongship and athwartship offsets
    offset_along, offset_athwart = get_offset(
        ds_b=ds_beam, fc=freq_center, freq_nominal=freq_nominal, ed=ed
    )

    # obtain the angle sensitivity values alongship and athwartship
    sens_along = ds_beam["angle_sensitivity_alongship"].isel(beam=0) * freq_center / freq_nominal
    sens_athwart = (
        ds_beam["angle_sensitivity_athwartship"].isel(beam=0) * freq_center / freq_nominal
    )

    # get complex representation of backscatter
    backscatter = ds_beam["backscatter_r"] + 1j * ds_beam["backscatter_i"]

    # initialize list that will hold split-beam angle data for each channel
    theta_channels = []
    phi_channels = []

    # obtain the split-beam angle data for each channel
    for chan_id in backscatter.channel.values:
        theta, phi = get_splitbeam(
            bs=backscatter.sel(channel=chan_id),
            beam_type=int(ds_beam["beam_type"].sel(channel=chan_id).isel(ping_time=0)),
            sens=[sens_along.sel(channel=chan_id), sens_athwart.sel(channel=chan_id)],
            offset=[offset_along.sel(channel=chan_id), offset_athwart.sel(channel=chan_id)],
        )

        theta_channels.append(theta)
        phi_channels.append(phi)

    # collect and construct final DataArrays for split-beam angle data
    theta = xr.DataArray(
        data=theta_channels,
        coords={
            "channel": backscatter.channel,
            "ping_time": theta_channels[0].ping_time,
            "range_sample": theta_channels[0].range_sample,
        },
    )

    phi = xr.DataArray(
        data=phi_channels,
        coords={
            "channel": backscatter.channel,
            "ping_time": phi_channels[0].ping_time,
            "range_sample": phi_channels[0].range_sample,
        },
    )

    return theta, phi


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
    theta: xr.Dataset
        The calculated split-beam alongship angle
    phi: xr.Dataset
        The calculated split-beam athwartship angle
    """

    # TODO: make sure to check that the appropriate beam_type is being used
    raise NotImplementedError(
        "Obtaining the split-beam angle data using " "pulse compression has not been implemented!"
    )

    return xr.DataArray(), xr.DataArray()


def _add_splitbeam_angle_to_ds(
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

    else:

        # add the split-beam angles to the provided Dataset
        ds["angle_alongship"] = theta
        ds["angle_athwartship"] = phi

    if return_dataset and (source_ds_path is not None):

        # open up and return Dataset in source_ds_path
        return xr.open_dataset(source_ds_path, engine=file_type, chunks={}, **storage_options)

    elif return_dataset:

        # return input dataset with split-beam angle data
        return ds
