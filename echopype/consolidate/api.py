import datetime
import pathlib
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr

from ..calibrate.ek80_complex import get_filter_coeff
from ..echodata import EchoData
from ..echodata.simrad import retrieve_correct_beam_group
from ..utils.io import get_file_format, open_source
from ..utils.log import _init_logger
from ..utils.prov import add_processing_level
from .split_beam_angle import get_angle_complex_samples, get_angle_power_samples

logger = _init_logger(__name__)

POSITION_VARIABLES = ["latitude", "longitude"]


def swap_dims_channel_frequency(ds: Union[xr.Dataset, str, pathlib.Path]) -> xr.Dataset:
    """
    Use frequency_nominal in place of channel to be dataset dimension and coorindate.

    This is useful because the nominal transducer frequencies are commonly used to
    refer to data collected from a specific transducer.

    Parameters
    ----------
    ds : xr.Dataset or str or pathlib.Path
        Dataset or path to a file containing the Dataset
        for which the dimension will be swapped

    Returns
    -------
    The input dataset with the dimension swapped

    Notes
    -----
    This operation is only possible when there are no duplicated frequencies present in the file.
    """
    ds = open_source(ds, "dataset", {})
    # Only possible if no duplicated frequencies
    if np.unique(ds["frequency_nominal"]).size == ds["frequency_nominal"].size:
        return (
            ds.set_coords("frequency_nominal")
            .swap_dims({"channel": "frequency_nominal"})
            .reset_coords("channel")
        )
    else:
        raise ValueError(
            "Duplicated transducer nominal frequencies exist in the file. "
            "Operation is not valid."
        )


def add_depth(
    ds: Union[xr.Dataset, str, pathlib.Path],
    depth_offset: float = 0,
    tilt: float = 0,
    downward: bool = True,
) -> xr.Dataset:
    """
    Create a depth data variable based on data in Sv dataset.

    The depth is generated based on whether the transducers are mounted vertically
    or with a polar angle to vertical, and whether the transducers were pointed
    up or down.

    Parameters
    ----------
    ds : xr.Dataset or str or pathlib.Path
        Source Sv dataset or path to a file containing the Source Sv dataset
        to which a depth variable will be added.
        Must contain `echo_range`.
    depth_offset : float
        Offset along the vertical (depth) dimension to account for actual transducer
        position in water, since `echo_range` is counted from transducer surface.
        Default is 0.
    tilt : float
        Transducer tilt angle [degree].
        Default is 0 (transducer mounted vertically).
    downward : bool
        Whether or not the transducers point downward.
        Default to True.

    Returns
    -------
    The input dataset with a `depth` variable (in meters) added

    Notes
    -----
    Currently this function only scalar inputs of depth_offset and tilt angle.
    In future expansion we plan to add the following options:

    * Allow inputs as xr.DataArray for time-varying variations of these variables
    * Use data stored in the EchoData object or raw-converted file from which the Sv is derived,
      specifically `water_level`, `vertical_offtset` and `tilt` in the `Platform` group.
    """
    # TODO: add options to use water_depth, vertical_offset, tilt stored in EchoData
    # # Water level has to come from somewhere
    # if depth_offset is None:
    #     if "water_level" in ds:
    #         depth_offset = ds["water_level"]
    #     else:
    #         raise ValueError(
    #             "water_level not found in dataset and needs to be supplied by the user"
    #         )

    # # If not vertical needs to have tilt
    # if not vertical:
    #     if tilt is None:
    #         if "tilt" in ds:
    #             tilt = ds["tilt"]
    #         else:
    #             raise ValueError(
    #                 "tilt not found in dataset and needs to be supplied by the user. "
    #                 "Required when vertical=False"
    #             )
    # else:
    #     tilt = 0

    ds = open_source(ds, "dataset", {})
    # Multiplication factor depending on if transducers are pointing downward
    mult = 1 if downward else -1

    # Compute depth
    ds["depth"] = mult * ds["echo_range"] * np.cos(tilt / 180 * np.pi) + depth_offset
    ds["depth"].attrs = {"long_name": "Depth", "standard_name": "depth", "units": "m"}

    # Add history attribute
    history_attr = (
        f"{datetime.datetime.utcnow()} +00:00. "
        "Added based on echo_range or other data in Sv dataset."  # noqa
    )
    ds["depth"] = ds["depth"].assign_attrs({"history": history_attr})

    return ds


@add_processing_level("L2A")
def add_location(
    ds: Union[xr.Dataset, str, pathlib.Path],
    echodata: Optional[Union[EchoData, str, pathlib.Path]],
    nmea_sentence: Optional[str] = None,
):
    """
    Add geographical location (latitude/longitude) to the Sv dataset.

    This function interpolates the location from the Platform group in the original data file
    based on the time when the latitude/longitude data are recorded and the time the acoustic
    data are recorded (`ping_time`).

    Parameters
    ----------
    ds : xr.Dataset or str or pathlib.Path
        An Sv or MVBS dataset or path to a file containing the Sv or MVBS
        dataset for which the geographical locations will be added to
    echodata : EchoData or str or pathlib.Path
        An ``EchoData`` object or path to a file containing the ``EchoData``
        object holding the raw data
    nmea_sentence
        NMEA sentence to select a subset of location data (optional)

    Returns
    -------
    The input dataset with the location data added
    """

    def sel_interp(var, time_dim_name):
        # NMEA sentence selection
        if nmea_sentence:
            position_var = echodata["Platform"][var][
                echodata["Platform"]["sentence_type"] == nmea_sentence
            ]
        else:
            position_var = echodata["Platform"][var]

        if len(position_var) == 1:
            # Propagate single, fixed-location coordinate
            return xr.DataArray(
                data=position_var.values[0] * np.ones(len(ds["ping_time"]), dtype=np.float64),
                dims=["ping_time"],
                attrs=position_var.attrs,
            )
        else:
            # Values may be nan if there are ping_time values outside the time_dim_name range
            return position_var.interp(**{time_dim_name: ds["ping_time"]})

    ds = open_source(ds, "dataset", {})
    echodata = open_source(echodata, "echodata", {})

    if "longitude" not in echodata["Platform"] or echodata["Platform"]["longitude"].isnull().all():
        raise ValueError("Coordinate variables not present or all nan")

    # Check if any latitude/longitude value is NaN/0
    contains_nan_lat_lon = (
        np.isnan(echodata["Platform"]["latitude"].values).any()
        or np.isnan(echodata["Platform"]["longitude"].values).any()
    )
    contains_zero_lat_lon = (echodata["Platform"]["latitude"].values == 0).any() or (
        echodata["Platform"]["longitude"].values == 0
    ).any()
    interp_msg = (
        "Interpolation may be negatively impacted, "
        "consider handling these values before calling ``add_location``."
    )
    if contains_nan_lat_lon:
        logger.warning(f"Latitude and/or longitude arrays contain NaNs. {interp_msg}")
    if contains_zero_lat_lon:
        logger.warning(f"Latitude and/or longitude arrays contain zeros. {interp_msg}")

    interp_ds = ds.copy()
    time_dim_name = list(echodata["Platform"]["longitude"].dims)[0]

    # Check if there are duplicates in time_dim_name
    if len(np.unique(echodata["Platform"][time_dim_name].data)) != len(
        echodata["Platform"][time_dim_name].data
    ):
        raise ValueError(
            f'The ``echodata["Platform"]["{time_dim_name}"]`` array contains duplicate values. '
            "Downstream interpolation on the position variables requires unique time values."
        )

    interp_ds["latitude"] = sel_interp("latitude", time_dim_name)
    interp_ds["longitude"] = sel_interp("longitude", time_dim_name)

    # Most attributes are attached automatically via interpolation
    # here we add the history
    history_attr = (
        f"{datetime.datetime.utcnow()} +00:00. "
        "Interpolated or propagated from Platform latitude/longitude."  # noqa
    )
    for da_name in POSITION_VARIABLES:
        interp_ds[da_name] = interp_ds[da_name].assign_attrs({"history": history_attr})

    if time_dim_name in interp_ds:
        interp_ds = interp_ds.drop_vars(time_dim_name)

    return interp_ds


def add_splitbeam_angle(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    echodata: Union[EchoData, str, pathlib.Path],
    waveform_mode: str,
    encode_mode: str,
    pulse_compression: bool = False,
    storage_options: dict = {},
    to_disk: bool = True,
) -> xr.Dataset:
    """
    Add split-beam (alongship/athwartship) angles into the Sv dataset.
    This function calculates the alongship/athwartship angle using data stored
    in the Sonar/Beam_groupX groups of an EchoData object.

    In cases when angle data does not already exist or cannot be computed from the data,
    an error is issued and no angle variables are added to the dataset.

    Parameters
    ----------
    source_Sv: xr.Dataset or str or pathlib.Path
        The Sv Dataset or path to a file containing the Sv Dataset,
        to which the split-beam angles will be added
    echodata: EchoData or str or pathlib.Path
        An ``EchoData`` object or path to a file containing the ``EchoData``
        object holding the raw data
    waveform_mode : {"CW", "BB"}
        Type of transmit waveform

        - ``"CW"`` for narrowband transmission,
          returned echoes recorded either as complex or power/angle samples
        - ``"BB"`` for broadband transmission,
          returned echoes recorded as complex samples

    encode_mode : {"complex", "power"}
        Type of encoded return echo data

        - ``"complex"`` for complex samples
        - ``"power"`` for power/angle samples, only allowed when
          the echosounder is configured for narrowband transmission
    pulse_compression: bool, False
        Whether pulse compression should be used (only valid for
        ``waveform_mode="BB"`` and ``encode_mode="complex"``)
    storage_options: dict, default={}
        Any additional parameters for the storage backend, corresponding to the
        path provided for ``source_Sv``
    to_disk: bool, default=True
        If ``False``, ``to_disk`` with split-beam angles added will be returned.
        ``to_disk=True`` is useful when ``source_Sv`` is a path and
        users only want to write the split-beam angle data to this path.

    Returns
    -------
    xr.Dataset or None
        If ``to_disk=False``, nothing will be returned.
        If ``to_disk=True``, either the input dataset ``source_Sv``
        or a lazy-loaded Dataset (from the path ``source_Sv``)
        with split-beam angles added will be returned.


    Raises
    ------
    ValueError
        If ``echodata`` has a sonar model that is not analogous to either EK60 or EK80
    ValueError
        If the input ``source_Sv`` does not have a ``channel`` dimension
    ValueError
        If ``source_Sv`` does not have appropriate dimension lengths in
        comparison to ``echodata`` data
    ValueError
        If the provided ``waveform_mode``, ``encode_mode``, and ``pulse_compression`` are not valid
    NotImplementedError
        If an unknown ``beam_type`` is encountered during the split-beam calculation

    Notes
    -----
    Split-beam angle data potentially exist for the Simrad EK60 or EK80 echosounders
    with split-beam transducers and configured to store angle data (along with power samples)
    or store raw complex samples.

    In most cases where the type of samples collected by the echosounder (power/angle
    samples or complex samples) and the transmit waveform (broadband or narrowband)
    are identical across all channels, the channels existing in ``source_Sv`` and `
    `echodata`` will be identical. If this is not the case, only angle data corresponding
    to channels existing in ``source_Sv`` will be added.
    """
    # ensure that when source_Sv is a Dataset then to_disk should be False
    if not isinstance(source_Sv, (str, Path)) and to_disk:
        raise ValueError(
            "The input source_Sv must be a path when to_disk=True, "
            "so that the split-beam angles can be written to disk!"
        )

    # obtain the file format of source_Sv if it is a path
    if isinstance(source_Sv, (str, Path)):
        source_Sv_type = get_file_format(source_Sv)

    source_Sv = open_source(source_Sv, "dataset", storage_options)
    echodata = open_source(echodata, "echodata", storage_options)

    # ensure that echodata was produced by EK60 or EK80-like sensors
    if echodata.sonar_model not in ["EK60", "ES70", "EK80", "ES80", "EA640"]:
        raise ValueError(
            "The sonar model that produced echodata does not have split-beam "
            "transducers, split-beam angles cannot be added to source_Sv!"
        )

    # raise not implemented error if source_Sv corresponds to MVBS
    if source_Sv.attrs["processing_function"] == "commongrid.compute_MVBS":
        raise NotImplementedError("Adding split-beam data to MVBS has not been implemented!")

    # check that the appropriate waveform and encode mode have been given
    # and obtain the echodata group path corresponding to encode_mode
    ed_beam_group = retrieve_correct_beam_group(echodata, waveform_mode, encode_mode)

    # check that source_Sv at least has a channel dimension
    if "channel" not in source_Sv.variables:
        raise ValueError("The input source_Sv Dataset must have a channel dimension!")

    # Select ds_beam channels from source_Sv
    ds_beam = echodata[ed_beam_group].sel(channel=source_Sv["channel"].values)

    # Assemble angle param dict
    angle_param_list = [
        "angle_sensitivity_alongship",
        "angle_sensitivity_athwartship",
        "angle_offset_alongship",
        "angle_offset_athwartship",
    ]
    angle_params = {}
    for p_name in angle_param_list:
        if p_name in source_Sv:
            angle_params[p_name] = source_Sv[p_name]
        else:
            raise ValueError(f"source_Sv does not contain the necessary parameter {p_name}!")

    # fail if source_Sv and ds_beam do not have the same lengths
    # for ping_time, range_sample, and channel
    same_dim_lens = [
        ds_beam.dims[dim] == source_Sv.dims[dim] for dim in ["channel", "ping_time", "range_sample"]
    ]
    if not same_dim_lens:
        raise ValueError(
            "The 'source_Sv' dataset does not have the same dimensions as data in 'echodata'!"
        )

    # obtain split-beam angles from
    # CW mode data
    if waveform_mode == "CW":
        if encode_mode == "power":  # power data
            theta, phi = get_angle_power_samples(ds_beam, angle_params)
        else:  # complex data
            # operation is identical with BB complex data
            theta, phi = get_angle_complex_samples(ds_beam, angle_params)
    # BB mode data
    else:
        if pulse_compression:  # with pulse compression
            # put receiver fs into the same dict for simplicity
            pc_params = get_filter_coeff(
                echodata["Vendor_specific"].sel(channel=source_Sv["channel"].values)
            )
            pc_params["receiver_sampling_frequency"] = source_Sv["receiver_sampling_frequency"]
            theta, phi = get_angle_complex_samples(ds_beam, angle_params, pc_params)
        else:  # without pulse compression
            # operation is identical with CW complex data
            theta, phi = get_angle_complex_samples(ds_beam, angle_params)

    # add theta and phi to source_Sv input
    theta.attrs["long_name"] = "split-beam alongship angle"
    phi.attrs["long_name"] = "split-beam athwartship angle"

    # add the split-beam angles to the provided Dataset
    source_Sv["angle_alongship"] = theta
    source_Sv["angle_athwartship"] = phi
    if to_disk:
        if source_Sv_type == "netcdf4":
            source_Sv.to_netcdf(mode="a", **storage_options)
        else:
            source_Sv.to_zarr(mode="a", **storage_options)
        source_Sv = open_source(source_Sv, "dataset", storage_options)

    # Add history attribute
    history_attr = (
        f"{datetime.datetime.utcnow()} +00:00. "
        "Calculated using data stored in the Beam groups of the echodata object."  # noqa
    )
    for da_name in ["angle_alongship", "angle_athwartship"]:
        source_Sv[da_name] = source_Sv[da_name].assign_attrs({"history": history_attr})

    return source_Sv
