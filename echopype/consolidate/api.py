import datetime
import pathlib
from typing import Optional, Union

import numpy as np
import xarray as xr

from ..calibrate.ek80_complex import get_filter_coeff
from ..echodata import EchoData
from ..echodata.simrad import retrieve_correct_beam_group
from ..utils.io import validate_source_ds_da
from .split_beam_angle import add_angle_to_ds, get_angle_complex_samples, get_angle_power_samples


def swap_dims_channel_frequency(ds: xr.Dataset) -> xr.Dataset:
    """
    Use frequency_nominal in place of channel to be dataset dimension and coorindate.

    This is useful because the nominal transducer frequencies are commonly used to
    refer to data collected from a specific transducer.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset for which the dimension will be swapped

    Returns
    -------
    The input dataset with the dimension swapped

    Note
    ----
    This operation is only possible when there are no duplicated frequencies present in the file.
    """
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
    ds: xr.Dataset,
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
    ds : xr.Dataset
        Source Sv dataset to which a depth variable will be added.
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


def add_location(ds: xr.Dataset, echodata: EchoData = None, nmea_sentence: Optional[str] = None):
    """
    Add geographical location (latitude/longitude) to the Sv dataset.

    This function interpolates the location from the Platform group in the original data file
    based on the time when the latitude/longitude data are recorded and the time the acoustic
    data are recorded (`ping_time`).

    Parameters
    ----------
    ds : xr.Dataset
        An Sv or MVBS dataset for which the geographical locations will be added to
    echodata
        An `EchoData` object holding the raw data
    nmea_sentence
        NMEA sentence to select a subset of location data (optional)

    Returns
    -------
    The input dataset with the the location data added
    """

    def sel_interp(var):
        # NMEA sentence selection
        if nmea_sentence:
            coord_var = echodata["Platform"][var][
                echodata["Platform"]["sentence_type"] == nmea_sentence
            ]
        else:
            coord_var = echodata["Platform"][var]

        if len(coord_var) == 1:
            # Propagate single, fixed-location coordinate
            return xr.DataArray(
                data=coord_var.values[0] * np.ones(len(ds["ping_time"]), dtype=np.float64),
                dims=["ping_time"],
                attrs=coord_var.attrs,
            )
        else:
            # Interpolation. time1 is always associated with location data
            return coord_var.interp(time1=ds["ping_time"])

    if "longitude" not in echodata["Platform"] or echodata["Platform"]["longitude"].isnull().all():
        raise ValueError("Coordinate variables not present or all nan")

    interp_ds = ds.copy()
    interp_ds["latitude"] = sel_interp("latitude")
    interp_ds["longitude"] = sel_interp("longitude")
    # Most attributes are attached automatically via interpolation
    # here we add the history
    history_attr = (
        f"{datetime.datetime.utcnow()} +00:00. "
        "Interpolated or propagated from Platform latitude/longitude."  # noqa
    )
    for da_name in ["latitude", "longitude"]:
        interp_ds[da_name] = interp_ds[da_name].assign_attrs({"history": history_attr})

    return interp_ds.drop_vars("time1")


def add_splitbeam_angle(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    echodata: EchoData,
    waveform_mode: str,
    encode_mode: str,
    pulse_compression: bool = False,
    storage_options: dict = {},
    return_dataset: bool = True,
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
    echodata: EchoData
        An ``EchoData`` object holding the raw data
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
    return_dataset: bool, default=True
        If ``True``, ``source_Sv`` with split-beam angles added will be returned.
        ``return_dataset=False`` is useful when ``source_Sv`` is a path and
        users only want to write the split-beam angle data to this path.

    Returns
    -------
    xr.Dataset or None
        If ``return_dataset=False``, nothing will be returned.
        If ``return_dataset=True``, either the input dataset ``source_Sv``
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

    # ensure that echodata was produced by EK60 or EK80-like sensors
    if echodata.sonar_model not in ["EK60", "ES70", "EK80", "ES80", "EA640"]:
        raise ValueError(
            "The sonar model that produced echodata does not have split-beam "
            "transducers, split-beam angles cannot be added to source_Sv!"
        )

    # validate the source_Sv type or path (if it is provided)
    source_Sv, file_type = validate_source_ds_da(source_Sv, storage_options)

    # initialize source_Sv_path
    source_Sv_path = None

    if isinstance(source_Sv, str):
        # store source_Sv path so we can use it to write to later
        source_Sv_path = source_Sv

        # TODO: In the future we can improve this by obtaining the variable names, channels,
        #  and dimension lengths directly from source_Sv using zarr or netcdf4. This would
        #  prevent the unnecessary loading in of the coordinates, which the below statement does.
        # open up Dataset using source_Sv path
        source_Sv = xr.open_dataset(source_Sv, engine=file_type, chunks={}, **storage_options)

    # raise not implemented error if source_Sv corresponds to MVBS
    if source_Sv.attrs["processing_function"] == "preprocess.compute_MVBS":
        raise NotImplementedError("Adding split-beam data to MVBS has not been implemented!")

    # check that the appropriate waveform and encode mode have been given
    # and obtain the echodata group path corresponding to encode_mode
    encode_mode_ed_group = retrieve_correct_beam_group(echodata, waveform_mode, encode_mode)

    # check that source_Sv at least has a channel dimension
    if "channel" not in source_Sv.variables:
        raise ValueError("The input source_Sv Dataset must have a channel dimension!")

    # Select ds_beam channels from source_Sv
    ds_beam = echodata[encode_mode_ed_group].sel(channel=source_Sv["channel"].values)

    # Assemble angle param dict
    angle_param_list = [
        "angle_sensitivity_alongship",
        "angle_sensitivity_athwartship",
        "angle_offset_alongship",
        "angle_offset_athwartship",
    ]
    angle_params = {}
    for p_name in angle_param_list:
        angle_params[p_name] = source_Sv[p_name]

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
    source_Sv = add_angle_to_ds(
        theta, phi, source_Sv, return_dataset, source_Sv_path, file_type, storage_options
    )

    # Add history attribute
    history_attr = (
        f"{datetime.datetime.utcnow()} +00:00. "
        "Calculated using data stored in the Beam groups of the echodata object."  # noqa
    )
    for da_name in ["angle_alongship", "angle_athwartship"]:
        source_Sv[da_name] = source_Sv[da_name].assign_attrs({"history": history_attr})

    return source_Sv
