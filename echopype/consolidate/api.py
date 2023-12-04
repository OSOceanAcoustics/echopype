import datetime
import pathlib
from typing import Optional, Union

import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation as R

from ..calibrate.ek80_complex import get_filter_coeff
from ..echodata import EchoData
from ..echodata.simrad import retrieve_correct_beam_group
from ..utils.io import validate_source_ds_da
from ..utils.prov import add_processing_level
from .split_beam_angle import add_angle_to_ds, get_angle_complex_samples, get_angle_power_samples

POSITION_VARIABLES = ["latitude", "longitude"]


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

    Notes
    -----
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
    echodata: Optional[EchoData] = None,
    echodata_strict: bool = False,
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
    echodata : EchoData, optional
        ``EchoData`` object from which the ``Sv` dataset originated.
        It must contain transducer position and orientation information.
    echodata_strict : bool, default=False
        If set to True and EchoData object is passed, check for invalid
        or suspect values in the source data arrays.
        CURRENTLY NOT BEING USED.
    depth_offset : float, default=0
        Offset along the vertical (depth) dimension to account for actual transducer
        position in water, since `echo_range` is counted from transducer surface.
    tilt : float, default=0
        Transducer tilt angle [degree]. 0 corresponds to a transducer pointing vertically.
    downward : bool, default=True
        The transducers point downward.

    Returns
    -------
    The input dataset with a `depth` variable (in meters) added

    Notes
    -----
    See https://echopype.readthedocs.io/en/stable/data-proc-additional.html#vertical-coordinate-z-axis-variables # noqa

    Currently this function only scalar inputs of depth_offset and tilt angle.
    In future expansion we plan to add the following options:

    * Allow inputs as xr.DataArray for time-varying variations of these variables
    * Use data stored in the EchoData object or raw-converted file from which the Sv is derived,
      specifically `water_level`, `vertical_offtset` and `tilt` in the `Platform` group.
    """

    if echodata is None:
        # depth_offset and tilt scalars passed as arguments to add_depth,
        transducer_depth = depth_offset
        echo_range_z_scaling = np.cos(np.deg2rad(tilt))
    else:
        # TODO: TODOs and notes for AZFP:
        #  - Assume tilt_x/y data has been translated into pitch & roll variables.
        #  - vertical_offset will typically not be populated. An external pressure variable is
        #    needed. Could it be added to the Environment group first, then look for it here?
        #    Other transducer position variables will be empty, too.
        #  - tilt_x/y: interpretation will depend on deployment configuration of the transducer
        #    relative to the cylinder where the inclinometer is located. Configurations are too
        #    variable to allow educated guesses

        sonar_model = echodata["Sonar"].attrs["sonar_model"]
        # For EK80 data with two Sonar/Beam_groupX groups, the Sv Dataset (ds) contains data
        # for channels from only one of the groups (depending on waveform_mode and encode_mode)
        if (
            sonar_model == "EK80"
            and "Sonar/Beam_group2" in echodata.group_paths
            and ds["channel"][0] in echodata["Sonar/Beam_group2"]
        ):
            beam_group_source = "Sonar/Beam_group2"
        else:
            beam_group_source = "Sonar/Beam_group1"

        def _z_var(z_var):
            """
            Returns a DataArray.
            If all nan, replace nan with 0 or 1
            """
            group = beam_group_source if z_var.startswith("beam_direction_") else "Platform"
            if sonar_model == "EK80" and z_var == "transducer_offset_z":
                # Platform group channel dimension includes channels from both beam groups.
                # Keep only the relevant channels.
                z_var_da = echodata[group][z_var].sel(channel=ds["channel"])
            else:
                z_var_da = echodata[group][z_var]

            # If the variable is not present, its dimensions would have to be hardwired here.
            # So, assume a very recent echopype version and that the variables exist.
            # TODO: Behavior should depend on echodata_strict.
            #  If False, return default values. If True, raise ValueError?
            #  Also, if True, treat 0 values in some source dataarrays skeptically?
            if not z_var_da.isnull().all():
                return z_var_da
            else:
                if z_var == "beam_direction_z":
                    # Returning 1 for beam_direction_z (and 0 for _x/y) will result in
                    # the z-axis unit vector, [0, 0, 1]
                    return xr.ones_like(echodata[group][z_var])
                else:
                    return xr.zeros_like(echodata[group][z_var])

        # Scalars
        water_level = _z_var("water_level")
        # With time dimension only
        vertical_offset = _z_var("vertical_offset")
        pitch = _z_var("pitch")
        roll = _z_var("roll")
        # With channel dimension only
        transducer_offset_z = _z_var("transducer_offset_z")
        beam_direction_x = _z_var("beam_direction_x")
        beam_direction_y = _z_var("beam_direction_y")
        beam_direction_z = _z_var("beam_direction_z")

        # 1. Perform z translation for transducer position vs water level
        # The dimensions of transducer_depth will be (channel x time), based on broadcasting
        # from transducer_offset_z (channel) and vertical_offset (time), in that order
        transducer_depth = transducer_offset_z - (water_level + vertical_offset)

        # Interpolate transducer_depth to ping_time
        da_time_dim_name = list(vertical_offset.dims)[0]
        transducer_depth = transducer_depth.interp(**{da_time_dim_name: ds["ping_time"]}).drop_vars(
            da_time_dim_name
        )

        # 2. Perform rotations
        # - Pitch & Roll. Set up stack of rotations, where each element (intrinsic euler angles)
        # corresponds to a pitch-roll timestep. rot_pitch_roll then has an implicit "time" dimension
        pitch_roll_euler_angles_stack = np.column_stack(
            (np.zeros_like(pitch.values), pitch.values, roll.values)
        )
        rot_pitch_roll = R.from_euler("ZYX", pitch_roll_euler_angles_stack, degrees=True)
        # - Beam direction. Set up stack of rotations, where each element (rotation matrix)
        # corresponds to a beam_direction channel. rot_beam_direction then has an implicit
        # "channel" dimension
        beam_dir_rotmatrix_stack = [
            [
                np.array([0, 0, beam_direction_x[c]]),
                np.array([0, 0, beam_direction_y[c]]),
                np.array([0, 0, beam_direction_z[c]]),
            ]
            for c in range(len(beam_direction_x))
        ]
        rot_beam_direction = R.from_matrix(beam_dir_rotmatrix_stack)

        # Total rotation
        # rot_total should have rotation elements for a channel x time 2D array
        # The two rotations will need to have aligned dimensions
        # Or use a ufunc?
        # https://stackoverflow.com/questions/71413808/understanding-xarray-apply-ufunc
        echo_range_z_scaling_bychannnel = []
        for c in range(len(rot_beam_direction)):
            # The beam direction rotation is a single rotation and pitch-roll is a
            # time-varying rotation stack.
            # rot_beam_direction[c] is applied to all elements of rot_pitch_roll
            rot_total_bychannel = rot_beam_direction[c] * rot_pitch_roll
            echo_range_z_scaling_bychannnel.append(rot_total_bychannel.as_matrix()[:, -1, -1])

        # Create echo_range_z_scaling DataArray with dimensions (channel x time)
        echo_range_z_scaling = xr.DataArray(
            np.column_stack(tuple([erz for erz in echo_range_z_scaling_bychannnel])).T,
            coords=dict(
                channel=("channel", beam_direction_z["channel"].data),
                time=("time", pitch[list(pitch.dims)[0]].data),
            ),
        )

        # Interpolate echo_range_z_scaling to ping_time
        echo_range_z_scaling = echo_range_z_scaling.interp(**{"time": ds["ping_time"]}).drop_vars(
            "time"
        )

    # TODO: Review nan and "extrapolation" handling on interpolation

    # TODO: What to do if both echodata and external data parameters
    #  (depth_offset, etc) are passed? Should specified external parameters be used
    #  instead of echodata variables?

    # Compute depth
    # Multiplication factor depending on if transducers are pointing downward
    orientation_mult = 1 if downward else -1
    # ds["echo_range"] dimensions: (channel, ping_time, range_sample)
    ds["depth"] = transducer_depth + orientation_mult * ds["echo_range"] * echo_range_z_scaling

    # Add attributes, including history attribute
    # TODO: In history_attr, specify whether the offset & angle data originated in
    #  external data or the source echodata object
    history_attr = (
        f"{datetime.datetime.utcnow()} +00:00. "
        "Added based on echo_range or other data in Sv dataset."  # noqa
    )
    ds["depth"] = ds["depth"].assign_attrs(
        {"long_name": "Depth", "standard_name": "depth", "units": "m", "history": history_attr}
    )

    return ds


@add_processing_level("L2A")
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

    if "longitude" not in echodata["Platform"] or echodata["Platform"]["longitude"].isnull().all():
        raise ValueError("Coordinate variables not present or all nan")

    interp_ds = ds.copy()
    time_dim_name = list(echodata["Platform"]["longitude"].dims)[0]
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
