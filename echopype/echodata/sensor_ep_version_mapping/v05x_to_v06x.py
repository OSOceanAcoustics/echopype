import numpy as np
import xarray as xr

# TODO: turn this into an absolute import!
from ...core import SONAR_MODELS
from ..convention import sonarnetcdf_1

_varattrs = sonarnetcdf_1.yaml_dict["variable_and_varattributes"]


def _range_bin_to_range_sample(ed_obj):
    """
    Renames the coordinate range_bin to range_sample.

    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x

    Notes
    -----
    The function directly modifies the input EchoData object.
    """

    for grp_path in ed_obj.group_paths:

        if "range_bin" in list(ed_obj[grp_path].coords):

            # renames range_bin in the dataset
            ed_obj[grp_path] = ed_obj[grp_path].rename(name_dict={"range_bin": "range_sample"})

            ed_obj[grp_path].range_sample.attrs["long_name"] = "Along-range sample number, base 0"


def _add_attrs_to_freq(ed_obj):
    """
    Makes the attributes of the ``frequency`` variable
    consistent for all groups. This is necessary because
    not all groups have the same attributes (some are
    missing them too) for the ``frequency`` variable.
    This variable is used to set the variable
    ``frequency_nominal`` later on.

    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x

    Notes
    -----
    The function directly modifies the input EchoData object.
    """

    for grp_path in ed_obj.group_paths:

        if "frequency" in list(ed_obj[grp_path].coords):

            # creates consistent frequency attributes
            # NOTE: the attribute standard_name will be removed in `frequency_nominal`
            ed_obj[grp_path]["frequency"] = ed_obj[grp_path].frequency.assign_attrs(
                {
                    "long_name": "Transducer frequency",
                    "standard_name": "sound_frequency",
                    "units": "Hz",
                    "valid_min": 0.0,
                }
            )


def _reorganize_beam_groups(ed_obj):
    """
    Maps Beam --> Sonar/Beam_group1 and Beam_power --> Sonar/Beam_group2.

    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x

    Notes
    -----
    The function directly modifies the input EchoData object.
    """

    # Map Beam --> Sonar/Beam_group1
    if "Beam" in ed_obj.group_paths:
        ed_obj._tree["Sonar"].add_child(ed_obj._tree["Beam"])
        ed_obj._tree["Sonar/Beam"].name = "Beam_group1"

    # Map Beam_power --> Sonar/Beam_group2
    if "Beam_power" in ed_obj.group_paths:
        ed_obj._tree["Sonar"].add_child(ed_obj._tree["Beam_power"])
        ed_obj._tree["Sonar/Beam_power"].name = "Beam_group2"


def get_channel_id(ed_obj, sensor):
    """
    Returns the channel_id for all non-unique frequencies.

    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x.
    sensor : str
        The sensor used to create the v0.5.x file.

    Returns
    -------
    A datarray specifying the channel_ids with dimension frequency.
    """
    if sensor == "AZFP":

        # create frequency_nominal variable
        freq_nom = ed_obj["Sonar/Beam_group1"].frequency

        # create unique channel_id for AZFP
        freq_as_str = (freq_nom / 1000.0).astype(int).astype(str).values
        channel_id_str = [
            str(ed_obj["Sonar"].sonar_serial_number) + "-" + freq_as_str[i] + "-" + str(i + 1)
            for i in range(len(freq_as_str))
        ]
        channel_id = xr.DataArray(
            data=channel_id_str, dims=["frequency"], coords={"frequency": freq_nom}
        )

    else:

        channel_id = xr.concat(
            [child.ds.channel_id for child in ed_obj._tree["Sonar"].children], dim="frequency"
        )

    return channel_id


def _frequency_to_channel(ed_obj, sensor):
    """
    1. In all groups that it appears, changes the dimension
    ``frequency`` to ``channel`` whose values are based on
    ``channel_id`` for EK60/EK80 and are a custom string
    for AZFP.
    2. Removes channel_id if it appears as a variable
    3. Adds the variable ``frequency_nominal`` to all
    Datasets that have dimension ``channel``

    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x.
    sensor : str
        The sensor used to create the v0.5.x file.

    Notes
    -----
    The function directly modifies the input EchoData object.
    """

    channel_id = get_channel_id(ed_obj, sensor)  # all channel ids

    for grp_path in ed_obj.group_paths:

        if "frequency" in ed_obj[grp_path]:

            # add frequency_nominal, remove standard_name attribute, and
            # rename frequency to channel
            ed_obj[grp_path]["frequency_nominal"] = ed_obj[grp_path].frequency

            if "standard_name" in ed_obj[grp_path].frequency_nominal.attrs:
                del ed_obj[grp_path].frequency_nominal.attrs["standard_name"]
            ed_obj[grp_path] = ed_obj[grp_path].rename({"frequency": "channel"})

            # set values for channel
            if "channel_id" in ed_obj[grp_path]:
                print(f"here = {grp_path}")
                ed_obj[grp_path]["channel"] = ed_obj[grp_path].channel_id.values
                ed_obj[grp_path] = ed_obj._tree[grp_path].ds.drop("channel_id")

            else:
                ed_obj[grp_path]["channel"] = channel_id.sel(
                    frequency=ed_obj[grp_path].frequency_nominal
                ).values

            # set attributes for channel
            ed_obj[grp_path]["channel"] = ed_obj[grp_path]["channel"].assign_attrs(
                _varattrs["beam_coord_default"]["channel"]
            )


def _change_beam_var_names(ed_obj, sensor):
    """
    For EK60 ``Beam_group1``
    1. Rename ``beamwidth_receive_alongship`` to
    ``beamwidth_twoway_alongship`` and change the attribute
    ``long_name``
    2. Rename ``beamwidth_transmit_athwartship`` to
    ``beamwidth_twoway_athwartship`` and change the attribute
    ``long_name``
    3. Remove the variables ``beamwidth_receive_athwartship``
    and ``beamwidth_transmit_alongship``
    4. Change the attribute ``long_name`` in the variables
    ``angle_offsetalongship/athwartship`` and
    ``angle_sensitivity_alongship/athwartship``

    For EK80 ``Beam_group1``
    1. Change the attribute ``long_name`` in the variables
    ``angle_offset_alongship/athwartship``

    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x.
    sensor : str
        Variable specifying the sensor that created the file.

    Notes
    -----
    The function directly modifies the input EchoData object.
    """

    if sensor == "EK60":

        ed_obj["Sonar/Beam_group1"] = ed_obj["Sonar/Beam_group1"].rename(
            {"beamwidth_receive_alongship": "beamwidth_twoway_alongship"}
        )

        ed_obj["Sonar/Beam_group1"].beamwidth_twoway_alongship.attrs[
            "long_name"
        ] = "Half power two-way beam width along alongship axis of beam"

        ed_obj["Sonar/Beam_group1"] = ed_obj["Sonar/Beam_group1"].rename(
            {"beamwidth_transmit_athwartship": "beamwidth_twoway_athwartship"}
        )

        ed_obj["Sonar/Beam_group1"].beamwidth_twoway_athwartship.attrs[
            "long_name"
        ] = "Half power two-way beam width along athwartship axis of beam"

        ed_obj["Sonar/Beam_group1"] = ed_obj["Sonar/Beam_group1"].drop(
            ["beamwidth_receive_athwartship", "beamwidth_transmit_alongship"]
        )

        ed_obj["Sonar/Beam_group1"].angle_sensitivity_alongship.attrs[
            "long_name"
        ] = "alongship angle sensitivity of the transducer"

        ed_obj["Sonar/Beam_group1"].angle_sensitivity_athwartship.attrs[
            "long_name"
        ] = "athwartship angle sensitivity of the transducer"

    if sensor in ["EK60", "EK80"]:

        for beam_group in ed_obj._tree["Sonar"].children:

            beam_group.ds.angle_offset_alongship.attrs[
                "long_name"
            ] = "electrical alongship angle offset of the transducer"

            beam_group.ds.angle_offset_athwartship.attrs[
                "long_name"
            ] = "electrical athwartship angle offset of the transducer"


def _add_comment_to_beam_vars(ed_obj, sensor):
    """
    For EK60 and EK80
    Add the ``comment`` attribute to the variables
    ``beamwidth_twoway_alongship/athwartship``,
    ``angle_offset_alongship/athwartship``,
    ``angle_sensitivity_alongship/athwartship``,
    ``angle_athwartship/alongship``,
    ``beamwidth_twoway_alongship/athwartship``,
    ``angle_athwartship/alongship``

    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x.
    sensor : str
        Variable specifying the sensor that created the file.

    Notes
    -----
    The function directly modifies the input EchoData object.

    """

    if sensor in ["EK60", "EK80"]:

        for beam_group in ed_obj._tree["Sonar"].children:
            beam_group.ds.beamwidth_twoway_alongship.attrs["comment"] = (
                "Introduced in echopype for Simrad echosounders to avoid "
                "potential confusion with convention definitions. The alongship "
                "angle corresponds to the minor angle in SONAR-netCDF4 vers 2. The "
                "convention defines one-way transmit or receive beamwidth "
                "(beamwidth_receive_minor and beamwidth_transmit_minor), but Simrad "
                "echosounders record two-way beamwidth in the data."
            )

            beam_group.ds.beamwidth_twoway_athwartship.attrs["comment"] = (
                "Introduced in echopype for Simrad echosounders to avoid "
                "potential confusion with convention definitions. The athwartship "
                "angle corresponds to the major angle in SONAR-netCDF4 vers 2. The "
                "convention defines one-way transmit or receive beamwidth "
                "(beamwidth_receive_major and beamwidth_transmit_major), but Simrad "
                "echosounders record two-way beamwidth in the data."
            )

            beam_group.ds.angle_offset_alongship.attrs["comment"] = (
                "Introduced in echopype for Simrad echosounders. The alongship "
                "angle corresponds to the minor angle in SONAR-netCDF4 vers 2. "
            )

            beam_group.ds.angle_offset_athwartship.attrs["comment"] = (
                "Introduced in echopype for Simrad echosounders. The athwartship "
                "angle corresponds to the major angle in SONAR-netCDF4 vers 2. "
            )

            beam_group.ds.angle_sensitivity_alongship.attrs[
                "comment"
            ] = beam_group.ds.angle_offset_alongship.attrs["comment"]

            beam_group.ds.angle_sensitivity_athwartship.attrs[
                "comment"
            ] = beam_group.ds.angle_offset_athwartship.attrs["comment"]

            if "angle_alongship" in beam_group.ds:
                beam_group.ds.angle_alongship.attrs[
                    "comment"
                ] = beam_group.ds.angle_offset_alongship.attrs["comment"]

            if "angle_athwartship" in beam_group.ds:
                beam_group.ds.angle_athwartship.attrs[
                    "comment"
                ] = beam_group.ds.angle_offset_athwartship.attrs["comment"]


def _beam_groups_to_convention(ed_obj, set_grp_cls):
    """
    Adds ``beam`` and ``ping_time`` dimensions to variables
    in ``Beam_groupX`` so that they comply with the convention.
    Additionally, it will change the ``quadrant`` dimension to
    ``beam`` with string values starting at 1 and set its
    attributes.

    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x
    set_grp_cls : SetGroupsBase object
        The set groups class of the sensor being considered

    Notes
    -----
    The function directly modifies the input EchoData object.
    """

    for beam_group in ed_obj._tree["Sonar"].children:

        if "quadrant" in beam_group.ds:

            # change quadrant to beam, assign its values
            # to a string starting at 1, and set attributes
            beam_group.ds = beam_group.ds.rename({"quadrant": "beam"})
            beam_group.ds["beam"] = (beam_group.ds.beam + 1).astype(str)
            beam_group.ds.beam.attrs["long_name"] = "Beam name"

        set_grp_cls.beamgroups_to_convention(
            set_grp_cls,
            beam_group.ds,
            set_grp_cls.beam_only_names,
            set_grp_cls.beam_ping_time_names,
            set_grp_cls.ping_time_only_names,
        )


def _modify_sonar_group(ed_obj, sensor):
    """
    1. Renames ``quadrant`` to ``beam``, sets the
    values to strings starting at 1, and sets
    attributes, if necessary.
    2. Adds ``beam_group`` coordinate to ``Sonar`` group
    for all sensors
    3. Adds the variable ``beam_group_descr`` to the
    ``Sonar`` group for all sensors
    4. Adds the variable ``sonar_serial_number`` to the
    ``Sonar`` group and fills it with NaNs (it is missing
    information) for the EK80 sensor only.


    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x.
    sensor : str
        The sensor used to create the v0.5.x file.

    Notes
    -----
    The function directly modifies the input EchoData object.
    """

    set_groups_cls = SONAR_MODELS[sensor]["set_groups"]

    _beam_groups_to_convention(ed_obj, set_groups_cls)

    # add beam_group coordinate and beam_group_descr variable
    num_beams = len(ed_obj._tree["Sonar"].children)
    set_groups_cls._beamgroups = set_groups_cls.beamgroups_possible[:num_beams]
    beam_groups_vars, beam_groups_coord = set_groups_cls._beam_groups_vars(set_groups_cls)

    ed_obj["Sonar"] = ed_obj["Sonar"].assign_coords(beam_groups_coord)
    ed_obj["Sonar"] = ed_obj["Sonar"].assign(**beam_groups_vars)

    # add sonar_serial_number to EK80 Sonar group
    if sensor == "EK80":

        ed_obj["Sonar"] = ed_obj["Sonar"].assign(
            {
                "sonar_serial_number": (
                    ["channel"],
                    np.full_like(ed_obj["Sonar"].frequency_nominal.values, np.nan),
                )
            }
        )

    # TODO: Do we need to make attributes into variables
    #  for EK60 and AZFP #681?


def _move_transducer_offset_vars(ed_obj, sensor):
    """
    Moves transducer_offset_x/y/z from beam groups to Platform
    group for EK60 and EK80. If more than one beam group exists,
    then the variables are first collected and then moved to
    Platform. Additionally, adds ``frequency_nominal`` to
    Platform for the EK80 sensor.

    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x.
    sensor : str
        Variable specifying the sensor that created the file.

    Notes
    -----
    The function directly modifies the input EchoData object.
    """

    if sensor in ["EK60", "EK80"]:
        full_transducer_vars = {"x": [], "y": [], "z": []}

        # collect transducser_offset_x/y/z from the beam groups
        for beam_group in ed_obj._tree["Sonar"].children:
            for spatial in full_transducer_vars.keys():
                full_transducer_vars[spatial].append(beam_group.ds["transducer_offset_" + spatial])

                # remove transducer_offset_x/y/z from the beam group
                beam_group.ds = beam_group.ds.drop("transducer_offset_" + spatial)

        # transfer transducser_offset_x/y/z to Platform
        for spatial in full_transducer_vars.keys():
            ed_obj["Platform"]["transducer_offset_" + spatial] = xr.concat(
                full_transducer_vars[spatial], dim="channel"
            )

    if sensor == "EK80":
        ed_obj["Platform"]["frequency_nominal"] = ed_obj["Vendor"].frequency_nominal.sel(
            channel=ed_obj["Platform"].channel
        )


def _add_vars_to_platform(ed_obj, sensor):
    """
    1.Adds ``MRU_offset_x/y/z``, ``MRU_rotation_x/y/z``, and
    ``position_offset_x/y/z`` to the ``Platform`` group
    for the EK60/EK80/AZFP sensors.
    2. Renames ``heave`` to ``vertical_offset`` for the EK60
    and EK80.
    3. Adds ``transducer_offset_x/y/z``, ``vertical_offset``,
    ``water_level`` to the ``Platform`` group for the AZFP
    sensor only.
    4. Adds the coordinate ``time3`` to the ``Platform`` group
    for the EK80 sensor only.
    5. Adds the variables ``drop_keel_offset(time3)`` (currently in
    the attribute), ``drop_keel_offset_is_manual(time3)``, and
    ``water_level_draft_is_manual(time3)`` to the ``Platform``
    group for the EK80 sensor only.
    6. Adds the coordinate ``time3`` to the variable ``water_level``
    in the ``Platform`` group for the EK80 sensor only.

    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x.
    sensor : str
        Variable specifying the sensor that created the file.

    Notes
    -----
    The function directly modifies the input EchoData object.
    """

    ds_tmp = xr.Dataset(
        {
            var: ([], np.nan, _varattrs["platform_var_default"][var])
            for var in [
                "MRU_offset_x",
                "MRU_offset_y",
                "MRU_offset_z",
                "MRU_rotation_x",
                "MRU_rotation_y",
                "MRU_rotation_z",
                "position_offset_x",
                "position_offset_y",
                "position_offset_z",
            ]
        }
    )

    if sensor == "EK60":
        ds_tmp = ds_tmp.expand_dims({"channel": ed_obj["Platform"].channel})
        ds_tmp["channel"] = ds_tmp["channel"].assign_attrs(
            _varattrs["beam_coord_default"]["channel"]
        )

    ed_obj["Platform"] = xr.merge([ed_obj["Platform"], ds_tmp])

    if sensor != "AZFP":
        ed_obj["Platform"] = ed_obj["Platform"].rename({"heave": "vertical_offset"})

    if sensor == "EK80":

        ed_obj["Platform"]["drop_keel_offset"] = xr.DataArray(
            data=[ed_obj["Platform"].attrs["drop_keel_offset"]], dims=["time3"]
        )

        del ed_obj["Platform"].attrs["drop_keel_offset"]

        ed_obj["Platform"]["drop_keel_offset_is_manual"] = xr.DataArray(
            data=[np.nan], dims=["time3"]
        )

        ed_obj["Platform"]["water_level_draft_is_manual"] = xr.DataArray(
            data=[np.nan], dims=["time3"]
        )

        ed_obj["Platform"]["water_level"] = ed_obj["Platform"]["water_level"].expand_dims(
            dim=["time3"]
        )

        ed_obj["Platform"] = ed_obj["Platform"].assign_coords(
            {
                "time3": (
                    ["time3"],
                    ed_obj["Environment"].ping_time.values,
                    {
                        "axis": "T",
                        "long_name": "Timestamps for Environment XML datagrams",
                        "standard_name": "time",
                    },
                )
            }
        )

    if sensor == "AZFP":
        ds_tmp = xr.Dataset(
            {
                var: ([], np.nan, _varattrs["platform_var_default"][var])
                for var in [
                    "transducer_offset_x",
                    "transducer_offset_y",
                    "transducer_offset_z",
                    "vertical_offset",
                    "water_level",
                ]
            }
        )

        ed_obj["Platform"] = xr.merge([ed_obj["Platform"], ds_tmp])


def _add_vars_coords_to_environment(ed_obj, sensor):
    """
    For EK80
    1. Adds the length one NaN coordinate ``sound_velocity_profile_depth``
    to the ``Environment`` group (this data is missing in v0.5.x).
    2. Adds the variables
    ``sound_velocity_profile(time1, sound_velocity_profile_depth)``,
    ``sound_velocity_source(time1)``, ``transducer_name(time1)``,
    ``transducer_sound_speed(time1) to the ``Environment`` group.

    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x.
    sensor : str
        Variable specifying the sensor that created the file.

    Notes
    -----
    The function directly modifies the input EchoData object.
    """

    if sensor == "EK80":

        ed_obj["Environment"]["sound_velocity_source"] = (
            ["ping_time"],
            np.array(len(ed_obj["Environment"].ping_time) * ["None"]),
        )

        ed_obj["Environment"]["transducer_name"] = (
            ["ping_time"],
            np.array(len(ed_obj["Environment"].ping_time) * ["None"]),
        )

        ed_obj["Environment"]["transducer_sound_speed"] = (
            ["ping_time"],
            np.array(len(ed_obj["Environment"].ping_time) * [np.nan]),
        )

        ed_obj["Environment"]["sound_velocity_profile"] = (
            ["ping_time", "sound_velocity_profile_depth"],
            np.nan * np.ones((len(ed_obj["Environment"].ping_time), 1)),
            {
                "long_name": "sound velocity profile",
                "standard_name": "speed_of_sound_in_sea_water",
                "units": "m/s",
                "valid_min": 0.0,
                "comment": "parsed from raw data files as (depth, sound_speed) value pairs",
            },
        )

        ed_obj["Environment"] = ed_obj["Environment"].assign_coords(
            {
                "sound_velocity_profile_depth": (
                    ["sound_velocity_profile_depth"],
                    [np.nan],
                    {
                        "standard_name": "depth",
                        "units": "m",
                        "axis": "Z",
                        "positive": "down",
                        "valid_min": 0.0,
                    },
                )
            }
        )


def _rearrange_azfp_attrs_vars(ed_obj, sensor):
    """
    Makes alterations to AZFP variables. Specifically,
    variables in ``Beam_group1``.
    1. Moves ``tilt_x/y(ping_time)`` to the `Platform` group.
    2. Moves ``temperature_counts(ping_time)``,
    ``tilt_x/y_count(ping_time)``, ``DS(channel)``, ``EL(channel)``,
    ``TVR(channel)``, ``VTX(channel)``, ``Sv_offset(channel)``,
    ``number_of_samples_digitized_per_pings(channel)``,
    ``number_of_digitized_samples_averaged_per_pings(channel)``
    to the `Vendor` group:
    3. Removes the variable `cos_tilt_mag(ping_time)`
    4. Moves the following attributes to the ``Vendor`` group:
    ``tilt_X_a/b/c/d``, ``tilt_Y_a/b/c/d``,
    ``temperature_ka/kb/kc/A/B/C``, ``number_of_frequency``,
    ``number_of_pings_per_burst``, ``average_burst_pings_flag``

    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x.
    sensor : str
        Variable specifying the sensor that created the file.

    Notes
    -----
    The function directly modifies the input EchoData object.
    """

    if sensor == "AZFP":

        beam_to_plat_vars = ["tilt_x", "tilt_y"]

        for var_name in beam_to_plat_vars:
            ed_obj["Platform"][var_name] = ed_obj["Sonar/Beam_group1"][var_name]

        beam_to_vendor_vars = [
            "temperature_counts",
            "tilt_x_count",
            "tilt_y_count",
            "DS",
            "EL",
            "TVR",
            "VTX",
            "Sv_offset",
            "number_of_samples_digitized_per_pings",
            "number_of_digitized_samples_averaged_per_pings",
        ]

        for var_name in beam_to_vendor_vars:
            ed_obj["Vendor"][var_name] = ed_obj["Sonar/Beam_group1"][var_name]

        beam_to_vendor_attrs = ed_obj["Sonar/Beam_group1"].attrs.copy()
        del beam_to_vendor_attrs["beam_mode"]
        del beam_to_vendor_attrs["conversion_equation_t"]

        for key, val in beam_to_vendor_attrs.items():

            ed_obj["Vendor"].attrs[key] = val
            del ed_obj["Sonar/Beam_group1"].attrs[key]

        ed_obj["Sonar/Beam_group1"] = ed_obj["Sonar/Beam_group1"].drop(
            ["cos_tilt_mag"] + beam_to_plat_vars + beam_to_vendor_vars
        )


def _rename_mru_time_location_time(ed_obj):
    """
    Renames location_time to time1 and mru_time to
    time2 wherever it occurs.

    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x.

    Notes
    -----
    The function directly modifies the input EchoData object.
    """

    for grp_path in ed_obj.group_paths:

        if "location_time" in list(ed_obj[grp_path].coords):

            # renames location_time to time1
            ed_obj[grp_path] = ed_obj[grp_path].rename(name_dict={"location_time": "time1"})

        if "mru_time" in list(ed_obj[grp_path].coords):

            # renames mru_time to time2
            ed_obj[grp_path] = ed_obj[grp_path].rename(name_dict={"mru_time": "time2"})


def _rename_and_add_time_vars_ek60(ed_obj):
    """
    1. For EK60's ``Platform`` group this function adds
    the variable time3, renames the variable ``water_level``
    time coordinate to time3, and changes ``ping_time`` to
    ``time2`` for the variables ``pitch/roll/vertical_offset``.
    2. For EK60's ``Envrionment`` group this function renames
    ``ping_time`` to ``time1``.

    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x.

    Notes
    -----
    The function directly modifies the input EchoData object.
    """

    ed_obj["Platform"]["water_level"] = ed_obj["Platform"]["water_level"].rename(
        {"ping_time": "time3"}
    )
    ed_obj["Platform"] = ed_obj["Platform"].rename({"ping_time": "time2"})

    ed_obj["Environment"] = ed_obj["Environment"].rename({"ping_time": "time1"})

    ed_obj["Platform"] = ed_obj["Platform"].assign_coords(
        {
            "time3": (
                ["time3"],
                ed_obj["Platform"].time3.values,
                {
                    "axis": "T",
                    "long_name": "Timestamps for position datagrams",
                    "standard_name": "time",
                },
            )
        }
    )


def _add_time_comment_in_platform(ed_obj, sensor):
    """
    Adds comments to ``time1``, ``time2``, and
    ``time3`` in the ``Platform`` group.

    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x.
    sensor : str
        Variable specifying the sensor that created the file.

    Notes
    -----
    The function directly modifies the input EchoData object.
    """

    if "time1" in ed_obj["Platform"]:
        ed_obj["Platform"].time1.attrs["comment"] = "Time coordinate corresponding to GPS location."

    ed_obj["Platform"].time2.attrs["comment"] = "Time coordinate corresponding to platform sensors."

    if sensor == "EK80":
        ed_obj["Platform"].time3.attrs["comment"] = (
            "Time coordinate corresponding to "
            "environmental variables. Note that "
            "Platform.time3 is the same as Environment.time1."
        )
    else:
        if "time3" in ed_obj["Platform"]:
            ed_obj["Platform"].time3.attrs[
                "comment"
            ] = "Time coordinate corresponding to environmental variables."


def _add_time_comment_in_environment(ed_obj, sensor):
    """
    Adds comments to ``time1`` in the ``Environment`` group.

    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x.
    sensor : str
        Variable specifying the sensor that created the file.

    Notes
    -----
    The function directly modifies the input EchoData object.

    """

    if sensor == "EK80":
        ed_obj["Environment"].time1.attrs["comment"] = (
            "Time coordinate corresponding to "
            "environmental variables. Note that "
            "Platform.time3 is the same as Environment.time1"
        )
    else:
        ed_obj["Environment"].time1.attrs[
            "comment"
        ] = "Time coordinate corresponding to environmental variables."


def _make_time_coords_consistent(ed_obj, sensor):
    """
    1. Renames location_time to time1 and mru_time to
    time2 wherever it occurs.
    2. For EK60 adds and modifies the ``Platform`` group's
    time variables.
    3. For EK80 renames ``ping_time`` to ``time1`` in the
    ``Environment`` group.
    4. For AZFP renames ``ping_time`` to ``time2`` in the
    ``Platform`` group and ``ping_time`` to ``time1`` in the
    ``Environment`` group.
    5. Adds time comments to the ``Platform``, ``Platform/NMEA``,
    and ``Environment`` groups.
    6. TODO: add descr


    Parameters
    ----------
    ed_obj : EchoData
        EchoData object that was created using echopype version 0.5.x.
    sensor : str
        Variable specifying the sensor that created the file.

    Notes
    -----
    The function directly modifies the input EchoData object.
    """

    _rename_mru_time_location_time(ed_obj)

    if sensor == "EK60":
        _rename_and_add_time_vars_ek60(ed_obj)

    if sensor == "EK80":
        ed_obj["Environment"] = ed_obj["Environment"].rename({"ping_time": "time1"})

    if sensor == "AZFP":
        ed_obj["Platform"] = ed_obj["Platform"].rename({"ping_time": "time2"})
        ed_obj["Environment"] = ed_obj["Environment"].rename({"ping_time": "time1"})

    _add_time_comment_in_platform(ed_obj, sensor)
    _add_time_comment_in_environment(ed_obj, sensor)

    if "Platform/NMEA" in ed_obj.group_paths:
        ed_obj["Platform/NMEA"].time1.attrs[
            "comment"
        ] = "Time coordinate corresponding to GPS location."

    # TODO: modify Platform/NMEA time coordinates


def convert_v05x_to_v06x(echodata_obj):
    """
    This function converts the EchoData structure created in echopype
    version 0.5.x to the EchoData structure created in echopype version
    0.6.x. Specifically, the following items are completed:

    1. Rename the coordinate ``range_bin`` to ``range_sample``
    2. Add attributes to `frequency` dimension throughout
    all sensors.
    3. Map ``Beam`` to ``Sonar/Beam_group1``
    4. Map ``Beam_power`` to ``Sonar/Beam_group2``
    5. Adds ``beam`` and ``ping_time`` dimensions to
    certain variables within the beam groups.
    6. Renames ``quadrant`` to ``beam``, sets the
    values to strings starting at 1, and sets
    attributes, if necessary.
    7. Add comment attribute to all _alongship/_athwartship
    variables and use two-way beamwidth variables.
    8. Adds ``beam_group`` dimension to ``Sonar`` group
    9. Adds ``sonar_serial_number``, ``beam_group_name``,
    and ``beam_group_descr`` to ``Sonar`` group.
    10. Renames ``frequency`` to ``channel`` and adds the
    variable ``frequency_nominal`` to every group that
    needs it.
    11. Move ``transducer_offset_x/y/z`` from beam groups
    to the ``Platform`` group (for EK60 and EK80 only).
    12. Add variables to the `Platform` group and rename
    ``heave`` to ``vertical_offset`` (if necessary).
    13. Add variables and coordinate to the ``Environment``
    group for EK80 only.
    14. Move AZFP attributes and variables from ``Beam_group1``
    to the ``Vendor`` and ``Platform`` groups. Additionally,
    remove the variable ``cos_tilt_mag``, if it exists.
    15. Make the names of the time coordinates in the `Platform`
    and `Environment` groups consistent and add new the attribute
    comment to these time coordinates.

    Parameters
    ----------
    echodata_obj : EchoData
        EchoData object that was created using echopype version 0.5.x

    Notes
    -----
    The function directly modifies the input EchoData object.
    No actions are taken for AD2CP.
    """

    # TODO: make this a warning that links to the documentation
    print(
        "Converting echopype version 0.5.x file to 0.6.x."
        + " For specific details on how items have been changed,"
        + " please see ... . It is recommended that one creates "
        + "the file using open_raw again, rather than relying on this conversion"
    )

    # get the sensor used to create the v0.5.x file.
    sensor = echodata_obj["Top-level"].keywords

    if sensor == "AD2CP":
        pass
    else:
        _range_bin_to_range_sample(echodata_obj)

        _add_attrs_to_freq(echodata_obj)

        _reorganize_beam_groups(echodata_obj)

        _frequency_to_channel(echodata_obj, sensor)

        _change_beam_var_names(echodata_obj, sensor)

        _add_comment_to_beam_vars(echodata_obj, sensor)

        _modify_sonar_group(echodata_obj, sensor)

        _move_transducer_offset_vars(echodata_obj, sensor)

        _add_vars_to_platform(echodata_obj, sensor)

        _add_vars_coords_to_environment(echodata_obj, sensor)

        _rearrange_azfp_attrs_vars(echodata_obj, sensor)

        _make_time_coords_consistent(echodata_obj, sensor)

        # Change src_filenames string attribute to source_filenames
        # list-of-strings variable in Platform (#620, #621)

        # addition of missing required variables in Platform
        # groups (#592, #649)

        # Rename `Vendor` group to `Vendor_specific` (Issue #675)
