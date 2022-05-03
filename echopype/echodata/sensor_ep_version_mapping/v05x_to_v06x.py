import xarray as xr

# TODO: turn this into an absolute import!
from ...core import SONAR_MODELS
from ..convention import sonarnetcdf_1


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
            ed_obj._tree[grp_path].ds = ed_obj._tree[grp_path].ds.rename(
                name_dict={"range_bin": "range_sample"}
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
    2. Adds ``beam_group`` dimension to ``Sonar`` group
    3. Adds ``sonar_serial_number``, ``beam_group_name``,
    and ``beam_group_descr`` to ``Sonar`` group.

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

    # TODO: add beam_group dimension to Sonar group
    #  add variables to sonar group
    #  use set_groups_cls


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
    varattrs = sonarnetcdf_1.yaml_dict["variable_and_varattributes"]

    for grp_path in ed_obj.group_paths:

        if "frequency" in ed_obj[grp_path]:

            # add frequency_nominal and rename frequency to channel
            ed_obj[grp_path]["frequency_nominal"] = ed_obj[grp_path].frequency
            ed_obj._tree[grp_path].ds = ed_obj[grp_path].rename({"frequency": "channel"})

            # set values for channel
            if "channel_id" in ed_obj[grp_path]:
                ed_obj[grp_path]["channel"] = ed_obj[grp_path].channel_id.values
                ed_obj._tree[grp_path].ds = ed_obj._tree[grp_path].ds.drop("channel_id")

            else:
                ed_obj[grp_path]["channel"] = channel_id.sel(
                    frequency=ed_obj[grp_path].frequency_nominal
                ).values

            # set attributes for channel
            ed_obj[grp_path]["channel"] = ed_obj[grp_path]["channel"].assign_attrs(
                varattrs["beam_coord_default"]["channel"]
            )


def convert_v05x_to_v06x(echodata_obj):
    """
    This function converts the EchoData structure created in echopype
    version 0.5.x to the EchoData structure created in echopype version
    0.6.x. Specifically, the following items are completed:

    1. Rename the coordinate ``range_bin`` to ``range_sample``
    2. Map ``Beam`` to ``Sonar/Beam_group1``
    3. Map ``Beam_power`` to ``Sonar/Beam_group2``
    4. Adds ``beam`` and ``ping_time`` dimensions to
    certain variables within the beam groups.
    5. Renames ``quadrant`` to ``beam``, sets the
    values to strings starting at 1, and sets
    attributes, if necessary.
    6. Adds ``beam_group`` dimension to ``Sonar`` group
    7. Adds ``sonar_serial_number``, ``beam_group_name``,
    and ``beam_group_descr`` to ``Sonar`` group.
    8. Renames ``frequency`` to ``channel`` and adds the
    variable ``frequency_nominal`` to every group that
    needs it.

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

        _reorganize_beam_groups(echodata_obj)

        _modify_sonar_group(echodata_obj, sensor)

        _frequency_to_channel(echodata_obj, sensor)
