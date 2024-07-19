import itertools
import re
from collections import ChainMap
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from warnings import warn

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from datatree import DataTree

from ..utils.io import validate_output_path
from ..utils.log import _init_logger
from ..utils.prov import echopype_prov_attrs
from .echodata import EchoData

logger = _init_logger(__name__)

POSSIBLE_TIME_DIMS = {"time1", "time2", "time3", "time4", "ping_time"}
APPEND_DIMS = {"filenames"}.union(POSSIBLE_TIME_DIMS)
DATE_CREATED_ATTR = "date_created"
CONVERSION_TIME_ATTR = "conversion_time"
ED_GROUP = "echodata_group"
ED_FILENAME = "echodata_filename"
FILENAMES = "filenames"


def check_zarr_path(
    zarr_path: Union[str, Path], storage_options: Dict[str, Any] = {}, overwrite: bool = False
) -> str:
    """
    Checks that the zarr path provided to ``combine``
    is valid.

    Parameters
    ----------
    zarr_path: str or Path
        The full save path to the final combined zarr store
    storage_options: dict
        Any additional parameters for the storage
        backend (ignored for local paths)
    overwrite: bool
        If True, will overwrite the zarr store specified by
        ``zarr_path`` if it already exists, otherwise an error
        will be returned if the file already exists.

    Returns
    -------
    str
        The validated zarr path

    Raises
    ------
    ValueError
        If the provided zarr path does not point to a zarr file
    RuntimeError
        If ``zarr_path`` already exists and ``overwrite=False``
    """

    if zarr_path is not None:
        # ensure that zarr_path is a string or Path object, throw an error otherwise
        if not isinstance(zarr_path, (str, Path)):
            raise TypeError("The provided zarr_path input must be of type string or pathlib.Path!")

        # check that the appropriate suffix was provided
        if not (Path(zarr_path).suffix == ".zarr"):
            raise ValueError("The provided zarr_path input must have a '.zarr' suffix!")

    # set default source_file name (will be used only if zarr_path is None)
    source_file = "combined_echodata.zarr"

    validated_path = validate_output_path(
        source_file=source_file,
        engine="zarr",
        output_storage_options=storage_options,
        save_path=zarr_path,
    )

    # convert created validated_path to a string if it is in other formats,
    # since fsspec only accepts strings
    if isinstance(validated_path, Path):
        validated_path = str(validated_path.absolute())

    # check if validated_path already exists
    fs = fsspec.get_mapper(validated_path, **storage_options).fs  # get file system
    exists = True if fs.exists(validated_path) else False

    if exists and not overwrite:
        raise RuntimeError(
            f"{validated_path} already exists, please provide a "
            "different path or set overwrite=True."
        )
    elif exists and overwrite:
        logger.info(f"overwriting {validated_path}")

        # remove zarr file
        fs.rm(validated_path, recursive=True)

    return validated_path


def _check_channel_selection_form(
    channel_selection: Optional[Union[List, Dict[str, list]]] = None
) -> None:
    """
    Ensures that the provided user input ``channel_selection`` is in
    an acceptable form.

    Parameters
    ----------
    channel_selection: list of str or dict, optional
        Specifies what channels should be selected for an ``EchoData`` group
        with a ``channel`` dimension (before combination).
    """

    # check that channel selection is None, a list, or a dict
    if not isinstance(channel_selection, (type(None), list, dict)):
        raise TypeError("The input channel_selection does not have an acceptable type!")

    if isinstance(channel_selection, list):
        # make sure each element is a string
        are_elem_str = [isinstance(elem, str) for elem in channel_selection]
        if not all(are_elem_str):
            raise TypeError("Each element of channel_selection must be a string!")

    if isinstance(channel_selection, dict):
        # make sure all keys are strings
        are_keys_str = [isinstance(elem, str) for elem in channel_selection.keys()]
        if not all(are_keys_str):
            raise TypeError("Each key of channel_selection must be a string!")

        # make sure all keys are of the form Sonar/Beam_group using regular expression
        are_keys_right_form = [
            True if re.match("Sonar/Beam_group(\d{1})", elem) else False  # noqa
            for elem in channel_selection.keys()
        ]
        if not all(are_keys_right_form):
            raise TypeError(
                "Each key of channel_selection can only be a beam group path of "
                "the form Sonar/Beam_group!"
            )

        # make sure all values are a list
        are_vals_list = [isinstance(elem, list) for elem in channel_selection.values()]
        if not all(are_vals_list):
            raise TypeError("Each value of channel_selection must be a list!")

        # make sure all values are a list of strings
        are_vals_list_str = [set(map(type, elem)) == {str} for elem in channel_selection]
        if not all(are_vals_list_str):
            raise TypeError("Each value of channel_selection must be a list of strings!")


def check_eds(echodata_list: List[EchoData]) -> Tuple[str, List[str]]:
    """
    Ensures that the input list of ``EchoData`` objects for ``combine_echodata``
    is in the correct form and all necessary items exist.

    Parameters
    ----------
    echodata_list: list of EchoData object
        The list of `EchoData` objects to be combined.

    Returns
    -------
    sonar_model : str
        The sonar model used for all values in ``echodata_list``
    echodata_filenames : list of str
        The source files names for all values in ``echodata_list``

    Raises
    ------
    TypeError
        If a list of ``EchoData`` objects are not provided
    ValueError
        If any ``EchoData`` object's ``sonar_model`` is ``None``
    ValueError
        If and ``EchoData`` object does not have a file path
    ValueError
        If the provided ``EchoData`` objects have the same filenames
    """

    # make sure that the input is a list of EchoData objects
    if not isinstance(echodata_list, list) and all(
        [isinstance(ed, EchoData) for ed in echodata_list]
    ):
        raise TypeError("The input, eds, must be a list of EchoData objects!")

    # get the sonar model for the combined object
    if echodata_list[0].sonar_model is None:
        raise ValueError("all EchoData objects must have non-None sonar_model values")
    else:
        sonar_model = echodata_list[0].sonar_model

    echodata_filenames = []
    for ed in echodata_list:
        # check sonar model
        if ed.sonar_model is None:
            raise ValueError("all EchoData objects must have non-None sonar_model values")
        elif ed.sonar_model != sonar_model:
            raise ValueError("all EchoData objects must have the same sonar_model value")

        # check for file names and store them
        if ed.source_file is not None:
            filepath = ed.source_file
        elif ed.converted_raw_path is not None:
            filepath = ed.converted_raw_path
        else:
            # defaulting to none, must be from memory
            filepath = None

        # set default filename to internal memory
        filename = "internal-memory"
        if filepath is not None:
            filename = Path(filepath).name
            if filename in echodata_filenames:
                raise ValueError("EchoData objects have conflicting filenames")

        echodata_filenames.append(filename)

    return sonar_model, echodata_filenames


def _check_channel_consistency(
    all_chan_list: List, ed_group: str, channel_selection: Optional[List[str]] = None
) -> None:
    """
    If ``channel_selection = None``, checks that each element in ``all_chan_list`` are
    the same, else makes sure that each element in ``all_chan_list`` contains all channel
    names in ``channel_selection``.

    Parameters
    ----------
    all_chan_list: list of list
        A list whose elements correspond to the Datasets to be combined with
        their values set as a list of the channel dimension names in the Dataset
    ed_group: str
        The EchoData group path that produced ``all_chan_list``
    channel_selection: list of str, optional
        A list of channel names, which should be a subset of each
        element in ``all_chan_list``

    Raises
    ------
    RuntimeError
        If ``channel_selection=None`` and all ``channel`` dimensions are not the
        same across all Datasets.
    NotImplementedError
        If ``channel_selection`` is a list and the listed channels are not contained
        in the ``EchoData`` group for all Datasets and need to be created and
        padded with NaN. This "expansion" type of combination has not been implemented.
    """

    if channel_selection is None:
        # sort each element in list, so correct comparison can be made
        all_chan_list = list(map(sorted, all_chan_list))

        # determine if the channels are the same across all Datasets
        all_chans_equal = [all_chan_list[0]] * len(all_chan_list) == all_chan_list

        if not all_chans_equal:
            # obtain all unique channel names
            unique_channels = set(itertools.chain.from_iterable(all_chan_list))

            # raise an error if we have varying channel lengths
            raise RuntimeError(
                f"For the EchoData group {ed_group} the channels: {unique_channels} are "
                f"not found in all EchoData objects being combined. Select which "
                f"channels should be included in the combination using the keyword argument "
                f"channel_selection in combine_echodata."
            )

    else:
        # make channel_selection a set, so it is easier to use
        channel_selection = set(channel_selection)

        # TODO: if we will allow for expansion, then the below code should be
        #  replaced with a code section that makes sure the selected channels
        #  appear at least once in one of the other Datasets

        # determine if channel selection is in each element of all_chan_list
        eds_num_chan = [
            channel_selection.intersection(set(ed_chans)) == channel_selection
            for ed_chans in all_chan_list
        ]

        if not all(eds_num_chan):
            # raise a not implemented error if expansion (i.e. padding is necessary)
            raise NotImplementedError(
                f"For the EchoData group {ed_group}, some EchoData objects do "
                f"not contain the selected channels. This type of combine is "
                f"not currently implemented."
            )


def _create_channel_selection_dict(
    sonar_model: str,
    has_chan_dim: Dict[str, bool],
    user_channel_selection: Optional[Union[List, Dict[str, list]]] = None,
) -> Dict[str, Optional[list]]:
    """
    Constructs the dictionary ``channel_selection_dict``, which specifies
    the ``channel`` dimension names that should be selected for each
    ``EchoData`` group. If a group does not have a ``channel`` dimension
    the dictionary value will be set to ``None``

    Parameters
    ----------
    sonar_model: str
        The name of the sonar model corresponding to ``has_chan_dim``
    has_chan_dim: dict
        A dictionary created using an ``EchoData`` object whose keys are
        the ``EchoData`` groups and whose values specify if that
        particular group has a ``channel`` dimension
    user_channel_selection: list or dict, optional
        A user provided input that will be used to construct the values of
        ``channel_selection_dict`` (see below for further details)

    Returns
    -------
    channel_selection_dict : dict
        A dictionary with the same keys as ``has_chan_dim`` and values
        determined by ``sonar_model`` and ``user_channel_selection`` as follows:
        - If ``user_channel_selection=None``, then the values of the dictionary
        will be set to ``None``
        - If ``user_channel_selection`` is a list, then all keys corresponding to
        an ``EchoData`` group with a ``channel`` dimension will have their values
        set to the provided list and all other groups will be set to ``None``
        - If ``user_channel_selection`` is a dictionary, then all keys corresponding to
        an ``EchoData`` group without a ``channel`` dimension will have their values
        set as ``None`` and the other group's values will be set as follows:
            - If ``sonar_model`` is not EK80-like then all values will be set to
            the union of the values of ``user_channel_selection``
            - If ``sonar_model`` is EK80-like then the groups ``Sonar, Platform, Vendor_specific``
            will be set to the union of the values of ``user_channel_selection`` and the rest of
            the groups will be set to the same value in ``user_channel_selection`` with the same key

    Notes
    -----
    See ``tests/echodata/test_echodata_combine.py::test_create_channel_selection_dict`` for example
    outputs from this function.
    """

    # base case where the user did not provide selected channels (will be used downstream)
    if user_channel_selection is None:
        return {grp: None for grp in has_chan_dim.keys()}

    # obtain the union of all channels for each beam group
    if isinstance(user_channel_selection, list):
        union_beam_chans = user_channel_selection[:]
    else:
        union_beam_chans = list(set(itertools.chain.from_iterable(user_channel_selection.values())))

    # make channel_selection dictionary where the keys are the EchoData groups and the
    # values are based on the user provided input user_channel_selection
    channel_selection_dict = dict()
    for ed_group, has_chan in has_chan_dim.items():
        # if there are no channel dimensions in the group, set the value to None
        if has_chan:
            if (
                (not isinstance(user_channel_selection, list))
                and (sonar_model in ["EK80", "ES80", "EA640"])
                and (ed_group not in ["Sonar", "Platform", "Vendor_specific"])
            ):
                # set value to the user provided input with the same key
                channel_selection_dict[ed_group] = user_channel_selection[ed_group]

            else:
                # set value to the union of the values of user_channel_selection
                channel_selection_dict[ed_group] = union_beam_chans

            # sort channel names to produce consistent output (since we may be using sets)
            channel_selection_dict[ed_group].sort()

        else:
            channel_selection_dict[ed_group] = None

    return channel_selection_dict


def _check_echodata_channels(
    echodata_list: List[EchoData],
    user_channel_selection: Optional[Union[List, Dict[str, list]]] = None,
) -> Dict[str, Optional[List[str]]]:
    """
    Coordinates the routines that check to make sure each ``EchoData`` group with a ``channel``
    dimension has consistent channels for all elements in ``echodata_list``, taking into account
    the input ``user_channel_selection``.

    Parameters
    ----------
    echodata_list: list of EchoData object
        The list of ``EchoData`` objects to be combined
    user_channel_selection: list or dict, optional
        A user provided input that will be used to specify which channels will be
        selected for each ``EchoData`` group

    Returns
    -------
    dict
        A dictionary with keys corresponding to the ``EchoData`` groups and
        values specifying the channels that should be selected within that group.
        For more information on this dictionary see the function ``_create_channel_selection_dict``.

    Raises
    ------
    RuntimeError
        If any ``EchoData`` group has a ``channel`` dimension value
        with a duplicate value.

    Notes
    -----
    For further information on what is deemed consistent, please see the
    function ``_check_channel_consistency``.
    """

    # determine if the EchoData group contains a channel dimension
    has_chan_dim = {
        grp: "channel" in echodata_list[0][grp].dims for grp in echodata_list[0].group_paths
    }

    # create dictionary specifying the channels that should be selected for each group
    channel_selection = _create_channel_selection_dict(
        echodata_list[0].sonar_model, has_chan_dim, user_channel_selection
    )

    for ed_group in echodata_list[0].group_paths:
        if "channel" in echodata_list[0][ed_group].dims:
            # get each EchoData's channels as a list of list
            all_chan_list = [list(ed[ed_group].channel.values) for ed in echodata_list]

            # make sure each EchoData does not have repeating channels
            all_chan_unique = [len(set(ed_chans)) == len(ed_chans) for ed_chans in all_chan_list]

            if not all(all_chan_unique):
                # get indices of EchoData objects with repeating channel names
                false_ind = [ind for ind, x in enumerate(all_chan_unique) if not x]

                # get files that produced the EchoData objects with repeated channels
                files_w_rep_chan = [
                    echodata_list[ind]["Provenance"].source_filenames.values[0] for ind in false_ind
                ]

                raise RuntimeError(
                    f"The EchoData objects produced by the following files "
                    f"have a channel dimension with repeating values, "
                    f"combine cannot be used: {files_w_rep_chan}"
                )

            # perform a consistency check for the channel dims across all Datasets
            _check_channel_consistency(all_chan_list, ed_group, channel_selection[ed_group])

    return channel_selection


def _check_ascending_ds_times(ds_list: List[xr.Dataset], ed_group: str) -> None:
    """
    A minimal check that the first time value of each Dataset is less than
    the first time value of the subsequent Dataset. If each first time value
    is NaT, then this check is skipped.

    Parameters
    ----------
    ds_list: list of xr.Dataset
        List of Datasets to be combined
    ed_group: str
        The name of the ``EchoData`` group being combined

    Returns
    -------
    None


    Raises
    ------
    RuntimeError
        If the timeX dimension is not in ascending order
        for the specified echodata group
    """

    # get all time dimensions of the input Datasets
    ed_time_dim = set(ds_list[0].dims).intersection(POSSIBLE_TIME_DIMS)

    for time in ed_time_dim:
        # gather the first time of each Dataset
        first_times = []
        for ds in ds_list:
            times = ds[time].values
            if isinstance(times, np.ndarray):
                # store first time if we have an array
                first_times.append(times[0])
            else:
                # store first time if we have a single value
                first_times.append(times)

        first_times = np.array(first_times)

        # skip check if all first times are NaT
        if not np.isnan(first_times).all():
            is_descending = (np.diff(first_times) < np.timedelta64(0, "ns")).any()

            if is_descending:
                raise RuntimeError(
                    f"The coordinate {time} is not in ascending order for "
                    f"group {ed_group}, combine cannot be used!"
                )


def _check_no_append_vendor_params(
    ds_list: List[xr.Dataset], ed_group: Literal["Vendor_specific"], ds_append_dims: set
) -> None:
    """
    Check for identical params for all inputs without an
    appending dimension in Vendor specific group

    Parameters
    ----------
    ds_list: list of xr.Dataset
        List of Datasets to be combined
    ed_group: "Vendor_specific"
        The name of the ``EchoData`` group being combined,
        this only works for "Vendor_specific" group.
    ds_append_dims: set
        A set of datasets append dimensions

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``ed_group`` is not ``Vendor_specific``.
    RuntimeError
        If non identical filter parameters is found.
    """
    if ed_group != "Vendor_specific":
        raise ValueError("Group must be `Vendor_specific`!")

    if len(ds_append_dims) > 0:
        # If there's a dataset appending dimension, drop for comparison
        # of the other values... everything else should be identical
        ds_list = [ds.drop_dims(ds_append_dims) for ds in ds_list]

    it = iter(ds_list)
    # Init as identical, must stay True.
    is_identical = True
    dataset = next(it)
    for next_dataset in it:
        is_identical = dataset.identical(next_dataset)
        if not is_identical:
            raise RuntimeError(
                f"Non identical filter parameters in {ed_group} group. " "Objects cannot be merged!"
            )
        dataset = next_dataset


def _merge_attributes(attributes: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Merge a list of attributes dictionary

    Parameters
    ----------
    attributes : list of dict
        List of attributes dictionary
        E.g. [{'attr1': 'val1'}, {'attr2': 'val2'}, ...]

    Returns
    -------
    dict
        The merged attribute dictionary
    """
    merged_dict = {}
    for attribute in attributes:
        for key, value in attribute.items():
            if key not in merged_dict:
                # if current key is not in merged attribute,
                # then save the value for that key
                merged_dict[key] = value
            elif merged_dict[key] == "":
                # if current key is already in merged attribute,
                # check if the value of that key is empty,
                # in this case overwrite the value with current value
                merged_dict[key] = value
            # By default the rest of the behavior
            # will keep the first non-empty value it sees

            # NOTE: @lsetiawan (6/2/2023) - Comment this out for now until
            # attributes are fully evaluated by @leewujung and @emiliom
            # if value == "" and key not in merged_dict:
            #     # checks if current attr value is empty,
            #     # and doesn't exist in merged attribute,
            #     # saving the first non empty value only
            #     merged_dict[key] = value
            # elif value != "":
            #     # if current attr value is not empty,
            #     # then overwrite the merged attribute,
            #     # keeping attribute from latest value
            #     merged_dict[key] = value
    return merged_dict


def _capture_prov_attrs(
    attrs_dict: Dict[str, List[Dict[str, str]]], echodata_filenames: List[str], sonar_model: str
) -> xr.Dataset:
    """
    Capture and create provenance dataset,
    from the combined attribute values.

    Parameters
    ----------
    attrs_dict : dict of list
        Dictionary of attributes for each of the group.
        E.g. {'Group': [{'attr1': 'val1'}, {'attr2': 'val2'}, ...]}
    echodata_filenames : list of str
        The filenames of the echodata objects
    sonar_model : str
        The sonar model

    Returns
    -------
    xr.Dataset
        The provenance dataset for all attribute values from
        the list of echodata objects that are combined.

    """
    ds_list = []
    for group, attributes in attrs_dict.items():
        df = pd.DataFrame.from_records(attributes)
        df.loc[:, ED_FILENAME] = echodata_filenames
        df = df.set_index(ED_FILENAME)

        group_ds = df.to_xarray()
        for _, var in group_ds.data_vars.items():
            var.attrs.update({ED_GROUP: group})
        ds_list.append(group_ds)

    prov_ds = xr.merge(ds_list)
    # Set these provenance as string
    prov_ds = prov_ds.fillna("").astype(str)
    prov_ds[ED_FILENAME] = prov_ds[ED_FILENAME].astype(str)
    return prov_ds


def _get_prov_attrs(
    ds: xr.Dataset, is_combined: bool = True
) -> Optional[Dict[str, List[Dict[str, str]]]]:
    """
    Get the provenance attributes from the dataset.
    This function is meant to be used on an already combined dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The Provenance group dataset to get attributes from
    is_combined: bool
        The flag to indicate if it's combined

    Returns
    -------
    Dict[str, List[Dict[str, str]]]
        The provenance attributes
    """

    if is_combined:
        attrs_dict = {}
        for k, v in ds.data_vars.items():
            # Go through each data variable and extract the attribute values
            # based on the echodata group as stored in the variable attribute
            if ED_GROUP in v.attrs:
                ed_group = v.attrs[ED_GROUP]
                if ed_group not in attrs_dict:
                    attrs_dict[ed_group] = []
                # Store the values as a list of dictionary for each group
                attrs_dict[ed_group].append([{k: i} for i in v.values])

        # Merge the attributes for each group so it matches the
        # attributes dict for later merging
        return {
            ed_group: [
                dict(ChainMap(*v))
                for _, v in pd.DataFrame.from_dict(attrs).to_dict(orient="list").items()
            ]
            for ed_group, attrs in attrs_dict.items()
        }
    return None


def _combine(
    sonar_model: str,
    eds: List[EchoData] = [],
    echodata_filenames: List[str] = [],
    ed_group_chan_sel: Dict[str, Optional[List[str]]] = {},
) -> Dict[str, xr.Dataset]:
    """
    Combines the echodata objects and export to a dictionary tree.

    Parameters
    ----------
    sonar_model : str
        The sonar model used for all elements in ``eds``
    eds: list of EchoData object
        The list of ``EchoData`` objects to be combined
    echodata_filenames : list of str
        The filenames of the echodata objects
    ed_group_chan_sel: dict
        A dictionary with keys corresponding to the ``EchoData`` groups
        and values specify what channels should be selected within that
        group. If a value is ``None``, then a subset of channels should
        not be selected.

    Returns
    -------
    dict of xr.Dataset
        The dictionary tree containing the xarray dataset
        for each of the combined group

    """
    all_group_paths = dict.fromkeys(
        itertools.chain.from_iterable([list(ed.group_paths) for ed in eds])
    ).keys()
    # For dealing with attributes
    attrs_dict = {}

    # Check if input data are combined datasets
    # Create combined mapping for later use
    combined_mapping = []
    for idx, ed in enumerate(eds):
        is_combined = ed["Provenance"].attrs.get("is_combined", False)
        combined_mapping.append(
            {
                "is_combined": is_combined,
                "attrs_dict": _get_prov_attrs(ed["Provenance"], is_combined),
                "echodata_filename": (
                    [str(s) for s in ed["Provenance"][ED_FILENAME].values]
                    if is_combined
                    else [echodata_filenames[idx]]
                ),
            }
        )
    # Get single boolean value to see if there's any combined files
    any_combined = any(d["is_combined"] for d in combined_mapping)

    if any_combined:
        # Fetches the true echodata filenames if there are any combined files
        echodata_filenames = list(
            itertools.chain.from_iterable([d[ED_FILENAME] for d in combined_mapping])
        )

    # Create Echodata tree dict
    tree_dict = {}
    for ed_group in all_group_paths:
        # collect the group Dataset from all eds that have their channels unselected
        all_chan_ds_list = [ed[ed_group] for ed in eds]

        # select only the appropriate channels from each Dataset
        ds_list = [
            (
                ds.sel(channel=ed_group_chan_sel[ed_group])
                if ed_group_chan_sel[ed_group] is not None
                else ds
            )
            for ds in all_chan_ds_list
        ]

        if ds_list:
            if not any_combined:
                # Get all of the keys and attributes
                # for regular non combined echodata object
                ds_attrs = [ds.attrs for ds in ds_list]
            else:
                # If there are any combined files,
                # iterate through from mapping above
                ds_attrs = []
                for idx, ds in enumerate(ds_list):
                    # Retrieve the echodata attrs dict
                    # parsed from provenance group above
                    ed_attrs_dict = combined_mapping[idx]["attrs_dict"]
                    if ed_attrs_dict is not None:
                        # Set attributes to the appropriate group
                        # from echodata attrs provenance,
                        # set default empty dict for missing group
                        attrs = ed_attrs_dict.get(ed_group, {})
                    else:
                        # This is for non combined echodata object
                        attrs = [ds.attrs]
                    ds_attrs += attrs

            # Attribute holding
            attrs_dict[ed_group] = ds_attrs

            # Checks for ascending time in dataset list
            _check_ascending_ds_times(ds_list, ed_group)

            # get all dimensions in ds that are append dimensions
            ds_append_dims = set(ds_list[0].dims).intersection(APPEND_DIMS)

            # Checks for filter parameters for "Vendor_specific" ONLY
            if ed_group == "Vendor_specific":
                _check_no_append_vendor_params(ds_list, ed_group, ds_append_dims)

            if len(ds_append_dims) == 0:
                combined_ds = ds_list[0]
            else:
                combined_ds = xr.Dataset()
                for dim in ds_append_dims:
                    drop_dims = [c_dim for c_dim in ds_append_dims if c_dim != dim]
                    sub_ds = xr.concat(
                        [ds.drop_dims(drop_dims) for ds in ds_list],
                        dim=dim,
                        coords="minimal",
                        data_vars="minimal",
                        compat="no_conflicts",
                    )
                    combined_ds = combined_ds.assign(sub_ds.variables)

            # Modify default attrs
            if ed_group == "Top-level":
                ed_group = "/"

            # Merge attributes and set to dataset
            group_attrs = _merge_attributes(ds_attrs)

            # Empty out attributes for now, will be refilled later
            combined_ds.attrs = group_attrs

            # Add combined flag and update conversion time for Provenance
            if ed_group == "Provenance":
                combined_ds.attrs.update(
                    {
                        "is_combined": True,
                        "conversion_software_name": group_attrs["conversion_software_name"],
                        "conversion_software_version": group_attrs["conversion_software_version"],
                        "conversion_time": group_attrs["conversion_time"],
                    }
                )
                prov_dict = echopype_prov_attrs(process_type="combination")
                combined_ds = combined_ds.assign_attrs(prov_dict)

            # Data holding
            tree_dict[ed_group] = combined_ds

    # Capture provenance for all the attributes
    prov_ds = _capture_prov_attrs(attrs_dict, echodata_filenames, sonar_model)
    if not any_combined:
        # Update the provenance dataset with the captured data
        prov_ds = tree_dict["Provenance"].assign(prov_ds)
    else:
        prov_ds = tree_dict["Provenance"].drop_dims(ED_FILENAME).assign(prov_ds)

    # Update filenames to iter integers
    prov_ds[FILENAMES] = prov_ds[FILENAMES].copy(data=np.arange(*prov_ds[FILENAMES].shape))  # noqa
    tree_dict["Provenance"] = prov_ds

    return tree_dict


def combine_echodata(
    echodata_list: List[EchoData] = None,
    channel_selection: Optional[Union[List, Dict[str, list]]] = None,
) -> EchoData:
    """
    Combines multiple ``EchoData`` objects into a single ``EchoData`` object.

    Parameters
    ----------
    echodata_list : list of EchoData object
        The list of ``EchoData`` objects to be combined
    channel_selection: list of str or dict, optional
        Specifies what channels should be selected for an ``EchoData`` group
        with a ``channel`` dimension (before combination).

        - if a list is provided, then each ``EchoData`` group with a ``channel`` dimension
        will only contain the channels in the provided list
        - if a dictionary is provided, the dictionary should have keys specifying only beam
        groups (e.g. "Sonar/Beam_group1") and values as a list of channel names to select
        within that beam group. The rest of the ``EchoData`` groups with a ``channel`` dimension
        will have their selected channels chosen automatically.

    Returns
    -------
    EchoData
        A lazy loaded ``EchoData`` object,
        with all data from the input ``EchoData`` objects combined.

    Raises
    ------
    ValueError
        If the provided zarr path does not point to a zarr file
    TypeError
        If a list of ``EchoData`` objects are not provided
    ValueError
        If any ``EchoData`` object's ``sonar_model`` is ``None``
    ValueError
        If any ``EchoData`` object does not have a file path
    ValueError
        If the provided ``EchoData`` objects have the same filenames
    RuntimeError
        If the first time value of each ``EchoData`` group is not less
        than the first time value of the subsequent corresponding
        ``EchoData`` group, with respect to the order in ``echodata_list``
    RuntimeError
        If the same ``EchoData`` groups in ``echodata_list`` do not
        have the same number of channels and the same name for each
        of these channels.
    RuntimeError
        If any of the following attribute checks are not met
        amongst the combined ``EchoData`` groups:

        - the keys are not the same
        - the values are not identical
        - the keys ``date_created`` or ``conversion_time``
          do not have the same types
    RuntimeError
        If any ``EchoData`` group has a ``channel`` dimension value
        with a duplicate value.
    RuntimeError
        If ``channel_selection=None`` and the ``channel`` dimensions are not the
        same across the same group under each object in ``echodata_list``.
    NotImplementedError
        If ``channel_selection`` is a list and the listed channels are not contained
        in the ``EchoData`` group across all objects in ``echodata_list``.

    Notes
    -----
    * ``EchoData`` objects are combined by appending their groups individually.
    * All attributes (besides attributes whose values are arrays) from all groups before the
      combination will be stored in the ``Provenance`` group.

    Examples
    --------
    Combine lazy loaded ``EchoData`` objects:

    >>> ed1 = echopype.open_converted("file1.zarr")
    >>> ed2 = echopype.open_converted("file2.zarr")
    >>> combined = echopype.combine_echodata(echodata_list=[ed1, ed2])

    Combine in-memory ``EchoData`` objects:

    >>> ed1 = echopype.open_raw(raw_file="EK60_file1.raw", sonar_model="EK60")
    >>> ed2 = echopype.open_raw(raw_file="EK60_file2.raw", sonar_model="EK60")
    >>> combined = echopype.combine_echodata(echodata_list=[ed1, ed2])
    """
    # return empty EchoData object, if no EchoData objects are provided
    if echodata_list is None:
        warn("No EchoData objects were provided, returning an empty EchoData object.")
        return EchoData()

    # Ensure the list of all EchoData objects to be combined are valid
    sonar_model, echodata_filenames = check_eds(echodata_list)

    # make sure channel_selection is the appropriate type and only contains the beam groups
    _check_channel_selection_form(channel_selection)

    # perform channel check and get channel selection for each EchoData group
    ed_group_chan_sel = _check_echodata_channels(echodata_list, channel_selection)

    # combine the echodata objects and get the tree dict
    tree_dict = _combine(
        sonar_model=sonar_model,
        eds=echodata_list,
        echodata_filenames=echodata_filenames,
        ed_group_chan_sel=ed_group_chan_sel,
    )

    # create datatree from tree dictionary
    tree = DataTree.from_dict(tree_dict, name="root")

    # create echodata object from datatree
    ed_comb = EchoData(sonar_model=sonar_model)
    ed_comb._set_tree(tree)
    ed_comb._load_tree()

    return ed_comb
