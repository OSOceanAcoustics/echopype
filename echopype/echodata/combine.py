import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import dask.distributed
import fsspec
from dask.distributed import Client

from ..utils.io import validate_output_path
from ..utils.log import _init_logger
from .echodata import EchoData
from .zarr_combine import ZarrCombine

logger = _init_logger(__name__)


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
    source_file = "combined_echodatas.zarr"

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


def check_echodatas_input(echodatas: List[EchoData]) -> Tuple[str, List[str]]:
    """
    Ensures that the input list of ``EchoData`` objects for ``combine_echodata``
    is in the correct form and all necessary items exist.

    Parameters
    ----------
    echodatas: list of EchoData object
        The list of `EchoData` objects to be combined.

    Returns
    -------
    sonar_model : str
        The sonar model used for all values in ``echodatas``
    echodata_filenames : list of str
        The source files names for all values in ``echodatas``

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
    if not isinstance(echodatas, list) and all([isinstance(ed, EchoData) for ed in echodatas]):
        raise TypeError("The input, eds, must be a list of EchoData objects!")

    # get the sonar model for the combined object
    if echodatas[0].sonar_model is None:
        raise ValueError("all EchoData objects must have non-None sonar_model values")
    else:
        sonar_model = echodatas[0].sonar_model

    echodata_filenames = []
    for ed in echodatas:

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
            # unreachable
            raise ValueError("EchoData object does not have a file path")

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
        If ``channel_selection`` is a list and the list of channels are not contained
        in the ``EchoData`` group for all Datasets
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
                f"For the EchoData group {ed_group} the channels: {unique_channels}, are "
                f"not provided in each EchoData object being combined. One can select which "
                f"channels should be included using the keyword argument channel_selection in "
                f"combine_echodata."
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
        the groups of the ``EchoData`` object and values specify if that
        particular group has a ``channel`` dimension
    user_channel_selection: list or dict, optional
        A user provided input that will be used to construct the values of
        ``channel_selection_dict`` (see below for further details)

    Returns
    -------
    dict
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

            # if the model is not EK80-like set value to the union of the values
            # of user_channel_selection
            if sonar_model in ["EK80", "ES80", "EA640"]:

                # if the model is EK80-like
                if (ed_group in ["Sonar", "Platform", "Vendor_specific"]) or isinstance(
                    user_channel_selection, list
                ):
                    # set value to the union of the values of user_channel_selection
                    channel_selection_dict[ed_group] = union_beam_chans

                else:

                    # set value to the user provided input with the same key
                    channel_selection_dict[ed_group] = user_channel_selection[ed_group]

            else:
                channel_selection_dict[ed_group] = union_beam_chans

            # sort channel names to produce consistent output (since we may be using sets)
            channel_selection_dict[ed_group].sort()

        else:
            channel_selection_dict[ed_group] = None

    return channel_selection_dict


def _check_echodata_channels(
    echodatas: List[EchoData], user_channel_selection: Optional[Union[List, Dict[str, list]]] = None
) -> Dict[str, Optional[List[str]]]:
    """
    Coordinates the routines that check to make sure each ``EchoData`` group with a ``channel``
    dimension has consistent channels for all elements in ``echodatas``, taking into account
    the input ``user_channel_selection``.

    Parameters
    ----------
    echodatas: list of EchoData object
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
    has_chan_dim = {grp: "channel" in echodatas[0][grp].dims for grp in echodatas[0].group_paths}

    # create dictionary specifying the channels that should be selected for each group
    channel_selection = _create_channel_selection_dict(
        echodatas[0].sonar_model, has_chan_dim, user_channel_selection
    )

    for ed_group in echodatas[0].group_paths:

        if "channel" in echodatas[0][ed_group].dims:

            # get each EchoData's channels as a list of list
            all_chan_list = [list(ed[ed_group].channel.values) for ed in echodatas]

            # make sure each EchoData does not have repeating channels
            all_chan_unique = [len(set(ed_chans)) == len(ed_chans) for ed_chans in all_chan_list]

            if not all(all_chan_unique):
                # get indices of EchoData objects with repeating channel names
                false_ind = [ind for ind, x in enumerate(all_chan_unique) if not x]

                # get files that produced the EchoData objects with repeated channels
                files_w_rep_chan = [
                    echodatas[ind]["Provenance"].source_filenames.values[0] for ind in false_ind
                ]

                raise RuntimeError(
                    f"The EchoData objects produced by the following files "
                    f"have a channel dimension with repeating values, "
                    f"combine cannot be used: {files_w_rep_chan}"
                )

            # perform a consistency check for the channel dims across all Datasets
            _check_channel_consistency(all_chan_list, ed_group, channel_selection[ed_group])

    return channel_selection


def combine_echodata(
    echodatas: List[EchoData] = None,
    zarr_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    storage_options: Dict[str, Any] = {},
    client: Optional[dask.distributed.Client] = None,
) -> EchoData:
    """
    Combines multiple ``EchoData`` objects into a single ``EchoData`` object.
    This is accomplished by writing each element of ``echodatas`` in parallel
    (using Dask) to the zarr store specified by ``zarr_path``.

    Parameters
    ----------
    echodatas : list of EchoData object
        The list of ``EchoData`` objects to be combined
    zarr_path: str or pathlib.Path, optional
        The full save path to the final combined zarr store
    overwrite: bool
        If True, will overwrite the zarr store specified by
        ``zarr_path`` if it already exists, otherwise an error
        will be returned if the file already exists.
    storage_options: dict
        Any additional parameters for the storage
        backend (ignored for local paths)
    client: dask.distributed.Client, optional
        An initialized Dask distributed client

    Returns
    -------
    EchoData
        A lazy loaded ``EchoData`` object obtained from ``zarr_path``,
        with all data from the input ``EchoData`` objects combined.

    Raises
    ------
    ValueError
        If the provided zarr path does not point to a zarr file
    TypeError
        If the a provided `zarr_path` input is not of type string or pathlib.Path
    RuntimeError
        If ``zarr_path`` already exists and ``overwrite=False``
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
        ``EchoData`` group, with respect to the order in ``echodatas``
    RuntimeError
        If the same ``EchoData`` groups in ``echodatas`` do not
        have the same number of channels and the same name for each
        of these channels.
    RuntimeError
        If any of the following attribute checks are not met
        amongst the combined ``EchoData`` groups:

        - the keys are not the same
        - the values are not identical
        - the keys ``date_created`` or ``conversion_time``
          do not have the same types

          # TODO: add the following error descriptions
    RuntimeError
        If any ``EchoData`` group has a ``channel`` dimension value
        with a duplicate value.
    RuntimeError
        If ``channel_selection=None`` and all ``channel`` dimensions are not the
        same across all Datasets.
    NotImplementedError
        If ``channel_selection`` is a list and the list of channels are not contained
        in the ``EchoData`` group for all Datasets

    Notes
    -----
    * ``EchoData`` objects are combined by appending their groups individually to a zarr store.
    * All attributes (besides attributes whose values are arrays) from all groups before the
      combination will be stored in the ``Provenance`` group.
    * The instance attributes ``source_file`` and ``converted_raw_path`` of the combined
      ``EchoData`` object will be copied from the first ``EchoData`` object in the given list.
    * If no ``zarr_path`` is provided, the combined zarr file will be
      ``'temp_echopype_output/combined_echodatas.zarr'`` under the current working directory.
    * If no ``client`` is provided, then a client with a local scheduler will be used. The
      created scheduler and client will be shutdown once computation has finished.
    * For each run of this function, we print our the client dashboard link.

    Examples
    --------
    Combine lazy loaded ``EchoData`` objects:

    >>> ed1 = echopype.open_converted("file1.nc")
    >>> ed2 = echopype.open_converted("file2.zarr")
    >>> combined = echopype.combine_echodata(echodatas=[ed1, ed2],
    >>>                                      zarr_path="path/to/combined.zarr",
    >>>                                      storage_options=my_storage_options)

    Combine in-memory ``EchoData`` objects:

    >>> ed1 = echopype.open_raw(raw_file="EK60_file1.raw", sonar_model="EK60")
    >>> ed2 = echopype.open_raw(raw_file="EK60_file2.raw", sonar_model="EK60")
    >>> combined = echopype.combine_echodata(echodatas=[ed1, ed2],
    >>>                                      zarr_path="path/to/combined.zarr",
    >>>                                      storage_options=my_storage_options)
    """

    # set flag specifying that a client was not created
    client_created = False

    # check the client input and print dashboard link
    if client is None:
        # set flag specifying that a client was created
        client_created = True

        client = Client()  # create client with local scheduler
        logger.info(f"Client dashboard link: {client.dashboard_link}")
    elif isinstance(client, Client):
        logger.info(f"Client dashboard link: {client.dashboard_link}")
    else:
        raise TypeError(f"The input client is not of type {type(Client)}!")

    # Check the provided zarr_path is valid, or create a temp zarr_path if not provided
    zarr_path = check_zarr_path(zarr_path, storage_options, overwrite)

    # return empty EchoData object, if no EchoData objects are provided
    if echodatas is None:
        warn("No EchoData objects were provided, returning an empty EchoData object.")
        return EchoData()

    # Ensure the list of all EchoData objects to be combined are valid
    sonar_model, echodata_filenames = check_echodatas_input(echodatas)

    # TODO: make sure user_channel_selection is the appropriate type and
    #  only contains the Beam_groups

    ed_group_chan_sel = _check_echodata_channels(echodatas)
    print(ed_group_chan_sel)

    # initiate ZarrCombine object
    comb = ZarrCombine()

    # combine all elements in echodatas by writing to a zarr store
    ed_comb = comb.combine(
        zarr_path,
        echodatas,
        storage_options=storage_options,
        sonar_model=sonar_model,
        echodata_filenames=echodata_filenames,
    )

    if client_created:
        # close client
        client.close()

    return ed_comb
