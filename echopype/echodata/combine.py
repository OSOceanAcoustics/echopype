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


def _check_channel_consistency(all_chan_list, channel_selection, ed_group) -> None:
    """
    asdfsd
    """

    # TODO: document!

    # get the number of channels in each Dataset being combined
    eds_num_chan = set(map(len, all_chan_list))

    if len(eds_num_chan) > 1 and (channel_selection is None):

        # obtain all unique channel names
        unique_channels = set(itertools.chain.from_iterable(all_chan_list))

        # raise an error if we have varying channel lengths
        raise RuntimeError(
            f"For the EchoData group {ed_group} all EchoData objects being "
            f"combined do not have the following channels: {unique_channels}. "
            "One can select which channels should be included using the keyword "
            "argument channel_selection in combine_echodata."
        )

    elif channel_selection is not None:

        # make channel_selection a set, so it is easier to use
        channel_selection = set(channel_selection)

        # determine if the selected channels are in each Dataset being combined
        eds_num_chan_after_select = {
            len(ed_chans.intersection(channel_selection)) for ed_chans in all_chan_list
        }

        # TODO: if we will allow for expansion, then the below code should be
        #  replaced with a section that makes sure the selected channels appear
        #  at least once in one of the other Datasets

        padding_needed = False
        if len(eds_num_chan_after_select) > 1:

            # padding needed for scenarios such as eds_num_chan_after_select = {3, 4}
            padding_needed = True

        elif len(channel_selection) not in eds_num_chan_after_select:

            # padding needed for scenarios such as
            # eds_num_chan_after_select = {3}, len(channel_selection)=4
            padding_needed = True

        if padding_needed:

            # raise a not implemented error if expansion (i.e. padding is necessary)
            raise NotImplementedError("")

        # TODO: make sure logic is correct


def _check_echodata_channels(echodatas: List[EchoData], channel_selection: Optional[List] = None):
    """
    Ensures that all channels are the same


    """

    # TODO: make sure channel_selection is the appropriate type

    for ed_group in echodatas[0].group_paths:

        if "channel" in echodatas[0][ed_group].dims:

            # get each EchoData's channels as a list of sets
            all_chan_list = [set(ed[ed_group].channel.values) for ed in echodatas]

            print(all_chan_list)


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
