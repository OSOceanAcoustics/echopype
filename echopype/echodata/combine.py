from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
    zarr_path: str, storage_options: Dict[str, Any] = {}, overwrite: bool = False
) -> str:
    """
    Checks that the zarr path provided to ``combine``
    is valid.

    Parameters
    ----------
    zarr_path: str
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
        # check that zarr_path is a string
        if not isinstance(zarr_path, str):
            raise TypeError(f"zarr_path must be of type {str}")

        # check that the appropriate suffix was provided
        if not zarr_path.strip("/").endswith(".zarr"):
            raise ValueError("The provided zarr_path input must have '.zarr' suffix!")

    # set default source_file name (will be used only if zarr_path is None)
    source_file = "combined_echodatas.zarr"

    validated_path = validate_output_path(
        source_file=source_file,
        engine="zarr",
        output_storage_options=storage_options,
        save_path=zarr_path,
    )

    # check if validated_path already exists
    fs = fsspec.get_mapper(validated_path, **storage_options).fs  # get file system
    exists = True if fs.exists(validated_path) else False

    if exists and not overwrite:
        raise RuntimeError(
            f"{zarr_path} already exists, please provide a different path or set overwrite=True."
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


def combine_echodata(
    echodatas: List[EchoData] = None,
    zarr_path: Optional[str] = None,
    overwrite: bool = False,
    storage_options: Dict[str, Any] = {},
    client: Optional[dask.distributed.Client] = None,
    consolidated: bool = True,
) -> EchoData:
    """
    Combines multiple ``EchoData`` objects into a single ``EchoData`` object.
    This is accomplished by writing each element of ``echodatas`` in parallel
    (using Dask) to the zarr store specified by ``zarr_path``.

    Parameters
    ----------
    echodatas : list of EchoData object
        The list of ``EchoData`` objects to be combined
    zarr_path: str, optional
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
    consolidated: bool
        Flag to consolidate zarr metadata.
        Defaults to ``True``

    Returns
    -------
    EchoData
        A lazy loaded ``EchoData`` object obtained from ``zarr_path``,
        with all data from the input ``EchoData`` objects combined.

    Raises
    ------
    ValueError
        If the provided zarr path does not point to a zarr file
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
        consolidated=consolidated,
    )

    if client_created:
        # close client
        client.close()

    return ed_comb
