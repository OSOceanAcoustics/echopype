from pathlib import Path
from typing import List, Optional, Tuple
from warnings import warn

import xarray as xr
from dask.distributed import Client

from ..qc import coerce_increasing_time, exist_reversed_time
from ..utils.io import validate_output_path
from ..utils.log import _init_logger
from .echodata import EchoData
from .zarr_combine import ZarrCombine

logger = _init_logger(__name__)


def check_zarr_path(zarr_path: str, storage_options: Optional[dict]) -> str:
    """
    Checks that the zarr path provided to ``combine``
    is valid.

    Parameters
    ----------
    zarr_path: str
        The full save path to the final combined zarr store
    storage_options: Optional[dict]
        Any additional parameters for the storage
        backend (ignored for local paths)

    Returns
    -------
    str
        The validated zarr path

    Raises
    ------
    ValueError
        If the provided zarr path does not point to a zarr file
    """

    if zarr_path is None:

        # assign values, if no zarr path has been provided
        source_file = "combined_echodatas.zarr"
        save_path = None
    else:

        # turn string path into Path object
        path_obj = Path(zarr_path)
        if path_obj.suffix != ".zarr":
            raise ValueError("The provided zarr_path input must point to a zarr file!")
        else:

            # assign values based on zarr path
            source_file = path_obj.parts[-1]
            save_path = path_obj.parent

    return validate_output_path(
        source_file=source_file,
        engine="zarr",
        output_storage_options=storage_options,
        save_path=save_path,
    )


def check_echodatas_input(echodatas: List[EchoData]) -> Tuple[str, List[str]]:
    """
    Ensures that the input list of ``EchoData`` objects for ``combine_echodata``
    is in the correct form and all necessary items exist.

    Parameters
    ----------
    echodatas: List[EchoData]
        The list of `EchoData` objects to be combined.

    Returns
    -------
    sonar_model : str
        The sonar model used for all values in ``echodatas``
    echodata_filenames : List[str]
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


def check_and_correct_reversed_time(
    combined_group: xr.Dataset, time_str: str, ed_group: str
) -> Optional[xr.DataArray]:
    """
    Makes sure that the time coordinate ``time_str`` in
    ``combined_group`` is in the correct order and corrects
    it, if it is not. If coercion is necessary, the input
    `combined_group` will be directly modified.

    Parameters
    ----------
    combined_group : xr.Dataset
        Dataset representing a combined EchoData group
    time_str : str
        Name of time coordinate to be checked and corrected
    ed_group : str
        Name of ``EchoData`` group name

    Returns
    -------
    old_time : Optional[xr.DataArray]
        If correction is necessary, returns the time before
        reversal correction, otherwise returns None

    Warns
    -----
    UserWarning
        If a time reversal is detected
    """

    if time_str in combined_group and exist_reversed_time(combined_group, time_str):

        logger.warning(
            f"{ed_group} {time_str} reversal detected; {time_str} will be corrected"  # noqa
            " (see https://github.com/OSOceanAcoustics/echopype/pull/297)"
        )
        old_time = combined_group[time_str].copy()
        coerce_increasing_time(combined_group, time_name=time_str)
    else:
        old_time = None

    return old_time


def create_old_time_array(group: str, old_time_in: xr.DataArray) -> xr.DataArray:
    """
    Creates an old time array with the appropriate values, name,
    attributes, and encoding.

    Parameters
    ----------
    group: str
        The name of the ``EchoData`` group that contained
        the old time
    old_time_in: xr.DataArray
        The uncorrected old time

    Returns
    -------
    old_time_array: xr.DataArray
        The newly created old time array
    """

    # make a copy, so we don't change the source array
    old_time = old_time_in.copy()

    # get name of old time and dim for Provenance group
    ed_name = group.replace("-", "_").replace("/", "_").lower()
    old_time_name = ed_name + "_old_" + old_time.name

    old_time_name_dim = old_time_name + "_dim"

    # construct old time attributes
    attributes = old_time.attrs
    attributes["comment"] = f"Uncorrected {old_time.name} from the combined group {group}."

    # create old time array
    old_time_array = xr.DataArray(
        data=old_time.values, dims=[old_time_name_dim], attrs=attributes, name=old_time_name
    )

    # set encodings
    old_time_array.encoding = old_time.encoding

    return old_time_array


def orchestrate_reverse_time_check(
    ed_comb: EchoData, zarr_store: str, possible_time_dims: List[str], storage_options: dict
) -> None:
    """
    Performs a reverse time check of all groups and
    each time dimension within the group. If a reversed
    time is found it will be corrected in ``ed_comb``,
    updated in the zarr store, the old time will be
    added to the ``Provenance`` group in ``ed_comb``,
    the old time will be written to the zarr store,
    and the attribute ``reversed_ping_times`` in the
    ``Provenance`` group will be set to ``1``.

    Parameters
    ----------
    ed_comb: EchoData
        ``EchoData`` object that has been constructed from
        combined ``EchoData`` objects
    zarr_store: str
        The zarr store containing the ``ed_comb`` data
    possible_time_dims: List[str]
        All possible time dimensions that can occur within
        ``ed_comb``, which should be checked
    storage_options: dict
        Additional keywords to pass to the filesystem class.

    Notes
    -----
    If correction is necessary, ``ed_comb`` will be
    directly modified.
    """

    # set Provenance attribute to zero in ed_comb
    ed_comb["Provenance"].attrs["reversed_ping_times"] = 0

    # set Provenance attribute to zero in zarr (Dataset needed for metadata creation)
    only_attrs_ds = xr.Dataset(attrs=ed_comb["Provenance"].attrs)
    only_attrs_ds.to_zarr(zarr_store, group="Provenance", mode="a", storage_options=storage_options)

    for group in ed_comb.group_paths:

        if group != "Platform/NMEA":
            # Platform/NMEA is skipped because we found that the times which correspond to
            # other non-GPS messages are often out of order and correcting them is not
            # possible with the current implementation of _clean_ping_time in qc.api due
            # to excessive recursion. There is also no obvious advantage in correcting
            # the order of these timestamps.

            # get all time dimensions of the group
            ed_comb_time_dims = set(ed_comb[group].dims).intersection(possible_time_dims)

            for time in ed_comb_time_dims:

                old_time = check_and_correct_reversed_time(
                    combined_group=ed_comb[group], time_str=time, ed_group=group
                )

                if old_time is not None:

                    old_time_array = create_old_time_array(group, old_time)

                    # put old times in Provenance and modify attribute
                    ed_comb["Provenance"][old_time_array.name] = old_time_array
                    ed_comb["Provenance"].attrs["reversed_ping_times"] = 1

                    # save old time to zarr store
                    old_time_ds = old_time_array.to_dataset()
                    old_time_ds.attrs = ed_comb["Provenance"].attrs
                    old_time_ds.to_zarr(
                        zarr_store, group="Provenance", mode="a", storage_options=storage_options
                    )

                    # save corrected time to zarr store
                    ed_comb[group][[time]].to_zarr(
                        zarr_store, group=group, mode="r+", storage_options=storage_options
                    )


def combine_echodata(
    echodatas: List[EchoData] = None,
    zarr_path: Optional[str] = None,
    storage_options: Optional[dict] = {},
    client: Optional[Client] = None,
) -> EchoData:
    """
    Combines multiple ``EchoData`` objects into a single ``EchoData`` object.
    This is accomplished by writing each element of ``echodatas`` in parallel
    (using Dask) to the zarr store specified by ``zarr_path``.

    Parameters
    ----------
    echodatas : List[EchoData]
        The list of ``EchoData`` objects to be combined
    zarr_path: str
        The full save path to the final combined zarr store
    storage_options: Optional[dict]
        Any additional parameters for the storage
        backend (ignored for local paths)
    client: Optional[dask.distributed.Client]
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

    Warns
    -----
    UserWarning
        If any time coordinate in a final combined group is not
        in ascending order (see Notes below for more details).

    Notes
    -----
    * ``EchoData`` objects are combined by appending their groups individually to a zarr store.
    * All attributes (besides attributes whose values are arrays) from all groups before the
      combination will be stored in the ``Provenance`` group.
    * The instance attributes ``source_file`` and ``converted_raw_path`` of the combined
      ``EchoData`` object will be copied from the first ``EchoData`` object in the given list.
    * If any time coordinate in a final combined group is not in ascending order, then it will
      be corrected according to `#297 <https://github.com/OSOceanAcoustics/echopype/pull/297>`_.
      Additionally, the uncorrected time coordinate will be stored in the ``Provenace`` group as
      a variable and the ``Provenance`` attribute ``reversed_ping_times`` will be set to ``1``.
    * If no ``zarr_path`` is provided, the combined zarr file will be
      ``'temp_echopype_output/combined_echodatas.zarr'`` under the current working directory.
    * If no ``client`` is provided, then a client with a local scheduler will be used.
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
    # TODO: change PR #297 reference to a link in our documentation

    # check the client input and print dashboard link
    if client is None:
        client = Client()  # create client with local scheduler
        print(f"Client dashboard link: {client.dashboard_link}")
    else:

        if isinstance(client, Client):
            print(f"Client dashboard link: {client.dashboard_link}")
        else:
            raise TypeError("The input client is not of type dask.distributed.Client!")

    # Check the provided zarr_path is valid, or create a temp zarr_path if not provided
    zarr_path = check_zarr_path(zarr_path, storage_options)

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

    orchestrate_reverse_time_check(ed_comb, zarr_path, comb.possible_time_dims, storage_options)

    return ed_comb
