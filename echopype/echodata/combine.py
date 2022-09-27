from pathlib import Path
from typing import List, Optional, Tuple
from warnings import warn

import xarray as xr

from ..qc import coerce_increasing_time, exist_reversed_time
from ..utils.log import _init_logger
from .echodata import EchoData
from .zarr_combine import ZarrCombine

logger = _init_logger(__name__)


def check_echodatas_input(echodatas: List[EchoData]) -> Tuple[str, List[str]]:
    """
    Ensures that the input ``echodatas`` for ``combine_echodata``
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
    """

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
    """

    if time_str in combined_group and exist_reversed_time(combined_group, time_str):

        logger.warning(
            f"{ed_group} {time_str} reversal detected; {time_str} will be corrected"  # noqa
            " (see https://github.com/OSOceanAcoustics/echopype/pull/297)"
        )
        old_time = combined_group[time_str]
        coerce_increasing_time(combined_group, time_name=time_str)

    else:
        old_time = None

    return old_time


def combine_echodata(echodatas: List[EchoData], zarr_store=None, storage_options={}) -> EchoData:
    """
    Combines multiple ``EchoData`` objects into a single ``EchoData`` object.

    Parameters
    ----------
    echodatas : List[EchoData]
        The list of ``EchoData`` objects to be combined.
    combine_attrs : str
        String indicating how to combine attrs of the ``EchoData`` objects being merged.
        This parameter matches the identically named xarray parameter
        (see https://xarray.pydata.org/en/latest/generated/xarray.combine_nested.html)
        with the exception of the "overwrite_conflicts" value. Possible options:
        * ``"override"``: Default. skip comparing and copy attrs from the first ``EchoData``
          object to the result.
        * ``"drop"``: empty attrs on returned ``EchoData`` object.
        * ``"identical"``: all attrs must be the same on every object.
        * ``"no_conflicts"``: attrs from all objects are combined,
          any that have the same name must also have the same value.
        * ``"overwrite_conflicts"``: attrs from all ``EchoData`` objects are combined,
          attrs with conflicting keys will be overwritten by later ``EchoData`` objects.
    in_memory : bool
        If True, creates an in-memory form of the combined ``EchoData`` object, otherwise
        a lazy ``EchoData`` object will be created (not currently implemented).

    Returns
    -------
    EchoData
        An ``EchoData`` object with all data from the input ``EchoData`` objects combined.

    Raises
    ------
    ValueError
        If ``echodatas`` contains ``EchoData`` objects with different or ``None``
        ``sonar_model`` values (i.e., all `EchoData` objects must have the same
        non-None ``sonar_model`` value).
    ValueError
        If EchoData objects have conflicting source file names.

    Warns
    -----
    UserWarning
        If the ``sonar_model`` of the input ``EchoData`` objects is ``"EK60"`` and any
        ``EchoData`` objects have non-monotonically increasing ``ping_time``, ``time1``
        or ``time2`` values, the corresponding values in the output ``EchoData`` object
        will be increased starting at the timestamp where the reversal occurs such that
        all values in the output are monotonically increasing. Additionally, the original
        ``ping_time``, ``time1`` or ``time2`` values will be stored in the ``Provenance``
        group, although this behavior may change in future versions.

    Warnings
    --------
    Changes in parameters between ``EchoData`` objects are not currently checked;
    however, they may raise an error in future versions.

    Notes
    -----
    * ``EchoData`` objects are combined by combining their groups individually.
    * Attributes from all groups before the combination will be stored in the provenance group,
      although this behavior may change in future versions.
    * The ``source_file`` and ``converted_raw_path`` attributes will be copied from the first
      ``EchoData`` object in the given list, but this may change in future versions.

    Examples
    --------
    >>> ed1 = echopype.open_converted("file1.nc")
    >>> ed2 = echopype.open_converted("file2.zarr")
    >>> combined = echopype.combine_echodata([ed1, ed2])
    """

    if zarr_store is None:
        zarr_store = "/Users/brandonreyes/UW_work/Echopype_work/code_playing_around/test.zarr"
        raise RuntimeError("You need to provide a path!")  # TODO: use Don's path

    if not isinstance(echodatas, list):
        raise TypeError("The input, eds, must be a list of EchoData objects!")

    if not isinstance(zarr_store, str):  # TODO: change this in the future
        raise TypeError("The input, store, must be a string!")

    # return empty EchoData object, if no EchoData objects are provided
    if not echodatas:
        warn("No EchoData objects were provided, returning an empty EchoData object.")
        return EchoData()

    sonar_model, echodata_filenames = check_echodatas_input(echodatas)

    comb = ZarrCombine()
    ed_comb = comb.combine(
        zarr_store,
        echodatas,
        storage_options=storage_options,
        sonar_model=sonar_model,
        echodata_filenames=echodata_filenames,
    )

    # TODO: perform time check, put this in its own function
    for group in ed_comb.group_paths:

        if group != "Platform/NMEA":
            # Platform/NMEA is skipped because we found that the times correspond to other
            # messages besides GPS. This causes multiple times to be out of order and
            # correcting them is not possible with the current implementation of
            # _clean_ping_time in qc.api

            # get all time dimensions of the group
            ed_comb_time_dims = set(ed_comb[group].dims).intersection(comb.possible_time_dims)

            for time in ed_comb_time_dims:

                old_time = check_and_correct_reversed_time(
                    combined_group=ed_comb[group], time_str=time, ed_group=group
                )

                if old_time is not None:

                    # get name of old time and dim for Provenance group
                    ed_name = group.replace("-", "_").replace("/", "_").lower()
                    old_time_name = ed_name + "_old_" + time
                    old_time_name_dim = old_time_name + "_dim"

                    # put old times in Provenance and modify attribute
                    # TODO: should we give old time a long name?
                    old_time_array = xr.DataArray(data=old_time.values, dims=[old_time_name_dim])
                    ed_comb["Provenance"][old_time_name] = old_time_array
                    ed_comb["Provenance"].attrs["reversed_ping_times"] = 1

                    # TODO: save new time and old time to zarr store

    return ed_comb
