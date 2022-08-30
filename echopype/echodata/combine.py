from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import xarray as xr
from datatree import DataTree

from ..core import SONAR_MODELS
from ..qc import coerce_increasing_time, exist_reversed_time
from ..utils.coding import set_encodings
from ..utils.log import _init_logger
from ..utils.prov import echopype_prov_attrs, source_files_vars
from .echodata import EchoData

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
    combined_group: xr.Dataset, time_str: str, sonar_model: str
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
    sonar_model : str
        Name of sonar model

    Returns
    -------
    old_time : Optional[xr.DataArray]
        If correction is necessary, returns the time before
        reversal correction, otherwise returns None
    """

    if time_str in combined_group and exist_reversed_time(combined_group, time_str):

        logger.warning(
            f"{sonar_model} {time_str} reversal detected; {time_str} will be corrected"  # noqa
            " (see https://github.com/OSOceanAcoustics/echopype/pull/297)"
        )
        old_time = combined_group[time_str]
        coerce_increasing_time(combined_group, time_name=time_str)

    else:
        old_time = None

    return old_time


def assemble_combined_provenance(input_paths):
    return xr.Dataset(
        data_vars=source_files_vars(input_paths),
        attrs=echopype_prov_attrs(process_type="conversion"),
    )


def union_attrs(datasets: List[xr.Dataset]) -> Dict[str, Any]:
    """
    Merges attrs from a list of datasets.
    Prioritizes keys from later datasets.
    """

    total_attrs = dict()
    for ds in datasets:
        total_attrs.update(ds.attrs)
    return total_attrs


def examine_group_time_coords(
    combined_group: xr.Dataset,
    group: str,
    sonar_model: str,
    old_times: Dict[str, Optional[xr.DataArray]],
) -> None:
    """
    Ensures that the time coords for each group are in the
    correct order.

    Parameters
    ----------
    combined_group: xr.Dataset
        Dataset representing a combined ``EchoData`` group
    group: str
        Group name of ``combined_group`` obtained from ``EchoData.group_map``
    sonar_model: str
        Name of sonar model
    old_times: Dict[str, Optional[xr.DataArray]]
        Dictionary that holds times before they are corrected

    Notes
    -----
    If old time coordinates need to be stored, the input variable ``old_times``
    will be directly modified.

    This does not check the AD2CP time coordinates!
    """

    if sonar_model != "AD2CP":

        old_times["old_ping_time"] = check_and_correct_reversed_time(
            combined_group, "ping_time", sonar_model
        )

        if group != "nmea":
            old_times["old_time1"] = check_and_correct_reversed_time(
                combined_group, "time1", sonar_model
            )

        old_times["old_time2"] = check_and_correct_reversed_time(
            combined_group, "time2", sonar_model
        )
        old_times["old_time3"] = check_and_correct_reversed_time(
            combined_group, "time3", sonar_model
        )


def store_old_attrs(
    result: EchoData,
    old_attrs: Dict[str, List[Dict[str, Any]]],
    echodata_filenames: List[str],
    sonar_model: str,
) -> None:
    """
    Stores all attributes of the groups in ``echodatas`` before
    they were combined in the ``Provenance`` group of ``result``
    and specifies the sonar model of the new combined data.

    Parameters
    ----------
    result: EchoData
        The final ``EchoData`` object representing the combined data
    old_attrs: Dict[str, List[Dict[str, Any]]]
        All attributes before combination
    echodata_filenames : List[str]
        The source files names for all values in ``echodatas``
    sonar_model : str
        The sonar model used for all values in ``echodatas``

    Notes
    -----
    The input ``result`` will be directly modified.
    """

    # store all old attributes
    for group in old_attrs:
        all_group_attrs = set()
        for group_attrs in old_attrs[group]:
            for attr in group_attrs:
                all_group_attrs.add(attr)
        attrs = xr.DataArray(
            [
                [group_attrs.get(attr) for attr in all_group_attrs]
                for group_attrs in old_attrs[group]
            ],
            coords={
                "echodata_filename": echodata_filenames,
                f"{group}_attr_key": list(all_group_attrs),
            },
            dims=["echodata_filename", f"{group}_attr_key"],
        )
        result["Provenance"] = result["Provenance"].assign({f"{group}_attrs": attrs})

    # Add back sonar model
    result.sonar_model = sonar_model


def in_memory_combine(
    echodatas: List[EchoData],
    sonar_model: str,
    combine_attrs: str,
    old_attrs: Dict[str, List[Dict[str, Any]]],
    old_times: Dict[str, Optional[xr.DataArray]],
) -> EchoData:
    """
    Creates an in-memory (i.e. in RAM) combined ``EchoData``
    object from the values in ``echodatas``.

    Parameters
    ----------
    echodatas : List[EchoData]
        The list of ``EchoData`` objects to be combined.
    sonar_model: str
        The sonar model used for all values in ``echodatas``
    combine_attrs : str
        String indicating how to combine attrs of the ``EchoData`` objects being merged.
    old_attrs: Dict[str, List[Dict[str, Any]]]
        All attributes before combination
    old_times: Dict[str, Optional[xr.DataArray]]
        Dictionary that holds times before they are corrected

    Returns
    -------
    result : EchoData
        An in-memory ``EchoData`` object with all data from the input
        ``EchoData`` objects combined.

    Notes
    -----
    If necessary, the input variables ``old_attrs`` and ``old_times``
    will be directly modified.
    """

    # initialize EchoData object and tree that will store the final result
    tree_dict = {}
    result = EchoData()

    # assign EchoData class variables
    result.source_file = echodatas[0].source_file
    result.converted_raw_path = echodatas[0].converted_raw_path

    # Specification for Echodata.group_map can be found in
    # echopype/echodata/convention/1.0.yml
    for group, value in EchoData.group_map.items():
        group_datasets = []
        group_path = value["ep_group"]
        if group_path is None:
            group_path = "Top-level"

        for echodata in echodatas:
            if echodata[group_path] is not None:
                group_datasets.append(echodata[group_path])

        if group in ("top", "sonar"):
            combined_group = echodatas[0][group_path]
        elif group == "provenance":
            combined_group = assemble_combined_provenance(
                [
                    echodata.source_file
                    if echodata.source_file is not None
                    else echodata.converted_raw_path
                    for echodata in echodatas
                ]
            )
        else:
            if len(group_datasets) == 0:
                continue

            concat_dim = SONAR_MODELS[sonar_model]["concat_dims"].get(
                group, SONAR_MODELS[sonar_model]["concat_dims"]["default"]
            )
            concat_data_vars = SONAR_MODELS[sonar_model]["concat_data_vars"].get(
                group, SONAR_MODELS[sonar_model]["concat_data_vars"]["default"]
            )
            combined_group = xr.combine_nested(
                group_datasets,
                [concat_dim],
                data_vars=concat_data_vars,
                coords="minimal",
                combine_attrs="drop" if combine_attrs == "overwrite_conflicts" else combine_attrs,
            )
            if combine_attrs == "overwrite_conflicts":
                combined_group.attrs.update(union_attrs(group_datasets))

            if group == "beam":
                if sonar_model == "EK80":
                    combined_group["transceiver_software_version"] = combined_group[
                        "transceiver_software_version"
                    ].astype("<U10")
                    combined_group["channel"] = combined_group["channel"].astype("<U50")
                elif sonar_model == "EK60":
                    combined_group["gpt_software_version"] = combined_group[
                        "gpt_software_version"
                    ].astype("<U10")

                    # TODO: investigate further why we need to do .astype("<U50")
                    combined_group["channel"] = combined_group["channel"].astype("<U50")

            examine_group_time_coords(combined_group, group, sonar_model, old_times)

        if len(group_datasets) > 1:
            old_attrs[group] = [group_dataset.attrs for group_dataset in group_datasets]
        if combined_group is not None:
            # xarray inserts this dimension when concatenating along multiple dimensions
            combined_group = combined_group.drop_dims("concat_dim", errors="ignore")

        combined_group = set_encodings(combined_group)
        if value["ep_group"] is None:
            tree_dict["/"] = combined_group
        else:
            tree_dict[value["ep_group"]] = combined_group

    # Set tree into echodata object
    result._set_tree(tree=DataTree.from_dict(tree_dict, name="root"))
    result._load_tree()

    return result


def combine_echodata(
    echodatas: List[EchoData], combine_attrs: str = "override", in_memory: bool = True
) -> EchoData:
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

    if len(echodatas) == 0:
        return EchoData()

    sonar_model, echodata_filenames = check_echodatas_input(echodatas)

    # all attributes before combination
    # { group1: [echodata1 attrs, echodata2 attrs, ...], ... }
    old_attrs: Dict[str, List[Dict[str, Any]]] = dict()

    # dict that holds times before they are corrected
    old_times: Dict[str, Optional[xr.DataArray]] = {
        "old_ping_time": None,
        "old_time1": None,
        "old_time2": None,
        "old_time3": None,
    }

    if in_memory:
        result = in_memory_combine(echodatas, sonar_model, combine_attrs, old_attrs, old_times)
    else:
        raise NotImplementedError(
            "Lazy representation of combined EchoData object has not been implemented yet."
        )

    # save times before reversal correction
    for key, val in old_times.items():
        if val is not None:
            result["Provenance"][key] = val
            result["Provenance"].attrs["reversed_ping_times"] = 1

    # save attrs from before combination
    store_old_attrs(result, old_attrs, echodata_filenames, sonar_model)

    # TODO: possible parameter to disable original attributes and original ping_time storage
    #  in provenance group?

    return result
