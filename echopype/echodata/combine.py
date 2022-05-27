import warnings
from pathlib import Path
from typing import Any, Dict, List

import xarray as xr
from datatree import DataTree

from ..core import SONAR_MODELS
from ..qc import coerce_increasing_time, exist_reversed_time
from ..utils.coding import set_encodings
from ..utils.prov import echopype_prov_attrs, source_files_vars
from .echodata import EchoData


def union_attrs(datasets: List[xr.Dataset]) -> Dict[str, Any]:
    """
    Merges attrs from a list of datasets.
    Prioritizes keys from later datasets.
    """

    total_attrs = dict()
    for ds in datasets:
        total_attrs.update(ds.attrs)
    return total_attrs


def assemble_combined_provenance(input_paths):
    return xr.Dataset(
        data_vars=source_files_vars(input_paths),
        attrs=echopype_prov_attrs(process_type="conversion"),
    )


def check_and_correct_reversed_time(combined_group, old_time, new_time, time_str, sonar_model):
    """
    Makes sure that the time coordinate ``time_str`` in
    ``combined_group`` is in the correct order and corrects
    it, if it is not.

    Parameters
    ----------
    combined_group : xr.Dataset
        Dataset representing a combined EchoData group
    old_time : xr.DataArray
        Time before reversal correction
    new_time : xr.DataArray
        Time after reversal correction
    time_str : str
        Name of time coordinate to be checked and corrected
    sonar_model : str
        Name of sonar model

    Returns
    -------
    Combined group with monotonically increasing ``time_str``
    coordinate, the time coordinate before correction, and
    the time coordinate after correction.
    """

    if time_str in combined_group and exist_reversed_time(combined_group, time_str):
        if old_time is None:
            warnings.warn(
                f"{sonar_model} {time_str} reversal detected; {time_str} will be corrected"  # noqa
                " (see https://github.com/OSOceanAcoustics/echopype/pull/297)"
            )
            old_time = combined_group[time_str]
            coerce_increasing_time(combined_group, time_name=time_str)
            new_time = combined_group[time_str]
        else:
            combined_group[time_str] = new_time

    return combined_group, old_time, new_time


def combine_echodata(echodatas: List[EchoData], combine_attrs="override") -> EchoData:
    """
    Combines multiple `EchoData` objects into a single `EchoData` object.

    Parameters
    ----------
    echodatas: List[EchoData]
        The list of `EchoData` objects to be combined.
    combine_attrs: { "override", "drop", "identical", "no_conflicts", "overwrite_conflicts" }
        String indicating how to combine attrs of the `EchoData` objects being merged.
        This parameter matches the identically named xarray parameter
        (see https://xarray.pydata.org/en/latest/generated/xarray.combine_nested.html)
        with the exception of the "overwrite_conflicts" value.

        * "override": Default. skip comparing and copy attrs from the first `EchoData`
          object to the result.
        * "drop": empty attrs on returned `EchoData` object.
        * "identical": all attrs must be the same on every object.
        * "no_conflicts": attrs from all objects are combined,
          any that have the same name must also have the same value.
        * "overwrite_conflicts": attrs from all `EchoData` objects are combined,
          attrs with conflicting keys will be overwritten by later `EchoData` objects.

    Returns
    -------
    EchoData
        An `EchoData` object with all of the data from the input `EchoData` objects combined.

    Raises
    ------
    ValueError
        If `echodatas` contains `EchoData` objects with different or `None` `sonar_model` values
        (i.e., all `EchoData` objects must have the same non-None `sonar_model` value).
    ValueError
        If EchoData objects have conflicting source file names.

    Warns
    -----
    UserWarning
        If the `sonar_model` of the input `EchoData` objects is `"EK60"` and any `EchoData` objects
        have non-monotonically increasing `ping_time`, `time1` or `time2` values,
        the corresponding values in the output `EchoData` object will be increased starting at the
        timestamp where the reversal occurs such that all values in the output are monotonically
        increasing. Additionally, the original `ping_time`, `time1` or `time2` values
        will be stored in the `Provenance` group, although this behavior may change in future
        versions.

    Warnings
    --------
    Changes in parameters between `EchoData` objects are not currently checked;
    however, they may raise an error in future versions.

    Notes
    -----
    * `EchoData` objects are combined by combining their groups individually.
    * Attributes from all groups before the combination will be stored in the provenance group,
      although this behavior may change in future versions.
    * The `source_file` and `converted_raw_path` attributes will be copied from the first
      `EchoData` object in the given list, but this may change in future versions.

    Examples
    --------
    >>> ed1 = echopype.open_converted("file1.nc")
    >>> ed2 = echopype.open_converted("file2.zarr")
    >>> combined = echopype.combine_echodata([ed1, ed2])
    """

    tree_dict = {}
    result = EchoData()
    if len(echodatas) == 0:
        return result
    result.source_file = echodatas[0].source_file
    result.converted_raw_path = echodatas[0].converted_raw_path

    sonar_model = None
    for echodata in echodatas:
        if echodata.sonar_model is None:
            raise ValueError("all EchoData objects must have non-None sonar_model values")
        elif sonar_model is None:
            sonar_model = echodata.sonar_model
        elif echodata.sonar_model != sonar_model:
            raise ValueError("all EchoData objects must have the same sonar_model value")

    # ping time before reversal correction
    old_ping_time = None
    # ping time after reversal correction
    new_ping_time = None
    # location time before reversal correction
    old_time1 = None
    # location time after reversal correction
    new_time1 = None
    # mru time before reversal correction
    old_time2 = None
    # mru time after reversal correction
    new_time2 = None
    # time3 before reversal correction
    old_time3 = None
    # time3 after reversal correction
    new_time3 = None

    # all attributes before combination
    # { group1: [echodata1 attrs, echodata2 attrs, ...], ... }
    old_attrs: Dict[str, List[Dict[str, Any]]] = dict()

    for group, value in EchoData.group_map.items():
        group_datasets = [
            getattr(echodata, group)
            for echodata in echodatas
            if getattr(echodata, group) is not None
        ]
        if group in ("top", "sonar"):
            combined_group = getattr(echodatas[0], group)
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
                setattr(result, group, None)
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

            if sonar_model != "AD2CP":

                combined_group, old_ping_time, new_ping_time = check_and_correct_reversed_time(
                    combined_group, old_ping_time, new_ping_time, "ping_time", sonar_model
                )

                if group != "nmea":
                    combined_group, old_time1, new_time1 = check_and_correct_reversed_time(
                        combined_group, old_time1, new_time1, "time1", sonar_model
                    )

                combined_group, old_time2, new_time2 = check_and_correct_reversed_time(
                    combined_group, old_time2, new_time2, "time2", sonar_model
                )

                combined_group, old_time3, new_time3 = check_and_correct_reversed_time(
                    combined_group, old_time3, new_time3, "time3", sonar_model
                )

        if len(group_datasets) > 1:
            old_attrs[group] = [group_dataset.attrs for group_dataset in group_datasets]
        if combined_group is not None:
            # xarray inserts this dimension when concatenating along multiple dimensions
            combined_group = combined_group.drop_dims("concat_dim", errors="ignore")

        combined_group = set_encodings(combined_group)
        if value["ep_group"] is None:
            tree_dict["root"] = combined_group
        else:
            tree_dict[value["ep_group"]] = combined_group

    # Set tree into echodata object
    result._set_tree(tree=DataTree.from_dict(tree_dict))
    result._load_tree()

    # save ping time before reversal correction
    if old_ping_time is not None:
        result.provenance["old_ping_time"] = old_ping_time
        result.provenance.attrs["reversed_ping_times"] = 1
    # save location time before reversal correction
    if old_time1 is not None:
        result.provenance["old_time1"] = old_time1
        result.provenance.attrs["reversed_ping_times"] = 1
    # save mru time before reversal correction
    if old_time2 is not None:
        result.provenance["old_time2"] = old_time2
        result.provenance.attrs["reversed_ping_times"] = 1
    # save time3 before reversal correction
    if old_time3 is not None:
        result.provenance["old_time3"] = old_time3
        result.provenance.attrs["reversed_ping_times"] = 1
    # TODO: possible parameter to disable original attributes and original ping_time storage
    # in provenance group?
    # save attrs from before combination
    for group in old_attrs:
        all_group_attrs = set()
        for group_attrs in old_attrs[group]:
            for attr in group_attrs:
                all_group_attrs.add(attr)
        echodata_filenames = []
        for ed in echodatas:
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
        result.provenance = result.provenance.assign({f"{group}_attrs": attrs})

    # Add back sonar model
    result.sonar_model = sonar_model

    return result
