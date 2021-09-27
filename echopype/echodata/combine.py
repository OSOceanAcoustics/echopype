import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import xarray as xr
from _echopype_version import version as ECHOPYPE_VERSION

from ..core import SONAR_MODELS
from ..qc import coerce_increasing_time, exist_reversed_time
from .echodata import EchoData


def union_attrs(datasets: List[xr.Dataset]) -> Dict[str, Any]:
    """
    Merges attrs from a list of datasets.
    Prioritizes keys from later datsets.
    """

    total_attrs = dict()
    for ds in datasets:
        total_attrs.update(ds.attrs)
    return total_attrs


def assemble_combined_provenance(input_paths):
    return xr.Dataset(
        data_vars={
            "src_filenames": ("file", input_paths),
        },
        attrs={
            "conversion_software_name": "echopype",
            "conversion_software_version": ECHOPYPE_VERSION,
            "conversion_time": datetime.utcnow().isoformat(timespec="seconds")
            + "Z",  # use UTC time
        },
    )


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
        have non-monotonically increasing `ping_time`, `location_time` or `mru_time` values,
        the corresponding values in the output `EchoData` object will be increased starting at the
        timestamp where the reversal occurs such that all values in the output are monotonically
        increasing. Additionally, the original `ping_time`, `location_time` or `mru_time` values
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

    result = EchoData()
    if len(echodatas) == 0:
        return result
    result.source_file = echodatas[0].source_file
    result.converted_raw_path = echodatas[0].converted_raw_path

    sonar_model = None
    for echodata in echodatas:
        if echodata.sonar_model is None:
            raise ValueError(
                "all EchoData objects must have non-None sonar_model values"
            )
        elif sonar_model is None:
            sonar_model = echodata.sonar_model
        elif echodata.sonar_model != sonar_model:
            raise ValueError(
                "all EchoData objects must have the same sonar_model value"
            )

    # ping time before reversal correction
    old_ping_time = None
    # ping time after reversal correction
    new_ping_time = None
    # location time before reversal correction
    old_location_time = None
    # location time after reversal correction
    new_location_time = None
    # mru time before reversal correction
    old_mru_time = None
    # mru time after reversal correction
    new_mru_time = None

    # all attributes before combination
    # { group1: [echodata1 attrs, echodata2 attrs, ...], ... }
    old_attrs: Dict[str, List[Dict[str, Any]]] = dict()

    for group in EchoData.group_map:
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
                combine_attrs="drop"
                if combine_attrs == "overwrite_conflicts"
                else combine_attrs,
            )
            if combine_attrs == "overwrite_conflicts":
                combined_group.attrs.update(union_attrs(group_datasets))

            if group == "beam":
                if sonar_model == "EK80":
                    combined_group["transceiver_software_version"] = combined_group[
                        "transceiver_software_version"
                    ].astype("<U10")
                    combined_group["channel_id"] = combined_group["channel_id"].astype(
                        "<U50"
                    )
                elif sonar_model == "EK60":
                    combined_group["gpt_software_version"] = combined_group[
                        "gpt_software_version"
                    ].astype("<U10")
                    combined_group["channel_id"] = combined_group["channel_id"].astype(
                        "<U50"
                    )

            if sonar_model in ("EK60", "EK80"):
                if "ping_time" in combined_group and exist_reversed_time(
                    combined_group, "ping_time"
                ):
                    if old_ping_time is None:
                        warnings.warn(
                            f"{sonar_model} ping_time reversal detected; the ping times will be corrected"  # noqa
                            " (see https://github.com/OSOceanAcoustics/echopype/pull/297)"
                        )
                        old_ping_time = combined_group["ping_time"]
                        coerce_increasing_time(combined_group, time_name="ping_time")
                        new_ping_time = combined_group["ping_time"]
                    else:
                        combined_group["ping_time"] = new_ping_time
                if "location_time" in combined_group and exist_reversed_time(
                    combined_group, "location_time"
                ):
                    if group != "nmea":
                        if old_location_time is None:
                            warnings.warn(
                                f"{sonar_model} location_time reversal detected; the location times will be corrected"  # noqa
                                " (see https://github.com/OSOceanAcoustics/echopype/pull/297)"
                            )
                            old_location_time = combined_group["location_time"]
                            coerce_increasing_time(
                                combined_group, time_name="location_time"
                            )
                            new_location_time = combined_group["location_time"]
                        else:
                            combined_group["location_time"] = new_location_time
            if sonar_model == "EK80":
                if "mru_time" in combined_group and exist_reversed_time(
                    combined_group, "mru_time"
                ):
                    if old_mru_time is None:
                        warnings.warn(
                            f"{sonar_model} mru_time reversal detected; the mru times will be corrected"  # noqa
                            " (see https://github.com/OSOceanAcoustics/echopype/pull/297)"
                        )
                        old_mru_time = combined_group["mru_time"]
                        coerce_increasing_time(combined_group, time_name="mru_time")
                        new_mru_time = combined_group["mru_time"]
                    else:
                        combined_group["mru_time"] = new_mru_time

        if len(group_datasets) > 1:
            old_attrs[group] = [group_dataset.attrs for group_dataset in group_datasets]
        if combined_group is not None:
            # xarray inserts this dimension when concating along multiple dimensions
            combined_group = combined_group.drop_dims("concat_dim", errors="ignore")
        setattr(result, group, combined_group)

    # save ping time before reversal correction
    if old_ping_time is not None:
        result.provenance["old_ping_time"] = old_ping_time
        result.provenance.attrs["reversed_ping_times"] = 1
    # save location time before reversal correction
    if old_location_time is not None:
        result.provenance["old_location_time"] = old_location_time
        result.provenance.attrs["reversed_ping_times"] = 1
    # save mru time before reversal correction
    if old_mru_time is not None:
        result.provenance["old_mru_time"] = old_mru_time
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
