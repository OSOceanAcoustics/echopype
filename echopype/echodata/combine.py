import warnings
from datetime import datetime
from typing import List

import xarray as xr
from _echopype_version import version as ECHOPYPE_VERSION

from ..core import SONAR_MODELS
from ..qc import coerce_increasing_time, exist_reversed_time
from .echodata import EchoData


def union_attrs(datasets: List[xr.Dataset]) -> List[xr.Dataset]:
    """
    Merges attrs from a list of datasets.
    Prioritizes keys from later datsets.
    """

    total_attrs = dict()
    for ds in datasets:
        total_attrs.update(ds.attrs)
    for ds in datasets:
        ds.attrs = total_attrs
    return datasets


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


def combine_echodata(echodatas: List[EchoData]) -> EchoData:
    """
    Combines multiple `EchoData` objects into a single `EchoData` object

    Parameters
    ----------
    echodatas: List[EchoData]
        The list of `EchoData` objects to be combined.

    Returns
    -------
    EchoData
        An `EchoData` object with all of the data from the input `EchoData` objects combined.

    Raises
    ------
    ValueError
        If `echodatas` contains `EchoData` objects with different or `None` `sonar_model` values
        (i.e., all `EchoData` objects must have the same non-None `sonar_model` value).

    Warns
    -----
    UserWarning
        If the `sonar_model` of the input `EchoData` objects is `"EK60"` and any `EchoData` objects
        have non-monotonically increasing `ping_time`s, the `ping_time`s in the output `EchoData`
        object will be increased starting at the timestamp where the reversal occurs such that all
        `ping_time`s in the output are monotonically increasing.

    Warnings
    --------
    Changes in parameters between `EchoData` objects are not currently checked;
    however, they may raise an error in future versions.

    Notes
    -----
    `EchoData` objects are combined by combining their groups individually.

    Examples
    --------
    >>> ed1 = echopype.open_converted("file1")
    >>> ed2 = echopype.open_converted("file2")
    >>> combined = echopype.combine_echodata([ed1, ed2])
    """

    result = EchoData()
    if len(echodatas) == 0:
        return result

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

    for group in EchoData.group_map:
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
            group_datasets = [
                getattr(echodata, group)
                for echodata in echodatas
                if getattr(echodata, group) is not None
            ]
            if len(group_datasets) == 0:
                setattr(result, group, None)
                continue
            group_datasets = union_attrs(group_datasets)

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
            )

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

            if "ping_time" in combined_group and exist_reversed_time(
                combined_group, "ping_time"
            ):
                if old_ping_time is None:
                    warnings.warn(
                        "EK60 ping_time reversal detected; the ping times will be corrected"
                        " (see https://github.com/OSOceanAcoustics/echopype/pull/297)"
                    )
                    old_ping_time = combined_group["ping_time"]
                    coerce_increasing_time(combined_group)
                    new_ping_time = combined_group["ping_time"]
                else:
                    combined_group["ping_time"] = new_ping_time

        if combined_group is not None:
            # xarray inserts this dimension when concating along multiple dimensions
            combined_group = combined_group.drop_dims("concat_dim", errors="ignore")
        setattr(result, group, combined_group)

    # save ping time before reversal correction
    if old_ping_time is not None:
        result.provenance["old_ping_time"] = old_ping_time

    return result
