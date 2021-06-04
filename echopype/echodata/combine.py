import warnings
from datetime import datetime
from typing import List

import xarray as xr
from _echopype_version import version as ECHOPYPE_VERSION

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
    prov_dict = {
        "conversion_software_name": "echopype",
        "conversion_software_version": ECHOPYPE_VERSION,
        "conversion_time": datetime.utcnow().isoformat(timespec="seconds")
        + "Z",  # use UTC time
        "src_filenames": input_paths,
    }
    ds = xr.Dataset()
    return ds.assign_attrs(prov_dict)


def _combine_echodata(echodatas: List[EchoData], sonar_model) -> EchoData:
    """
    Combines multiple echodata objects into a single EchoData by combining
        their groups individually.
    """

    result = EchoData()
    for group in EchoData.group_map:
        if group.casefold() in ("top", "sonar"):
            combined_group = getattr(echodatas[0], group)
        elif group.casefold() == "provenance":
            combined_group = assemble_combined_provenance(
                echodata.source_file for echodata in echodatas
            )
        else:
            group_datasets = [
                getattr(echodata, group)
                for echodata in echodatas
                if getattr(echodata, group) is not None
            ]
            group_datasets = union_attrs(group_datasets)

            if group.casefold() == "platform":
                if sonar_model == "AZFP":
                    concat_dim = [None]
                    data_vars = "all"
                elif sonar_model == "EK60":
                    concat_dim = ["location_time", "ping_time"]
                    data_vars = "minimal"
                elif sonar_model in ("EK80", "EA640"):
                    concat_dim = ["location_time", "mru_time"]
                    data_vars = "minimal"
                elif sonar_model == "AD2CP":
                    concat_dim = "ping_time"
                    data_vars = "minimal"
            elif group.casefold() == "nmea":
                concat_dim = "location_time"
                data_vars = "minimal"
            elif group.casefold() == "vendor":
                if sonar_model == "AZFP":
                    concat_dim = ["ping_time", "frequency"]
                    data_vars = "minimal"
                else:
                    concat_dim = [None]
                    data_vars = "minimal"
            else:
                # eg beam, environment, beam_complex
                concat_dim = "ping_time"
                data_vars = "minimal"
            combined_group = xr.combine_nested(
                group_datasets, concat_dim, data_vars=data_vars
            )

            if group.casefold() == "beam":
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

                    if exist_reversed_time(combined_group, "ping_time"):
                        warnings.warn(
                            "EK60 ping_time reversal detected; the ping times will be corrected"
                            " (see https://github.com/OSOceanAcoustics/echopype/pull/297)"
                        )
                        coerce_increasing_time(combined_group)

        setattr(result, group, combined_group)

    return result
