from typing import List, Optional

import numpy as np
import xarray as xr

from .parse_ad2cp import Ad2cpDataPacket, Field, HeaderOrDataRecordFormats
from .set_groups_base import SetGroupsBase, set_encodings


def merge_attrs(datasets: List[xr.Dataset]) -> List[xr.Dataset]:
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


class SetGroupsAd2cp(SetGroupsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pulse_compressed = self.parser_obj.get_pulse_compressed()
        self.combine_packets()

    def combine_packets(self):
        self.ds = None

        # # TODO: where to put string data in output?

        # pad raw samples so that "sample" dimenion has same length
        max_samples = 0
        for packet in self.parser_obj.echosounder_raw_packets:
            # both _r and _i have same dimensions
            max_samples = max(
                max_samples, packet.data["echosounder_raw_samples_i"].shape[0]
            )
        for packet in self.parser_obj.echosounder_raw_packets:
            packet.data["echosounder_raw_samples_i"] = np.pad(
                packet.data["echosounder_raw_samples_i"],
                ((0, max_samples - packet.data["echosounder_raw_samples_i"].shape[0])),
            )
            packet.data["echosounder_raw_samples_q"] = np.pad(
                packet.data["echosounder_raw_samples_q"],
                ((0, max_samples - packet.data["echosounder_raw_samples_q"].shape[0])),
            )

        def make_dataset(
            packets: List[Ad2cpDataPacket], ping_time_dim: str
        ) -> Optional[xr.Dataset]:
            for i in range(len(packets)):
                packet = packets[i]
                data_vars = dict()
                for field_name, field_value in packet.data.items():
                    # add dimension names to data vars for xarray
                    # TODO might not work with altimeter_spare
                    field = HeaderOrDataRecordFormats.data_record_format(
                        packet.data_record_type
                    ).get_field(field_name)
                    if field is not None:
                        dims = field.dimensions(packet.data_record_type)
                        units = field.units()
                    else:
                        dims = Field.default_dimensions()
                        units = None
                    if units:
                        data_vars[field_name] = (
                            tuple(dim.value for dim in dims),
                            [field_value],
                            {"Units": units},
                        )
                    else:
                        data_vars[field_name] = (
                            tuple(dim.value for dim in dims),
                            [field_value],
                        )
                coords = {
                    "ping_time": [packet.timestamp],
                    ping_time_dim: [packet.timestamp],
                }
                if "beams" in packet.data_exclude:
                    coords["beam"] = packet.data_exclude["beams"]
                new_packet = xr.Dataset(data_vars=data_vars, coords=coords)

                # modify in place to reduce memory consumption
                packets[i] = new_packet
            if len(packets) > 0:
                packets = merge_attrs(packets)
                return xr.combine_by_coords(
                    packets,
                    data_vars="minimal",
                    coords="minimal",
                    combine_attrs="override",
                )
            else:
                return None

        burst_ds = make_dataset(
            self.parser_obj.burst_packets, ping_time_dim="ping_time_burst"
        )
        average_ds = make_dataset(
            self.parser_obj.average_packets, ping_time_dim="ping_time_average"
        )
        echosounder_ds = make_dataset(
            self.parser_obj.echosounder_packets, ping_time_dim="ping_time_echosounder"
        )
        echosounder_raw_ds = make_dataset(
            self.parser_obj.echosounder_raw_packets,
            ping_time_dim="ping_time_echosounder_raw",
        )
        echosounder_raw_transmit_ds = make_dataset(
            self.parser_obj.echosounder_raw_transmit_packets,
            ping_time_dim="ping_time_echosounder_raw_transmit",
        )

        datasets = [
            ds
            for ds in (
                burst_ds,
                average_ds,
                echosounder_ds,
                echosounder_raw_ds,
                echosounder_raw_transmit_ds,
            )
            if ds
        ]

        for dataset in datasets:
            if "offset_of_data" in dataset:
                print(dataset["offset_of_data"])

        datasets = merge_attrs(datasets)
        self.ds = xr.merge(datasets)

    def set_env(self) -> xr.Dataset:
        ds = xr.Dataset(
            data_vars={
                "sound_speed_indicative": self.ds.get("speed_of_sound"),
                "temperature": self.ds.get("temperature"),
                "pressure": self.ds.get("pressure"),
            },
            coords={
                "ping_time": self.ds.get("ping_time"),
                "ping_time_burst": self.ds.get("ping_time_burst", []),
                "ping_time_average": self.ds.get("ping_time_average", []),
                "ping_time_echosounder": self.ds.get("ping_time_echosounder", []),
            },
        )

        # FIXME: this is a hack because the current file saving
        # mechanism requires that the env group have ping_time as a dimension,
        # but ping_time might not be a dimension if the dataset is completely
        # empty
        if "ping_time" not in ds.dims:
            ds = ds.expand_dims(dim="ping_time")

        return set_encodings(ds)

    def set_platform(self) -> xr.Dataset:
        ds = xr.Dataset(
            data_vars={
                "heading": self.ds.get("heading"),
                "pitch": self.ds.get("pitch"),
                "roll": self.ds.get("roll"),
                "magnetometer_raw_x": self.ds.get("magnetometer_raw_x"),
                "magnetometer_raw_y": self.ds.get("magnetometer_raw_y"),
                "magnetometer_raw_z": self.ds.get("magnetometer_raw_z"),
            },
            coords={
                "ping_time": self.ds.get("ping_time"),
                "ping_time_burst": self.ds.get("ping_time_burst"),
                "ping_time_average": self.ds.get("ping_time_average"),
                "ping_time_echosounder": self.ds.get("ping_time_echosounder"),
                "beam": self.ds.get("beam"),
                "range_bin_burst": self.ds.get("range_bin_burst"),
                "range_bin_average": self.ds.get("range_bin_average"),
                "range_bin_echosounder": self.ds.get("range_bin_echosounder"),
            },
            attrs={
                "platform_name": self.ui_param["platform_name"],
                "platform_type": self.ui_param["platform_type"],
                "platform_code_ICES": self.ui_param["platform_code_ICES"],
            },
        )
        return set_encodings(ds)

    def set_beam(self) -> xr.Dataset:
        # TODO: should we divide beam into burst/average (e.g., beam_burst, beam_average)
        # like was done for range_bin (we have range_bin_burst, range_bin_average,
        # and range_bin_echosounder)?
        data_vars = {
            "number_of_beams": self.ds.get("num_beams"),
            "coordinate_system": self.ds.get("coordinate_system"),
            "number_of_cells": self.ds.get("num_cells"),
            "blanking": self.ds.get("blanking"),
            "cell_size": self.ds.get("cell_size"),
            "velocity_range": self.ds.get("velocity_range"),
            "echosounder_frequency": self.ds.get("echosounder_frequency"),
            "ambiguity_velocity": self.ds.get("ambiguity_velocity"),
            "data_set_description": self.ds.get("dataset_description"),
            "transmit_energy": self.ds.get("transmit_energy"),
            "velocity_scaling": self.ds.get("velocity_scaling"),
            "velocity_burst": self.ds.get("velocity_data_burst"),
            "velocity_average": self.ds.get("velocity_data_average"),
            # "velocity_echosounder": self.ds.get("velocity_data_echosounder"),
            "amplitude_burst": self.ds.get("amplitude_data_burst"),
            "amplitude_average": self.ds.get("amplitude_data_average"),
            # "amplitude_echosounder": self.ds.get("amplitude_data_echosounder"),
            "correlation_burst": self.ds.get("correlation_data_burst"),
            "correlation_average": self.ds.get("correlation_data_average"),
            "correlation_echosounder": self.ds.get("correlation_data_echosounder"),
            # "echosounder": self.ds.get("echosounder_data"),
            "amplitude_echosounder": self.ds.get("echosounder_data"),
            "figure_of_merit": self.ds.get("figure_of_merit_data"),
            "altimeter_distance": self.ds.get("altimeter_distance"),
            "altimeter_quality": self.ds.get("altimeter_quality"),
            "ast_distance": self.ds.get("ast_distance"),
            "ast_quality": self.ds.get("ast_quality"),
            "ast_offset_100us": self.ds.get("ast_offset_100us"),
            "ast_pressure": self.ds.get("ast_pressure"),
            "altimeter_spare": self.ds.get("altimeter_spare"),
            "altimeter_raw_data_num_samples": self.ds.get(
                "altimeter_raw_data_num_samples"
            ),
            "altimeter_raw_data_sample_distance": self.ds.get(
                "altimeter_raw_data_sample_distance"
            ),
            "altimeter_raw_data_samples": self.ds.get("altimeter_raw_data_samples"),
        }

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "ping_time": self.ds.get("ping_time"),
                "ping_time_burst": self.ds.get("ping_time_burst"),
                "ping_time_average": self.ds.get("ping_time_average"),
                "ping_time_echosounder": self.ds.get("ping_time_echosounder"),
                "beam": self.ds.get("beam"),
                "range_bin_burst": self.ds.get("range_bin_burst"),
                "range_bin_average": self.ds.get("range_bin_average"),
                "range_bin_echosounder": self.ds.get("range_bin_echosounder"),
                "altimeter_sample_bin": self.ds.get("altimeter_sample_bin"),
            },
            attrs={"pulse_compressed": self.pulse_compressed},
        )

        # FIXME: this is a hack because the current file saving
        # mechanism requires that the beam group have ping_time as a dimension,
        # but ping_time might not be a dimension if the dataset is completely
        # empty
        if "ping_time" not in ds.dims:
            ds = ds.expand_dims(dim="ping_time")

        return set_encodings(ds)

    def set_vendor(self) -> xr.Dataset:
        attrs = {
            "pressure_sensor_valid": self.ds.get("pressure_sensor_valid"),
            "temperature_sensor_valid": self.ds.get("temperature_sensor_valid"),
            "compass_sensor_valid": self.ds.get("compass_sensor_valid"),
            "tilt_sensor_valid": self.ds.get("tilt_sensor_valid"),
        }
        attrs = {
            field_name: field_value.data[0]
            for field_name, field_value in attrs.items()
            if field_value is not None
        }
        ds = xr.Dataset(
            data_vars={
                "data_record_version": self.ds.get("version"),
                "error": self.ds.get("error"),
                "status": self.ds.get("status"),
                "status0": self.ds.get("status0"),
                "battery_voltage": self.ds.get("battery_voltage"),
                "power_level": self.ds.get("power_level"),
                "temperature_of_pressure_sensor": self.ds.get(
                    "temperature_from_pressure_sensor"
                ),
                "nominal_correlation": self.ds.get("nominal_correlation"),
                "magnetometer_temperature": self.ds.get("magnetometer_temperature"),
                "real_ping_time_clock_temperature": self.ds.get(
                    "real_ping_time_clock_temperature"
                ),
                "ensemble_counter": self.ds.get("ensemble_counter"),
                "ahrs_rotation_matrix_mij": (
                    ("mij", "ping_time")
                    if "ahrs_rotation_matrix_m11" in self.ds
                    else "mij",
                    [
                        self.ds.get("ahrs_rotation_matrix_m11"),
                        self.ds.get("ahrs_rotation_matrix_m12"),
                        self.ds.get("ahrs_rotation_matrix_m13"),
                        self.ds.get("ahrs_rotation_matrix_m21"),
                        self.ds.get("ahrs_rotation_matrix_m22"),
                        self.ds.get("ahrs_rotation_matrix_m23"),
                        self.ds.get("ahrs_rotation_matrix_m31"),
                        self.ds.get("ahrs_rotation_matrix_m32"),
                        self.ds.get("ahrs_rotation_matrix_m33"),
                    ],
                ),
                "ahrs_quaternions_wxyz": (
                    ("wxyz", "ping_time")
                    if "ahrs_quaternions_w" in self.ds
                    else "wxyz",
                    [
                        self.ds.get("ahrs_quaternions_w"),
                        self.ds.get("ahrs_quaternions_x"),
                        self.ds.get("ahrs_quaternions_y"),
                        self.ds.get("ahrs_quaternions_z"),
                    ],
                ),
                "ahrs_gyro_xyz": (
                    ("xyz", "ping_time") if "ahrs_gyro_x" in self.ds else "xyz",
                    [
                        self.ds.get("ahrs_gyro_x"),
                        self.ds.get("ahrs_gyro_y"),
                        self.ds.get("ahrs_gyro_z"),
                    ],
                ),
                "percentage_good_data": self.ds.get("percentage_good_data"),
                "std_dev_pitch": self.ds.get("std_dev_pitch"),
                "std_dev_roll": self.ds.get("std_dev_roll"),
                "std_dev_heading": self.ds.get("std_dev_heading"),
                "std_dev_pressure": self.ds.get("std_dev_pressure"),
                "echosounder_raw_samples_i": self.ds.get("echosounder_raw_samples_i"),
                "echosounder_raw_samples_q": self.ds.get("echosounder_raw_samples_q"),
                "echosounder_raw_transmit_samples_i": self.ds.get(
                    "echosounder_raw_transmit_samples_i"
                ),
                "echosounder_raw_transmit_samples_q": self.ds.get(
                    "echosounder_raw_transmit_samples_q"
                ),
                "echosounder_raw_beam": self.ds.get("echosounder_raw_beam"),
                "echosounder_raw_echogram": self.ds.get("echosounder_raw_echogram"),
            },
            coords={
                "ping_time": self.ds.get("ping_time"),
                "ping_time_burst": self.ds.get("ping_time_burst"),
                "ping_time_average": self.ds.get("ping_time_average"),
                "ping_time_echosounder": self.ds.get("ping_time_echosounder"),
                "ping_time_echosounder_raw": self.ds.get("ping_time_echosounder_raw"),
                "ping_time_echosounder_raw_transmit": self.ds.get(
                    "ping_time_echosounder_raw_transmit"
                ),
                "sample": self.ds.get("sample"),
                "sample_transmit": self.ds.get("sample_transmit"),
                "beam": self.ds.get("beam"),
                "range_bin_average": self.ds.get("range_bin_average"),
                "range_bin_burst": self.ds.get("range_bin_burst"),
                "range_bin_echosounder": self.ds.get("range_bin_echosounder"),
            },
            attrs={**attrs, "pulse_compressed": self.pulse_compressed},
        )
        ds = ds.reindex(
            {
                "mij": np.array(["11", "12", "13", "21", "22", "23", "31", "32", "33"]),
                "wxyz": np.array(["w", "x", "y", "z"]),
                "xyz": np.array(["x", "y", "z"]),
            }
        )

        # FIXME: this is a hack because the current file saving
        # mechanism requires that the vendor group have ping_time as a dimension,
        # but ping_time might not be a dimension if the dataset is completely
        # empty
        if "ping_time" not in ds.dims:
            ds = ds.expand_dims(dim="ping_time")

        return set_encodings(ds)

    def set_sonar(self) -> xr.Dataset:
        ds = xr.Dataset(
            attrs={
                "sonar_manufacturer": "Nortek",
                "sonar_model": "AD2CP",
                "sonar_serial_number": "",
                "sonar_software_name": "",
                "sonar_software_version": "",
                "sonar_firmware_version": "",
                "sonar_type": "acoustic Doppler current profiler (ADCP)",
            }
        )
        if "serial_number" in self.ds:
            ds.attrs["sonar_serial_number"] = int(self.ds["serial_number"].data[0])
        firmware_version = self.parser_obj.get_firmware_version()
        if firmware_version is not None:
            ds.attrs["sonar_firmware_version"] = ", ".join(
                [f"{k}:{v}" for k, v in firmware_version.items()]
            )
        return ds
