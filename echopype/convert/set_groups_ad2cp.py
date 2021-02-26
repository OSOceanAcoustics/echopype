import os
from typing import List, Optional

import xarray as xr
import numpy as np

from .set_groups_base import SetGroupsBase
from .parse_ad2cp import DataRecordType, Field, Ad2cpDataPacket
from ..utils import io


class SetGroupsAd2cp(SetGroupsBase):
    def write(self, ds: xr.Dataset, group: str):
        ds.attrs["Units"] = "velocity: m/s, temperature: deg Celsius, pressure: dBar, direction: degrees, length: m, voltage: V, magnitude: dB, echosounder data: dB/count"
        if os.path.exists(self.output_path):
            mode = "a"
        else:
            mode = "w"
        io.save_file(ds=ds, path=self.output_path, mode=mode, engine=self.engine, group=group)

    def save(self):
        self.combine_packets()
        self.set_toplevel("AD2CP", date_created=np.datetime64("now"))
        self.set_environment()
        self.set_platform()
        self.set_beam()
        self.set_vendor_specific()
        self.set_provenance()

    def combine_packets(self):
        self.ds = None
        string_data = dict()

        burst_packets = []
        average_packets = []
        echosounder_packets = []
        for packet in self.parser_obj.packets:
            if packet.data_record_type in (DataRecordType.BURST_VERSION2, DataRecordType.BURST_VERSION3):
                burst_packets.append(packet)
            elif packet.data_record_type in (DataRecordType.AVERAGE_VERSION2, DataRecordType.AVERAGE_VERSION3):
                average_packets.append(packet)
            elif packet.data_record_type == DataRecordType.ECHOSOUNDER:
                echosounder_packets.append(packet)
        
        def make_dataset(packets: List[Ad2cpDataPacket], time_dim: str) -> Optional[xr.Dataset]:
            for i in range(len(packets)):
                packet = packets[i]
                data_vars = dict()
                for field_name, field_value in packet.data.items():
                    # TODO might not work with altimeter_spare
                    # add dimension names to data vars for xarray
                    dims = Field.dimensions(field_name, packet.data_record_type)
                    if field_name in ("velocity_data", "amplitude_data", "correlation_data", "percentage_good_data"):
                        if packet.data_record_type in (DataRecordType.BURST_VERSION2, DataRecordType.BURST_VERSION3):
                            field_name += "_burst"
                        elif packet.data_record_type in (DataRecordType.AVERAGE_VERSION2, DataRecordType.AVERAGE_VERSION3):
                            field_name += "_average"
                    data_vars[field_name] = (tuple(dim.value for dim in dims), [field_value])
                new_packet = xr.Dataset(
                    data_vars=data_vars,
                    coords={
                        "time": [packet.timestamp],
                        time_dim: [packet.timestamp],
                        "beam": [b for b in [packet.data["beam0"], packet.data["beam1"], packet.data["beam2"], packet.data["beam3"], packet.data["beam4"]] if b > 0],
                        "complex": ["real", "imag"],
                    }
                )

                # modify in place to reduce memory consumption
                packets[i] = new_packet
            if len(packets) > 0:
                return xr.concat(packets, dim="time")
            else:
                return None

        burst_ds = make_dataset(burst_packets, time_dim="time_burst")
        average_ds = make_dataset(average_packets, time_dim="time_average")
        echosounder_ds = make_dataset(echosounder_packets, time_dim="time_echosounder")

        datasets = [ds for ds in (burst_ds, average_ds, echosounder_ds) if ds]
        self.ds = xr.merge(datasets)
        # TODO: where to put string data in output?
        self.ds.attrs["string_data"] = string_data

    def set_environment(self):
        ds = xr.Dataset(data_vars={
            "sound_speed_indicative": self.ds.get("speed_of_sound"),
            "temperature": self.ds.get("temperature"),
            "pressure": self.ds.get("pressure")
        },
            coords={
            "time_burst": self.ds.get("time_burst", []),
            "time_average": self.ds.get("time_average", []),
            "time_echosounder": self.ds.get("time_echosounder", [])
        })
        self.write(ds, "Environment")

    def set_platform(self):
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
                "time_burst": self.ds.get("time_burst"),
                "time_average": self.ds.get("time_average"),
                "time_echosounder": self.ds.get("time_echosounder"),
                "beam": self.ds.get("beam"),
                "range_bin_burst": self.ds.get("range_bin_burst"),
                "range_bin_average": self.ds.get("range_bin_average"),
                "range_bin_echosounder": self.ds.get("range_bin_echosounder")
            },
            attrs={
                "platform_name": self.ui_param["platform_name"],
                "platform_type": self.ui_param["platform_type"],
                "platform_code_ICES": self.ui_param["platform_code_ICES"]
            }
        )
        self.write(ds, "Platform")

    def set_beam(self):
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
            "velocity_echosounder": self.ds.get("velocity_data_echosounder"),
            "amplitude_burst": self.ds.get("amplitude_data_burst"),
            "amplitude_average": self.ds.get("amplitude_data_average"),
            "amplitude_echosounder": self.ds.get("amplitude_data_echosounder"),
            "correlation_burst": self.ds.get("correlation_data_burst"),
            "correlation_average": self.ds.get("correlation_data_average"),
            "correlation_echosounder": self.ds.get("correlation_data_echosounder"),
            "echosounder": self.ds.get("echosounder_data"),
            "figure_of_merit": self.ds.get("figure_of_merit_data"),
            "altimeter_distance": self.ds.get("altimeter_distance"),
            "altimeter_quality": self.ds.get("altimeter_quality"),
            "ast_distance": self.ds.get("ast_distance"),
            "ast_quality": self.ds.get("ast_quality"),
            "ast_offset_10us": self.ds.get("ast_offset_10us"),
            "ast_pressure": self.ds.get("ast_pressure"),
            "altimeter_spare": self.ds.get("altimeter_spare"),
            "altimeter_raw_data_num_samples": self.ds.get("altimeter_raw_data_num_samples"),
            "altimeter_raw_data_sample_distance": self.ds.get("altimeter_raw_data_sample_distance"),
            "altimeter_raw_data_samples": self.ds.get("altimeter_raw_data_samples")
        }

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time_burst": self.ds.get("time_burst"),
                "time_average": self.ds.get("time_average"),
                "time_echosounder": self.ds.get("time_echosounder"),
                "beam": self.ds.get("beam"),
                "range_bin_burst": self.ds.get("range_bin_burst"),
                "range_bin_average": self.ds.get("range_bin_average"),
                "range_bin_echosounder": self.ds.get("range_bin_echosounder"),
                "altimeter_sample_bin": self.ds.get("altimeter_sample_bin"),
            }
        )
        self.write(ds, "Beam")

    def set_vendor_specific(self):
        attrs = {
            # TODO: offset of data is only relevant to the raw data format
            # (says how many bytes the data is from the header)
            "offset_of_data": self.ds.get("offset_of_data"),
            "pressure_sensor_valid": self.ds.get("pressure_sensor_valid"),
            "temperature_sensor_valid": self.ds.get("temperature_sensor_valid"),
            "compass_sensor_valid": self.ds.get("compass_sensor_valid"),
            "tilt_sensor_valid": self.ds.get("tilt_sensor_valid"),
            "velocity_data_included": self.ds.get("velocity_data_included"),
            "amplitude_data_included": self.ds.get("amplitude_data_included"),
            "correlation_data_included": self.ds.get("correlation_data_included"),
            "altimeter_data_included": self.ds.get("altimeter_data_included"),
            "altimeter_raw_data_included": self.ds.get("altimeter_raw_data_included"),
            "ast_data_included": self.ds.get("ast_data_included"),
            "echosounder_data_included": self.ds.get("echosounder_data_included"),
            "ahrs_data_included": self.ds.get("ahrs_data_included"),
            "percentage_good_data_included": self.ds.get("percentage_good_data_included"),
            "std_dev_data_included": self.ds.get("std_dev_data_included"),
            "figure_of_merit_data_included": self.ds.get("figure_of_merit_data_included"),
        }
        attrs = {field_name: field_value.data[0] for field_name, field_value in attrs.items(
        ) if field_value is not None}
        ds = xr.Dataset(data_vars={
            "data_record_version": self.ds.get("version"),
            "error": self.ds.get("error"),
            "status": self.ds.get("status"),
            "status0": self.ds.get("status0"),
            "battery_voltage": self.ds.get("battery_voltage"),
            "power_level": self.ds.get("power_level"),
            "temperature_of_pressure_sensor": self.ds.get("temperature_from_pressure_sensor"),
            "nominal_correlation": self.ds.get("nominal_correlation"),
            "magnetometer_temperature": self.ds.get("magnetometer_temperature"),
            "real_time_clock_temperature": self.ds.get("real_time_clock_temperature"),
            "ensemble_counter": self.ds.get("ensemble_counter"),
            "ahrs_rotation_matrix_mij": ("mij", [
                self.ds.get("ahrs_rotation_matrix_m11"),
                self.ds.get("ahrs_rotation_matrix_m12"),
                self.ds.get("ahrs_rotation_matrix_m13"),
                self.ds.get("ahrs_rotation_matrix_m21"),
                self.ds.get("ahrs_rotation_matrix_m22"),
                self.ds.get("ahrs_rotation_matrix_m23"),
                self.ds.get("ahrs_rotation_matrix_m31"),
                self.ds.get("ahrs_rotation_matrix_m32"),
                self.ds.get("ahrs_rotation_matrix_m33")
            ]),
            "ahrs_quaternions_wxyz": ("wxyz", [
                self.ds.get("ahrs_quaternions_w"),
                self.ds.get("ahrs_quaternions_x"),
                self.ds.get("ahrs_quaternions_y"),
                self.ds.get("ahrs_quaternions_z"),
            ]),
            "ahrs_gyro_xyz": ("xyz", [
                self.ds.get("ahrs_gyro_x"),
                self.ds.get("ahrs_gyro_y"),
                self.ds.get("ahrs_gyro_z"),
            ]),
            "percentage_good_data": self.ds.get("percentage_good_data"),
            "std_dev_pitch": self.ds.get("std_dev_pitch"),
            "std_dev_roll": self.ds.get("std_dev_roll"),
            "std_dev_heading": self.ds.get("std_dev_heading"),
            "std_dev_pressure": self.ds.get("std_dev_pressure"),
        },
            coords={
            "time_burst": self.ds.get("time_burst"),
            "time_average": self.ds.get("time_average"),
            "time_echosounder": self.ds.get("time_echosounder"),
            "beam": self.ds.get("beam"),
            "range_bin_average": self.ds.get("range_bin_average"),
            "range_bin_burst": self.ds.get("range_bin_burst"),
            "range_bin_echosounder": self.ds.get("range_bin_echosounder"),
        }, attrs=attrs)
        ds = ds.reindex({
            "mij": np.array(["11", "12", "13", "21", "22", "23", "31", "32", "33"]),
            "wxyz": np.array(["w", "x", "y", "z"]),
            "xyz": np.array(["x", "y", "z"]),
        })
        self.write(ds, "Vendor")
