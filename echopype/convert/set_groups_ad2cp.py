import os

import xarray as xr
import numpy as np

from .set_groups_base import SetGroupsBase


class SetGroupsAd2cp(SetGroupsBase):
    def write(self, ds: xr.Dataset, group: str):
        # TODO: when to use "w" or "a"?
        if os.path.exists(self.output_path):
            mode = "a"
        else:
            mode = "w"
        ds.to_netcdf(path=self.output_path, mode=mode, group=group)

    def save(self):
        self.set_environment()
        self.set_platform()
        self.set_beam()
        self.set_vendor_specific()

    def set_environment(self):
        ds = xr.Dataset(data_vars={
            "sound_speed_indicative": self.convert_obj.ds.get("speed_of_sound"),
            "temperature": self.convert_obj.ds.get("temperature"),
            "pressure": self.convert_obj.ds.get("pressure")
        },
            coords={
            "time_burst": self.convert_obj.ds.get("time_burst", []),
            "time_average": self.convert_obj.ds.get("time_average", [])
        })
        self.write(ds, "Environment")

    def set_platform(self):
        ds = xr.Dataset(
            data_vars={
                "heading": self.convert_obj.ds.get("heading"),
                "pitch": self.convert_obj.ds.get("pitch"),
                "roll": self.convert_obj.ds.get("roll"),
                "magnetometer_raw_x": self.convert_obj.ds.get("magnetometer_raw_x"),
                "magnetometer_raw_y": self.convert_obj.ds.get("magnetometer_raw_y"),
                "magnetometer_raw_z": self.convert_obj.ds.get("magnetometer_raw_z"),
            },
            coords={
                "time_burst": self.convert_obj.ds.get("time_burst"),
                "time_average": self.convert_obj.ds.get("time_average"),
                "beam": self.convert_obj.ds.get("beam"),
                "range_bin": self.convert_obj.ds.get("range_bin")
            },
            attrs={
                "platform_name": self.ui_param["platform_name"],
                "platform_type": self.ui_param["platform_type"],
                "platform_code_ICES": self.ui_param["platform_code_ICES"]
            }
        )
        self.write(ds, "Platform")

    def set_beam(self):
        data_vars = {
            "number_of_beams": self.convert_obj.ds.get("num_beams"),
            "coordinate_system": self.convert_obj.ds.get("coordinate_system"),
            "number_of_cells": self.convert_obj.ds.get("num_cells"),
            "blanking": self.convert_obj.ds.get("blanking"),
            "cell_size": self.convert_obj.ds.get("cell_size"),
            "velocity_range": self.convert_obj.ds.get("velocity_range"),
            "echosounder_frequency": self.convert_obj.ds.get("echo_sounder_frequency"),
            "ambiguity_velocity": self.convert_obj.ds.get("ambiguity_velocity"),
            "data_set_description": self.convert_obj.ds.get("dataset_description"),
            "transmit_energy": self.convert_obj.ds.get("transmit_energy"),
            "velocity_scaling": self.convert_obj.ds.get("velocity_scaling"),
            "velocity": self.convert_obj.ds.get("velocity_data"),
            "amplitude": self.convert_obj.ds.get("amplitude_data"),
            "correlation": self.convert_obj.ds.get("correlation_data"),
            "echosounder": self.convert_obj.ds.get("echosounder_data"),
            "figure_of_merit": self.convert_obj.ds.get("figure_of_merit_data"),
            "altimeter_distance": self.convert_obj.ds.get("altimeter_distance"),
            "altimeter_quality": self.convert_obj.ds.get("altimeter_quality"),
            "ast_distance": self.convert_obj.ds.get("ast_distance"),
            "ast_quality": self.convert_obj.ds.get("ast_quality"),
            "ast_offset_10us": self.convert_obj.ds.get("ast_offset_10us"),
            "ast_pressure": self.convert_obj.ds.get("ast_pressure"),
            "altimeter_spare": self.convert_obj.ds.get("altimeter_spare"),
            "altimeter_raw_data_num_samples": self.convert_obj.ds.get("altimeter_raw_data_num_samples"),
            "altimeter_raw_data_sample_distance": self.convert_obj.ds.get("altimeter_raw_data_sample_distance"),
            "altimeter_raw_data_samples": self.convert_obj.ds.get("altimeter_raw_data_samples")
        }

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time_burst": self.convert_obj.ds.get("time_burst"),
                "time_average": self.convert_obj.ds.get("time_average"),
                "beam": self.convert_obj.ds.get("beam"),
                "range_bin": self.convert_obj.ds.get("range_bin"),
                "altimeter_sample_bin": self.convert_obj.ds.get("altimeter_sample_bin"),
                "range_bin_echosounder": self.convert_obj.ds.get("num_beams_and_coordinate_system_and_num_cells")
            }
        )
        self.write(ds, "Beam")

    def set_vendor_specific(self):
        attrs = {
            "offset_of_data": self.convert_obj.ds.get("offset_of_data"),
            "pressure_sensor_valid": self.convert_obj.ds.get("pressure_sensor_valid"),
            "temperature_sensor_valid": self.convert_obj.ds.get("temperature_sensor_valid"),
            "compass_sensor_valid": self.convert_obj.ds.get("compass_sensor_valid"),
            "tilt_sensor_valid": self.convert_obj.ds.get("tilt_sensor_valid"),
            "velocity_data_included": self.convert_obj.ds.get("velocity_data_included"),
            "amplitude_data_included": self.convert_obj.ds.get("amplitude_data_included"),
            "correlation_data_included": self.convert_obj.ds.get("correlation_data_included"),
            "altimeter_data_included": self.convert_obj.ds.get("altimeter_data_included"),
            "altimeter_raw_data_included": self.convert_obj.ds.get("altimeter_raw_data_included"),
            "ast_data_included": self.convert_obj.ds.get("ast_data_included"),
            "echo_sounder_data_included": self.convert_obj.ds.get("echo_sounder_data_included"),
            "ahrs_data_included": self.convert_obj.ds.get("ahrs_data_included"),
            "percentage_good_data_included": self.convert_obj.ds.get("percentage_good_data_included"),
            "std_dev_data_included": self.convert_obj.ds.get("std_dev_data_included"),
            "figure_of_merit_data_included": self.convert_obj.ds.get("figure_of_merit_data_included"),
        }
        attrs = {field_name: field_value.data[0] for field_name, field_value in attrs.items(
        ) if field_value is not None}
        ds = xr.Dataset(data_vars={
            "data_record_version": self.convert_obj.ds.get("version"),
            "error": self.convert_obj.ds.get("error"),
            "status": self.convert_obj.ds.get("status"),
            "status0": self.convert_obj.ds.get("status0"),
            "battery_voltage": self.convert_obj.ds.get("battery_voltage"),
            "power_level": self.convert_obj.ds.get("power_level"),
            "temperature_of_pressure_sensor": self.convert_obj.ds.get("temperature_from_pressure_sensor"),
            "nominal_correlation": self.convert_obj.ds.get("nominal_correlation"),
            "magnetometer_temperature": self.convert_obj.ds.get("magnetometer_temperature"),
            "real_time_clock_temperature": self.convert_obj.ds.get("real_time_clock_temperature"),
            "ensemble_counter": self.convert_obj.ds.get("ensemble_counter"),
            "ahrs_rotation_matrix_mij": ("mij", [
                self.convert_obj.ds.get("ahrs_rotation_matrix_m11"),
                self.convert_obj.ds.get("ahrs_rotation_matrix_m12"),
                self.convert_obj.ds.get("ahrs_rotation_matrix_m13"),
                self.convert_obj.ds.get("ahrs_rotation_matrix_m21"),
                self.convert_obj.ds.get("ahrs_rotation_matrix_m22"),
                self.convert_obj.ds.get("ahrs_rotation_matrix_m23"),
                self.convert_obj.ds.get("ahrs_rotation_matrix_m31"),
                self.convert_obj.ds.get("ahrs_rotation_matrix_m32"),
                self.convert_obj.ds.get("ahrs_rotation_matrix_m33")
            ]),
            "ahrs_quaternions_wxyz": ("wxyz", [
                self.convert_obj.ds.get("ahrs_quaternions_w"),
                self.convert_obj.ds.get("ahrs_quaternions_x"),
                self.convert_obj.ds.get("ahrs_quaternions_y"),
                self.convert_obj.ds.get("ahrs_quaternions_z"),
            ]),
            "ahrs_gyro_xyz": ("xyz", [
                self.convert_obj.ds.get("ahrs_gyro_x"),
                self.convert_obj.ds.get("ahrs_gyro_y"),
                self.convert_obj.ds.get("ahrs_gyro_z"),
            ]),
            "percentage_good_data": self.convert_obj.ds.get("percentage_good_data"),
            "std_dev_pitch": self.convert_obj.ds.get("std_dev_pitch"),
            "std_dev_roll": self.convert_obj.ds.get("std_dev_roll"),
            "std_dev_heading": self.convert_obj.ds.get("std_dev_heading"),
            "std_dev_pressure": self.convert_obj.ds.get("std_dev_pressure"),
        },
            coords={
            "time_burst": self.convert_obj.ds.get("time_burst"),
            "time_average": self.convert_obj.ds.get("time_average"),
            "beam": self.convert_obj.ds.get("beam"),
            "range_bin": self.convert_obj.ds.get("range_bin"),
        }, attrs=attrs)
        ds = ds.reindex({
            "mij": np.array(["11", "12", "13", "21", "22", "23", "31", "32", "33"]),
            "wxyz": np.array(["w", "x", "y", "z"]),
            "xyz": np.array(["x", "y", "z"]),
        })
        self.write(ds, "Vendor")
