from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import xarray as xr

from ..utils.coding import set_encodings
from .parse_ad2cp import DataType, Dimension, Field, HeaderOrDataRecordFormats
from .set_groups_base import SetGroupsBase

AHRS_COORDS: Dict[Dimension, np.ndarray] = {
    Dimension.MIJ: np.array(["11", "12", "13", "21", "22", "23", "31", "32", "33"]),
    Dimension.WXYZ: np.array(["w", "x", "y", "z"]),
    Dimension.XYZ: np.array(["x", "y", "z"]),
}


class SetGroupsAd2cp(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from Ad2cp data files."""

    beamgroups_possible = [
        {
            "name": "Beam_group1",
            "descr": (
                "contains velocity, correlation, and backscatter power (uncalibrated)"
                " data and other data derived from acoustic data."
            ),
        }
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pulse_compressed = self.parser_obj.get_pulse_compressed()
        self._make_time_coords()

    def _make_time_coords(self):
        timestamps = []
        times_idx = {
            Dimension.PING_TIME_AVERAGE: [],
            Dimension.PING_TIME_BURST: [],
            Dimension.PING_TIME_ECHOSOUNDER: [],
            Dimension.PING_TIME_ECHOSOUNDER_RAW: [],
            Dimension.PING_TIME_ECHOSOUNDER_RAW_TRANSMIT: [],
        }

        for packet in self.parser_obj.packets:
            if not packet.has_timestamp():
                continue
            timestamps.append(packet.timestamp)
            i = len(timestamps) - 1
            if packet.is_average():
                times_idx[Dimension.PING_TIME_AVERAGE].append(i)
            elif packet.is_burst():
                times_idx[Dimension.PING_TIME_BURST].append(i)
            elif packet.is_echosounder():
                times_idx[Dimension.PING_TIME_ECHOSOUNDER].append(i)
            elif packet.is_echosounder_raw():
                times_idx[Dimension.PING_TIME_ECHOSOUNDER_RAW].append(i)
            elif packet.is_echosounder_raw_transmit():
                times_idx[Dimension.PING_TIME_ECHOSOUNDER_RAW_TRANSMIT].append(i)

        self.times_idx = {
            time_dim: np.array(time_values, dtype="u8")
            for time_dim, time_values in times_idx.items()
        }
        self.timestamps = np.array(timestamps)
        _, unique_ping_time_idx = np.unique(self.timestamps, return_index=True)
        self.times_idx[Dimension.PING_TIME] = unique_ping_time_idx

    def _make_dataset(self, var_names: Dict[str, str]) -> xr.Dataset:
        """
        Constructs a dataset of the given variables using parser_obj data
        var_names maps parser_obj field names to output dataset variable names
        """

        # {field_name: [field_value]}
        #   [field_value] lines up with time_dim
        fields: Dict[str, List[np.ndarray]] = {field_name: [] for field_name in var_names.keys()}
        # {field_name: [Dimension]}
        dims: Dict[str, List[Dimension]] = dict()
        # {field_name: field dtype}
        dtypes: Dict[str, np.dtype] = dict()
        # {field_name: units}
        units: Dict[str, Optional[str]] = dict()
        # {field_name: [idx of padding]}
        pad_idx: Dict[str, List[int]] = {field_name: [] for field_name in var_names.keys()}
        # {field_name: field exists}
        field_exists: Dict[str, bool] = {field_name: False for field_name in var_names.keys()}
        beam_coords: Optional[np.ndarray] = None
        # separate by time dim
        for packet in self.parser_obj.packets:
            if not packet.has_timestamp():
                continue
            if "beams" in packet.data:
                if beam_coords is None:
                    beam_coords = packet.data["beams"]
                else:
                    beam_coords = max(beam_coords, packet.data["beams"], key=lambda x: len(x))
            data_record_format = HeaderOrDataRecordFormats.data_record_format(
                packet.data_record_type
            )
            for field_name in var_names.keys():
                field = data_record_format.get_field(field_name)
                if field is None:
                    field_dimensions = Field.default_dimensions()
                    # can't store in dims yet because there might be another data record format
                    #   which does have this field
                else:
                    field_dimensions = field.dimensions(packet.data_record_type)

                    if field_name not in dims:
                        dims[field_name] = field_dimensions
                    if field_name not in dtypes:
                        field_entry_size_bytes = field.field_entry_size_bytes
                        if callable(field_entry_size_bytes):
                            field_entry_size_bytes = field_entry_size_bytes(packet)
                        dtypes[field_name] = field.field_entry_data_type.dtype(
                            field_entry_size_bytes
                        )
                    if field_name not in units:
                        units[field_name] = field.units()

                if field_name in packet.data:  # field is in this packet
                    fields[field_name].append(packet.data[field_name])
                    field_exists[field_name] = True
                else:  # field is not in this packet
                    # pad the list of field values with an empty array so that
                    #   the time dimension still lines up with the field values
                    fields[field_name].append(np.array(0))
                    pad_idx[field_name].append(len(fields[field_name]) - 1)

        for field_name in fields.keys():
            # add dimensions to dims if they were not found
            #   (the desired fields did not exist in any of the packet's data records
            #   because they are in a different packet OR it is a field created by echopype
            #   from a bitfield, etc.)
            if field_name not in dims:
                dims[field_name] = Field.default_dimensions()
            # add dtypes to dtypes if they were not found
            #   (the desired fields did not exist in any of the packet's data records
            #   because they are in a different packet OR it is a field created by echopype
            #   from a bitfield, etc.)
            if field_name not in dtypes:
                dtypes[field_name] = DataType.default_dtype()

        # replace padding with correct shaped padding
        #   (earlier we padded along the time dimension but we didn't necessarily know the shape
        #   of the padding itself)
        for field_name, pad_idxs in pad_idx.items():
            for i in pad_idxs:
                fields[field_name][i] = np.zeros(
                    np.ones(len(dims[field_name]) - 1, dtype="u1"),  # type: ignore
                    dtype=dtypes[field_name],
                )

        # {field_name: field_value}
        #   field_value is now combined along time_dim
        combined_fields: Dict[str, np.ndarray] = dict()
        # pad to max shape and stack
        for field_name, field_values in fields.items():
            if field_exists[field_name]:
                if len(dims[field_name]) > 1:
                    shapes = [field_value.shape for field_value in field_values]
                    max_shape = np.amax(
                        np.stack(shapes),
                        axis=0,
                    )
                    field_values = [
                        np.pad(
                            field_value,
                            tuple(
                                (0, max_axis_len - field_value.shape[i])
                                for i, max_axis_len in enumerate(max_shape)  # type: ignore
                            ),
                        )
                        for field_value in field_values
                    ]
                field_values = np.stack(field_values)
                combined_fields[field_name] = field_values

        # slice fields to time_dim
        for field_name, field_value in combined_fields.items():
            combined_fields[field_name] = field_value[self.times_idx[dims[field_name][0]]]

        # make ds
        used_dims: Set[Dimension] = {
            dim
            for field_name, dims_list in dims.items()
            for dim in dims_list
            if field_exists[field_name]
        }
        data_vars: Dict[
            str,
            Union[Tuple[List[str], np.ndarray, Dict[str, str]], Tuple[Tuple[()], None]],
        ] = {
            var_name: (
                [dim.value for dim in dims[field_name]],
                combined_fields[field_name],
                {"Units": units[field_name]}
                if field_name in units and units[field_name] is not None
                else {},
            )
            if field_exists[field_name]
            else ((), None)
            for field_name, var_name in var_names.items()
        }  # type: ignore
        coords: Dict[str, np.ndarray] = dict()
        for time_dim, time_idxs in self.times_idx.items():
            if time_dim in used_dims:
                coords[time_dim.value] = self.timestamps[time_idxs]
        for ahrs_dim, ahrs_coords in AHRS_COORDS.items():
            if ahrs_dim in used_dims:
                coords[ahrs_dim.value] = ahrs_coords
        if Dimension.BEAM in used_dims and beam_coords is not None:
            coords[Dimension.BEAM.value] = beam_coords
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        # make arange coords for the remaining dims
        non_coord_dims = {dim.value for dim in used_dims} - set(ds.coords.keys())
        ds = ds.assign_coords({dim: np.arange(ds.dims[dim]) for dim in non_coord_dims})
        return ds

    def set_env(self) -> xr.Dataset:
        ds = self._make_dataset(
            {
                "speed_of_sound": "sound_speed_indicative",
                "temperature": "temperature",
                "pressure": "pressure",
            }
        )

        # FIXME: this is a hack because the current file saving
        # mechanism requires that the env group have ping_time as a dimension,
        # but ping_time might not be a dimension if the dataset is completely
        # empty
        if "ping_time" not in ds.dims:
            ds = ds.expand_dims(dim="ping_time")

        return set_encodings(ds)

    def set_platform(self) -> xr.Dataset:
        ds = self._make_dataset(
            {
                "heading": "heading",
                "pitch": "pitch",
                "roll": "roll",
                "magnetometer_raw": "magnetometer_raw",
            }
        )
        ds = ds.assign_attrs(
            {
                "platform_name": self.ui_param["platform_name"],  # type: ignore
                "platform_type": self.ui_param["platform_type"],  # type: ignore
                "platform_code_ICES": self.ui_param["platform_code_ICES"],  # type: ignore
            }
        )
        return set_encodings(ds)

    def set_beam(self) -> xr.Dataset:
        # TODO: should we divide beam into burst/average (e.g., beam_burst, beam_average)
        # like was done for range_bin (we have range_bin_burst, range_bin_average,
        # and range_bin_echosounder)?
        ds = self._make_dataset(
            {
                "num_beams": "number_of_beams",
                "coordinate_system": "coordinate_system",
                "num_cells": "number_of_cells",
                "blanking": "blanking",
                "cell_size": "cell_size",
                "velocity_range": "velocity_range",
                "echosounder_frequency": "echosounder_frequency",
                "ambiguity_velocity": "ambiguity_velocity",
                "dataset_description": "data_set_description",
                "transmit_energy": "transmit_energy",
                "velocity_scaling": "velocity_scaling",
                "velocity_data_burst": "velocity_burst",
                "velocity_data_average": "velocity_average",
                "amplitude_data_burst": "amplitude_burst",
                "amplitude_data_average": "amplitude_average",
                "correlation_data_burst": "correlation_burst",
                "correlation_data_average": "correlation_average",
                "correlation_data_echosounder": "correlation_echosounder",
                "echosounder_data": "amplitude_echosounder",
                "echosounder_raw_samples_i": "backscatter_r",
                "echosounder_raw_samples_q": "backscatter_i",
                "echosounder_raw_transmit_samples_i": "transmit_pulse_r",
                "echosounder_raw_transmit_samples_q": "transmit_pulse_i",
                "echosounder_raw_beam": "echosounder_raw_beam",
                "echosounder_raw_echogram": "echosounder_raw_echogram",
            }
        )
        ds = ds.assign_attrs({"pulse_compressed": self.pulse_compressed})

        # FIXME: this is a hack because the current file saving
        # mechanism requires that the beam group have ping_time as a dimension,
        # but ping_time might not be a dimension if the dataset is completely
        # empty
        if "ping_time" not in ds.dims:
            ds = ds.expand_dims(dim="ping_time")

        return set_encodings(ds)

    def set_vendor(self) -> xr.Dataset:
        ds = self._make_dataset(
            {
                "version": "data_record_version",
                "error": "error",
                "status": "status",
                "status0": "status0",
                "battery_voltage": "battery_voltage",
                "power_level": "power_level",
                "temperature_from_pressure_sensor": "temperature_of_pressure_sensor",
                "nominal_correlation": "nominal_correlation",
                "magnetometer_temperature": "magnetometer_temperature",
                "real_ping_time_clock_temperature": "real_ping_time_clock_temperature",
                "ensemble_counter": "ensemble_counter",
                "ahrs_rotation_matrix": "ahrs_rotation_matrix_mij",
                "ahrs_quaternions": "ahrs_quaternions_wxyz",
                "ahrs_gyro": "ahrs_gyro_xyz",
                "percentage_good_data": "percentage_good_data",
                "std_dev_pitch": "std_dev_pitch",
                "std_dev_roll": "std_dev_roll",
                "std_dev_heading": "std_dev_heading",
                "std_dev_pressure": "std_dev_pressure",
                "pressure_sensor_valid": "pressure_sensor_valid",
                "temperature_sensor_valid": "temperature_sensor_valid",
                "compass_sensor_valid": "compass_sensor_valid",
                "tilt_sensor_valid": "tilt_sensor_valid",
                "figure_of_merit_data": "figure_of_merit",
                "altimeter_distance": "altimeter_distance",
                "altimeter_quality": "altimeter_quality",
                "ast_distance": "ast_distance",
                "ast_quality": "ast_quality",
                "ast_offset_100us": "ast_offset_100us",
                "ast_pressure": "ast_pressure",
                "altimeter_spare": "altimeter_spare",
                "altimeter_raw_data_num_samples": "altimeter_raw_data_num_samples",
                "altimeter_raw_data_sample_distance": "altimeter_raw_data_sample_distance",
                "altimeter_raw_data_samples": "altimeter_raw_data_samples",
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
        """Set the Sonar group."""

        # Add beam_group and beam_group_descr variables sharing a common dimension
        # (beam_group), using the information from self._beamgroups
        self._beamgroups = self.beamgroups_possible
        beam_groups_vars, beam_groups_coord = self._beam_groups_vars()
        ds = xr.Dataset(beam_groups_vars, coords=beam_groups_coord)

        # Assemble sonar group global attribute dictionary
        sonar_attr_dict = {
            "sonar_manufacturer": "Nortek",
            "sonar_model": "AD2CP",
            "sonar_serial_number": "",
            "sonar_software_name": "",
            "sonar_software_version": "",
            "sonar_firmware_version": "",
            "sonar_type": "acoustic Doppler current profiler (ADCP)",
        }
        for packet in self.parser_obj.packets:
            if "serial_number" in packet.data:
                ds.attrs["sonar_serial_number"] = packet.data["serial_number"]
                break
        firmware_version = self.parser_obj.get_firmware_version()
        if firmware_version is not None:
            sonar_attr_dict["sonar_firmware_version"] = ", ".join(
                [f"{k}:{v}" for k, v in firmware_version.items()]
            )

        ds = ds.assign_attrs(sonar_attr_dict)

        return ds
