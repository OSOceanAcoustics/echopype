from enum import Enum, auto, unique
from importlib import resources
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import xarray as xr
import yaml

from .. import convert
from ..utils.coding import set_time_encodings
from .parse_ad2cp import DataType, Dimension, Field, HeaderOrDataRecordFormats
from .set_groups_base import SetGroupsBase

AHRS_COORDS: Dict[Dimension, np.ndarray] = {
    Dimension.MIJ: np.array(["11", "12", "13", "21", "22", "23", "31", "32", "33"]),
    Dimension.WXYZ: np.array(["w", "x", "y", "z"]),
    Dimension.XYZ: np.array(["x", "y", "z"]),
}


@unique
class BeamGroup(Enum):
    AVERAGE = auto()
    BURST = auto()
    ECHOSOUNDER = auto()
    ECHOSOUNDER_RAW = auto()


class SetGroupsAd2cp(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from Ad2cp data files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: bug: 0 if not exist in first string packet
        # resulting in index error in setting ds["pulse_compressed"]
        self.pulse_compressed = self.parser_obj.get_pulse_compressed()
        self._make_time_coords()
        with resources.open_text(convert, "ad2cp_fields.yaml") as f:
            self.field_attrs: Dict[str, Dict[str, Dict[str, str]]] = yaml.safe_load(f)  # type: ignore # noqa

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
        # {field_name: attrs}
        attrs: Dict[str, Dict[str, str]] = dict()
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

                    if field_name not in attrs:
                        if field_name in self.field_attrs["POSTPROCESSED"]:
                            attrs[field_name] = self.field_attrs["POSTPROCESSED"][field_name]
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
                    if field_name not in attrs:
                        attrs[field_name] = self.field_attrs[data_record_format.name][field_name]

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
                (
                    [dim.dimension_name() for dim in dims[field_name]],
                    combined_fields[field_name],
                    attrs.get(field_name, {}),
                )
                if field_exists[field_name]
                else ((), None)
            )
            for field_name, var_name in var_names.items()
        }  # type: ignore
        coords: Dict[str, np.ndarray] = dict()
        for time_dim, time_idxs in self.times_idx.items():
            if time_dim in used_dims:
                coords[time_dim.dimension_name()] = self.timestamps[time_idxs]
        for ahrs_dim, ahrs_coords in AHRS_COORDS.items():
            if ahrs_dim in used_dims:
                coords[ahrs_dim.dimension_name()] = ahrs_coords
        if Dimension.BEAM in used_dims and beam_coords is not None:
            coords[Dimension.BEAM.dimension_name()] = beam_coords
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        # make arange coords for the remaining dims
        non_coord_dims = {dim.dimension_name() for dim in used_dims} - set(ds.coords.keys())
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

        return set_time_encodings(ds)

    def set_platform(self) -> xr.Dataset:
        ds = self._make_dataset(
            {
                "heading": "heading",
                "pitch": "pitch",
                "roll": "roll",
            }
        )
        return set_time_encodings(ds)

    def set_beam(self) -> List[xr.Dataset]:
        # TODO: should we divide beam into burst/average (e.g., beam_burst, beam_average)
        # like was done for range_bin (we have range_bin_burst, range_bin_average,
        # and range_bin_echosounder)?
        beam_groups = []
        self._beamgroups = []
        beam_groups_exist = set()

        for packet in self.parser_obj.packets:
            if packet.is_average():
                beam_groups_exist.add(BeamGroup.AVERAGE)
            elif packet.is_burst():
                beam_groups_exist.add(BeamGroup.BURST)
            elif packet.is_echosounder():
                beam_groups_exist.add(BeamGroup.ECHOSOUNDER)
            elif packet.is_echosounder_raw():
                beam_groups_exist.add(BeamGroup.ECHOSOUNDER_RAW)

            if len(beam_groups_exist) == len(BeamGroup):
                break

        # average
        if BeamGroup.AVERAGE in beam_groups_exist:
            beam_groups.append(
                self._make_dataset(
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
                        "velocity_data_average": "velocity",
                        "amplitude_data_average": "amplitude",
                        "correlation_data_average": "correlation",
                    }
                )
            )

            self._beamgroups.append(
                {
                    "name": f"Beam_group{len(self._beamgroups) + 1}",
                    "descr": (
                        "contains echo intensity, velocity and correlation data "
                        "as well as other configuration parameters from the Average mode."
                    ),
                }
            )
        # burst
        if BeamGroup.BURST in beam_groups_exist:
            beam_groups.append(
                self._make_dataset(
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
                        "velocity_data_burst": "velocity",
                        "amplitude_data_burst": "amplitude",
                        "correlation_data_burst": "correlation",
                    }
                )
            )

            self._beamgroups.append(
                {
                    "name": f"Beam_group{len(self._beamgroups) + 1}",
                    "descr": (
                        "contains echo intensity, velocity and correlation data "
                        "as well as other configuration parameters from the Burst mode."
                    ),
                }
            )
        # echosounder
        if BeamGroup.ECHOSOUNDER in beam_groups_exist:
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
                    "correlation_data_echosounder": "correlation",
                    "echosounder_data": "amplitude",
                }
            )
            ds = ds.assign_coords({"echogram": np.arange(3)})
            pulse_compressed = np.zeros(3)
            # TODO: bug: if self.pulse_compress=0 this will set the last index to 1
            pulse_compressed[self.pulse_compressed - 1] = 1
            ds["pulse_compressed"] = (("echogram",), pulse_compressed)
            beam_groups.append(ds)

            self._beamgroups.append(
                {
                    "name": f"Beam_group{len(self._beamgroups) + 1}",
                    "descr": (
                        "contains backscatter echo intensity and other configuration "
                        "parameters from the Echosounder mode. "
                        "Data can be pulse compressed or raw intensity."
                    ),
                }
            )
        # echosounder raw
        if BeamGroup.ECHOSOUNDER_RAW in beam_groups_exist:
            beam_groups.append(
                self._make_dataset(
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
                        "num_complex_samples": "num_complex_samples",
                        "ind_start_samples": "ind_start_samples",
                        "freq_raw_sample_data": "freq_raw_sample_data",
                        "echosounder_raw_samples_i": "backscatter_r",
                        "echosounder_raw_samples_q": "backscatter_i",
                        "echosounder_raw_transmit_samples_i": "transmit_pulse_r",
                        "echosounder_raw_transmit_samples_q": "transmit_pulse_i",
                    }
                )
            )

            self._beamgroups.append(
                {
                    "name": f"Beam_group{len(self._beamgroups) + 1}",
                    "descr": (
                        "contains complex backscatter raw samples and other configuration "
                        "parameters from the Echosounder mode, "
                        "including complex data from the transmit pulse."
                    ),
                }
            )

        # FIXME: this is a hack because the current file saving
        # mechanism requires that the beam group have ping_time as a dimension,
        # but ping_time might not be a dimension if the dataset is completely
        # empty
        for i, ds in enumerate(beam_groups):
            if "ping_time" not in ds.dims:
                beam_groups[i] = ds.expand_dims(dim="ping_time")

        # remove time1 from beam groups
        for i, ds in enumerate(beam_groups):
            beam_groups[i] = ds.sel(time1=ds["ping_time"]).drop_vars("time1", errors="ignore")

        return [set_time_encodings(ds) for ds in beam_groups]

    def set_vendor(self) -> xr.Dataset:
        ds = self._make_dataset(
            {
                "version": "data_record_version",
                "pressure_sensor_valid": "pressure_sensor_valid",
                "temperature_sensor_valid": "temperature_sensor_valid",
                "compass_sensor_valid": "compass_sensor_valid",
                "tilt_sensor_valid": "tilt_sensor_valid",
                "velocity_data_included": "velocity_data_included",
                "amplitude_data_included": "amplitude_data_included",
                "correlation_data_included": "correlation_data_included",
                "altimeter_data_included": "altimeter_data_included",
                "altimeter_raw_data_included": "altimeter_raw_data_included",
                "ast_data_included": "ast_data_included",
                "echosounder_data_included": "echosounder_data_included",
                "ahrs_data_included": "ahrs_data_included",
                "percentage_good_data_included": "percentage_good_data_included",
                "std_dev_data_included": "std_dev_data_included",
                "distance_data_included": "distance_data_included",
                "figure_of_merit_data_included": "figure_of_merit_data_included",
                "error": "error",
                "status0": "status0",
                "procidle3": "procidle3",
                "procidle6": "procidle6",
                "procidle12": "procidle12",
                "status": "status",
                "wakeup_state": "wakeup_state",
                "orientation": "orientation",
                "autoorientation": "autoorientation",
                "previous_wakeup_state": "previous_wakeup_state",
                "last_measurement_low_voltage_skip": "last_measurement_low_voltage_skip",
                "active_configuration": "active_configuration",
                "echosounder_index": "echosounder_index",
                "telemetry_data": "telemetry_data",
                "boost_running": "boost_running",
                "echosounder_frequency_bin": "echosounder_frequency_bin",
                "bd_scaling": "bd_scaling",
                "battery_voltage": "battery_voltage",
                "power_level": "power_level",
                "temperature_from_pressure_sensor": "temperature_of_pressure_sensor",
                "nominal_correlation": "nominal_correlation",
                "magnetometer_temperature": "magnetometer_temperature",
                "real_time_clock_temperature": "real_time_clock_temperature",
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
                "magnetometer_raw": "magnetometer_raw",
            }
        )

        return set_time_encodings(ds)

    def set_sonar(self) -> xr.Dataset:
        """Set the Sonar group."""

        # Add beam_group and beam_group_descr variables sharing a common dimension
        # (beam_group), using the information from self._beamgroups
        beam_groups_vars, beam_groups_coord = self._beam_groups_vars()
        ds = xr.Dataset(beam_groups_vars, coords=beam_groups_coord)

        # Assemble sonar group global attribute dictionary
        sonar_attr_dict = {
            "sonar_manufacturer": "Nortek",
            "sonar_model": "AD2CP",
            "sonar_serial_number": ", ".join(
                np.unique(
                    [
                        str(packet.data["serial_number"])
                        for packet in self.parser_obj.packets
                        if "serial_number" in packet.data
                    ]
                )
            ),
            "sonar_software_name": "",
            "sonar_software_version": "",
            "sonar_firmware_version": "",
            "sonar_type": "acoustic Doppler current profiler (ADCP)",
        }
        firmware_version = self.parser_obj.get_firmware_version()
        if firmware_version is not None:
            sonar_attr_dict["sonar_firmware_version"] = ", ".join(
                [f"{k}:{v}" for k, v in firmware_version.items()]
            )

        ds = ds.assign_attrs(sonar_attr_dict)

        return set_time_encodings(ds)
