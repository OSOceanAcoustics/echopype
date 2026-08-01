import datetime
from typing import List

import numpy as np
import xarray as xr

from ..utils.coding import set_time_encodings
from ..utils.log import _init_logger
from ..utils.prov import echopype_prov_attrs, source_files_vars

# fmt: off
from .set_groups_base import SetGroupsBase

# fmt: on

logger = _init_logger(__name__)


class SetGroupsBI500(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from BI500 data files."""

    beamgroups_possible = [
        {
            "name": "Beam_group1",
            "descr": ("Contains BI500 channel, ping, and echogram geometry metadata."),
        }
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._beamgroups = self.beamgroups_possible

    @staticmethod
    def _build_ping_time(dates, times) -> np.ndarray:
        """Combine BI500 YYYYMMDD date and seconds-since-midnight time arrays."""
        ping_time = np.empty(len(dates), dtype="datetime64[ns]")
        for i, (date, time) in enumerate(zip(dates, times)):
            year = date // 10000
            month = (date // 100) % 100
            day = date % 100
            dt = datetime.datetime(
                year, month, day, tzinfo=datetime.timezone.utc
            ) + datetime.timedelta(seconds=int(time))
            ping_time[i] = np.datetime64(dt.replace(tzinfo=None), "ns")
        return ping_time

    def _get_ping_time(self) -> np.ndarray:
        ping_data = self.parser_obj.ping_data
        return self._build_ping_time(
            np.array(ping_data["date"], dtype=np.int64),
            np.array(ping_data["time"], dtype=np.int64),
        )

    def _get_ping_time_vlog(self) -> np.ndarray:
        vlog_data = self.parser_obj.vlog_data
        return self._build_ping_time(
            np.array(vlog_data["date"], dtype=np.int64),
            np.array(vlog_data["time"], dtype=np.int64),
        )

    def set_platform(self) -> xr.Dataset:
        """Set the Platform group."""
        ping_data = self.parser_obj.ping_data
        vlog_data = self.parser_obj.vlog_data
        ping_time = self._get_ping_time()
        ping_time_vlog = self._get_ping_time_vlog()

        platform_attrs = {
            "platform_name": "",
            "platform_type": "",
            "platform_code_ICES": "",
        }
        if self.parser_obj.parameters.get("ship"):
            platform_attrs["platform_code_ICES"] = str(self.parser_obj.parameters["ship"][0])

        ds = xr.Dataset(
            {
                "latitude": (
                    ["ping_time"],
                    np.array(ping_data["latitude"], dtype=np.float64),
                    self._varattrs["platform_var_default"]["latitude"],
                ),
                "latitude_vlog": (
                    ["ping_time_vlog"],
                    np.array(vlog_data["latitude"], dtype=np.float64),
                    {
                        "long_name": "Vessel log latitude",
                        "units": "degrees",
                    },
                ),
                "longitude": (
                    ["ping_time"],
                    np.array(ping_data["longitude"], dtype=np.float64),
                    self._varattrs["platform_var_default"]["longitude"],
                ),
                "longitude_vlog": (
                    ["ping_time_vlog"],
                    np.array(vlog_data["longitude"], dtype=np.float64),
                    {
                        "long_name": "Vessel log longitude",
                        "units": "degrees",
                    },
                ),
                "bottom_depth": (
                    ["ping_time"],
                    np.array(ping_data["bottom_depth"], dtype=np.float64),
                    {
                        "long_name": "Bottom depth",
                        "units": "m",
                        "positive": "down",
                    },
                ),
                "bottom_depth_vlog": (
                    ["ping_time_vlog"],
                    np.array(vlog_data["bottom_depth"], dtype=np.float64),
                    {
                        "long_name": "Bottom depth from vlog",
                        "units": "m",
                        "positive": "down",
                    },
                ),
                "vessel_log_distance": (
                    ["ping_time"],
                    np.array(ping_data["distance"], dtype=np.float64),
                    {
                        "long_name": "Vessel log distance",
                        "units": "m",
                        "comment": "Distance along track from the vessel log.",
                    },
                ),
                "vessel_log_distance_vlog": (
                    ["ping_time_vlog"],
                    np.array(vlog_data["distance"], dtype=np.float64),
                    {
                        "long_name": "Vessel log distance from vlog",
                        "units": "m",
                        "comment": "Distance along track from the vessel log from vlog.",
                    },
                ),
            },
            coords={
                "ping_time": (
                    ["ping_time"],
                    ping_time,
                    {
                        "axis": "T",
                        "long_name": "Timestamps for platform data",
                        "standard_name": "time",
                        "comment": "Combined from BI500 -Ping Date and Time fields.",
                    },
                ),
                "ping_time_vlog": (
                    ["ping_time_vlog"],
                    ping_time_vlog,
                    {
                        "axis": "T",
                        "long_name": "Timestamps for platform data from vlog",
                        "standard_name": "time",
                        "comment": "Combined from BI500 -Vlog Date and Time fields.",
                    },
                ),
            },
        )
        ds = ds.assign_attrs(platform_attrs)
        return set_time_encodings(ds)

    def _get_channel_id(self) -> str:
        """Return a single BI500 channel identifier."""
        parameters = self.parser_obj.parameters
        frequency = int(parameters["frequency"][0])
        transceiver = int(parameters["transceiver"][0])
        return f"BI500-F{frequency}-T{transceiver:02d}"

    def set_env(self) -> xr.Dataset:
        """Set the Environment group."""
        channel_id = self._get_channel_id()

        ds = xr.Dataset(
            {
                "absorption_indicative": (
                    ["channel"],
                    [np.nan],
                    {
                        "long_name": "Indicative acoustic absorption",
                        "units": "dB/m",
                        "valid_min": 0.0,
                    },
                ),
                "sound_speed_indicative": (
                    [],
                    np.nan,
                    {
                        "long_name": "Indicative sound speed",
                        "standard_name": "speed_of_sound_in_sea_water",
                        "units": "m/s",
                        "valid_min": 0.0,
                    },
                ),
            },
            coords={
                "channel": (
                    ["channel"],
                    [channel_id],
                    self._varattrs["beam_coord_default"]["channel"],
                ),
            },
        )
        return set_time_encodings(ds)

    def set_sonar(self) -> xr.Dataset:
        """Set the Sonar group."""
        parameters = self.parser_obj.parameters
        beam_groups_vars, beam_groups_coord = self._beam_groups_vars()
        ds = xr.Dataset(beam_groups_vars, coords=beam_groups_coord)

        sonar_attr_dict = {
            "sonar_manufacturer": "Bergen Integrator",
            "sonar_model": self.sonar_model,
            "sonar_serial_number": "",
            "sonar_software_name": "BI500",
            "sonar_software_version": str(int(parameters["release"][0])),
            "sonar_type": "echosounder",
        }
        ds = ds.assign_attrs(sonar_attr_dict)
        return set_time_encodings(ds)

    def set_calibrated(self) -> xr.Dataset:
        """Create a dataset containing calibrated BI500 echogram products."""
        parameters = self.parser_obj.parameters
        ping_data = self.parser_obj.ping_data
        unpacked_data = self.parser_obj.unpacked_data

        ping_time = self._get_ping_time()
        channel_id = self._get_channel_id()
        frequency = float(parameters["frequency"][0])

        sv = self._stack_samples(unpacked_data["pelagic"])
        sv_bottom = self._stack_samples(unpacked_data["bottom"])

        traces = self._collect_target_traces()
        n_targets = len(traces["single_target_range"])

        single_target = np.arange(
            n_targets,
            dtype=np.int64,
        )

        single_target_ping_index = np.asarray(
            traces["ping_index"],
            dtype=np.int64,
        )

        single_target_ping_time = ping_time[single_target_ping_index]

        ds = xr.Dataset(
            data_vars={
                "Sv": (
                    ["channel", "ping_time", "range_sample"],
                    sv[np.newaxis, :, :].astype(np.float32),
                    {
                        "long_name": "Volume backscattering strength",
                        "units": "dB",
                        "comment": (
                            "Nominally calibrated pelagic Sv recorded " "in the BI500 -Data file."
                        ),
                    },
                ),
                "Sv_bottom": (
                    ["channel", "ping_time", "range_sample_bottom"],
                    sv_bottom[np.newaxis, :, :].astype(np.float32),
                    {
                        "long_name": "Bottom volume backscattering strength",
                        "units": "dB",
                        "comment": (
                            "Nominally calibrated bottom Sv recorded " "in the BI500 -Data file."
                        ),
                    },
                ),
                "frequency_nominal": (
                    ["channel"],
                    [frequency],
                    {
                        "long_name": "Transducer frequency",
                        "standard_name": "sound_frequency",
                        "units": "Hz",
                        "valid_min": 0.0,
                    },
                ),
                "pelagic_upper": (
                    ["channel", "ping_time"],
                    np.asarray(
                        ping_data["pelagic_upper"],
                        dtype=np.float64,
                    )[np.newaxis, :],
                    {
                        "long_name": "Pelagic echogram upper depth bound",
                        "units": "m",
                    },
                ),
                "pelagic_lower": (
                    ["channel", "ping_time"],
                    np.asarray(
                        ping_data["pelagic_lower"],
                        dtype=np.float64,
                    )[np.newaxis, :],
                    {
                        "long_name": "Pelagic echogram lower depth bound",
                        "units": "m",
                    },
                ),
                "bottom_upper": (
                    ["channel", "ping_time"],
                    np.asarray(
                        ping_data["bottom_upper"],
                        dtype=np.float64,
                    )[np.newaxis, :],
                    {
                        "long_name": "Bottom echogram upper depth bound",
                        "units": "m",
                    },
                ),
                "bottom_lower": (
                    ["channel", "ping_time"],
                    np.asarray(
                        ping_data["bottom_lower"],
                        dtype=np.float64,
                    )[np.newaxis, :],
                    {
                        "long_name": "Bottom echogram lower depth bound",
                        "units": "m",
                    },
                ),
                "single_target_identifier": (
                    ["single_target"],
                    single_target,
                    {
                        "long_name": "Index of single target detected",
                    },
                ),
                "ping_index": (
                    ["single_target"],
                    single_target_ping_index,
                    {
                        "long_name": (
                            "Index of the BI500 ping containing the " "single-target detection"
                        ),
                    },
                ),
                "single_target_ping_time": (
                    ["single_target"],
                    single_target_ping_time,
                    {
                        "long_name": "Ping time of single-target detection",
                        "standard_name": "time",
                    },
                ),
                "single_target_range": (
                    ["single_target"],
                    np.asarray(
                        traces["single_target_range"],
                        dtype=np.float64,
                    ),
                    {
                        "long_name": "Range of single target detected",
                        "units": "m",
                    },
                ),
                "single_target_alongship_angle": (
                    ["single_target"],
                    np.asarray(
                        traces["single_target_alongship_angle"],
                        dtype=np.float64,
                    ),
                    {
                        "long_name": (
                            "Single target arrival angle in the " "minor beam coordinate"
                        ),
                        "units": "arc_degree",
                        "valid_range": [-180.0, 180.0],
                    },
                ),
                "single_target_athwartship_angle": (
                    ["single_target"],
                    np.asarray(
                        traces["single_target_athwartship_angle"],
                        dtype=np.float64,
                    ),
                    {
                        "long_name": (
                            "Single target arrival angle in the " "major beam coordinate"
                        ),
                        "units": "arc_degree",
                        "valid_range": [-180.0, 180.0],
                    },
                ),
                "Sp": (
                    ["single_target"],
                    np.asarray(
                        traces["Sp"],
                        dtype=np.float64,
                    ),
                    {
                        "long_name": (
                            "Calculated Target Strength (re 1 m2) "
                            "uncompensated for off-axis angle"
                        ),
                        "units": "dB",
                        "comment": (
                            "Uncompensated target strength generated " "directly by BI500."
                        ),
                    },
                ),
                "TS": (
                    ["single_target"],
                    np.asarray(
                        traces["TS"],
                        dtype=np.float64,
                    ),
                    {
                        "long_name": (
                            "Calculated Target Strength (re 1 m2) "
                            "after compensation for off-axis angle"
                        ),
                        "units": "dB",
                        "comment": (
                            "Beam-compensated target strength generated " "directly by BI500."
                        ),
                    },
                ),
            },
            coords={
                "channel": (
                    ["channel"],
                    [channel_id],
                    self._varattrs["beam_coord_default"]["channel"],
                ),
                "ping_time": (
                    ["ping_time"],
                    ping_time,
                    self._varattrs["beam_coord_default"]["ping_time"],
                ),
                "range_sample": (
                    ["range_sample"],
                    np.arange(sv.shape[1]),
                    self._varattrs["beam_coord_default"]["range_sample"],
                ),
                "range_sample_bottom": (
                    ["range_sample_bottom"],
                    np.arange(sv_bottom.shape[1]),
                    {
                        "long_name": "Along-range bottom sample number, base 0",
                    },
                ),
                "single_target": (
                    ["single_target"],
                    single_target,
                    {
                        "long_name": "Single-target detection index",
                    },
                ),
            },
            attrs={
                "processing_function": "open_raw",
                "source_sonar_model": "BI500",
                "comment": (
                    "The BI500 system generated these calibrated "
                    "quantities directly; echopype did not recalibrate them."
                ),
            },
        )

        return set_time_encodings(ds)

    @staticmethod
    def _stack_samples(samples: list) -> np.ndarray:
        """Stack per-ping sample arrays, padding shorter pings with NaN."""
        n_pings = len(samples)
        max_len = max(len(sample) for sample in samples)
        stacked = np.full((n_pings, max_len), np.nan, dtype=np.float64)
        for i, sample in enumerate(samples):
            stacked[i, : len(sample)] = sample
        return stacked

    def _collect_target_traces(self) -> dict:
        """Collect BI500 single-target values and their source ping indices."""
        trace_fields = {
            "single_target_range": "TargetDepth",
            "TS": "CompTS",
            "Sp": "UncompTS",
            "single_target_alongship_angle": "Alongship",
            "single_target_athwartship_angle": "Athwartship",
        }

        collected = {name: [] for name in trace_fields}
        collected["ping_index"] = []

        trace_idx = 0

        for ping_index, count in enumerate(self.parser_obj.index_counts["echotrace_count"]):
            # The parser inserts one zero placeholder when a ping has no traces.
            if count == 0:
                trace_idx += 1
                continue

            for _ in range(count):
                collected["ping_index"].append(ping_index)

                for output_name, parser_name in trace_fields.items():
                    collected[output_name].append(
                        float(self.parser_obj.unpacked_data[parser_name][trace_idx])
                    )

                trace_idx += 1

        return collected

    def set_beam(self) -> List[xr.Dataset]:
        """Set the Sonar/Beam_group1 group."""
        parameters = self.parser_obj.parameters
        ping_data = self.parser_obj.ping_data
        vlog_data = self.parser_obj.vlog_data

        ping_time = self._get_ping_time()
        ping_time_vlog = self._get_ping_time_vlog()
        channel_id = self._get_channel_id()
        frequency = float(parameters["frequency"][0])

        ds = xr.Dataset(
            {
                "frequency_nominal": (
                    ["channel"],
                    [frequency],
                    {
                        "units": "Hz",
                        "long_name": "Transducer frequency",
                        "valid_min": 0.0,
                        "standard_name": "sound_frequency",
                    },
                ),
                "transceiver_channel_number": (
                    ["channel"],
                    [int(parameters["transceiver"][0])],
                    {"long_name": "Transceiver channel number"},
                ),
                "beam_type": (
                    ["channel"],
                    [0],
                    {
                        "long_name": "Beam type",
                        "flag_values": [0, 1],
                        "flag_meanings": ["Single beam", "Split aperture beam"],
                    },
                ),
                "echogram_type": (
                    ["ping_time"],
                    np.array(ping_data["echogram_type"], dtype=np.int64),
                    {"long_name": "Echogram data type"},
                ),
                "echogram_type_vlog": (
                    ["ping_time_vlog"],
                    np.array(vlog_data["echogram_type"], dtype=np.int64),
                    {"long_name": "Echogram data type from vlog"},
                ),
                "pelagic_upper": (
                    ["channel", "ping_time"],
                    np.array(ping_data["pelagic_upper"], dtype=np.float64)[np.newaxis, :],
                    {"long_name": "Pelagic echogram upper depth bound", "units": "m"},
                ),
                "pelagic_lower": (
                    ["channel", "ping_time"],
                    np.array(ping_data["pelagic_lower"], dtype=np.float64)[np.newaxis, :],
                    {"long_name": "Pelagic echogram lower depth bound", "units": "m"},
                ),
                "pelagic_upper_vlog": (
                    ["channel", "ping_time_vlog"],
                    np.array(vlog_data["pelagic_upper"], dtype=np.float64)[np.newaxis, :],
                    {"long_name": "Pelagic echogram upper depth bound from vlog", "units": "m"},
                ),
                "pelagic_lower_vlog": (
                    ["channel", "ping_time_vlog"],
                    np.array(vlog_data["pelagic_lower"], dtype=np.float64)[np.newaxis, :],
                    {"long_name": "Pelagic echogram lower depth bound from vlog", "units": "m"},
                ),
                "bottom_upper": (
                    ["channel", "ping_time"],
                    np.array(ping_data["bottom_upper"], dtype=np.float64)[np.newaxis, :],
                    {"long_name": "Bottom echogram upper depth bound", "units": "m"},
                ),
                "bottom_lower": (
                    ["channel", "ping_time"],
                    np.array(ping_data["bottom_lower"], dtype=np.float64)[np.newaxis, :],
                    {"long_name": "Bottom echogram lower depth bound", "units": "m"},
                ),
                "bottom_upper_vlog": (
                    ["channel", "ping_time_vlog"],
                    np.array(vlog_data["bottom_upper"], dtype=np.float64)[np.newaxis, :],
                    {"long_name": "Bottom echogram upper depth bound from vlog", "units": "m"},
                ),
                "bottom_lower_vlog": (
                    ["channel", "ping_time_vlog"],
                    np.array(vlog_data["bottom_lower"], dtype=np.float64)[np.newaxis, :],
                    {"long_name": "Bottom echogram lower depth bound from vlog", "units": "m"},
                ),
            },
            coords={
                "channel": (
                    ["channel"],
                    [channel_id],
                    self._varattrs["beam_coord_default"]["channel"],
                ),
                "ping_time": (
                    ["ping_time"],
                    ping_time,
                    self._varattrs["beam_coord_default"]["ping_time"],
                ),
                "ping_time_vlog": (
                    ["ping_time_vlog"],
                    ping_time_vlog,
                    {
                        "axis": "T",
                        "long_name": "Timestamps for vlog beam metadata",
                        "standard_name": "time",
                    },
                ),
            },
        )
        return [set_time_encodings(ds)]

    def set_vendor(self) -> xr.Dataset:
        """Set the Vendor_specific group."""
        parameters = self.parser_obj.parameters

        return xr.Dataset(
            {
                "start_latitude": (
                    [],
                    float(parameters["start_latitude"][0]),
                    {},
                ),
                "start_longitude": (
                    [],
                    float(parameters["start_longitude"][0]),
                    {},
                ),
                "start_distance": (
                    [],
                    float(parameters["start_distance"][0]),
                    {},
                ),
                "stop_latitude": (
                    [],
                    float(parameters["stop_latitude"][0]),
                    {},
                ),
                "stop_longitude": (
                    [],
                    float(parameters["stop_longitude"][0]),
                    {},
                ),
                "stop_distance": (
                    [],
                    float(parameters["stop_distance"][0]),
                    {},
                ),
            }
        )

    def set_provenance(self) -> xr.Dataset:
        """Set the Provenance group."""
        prov_dict = echopype_prov_attrs(process_type="conversion")

        source_files = [
            self.parser_obj.file_type_map[file_type]
            for file_type in self.parser_obj.file_types
            if self.parser_obj.file_type_map.get(file_type)
        ]
        if not source_files:
            source_files = [self.input_file]

        files_vars = source_files_vars(source_files)
        parameters = self.parser_obj.parameters

        ds = xr.Dataset(
            data_vars={
                **files_vars["source_files_var"],
                "nation_code": (
                    [],
                    int(parameters["nation"][0]),
                    {
                        "long_name": "Nation code",
                        "comment": "Reference table nation code from BI500 -Info file.",
                    },
                ),
                "ship_code": (
                    [],
                    int(parameters["ship"][0]),
                    {
                        "long_name": "Ship code",
                        "comment": "Reference table ship code from BI500 -Info file.",
                    },
                ),
                "survey_code": (
                    [],
                    int(parameters["survey"][0]),
                    {
                        "long_name": "Survey code",
                        "comment": "Reference table survey code from BI500 -Info file.",
                    },
                ),
            },
            coords=files_vars["source_files_coord"],
            attrs=prov_dict,
        )
        return ds
