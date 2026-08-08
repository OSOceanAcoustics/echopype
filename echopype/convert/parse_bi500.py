import re
from collections import defaultdict
from struct import unpack

import fsspec
import numpy as np

from ..utils.log import _init_logger
from ..utils.misc import camelcase2snakecase
from .parse_base import ParseBase

logger = _init_logger(__name__)

FILENAME_DATETIME_BI500 = (
    "?(?<prefix>.*)?-?F(?P<frequency>\\w+)?-?T(?P<transducer>\\w+)?"
    "-?D(?P<date>\\w+)?-?T(?P<time>\\w+)"
)

BI500_FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>.+)"
    r"-F(?P<frequency>\d+)"
    r"-T(?P<transducer>\d+)"
    r"-D(?P<date>\d{8})"
    r"-T(?P<time>\d{6})"
    r"(?P<file_type>-(?:Data|Info|Ping|Vlog|Snap|Work))$"
)

FILE_TYPES = ["-Data", "-Info", "-Ping", "-Vlog", "-Snap", "-Work"]

REQUIRED_FILES = ["-Data", "-Info", "-Ping"]

# Common BI500 Ping and Vlog parameters for unpacking
PV_FILE_FORMAT = ">llfffflffllffllll"

PV_FILE_SIZE = 68

PV_FIELDS = (
    "Date",
    "Time",
    "Distance",
    "Latitude",
    "Longitude",
    "BottomDepth",
    "EchogramType",
    "PelagicUpper",
    "PelagicLower",
    "PelagicCount",
    "PelagicOffset",
    "BottomUpper",
    "BottomLower",
    "BottomCount",
    "BottomOffset",
    "EchotraceCount",
    "EchotraceOffset",
)


class ParseBI500(ParseBase):
    """Class for converting data from Bergen Integrator (BI500) software."""

    def __init__(
        self,
        file,
        storage_options={},
        sonar_model="BI500",
        file_meta=None,
        bot_file="",
        idx_file="",
        **kwargs,
    ):
        super().__init__(file, storage_options, sonar_model)

        self.timestamp_pattern = FILENAME_DATETIME_BI500
        self.file_types = FILE_TYPES

        # BI500 stores each transceiver/channel in its own companion-file set.
        # file_set_map groups those sets for a single acquisition.
        self.file_set_map = {}
        self.channel_data = {}
        self.acquisition_key = None

        # Keep the original single-channel attributes as aliases to the first
        # parsed channel. This preserves the existing BI500 code path/API while
        # allowing SetGroupsBI500 to iterate over all parsed channels.
        self.file_type_map = defaultdict(None)
        self.parameters = defaultdict(list)
        self.ping_data = defaultdict(list)
        self.vlog_data = defaultdict(list)
        self.index_counts = defaultdict(list)
        self.unpacked_data = defaultdict(list)

        self.fsmap = self._validate_folder_path(file)
        self.sonar_type = "BI500"

    def _validate_folder_path(self, folder_path):
        """Validate a folder containing one BI500 acquisition."""
        fsmap = fsspec.get_mapper(folder_path, **self.storage_options)
        try:
            all_files = fsmap.fs.ls(folder_path)
        except NotADirectoryError:
            raise ValueError(
                "Expecting a folder containing at least '-Data', '-Info' and '-Ping' files, "
                f"but got {folder_path}"
            )

        self._group_file_sets(all_files)
        return fsmap

    def _group_file_sets(self, all_files):
        """Group companion files by BI500 frequency/transceiver file set."""
        logger.info("Found the following files in the folder:")

        file_set_map = {}
        acquisition_keys = set()

        for file in all_files:
            file_name = file if isinstance(file, str) else file.get("name")
            basename = file_name.replace("\\", "/").rsplit("/", 1)[-1]

            match = BI500_FILENAME_PATTERN.match(basename)
            if match is None:
                continue

            info = match.groupdict()
            acquisition_key = (info["prefix"], info["date"], info["time"])
            file_set_key = (info["frequency"], info["transducer"])

            acquisition_keys.add(acquisition_key)
            file_set_map.setdefault(file_set_key, {})
            file_set_map[file_set_key][info["file_type"]] = file_name

            logger.info(f"Found file: {file_name}")

        if not file_set_map:
            raise ValueError(
                "Expecting a folder containing at least '-Data', '-Info' and '-Ping' files, "
                "but no BI500 file set was found."
            )

        if len(acquisition_keys) != 1:
            raise ValueError(
                "BI500 open_raw expects files from a single acquisition, "
                f"but found {len(acquisition_keys)} acquisition groups."
            )

        for file_set_key, file_type_map in file_set_map.items():
            missing = [file_type for file_type in REQUIRED_FILES if file_type not in file_type_map]
            if missing:
                raise ValueError(
                    f"BI500 file set {file_set_key} is missing required files: {missing}"
                )

        self.acquisition_key = next(iter(acquisition_keys))
        self.file_set_map = file_set_map

    def load_BI500_info(self, file_type_map=None, parameters=None):
        """Parse one BI500 Info file."""
        if file_type_map is None:
            file_type_map = self.file_type_map
        if parameters is None:
            parameters = self.parameters

        # BI500 Info file parameters for unpacking
        info_file_format = ">llllllllfffllfff"
        info_vars = (
            "Release",
            "Nation",
            "Ship",
            "Survey",
            "Frequency",
            "Transceiver",
            "StartDate",
            "StartTime",
            "StartLatitude",
            "StartLongitude",
            "StartDistance",
            "StopDate",
            "StopTime",
            "StopLatitude",
            "StopLongitude",
            "StopDistance",
        )

        with self.fsmap.fs.open(file_type_map["-Info"], mode="rb") as bi500_info:
            info_data = unpack(info_file_format, bi500_info.read())

        for name, data in zip(info_vars, info_data):
            parameters[camelcase2snakecase(name)].append(data)

    def load_BI500_ping(
        self,
        file_type_map=None,
        ping_data=None,
        index_counts=None,
    ):
        """Parse one BI500 Ping file."""
        if file_type_map is None:
            file_type_map = self.file_type_map
        if ping_data is None:
            ping_data = self.ping_data
        if index_counts is None:
            index_counts = self.index_counts

        with self.fsmap.fs.open(file_type_map["-Ping"], mode="rb") as bi500_ping:
            eof = False
            while not eof:
                data_read = bi500_ping.read(PV_FILE_SIZE)
                if data_read:
                    data = unpack(PV_FILE_FORMAT, data_read)
                    for name, value in zip(PV_FIELDS, data):
                        key = camelcase2snakecase(name)
                        if name in ("PelagicCount", "BottomCount", "EchotraceCount"):
                            index_counts[key].append(value)
                        ping_data[key].append(value)
                else:
                    eof = True

    def load_BI500_vlog(self, file_type_map=None, vlog_data=None):
        """Parse one BI500 Vlog file."""
        if file_type_map is None:
            file_type_map = self.file_type_map
        if vlog_data is None:
            vlog_data = self.vlog_data

        vlog_file = file_type_map.get("-Vlog")
        if vlog_file is None:
            return

        with self.fsmap.fs.open(vlog_file, mode="rb") as bi500_vlog:
            eof = False
            while not eof:
                data_read = bi500_vlog.read(PV_FILE_SIZE)
                if data_read:
                    data = unpack(PV_FILE_FORMAT, data_read)
                    for name, value in zip(PV_FIELDS, data):
                        vlog_data[camelcase2snakecase(name)].append(value)
                else:
                    eof = True

    def _load_BI500_data(self, file_type_map, index_counts, unpacked_data):
        """Parse one BI500 Data file."""
        start_format = ">"
        trace_vars = ("TargetDepth", "CompTS", "UncompTS", "Alongship", "Athwartship")

        with self.fsmap.fs.open(file_type_map["-Data"], mode="rb") as bi500_data:
            num_pings = len(index_counts["pelagic_count"])

            for i in range(num_pings):
                pelagic_count = index_counts["pelagic_count"][i]
                bottom_count = index_counts["bottom_count"][i]
                trace_count = index_counts["echotrace_count"][i]

                ping_size = pelagic_count * 2 + bottom_count * 2 + trace_count * 20
                loaded_data = bi500_data.read(ping_size)

                if not loaded_data:
                    break

                pelagic_format = start_format + "h" * pelagic_count
                bottom_format = "h" * bottom_count
                trace_format = "fffff" * trace_count
                ping_format = pelagic_format + bottom_format + trace_format

                values = np.asarray(unpack(ping_format, loaded_data), dtype=np.float64)

                # Convert pelagic/bottom power samples to dB units.
                values[: pelagic_count + bottom_count] *= 10 * np.log10(2) / 256

                unpacked_data["pelagic"].append(values[:pelagic_count])
                unpacked_data["bottom"].append(values[pelagic_count : pelagic_count + bottom_count])

                for trace_num in range(trace_count):
                    start = pelagic_count + bottom_count + trace_num * 5
                    trace_data = values[start : start + 5]

                    for name, value in zip(trace_vars, trace_data):
                        unpacked_data[name].append(value)

                # Preserve the existing placeholder convention when no
                # single-target trace data are available for a ping.
                if trace_count == 0:
                    for name in trace_vars:
                        unpacked_data[name].append(float(0))

    def parse_raw(self):
        """Parse all BI500 channel file sets for one acquisition."""
        for _, file_type_map in sorted(self.file_set_map.items()):
            parameters = defaultdict(list)
            ping_data = defaultdict(list)
            vlog_data = defaultdict(list)
            index_counts = defaultdict(list)
            unpacked_data = defaultdict(list)

            self.load_BI500_info(
                file_type_map=file_type_map,
                parameters=parameters,
            )
            self.load_BI500_ping(
                file_type_map=file_type_map,
                ping_data=ping_data,
                index_counts=index_counts,
            )
            self.load_BI500_vlog(
                file_type_map=file_type_map,
                vlog_data=vlog_data,
            )
            self._load_BI500_data(
                file_type_map=file_type_map,
                index_counts=index_counts,
                unpacked_data=unpacked_data,
            )

            frequency = int(parameters["frequency"][0])
            transceiver = int(parameters["transceiver"][0])
            channel_id = f"BI500-F{frequency}-T{transceiver:02d}"

            self.channel_data[channel_id] = {
                "file_type_map": file_type_map,
                "parameters": parameters,
                "ping_data": ping_data,
                "vlog_data": vlog_data,
                "index_counts": index_counts,
                "unpacked_data": unpacked_data,
            }

        # Preserve the current single-channel attributes as aliases to the
        # first channel so existing BI500 API/grouping code remains usable.
        first_channel_data = next(iter(self.channel_data.values()))
        self.file_type_map = first_channel_data["file_type_map"]
        self.parameters = first_channel_data["parameters"]
        self.ping_data = first_channel_data["ping_data"]
        self.vlog_data = first_channel_data["vlog_data"]
        self.index_counts = first_channel_data["index_counts"]
        self.unpacked_data = first_channel_data["unpacked_data"]
