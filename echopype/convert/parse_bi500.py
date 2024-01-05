from collections import defaultdict
from struct import unpack

import fsspec

from ..utils.log import _init_logger
from ..utils.misc import camelcase2snakecase
from .parse_base import ParseBase

logger = _init_logger(__name__)

FILENAME_DATETIME_BI500 = "?(?<prefix>.*)?-?F(?P<frequency>\\w+)?-?T(?P<transducer>\\w+)?- \
                        ?D(?P<date>\\w+)?-?T(?P<time>\\w+)"

FILE_TYPES = ["-Data", "-Info", "-Ping", "-Vlog", "-Snap", "-Work"]


class ParseBI500(ParseBase):
    """Class for converting data from Bergen Integrator (BI500) software."""

    def __init__(self, file, file_meta, storage_options={}, sonar_model="BI500"):
        super().__init__(file, storage_options, sonar_model)

        self.timestamp_pattern = FILENAME_DATETIME_BI500
        self.file_types = FILE_TYPES
        self.file_type_map = defaultdict(None)

        self.parameters = defaultdict(list)
        self.ping_counts = defaultdict(list)
        self.vlog_counts = defaultdict(list)
        self.index_counts = defaultdict(list)
        self.unpacked_data = defaultdict(list)
        self.fsmap = self._validate_folder_path(file)
        self.sonar_type = "BI500"

    def _validate_folder_path(self, folder_path):
        """Validate the folder path."""
        fsmap = fsspec.get_mapper(folder_path, **self.storage_options)
        try:
            all_files = fsmap.fs.ls(folder_path)
        except NotADirectoryError:
            raise ValueError(
                "Expecting a folder containing at least '-Data' and '-Info' files, "
                f"but got {folder_path}"
            )

        if isinstance(all_files[0], str):
            reqd_files = [
                file for file in all_files if file.endswith("-Data") or file.endswith("-Info")
            ]
        else:
            reqd_files = [
                file
                for file in all_files
                if file.get("name").endswith("-Data") or file.get("name").endswith("-Info")
            ]

        if len(reqd_files) < 2:
            raise ValueError(
                "Expecting a folder containing at least '-Data' and '-Info' files, "
                f"but got {folder_path} with at least one required file missing."
            )

        self._print_files(all_files)
        return fsmap

    def _print_files(self, all_files):
        """Prints all the files found to be parsed in the folder to console."""
        logger.info("Found the following files in the folder:")
        for file in all_files:
            file_name = file if isinstance(file, str) else file.get("name")
            for file_type in self.file_types:
                if file_name.endswith(file_type):
                    self.file_type_map[file_type] = file_name
                    logger.info(file_name)

    def load_BI500_info(self):
        """
        Parses the BI500 Info file.
        """

        # BI500 Info file parameters for unpacking
        INFO_FILE_FORMAT = ">llllllllfffllfff"
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

        bi500_info = self.fsmap.fs.open(self.file_type_map["-Info"], mode="rb")

        # Unpack the BI500 Info file
        info_data = unpack(INFO_FILE_FORMAT, bi500_info.read())
        for name, data in zip(info_vars, info_data):
            self.parameters[camelcase2snakecase(name)].append(data)

    def load_BI500_ping(self):
        """
        Parses the BI500 Ping file.
        """

        # BI500 Ping file parameters for unpacking
        PING_FILE_FORMAT = ">llfffflffllffllll"
        PING_FILE_SIZE = 68
        ping_vars = (
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
            "PelanvicOffset",
            "BottomUpper",
            "BottomLower",
            "BottomCount",
            "BottomOffset",
            "EchotraceCount",
            "EchotraceOffset",
        )

        bi500_ping = self.fsmap.fs.open(self.file_type_map["-Ping"], mode="rb")

        # Unpack the BI500 Ping file
        eof = False
        while not eof:
            data_read = bi500_ping.read(PING_FILE_SIZE)
            if data_read:
                data = unpack(PING_FILE_FORMAT, data_read)
                for name, data in zip(ping_vars, data):
                    if name == "PelagicCount" or name == "BottomCount" or name == "EchotraceCount":
                        self.ping_counts[camelcase2snakecase(name)].append(data)
                    else:
                        self.parameters[camelcase2snakecase(name)].append(data)
            else:
                eof = True
        # Set the index counts equal to the ping counts
        self.index_counts = self.ping_counts

    def load_BI500_vlog(self):
        """
        Parses the BI500 Vlog file.
        """

        # BI500 Info file parameters for unpacking
        VLOG_FILE_FORMAT = ">llfffflffllffllll"
        VLOG_FILE_SIZE = 68
        vlog_vars = (
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
            "PelanvicOffset",
            "BottomUpper",
            "BottomLower",
            "BottomCount",
            "BottomOffset",
            "EchotraceCount",
            "EchotraceOffset",
        )

        bi500_vlog = self.fsmap.fs.open(self.file_type_map["-Vlog"], mode="rb")

        # Unpack the BI500 info file
        eof = False
        while not eof:
            data_read = bi500_vlog.read(VLOG_FILE_SIZE)
            if data_read:
                data = unpack(VLOG_FILE_FORMAT, data_read)
                for name, data in zip(vlog_vars, data):
                    if name == "PelagicCount" or name == "BottomCount" or name == "EchotraceCount":
                        self.vlog_counts[camelcase2snakecase(name)].append(data)
                    else:
                        self.parameters[camelcase2snakecase(name)].append(data)
            else:
                eof = True

    def parse_raw(self):
        """
        Parses the BI500 Data file.
        """

        # BI500 Data file parameters for unpacking
        START_FORMAT = ">"
        trace_vars = ("TargetDepth", "CompTS", "UncompTS", "Alongship", "Athwartship")
        self.load_BI500_info()
        self.load_BI500_ping()
        bi500_data = self.fsmap.fs.open(self.file_type_map["-Data"], mode="rb")

        # Unpack the BI500 Data file
        num_pings = len(self.index_counts["pelagic_count"])
        for i in range(num_pings):
            PELAGIC_COUNT = self.index_counts["pelagic_count"][i]
            BOTTOM_COUNT = self.index_counts["bottom_count"][i]
            TRACE_COUNT = self.index_counts["echotrace_count"][i]
            ping_size = PELAGIC_COUNT * 2 + BOTTOM_COUNT * 2 + TRACE_COUNT * 20
            loaded_data = bi500_data.read(ping_size)
            if loaded_data:
                PELAGIC_FORMAT = START_FORMAT + "h" * PELAGIC_COUNT
                BOTTOM_FORMAT = "h" * BOTTOM_COUNT
                TRACE_FORMAT = "fffff" * TRACE_COUNT
                PING_FORMAT = PELAGIC_FORMAT + BOTTOM_FORMAT + TRACE_FORMAT
                unpacked_data = unpack(PING_FORMAT, loaded_data)
                self.unpacked_data["pelagic"].append(unpacked_data[:PELAGIC_COUNT])
                self.unpacked_data["bottom"].append(
                    unpacked_data[PELAGIC_COUNT : PELAGIC_COUNT + BOTTOM_COUNT]
                )
                for trace_num in range(TRACE_COUNT):
                    trace_data = unpacked_data[
                        PELAGIC_COUNT
                        + BOTTOM_COUNT
                        + trace_num * 5 : PELAGIC_COUNT
                        + BOTTOM_COUNT
                        + (trace_num + 1) * 5
                    ]
                    for name, data in zip(trace_vars, trace_data):
                        self.unpacked_data[name].append(data)

                # adding zeros when trace data is not available
                if TRACE_COUNT == 0:
                    for name in trace_vars:
                        self.unpacked_data[name].append(float(0))
            else:
                break
