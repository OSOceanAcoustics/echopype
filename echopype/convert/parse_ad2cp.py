from typing import BinaryIO, Union, Callable, List, Optional, Any
from enum import Enum, unique, auto
import struct
import math

import numpy as np
from numpy.core.numeric import NaN
import xarray as xr

from .parse_base import ParseBase


@unique
class BurstAverageDataRecordVersion(Enum):
    VERSION2 = auto()  # Burst/Average Data Record Definition (DF2)
    VERSION3 = auto()  # Burst/Average Data Record Definition (DF3)


@unique
class DataRecordType(Enum):
    BURST_VERSION2 = auto()
    BURST_VERSION3 = auto()
    AVERAGE_VERSION2 = auto()
    AVERAGE_VERSION3 = auto()
    BOTTOM_TRACK = auto()
    STRING = auto()


@unique
class DataType(Enum):
    RAW_BYTES = auto()
    STRING = auto()
    SIGNED_INTEGER = auto()
    UNSIGNED_INTEGER = auto()
    FLOAT = auto()
    # TODO: figure out what this is in binary
    SIGNED_FRACTION = auto()


RAW_BYTES = DataType.RAW_BYTES
STRING = DataType.STRING
SIGNED_INTEGER = DataType.SIGNED_INTEGER
UNSIGNED_INTEGER = DataType.UNSIGNED_INTEGER
FLOAT = DataType.FLOAT
SIGNED_FRACTION = DataType.SIGNED_FRACTION


@unique
class Dimension(Enum):
    TIME = "time"
    BEAM = "beam"
    RANGE_BIN = "range_bin"
    NUM_ALTIMETER_SAMPLES = "num_altimeter_samples"


class Field:
    def __init__(self, field_name: Optional[str], field_entry_size_bytes: Union[int, Callable[["Ad2cpDataPacket"], int]], field_entry_data_type: DataType, *, field_shape: Union[List[int], Callable[["Ad2cpDataPacket"], List[int]]] = [], field_exists_predicate: Callable[["Ad2cpDataPacket"], bool] = lambda _: True):
        """
        field_name: Name of the field. If None, the field is parsed but ignored
        field_entry_size_bytes: Size of each entry within the field, in bytes. 
            In most cases, the entry is the field itself, but sometimes the field
            contains a list of entries.
        field_entry_data_type: Data type of each entry in the field
        field_shape: Shape of entries within the field. 
            [] (the default) means the entry is the field itself,
            [n] means the field consists of a list of n entries,
            [n, m] means the field consists of a two dimensional array with
                n number of m length arrays,
            etc.
        field_exists_predicate: Tests to see whether the field should be parsed at all
        """

        self.field_name = field_name
        self.field_entry_size_bytes = field_entry_size_bytes
        self.field_entry_data_type = field_entry_data_type
        self.field_shape = field_shape
        self.field_exists_predicate = field_exists_predicate

    @staticmethod
    def dimensions(field_name: str, data_record_format: List["Field"]) -> List[Dimension]:
        # TODO: altimeter spare (but it's not included in final dataset)
        if data_record_format == Ad2cpDataPacket.BOTTOM_TRACK_DATA_RECORD_FORMAT:
            if field_name in ("velocity_data", "distance_data", "figure_of_merit_data"):
                return [Dimension.TIME, Dimension.BEAM]
        else:
            if field_name in ("velocity_data", "amplitude_data", "correlation_data"):
                return [Dimension.TIME, Dimension.BEAM, Dimension.RANGE_BIN]
            elif field_name in ("echo_sounder_data", "percentage_good_data"):
                return [Dimension.TIME, Dimension.RANGE_BIN]
            elif field_name == "altimeter_raw_data_samples":
                return [Dimension.TIME, Dimension.NUM_ALTIMETER_SAMPLES]
        return [Dimension.TIME]


F = Field  # use F instead of Field to make the repeated fields easier to read


class NoMorePackets(Exception):
    pass


class ParseAd2cp(ParseBase):
    def __init__(self, *args, burst_average_data_record_version: BurstAverageDataRecordVersion = BurstAverageDataRecordVersion.VERSION3, **kwargs):
        super().__init__(args, kwargs)
        self.burst_average_data_record_version = burst_average_data_record_version
        self.packets = []
        self.ds = None

    def parse_raw(self):
        # i = 0
        with open(self.source_file[0], "rb") as f:
            while True:
                try:
                    self.packets.append(Ad2cpDataPacket(f, self.burst_average_data_record_version))
                except NoMorePackets:
                    break
                # else:
                #     i += 1
                #     print(i)
                #     print(self.packets[0].data["id"])

        # self.ds = xr.Dataset(coords={"time": []}, attrs={"string_data": dict()})
        self.ds = None
        string_data = dict()

        # TODO: burst, average, and echosounder now have different range_bins

        max_beam_count = max(
            self.packets, key=lambda p: p.data.get("num_beams", 0)).data.get("num_beams", 0)
        max_range_bin_count = max(
            self.packets, key=lambda p: p.data.get("num_cells", 0)).data.get("num_cells", 0)
        max_range_bin_echo_sounder_count = max(
            self.packets, key=lambda p: p.data.get("num_echo_sounder_cells", 0)).data.get("num_echo_sounder_cells", 0)
        # fill packets with missing beams with nans up to the max beam count
        # TODO: this is awful
        for packet in self.packets:
            for field_name, field_value in packet.data.items():
                if isinstance(field_value, np.ndarray):
                    # if field_name in ("echo_sounder_data", "percentage_good_data"):
                    #     print(field_name)
                    #     print(field_value)
                    #     raise SystemExit
                    if field_name in ("echo_sounder_data", "percentage_good_data"):
                        # print(max_range_bin_echo_sounder_count - field_value.size)
                        if field_value.size < max_range_bin_echo_sounder_count:
                            packet.data[field_name] = np.concatenate((field_value, [np.nan] * (max_range_bin_echo_sounder_count - field_value.size)))
                    else:
                        if len(field_value[0]) < max_range_bin_count:
                            field_value = [np.concatenate(
                                (beam, [np.nan] * (max_range_bin_count - len(field_value[0])))) for beam in field_value]
                            packet.data[field_name] = np.array(field_value)
                        if len(field_value) < max_beam_count:
                            field_value = np.concatenate(
                                (field_value, [[np.nan] * max_range_bin_count for _ in range(max_beam_count - len(field_value))]))
                            packet.data[field_name] = np.array(field_value)

        for packet in self.packets:
            if packet.data_record_type == DataRecordType.STRING:
                # self.ds.attrs["string_data"][packet.data["string_data_id"]
                #                         ] = packet.data["string_data"]
                string_data[packet.data["string_data_id"]] = packet.data["string_data"]
            else:
                if packet.data_record_type in (DataRecordType.BURST_VERSION2, DataRecordType.BURST_VERSION3):
                    time = "time_burst"
                else:
                    time = "time_average"
                data_vars = dict()
                for field_name, field_value in packet.data.items():
                    # TODO might not work with altimeter_spare
                    # add dimension names to data vars for xarray
                    data_vars[field_name] = (tuple(dim.value for dim in Field.dimensions(
                        field_name, packet.data_record_format)), [field_value])
                
                new_packet = xr.Dataset(
                    data_vars=data_vars,
                    coords={"time": [packet.timestamp],
                            time: [packet.timestamp]}
                )
                # print(self.ds)
                # print(new_packet)
                if self.ds is None:
                    self.ds = new_packet
                else:
                    self.ds = xr.concat([
                        new_packet,
                        self.ds
                    ], dim="time")
                    # try:
                    #     self.ds = xr.concat([
                    #         new_packet,
                    #         self.ds
                    #     ], dim="time")
                    # except:
                        # pass
                        # print(self.ds)
                        # print(new_packet)
                        # print(packet.data.get("num_echo_sounder_cells"))
                        # raise SystemExit

                # self.ds = xr.merge([
                #     self.ds,
                #     new_packet
                # ], compat="override", combine_attrs="override")
        if self.ds is None:
            self.ds = xr.Dataset(attrs={"string_data": dict()})
        else:
            self.ds.attrs["string_data"] = string_data

class Ad2cpDataPacket:
    def __init__(self, f: BinaryIO, burst_average_data_record_version: BurstAverageDataRecordVersion):
        self.burst_average_data_record_version = burst_average_data_record_version
        self.data_record_type: Optional[DataRecordType] = None
        self.data = dict()
        self._read_data_record_header(f)
        self._read_data_record(f)

    @property
    def timestamp(self) -> np.datetime64:
        year = self.data["year"] + 1900
        month = self.data["month"] + 1
        day = self.data["day"]
        hour = self.data["hour"]
        minute = self.data["minute"]
        seconds = self.data["seconds"]
        microsec100 = self.data["microsec100"]
        return np.datetime64(f"{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{seconds:02}.{microsec100:04}")

    def _read_data_record_header(self, f: BinaryIO):
        """
        Reads the header part of the AD2CP packet from the stream
        """

        self.data_record_format = self.HEADER_FORMAT
        raw_header = self._read_data(f, self.data_record_format)
        # don't include the last 2 bytes, which is the header checksum itself
        calculated_checksum = self.checksum(raw_header[: -2])
        expected_checksum = self.data["header_checksum"]
        assert calculated_checksum == expected_checksum, f"invalid header checksum: found {calculated_checksum}, expected {expected_checksum}"

    def _read_data_record(self, f: BinaryIO):
        """
        Reads the data record part of the AD2CP packet from the stream
        """

        if self.data["id"] in (0x15, 0x18):  # burst
            if self.burst_average_data_record_version == BurstAverageDataRecordVersion.VERSION2:
                self.data_record_format = self.BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT
                self.data_record_type = DataRecordType.BURST_VERSION2
            elif self.burst_average_data_record_version == BurstAverageDataRecordVersion.VERSION3:
                self.data_record_format = self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
                self.data_record_type = DataRecordType.BURST_VERSION3
            else:
                raise ValueError("invalid burst/average data record version")
        elif self.data["id"] == 0x16:  # average
            if self.burst_average_data_record_version == BurstAverageDataRecordVersion.VERSION2:
                self.data_record_format = self.BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT
                self.data_record_type = DataRecordType.AVERAGE_VERSION2
            elif self.burst_average_data_record_version == BurstAverageDataRecordVersion.VERSION3:
                self.data_record_format = self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
                self.data_record_type = DataRecordType.AVERAGE_VERSION3
            else:
                raise ValueError("invalid burst/average data record version")
        elif self.data["id"] in (0x17, 0x1b):  # bottom track
            self.data_record_format = self.BOTTOM_TRACK_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.BOTTOM_TRACK
        elif self.data["id"] == 0x1a:  # burst altimeter
            # altimeter is only supported by burst/average version 3
            self.data_record_format = self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.BURST_VERSION3
        elif self.data["id"] == 0x1c:  # echosounder
            # echosounder is only supported by burst/average version 3
            self.data_record_format = self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.AVERAGE_VERSION3
        elif self.data["id"] == 0x1d:  # dvl water track record
            # TODO: is this correct?
            self.data_record_format = self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.AVERAGE_VERSION3
        elif self.data["id"] == 0x1e:  # altimeter
            # altimeter is only supported by burst/average version 3
            self.data_record_format = self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.AVERAGE_VERSION3
        elif self.data["id"] == 0x1f:  # average altimeter
            self.data_record_format = self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.AVERAGE_VERSION3
        elif self.data["id"] == 0xa0:  # string data
            self.data_record_format = self.STRING_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.STRING
        else:
            raise ValueError("invalid data record type id: 0x{:02x}".format(self.data["id"]))

        raw_data_record = self._read_data(f, self.data_record_format)
        calculated_checksum = self.checksum(raw_data_record)
        expected_checksum = self.data["data_record_checksum"]
        # if calculated_checksum != expected_checksum:
        #     print(self.data["data_record_size"])
        #     print(len(raw_data_record))
        #     print(self.data["num_cells"])
        assert calculated_checksum == expected_checksum, f"invalid data record checksum: found {calculated_checksum}, expected {expected_checksum}"

    def _read_data(self, f: BinaryIO, data_format: List[Field]) -> bytes:
        """
        Reads data from the stream, interpreting the data using the given format
        """

        raw_bytes = bytes()  # combination of all raw fields
        for field_format in data_format:
            field_name = field_format.field_name
            field_entry_size_bytes = field_format.field_entry_size_bytes
            field_entry_data_type = field_format.field_entry_data_type
            field_shape = field_format.field_shape
            field_exists_predicate = field_format.field_exists_predicate
            if not field_exists_predicate(self):
                continue
            if callable(field_entry_size_bytes):
                field_entry_size_bytes = field_entry_size_bytes(self)
            if callable(field_shape):
                field_shape = field_shape(self)

            raw_field = self._read_exact(
                f, field_entry_size_bytes * math.prod(field_shape))
            raw_bytes += raw_field
            if len(field_shape) == 0:
                parsed_field = self._parse(raw_field, field_entry_data_type)
            else:
                # split the field into entries of size field_entry_size_bytes
                raw_field_entries = [raw_field[i * field_entry_size_bytes:(
                    i + 1) * field_entry_size_bytes] for i in range(math.prod(field_shape))]
                # parse each entry individually
                parsed_field_entries = [self._parse(
                    raw_field_entry, field_entry_data_type) for raw_field_entry in raw_field_entries]
                # reshape the list of entries into the correct shape
                parsed_field = np.reshape(parsed_field_entries, field_shape)
            # we cannot check for this before reading because some fields are placeholder fields
            # which, if not read in the correct order with other fields, will offset the rest of the data
            if field_name is not None:
                self.data[field_name] = parsed_field
                self._postprocess(field_name)

        return raw_bytes

    @staticmethod
    def _parse(value: bytes, data_type: DataType) -> Any:
        """
        Parses raw bytes into a value given its data type
        """

        # all numbers are little endian
        if data_type == DataType.RAW_BYTES:
            return value
        elif data_type == DataType.STRING:
            return value.decode("utf-8")
        elif data_type == DataType.SIGNED_INTEGER:
            return int.from_bytes(
                value, byteorder="little", signed=True)
        elif data_type == DataType.UNSIGNED_INTEGER:
            return int.from_bytes(
                value, byteorder="little", signed=False)
        elif data_type == DataType.FLOAT and len(value) == 4:
            return struct.unpack("<f", value)
        elif data_type == DataType.FLOAT and len(value) == 8:
            return struct.unpack("<d", value)
        elif data_type == DataType.SIGNED_FRACTION:
            # TODO: ??????
            pass
        else:
            # unreachable
            raise RuntimeError

    @staticmethod
    def _read_exact(f: BinaryIO, total_num_bytes_to_read: int) -> bytes:
        """
        Drives a stream until an exact amount of bytes is read from it.
        This is necessary because a single read may not return the correct number of bytes.
        """

        all_bytes_read = bytes()
        if total_num_bytes_to_read <= 0:
            return all_bytes_read
        last_bytes_read = None
        while last_bytes_read is None or (len(last_bytes_read) > 0 and len(all_bytes_read) < total_num_bytes_to_read):
            last_bytes_read = f.read(
                total_num_bytes_to_read - len(all_bytes_read))
            if len(last_bytes_read) == 0:
                # 0 bytes read with non-0 bytes requested means eof
                raise NoMorePackets
            else:
                all_bytes_read += last_bytes_read
        return all_bytes_read

    def _postprocess(self, field_name):
        """
        Calculates values based on parsed data. This should be called immediately after
        parsing each field in a data record.
        """

        if self.data_record_format == self.BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT:
            if field_name == "configuration":
                self.data["pressure_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0001
                self.data["temperature_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0010
                self.data["compass_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0100
                self.data["tilt_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_1000
                self.data["velocity_data_included"] = self.data["configuration"] & 0b0000_0000_0010_0000
                self.data["amplitude_data_included"] = self.data["configuration"] & 0b0000_0000_0100_0000
                self.data["correlation_data_included"] = self.data["configuration"] & 0b0000_0000_1000_0000
                # self.data["pressure_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0001 > 0
                # self.data["temperature_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0010 > 0
                # self.data["compass_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0100 > 0
                # self.data["tilt_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_1000 > 0
                # self.data["velocity_data_included"] = self.data["configuration"] & 0b0000_0000_0010_0000 > 0
                # self.data["amplitude_data_included"] = self.data["configuration"] & 0b0000_0000_0100_0000 > 0
                # self.data["correlation_data_included"] = self.data["configuration"] & 0b0000_0000_1000_0000 > 0
            elif field_name == "num_beams_and_coordinate_system_and_num_cells":
                self.data["num_cells"] = self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b0000_0011_1111_1111
                self.data["coordinate_system"] = (
                    self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b0000_1100_0000_0000) >> 10
                self.data["num_beams"] = (
                    self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b1111_0000_0000_0000) >> 12
        elif self.data_record_format == self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT:
            if field_name == "configuration":
                self.data["pressure_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0001
                self.data["temperature_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0010
                self.data["compass_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0100
                self.data["tilt_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_1000
                self.data["velocity_data_included"] = self.data["configuration"] & 0b0000_0000_0010_0000
                self.data["amplitude_data_included"] = self.data["configuration"] & 0b0000_0000_0100_0000
                self.data["correlation_data_included"] = self.data["configuration"] & 0b0000_0000_1000_0000
                self.data["altimeter_data_included"] = self.data["configuration"] & 0b0000_0001_0000_0000
                self.data["altimeter_raw_data_included"] = self.data["configuration"] & 0b0000_0010_0000_0000
                self.data["ast_data_included"] = self.data["configuration"] & 0b0000_0100_0000_0000
                self.data["echo_sounder_data_included"] = self.data["configuration"] & 0b0000_1000_0000_0000
                self.data["ahrs_data_included"] = self.data["configuration"] & 0b0001_0000_0000_0000
                self.data["percentage_good_data_included"] = self.data["configuration"] & 0b0010_0000_0000_0000
                self.data["std_dev_data_included"] = self.data["configuration"] & 0b0100_0000_0000_0000
                # self.data["pressure_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0001 > 0
                # self.data["temperature_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0010 > 0
                # self.data["compass_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0100 > 0
                # self.data["tilt_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_1000 > 0
                # self.data["velocity_data_included"] = self.data["configuration"] & 0b0000_0000_0010_0000 > 0
                # self.data["amplitude_data_included"] = self.data["configuration"] & 0b0000_0000_0100_0000 > 0
                # self.data["correlation_data_included"] = self.data["configuration"] & 0b0000_0000_1000_0000 > 0
                # self.data["altimeter_data_included"] = self.data["configuration"] & 0b0000_0001_0000_0000 > 0
                # self.data["altimeter_raw_data_included"] = self.data["configuration"] & 0b0000_0010_0000_0000 > 0
                # self.data["ast_data_included"] = self.data["configuration"] & 0b0000_0100_0000_0000 > 0
                # self.data["echo_sounder_data_included"] = self.data["configuration"] & 0b0000_1000_0000_0000 > 0
                # self.data["ahrs_data_included"] = self.data["configuration"] & 0b0001_0000_0000_0000 > 0
                # self.data["percentage_good_data_included"] = self.data["configuration"] & 0b0010_0000_0000_0000 > 0
                # self.data["std_dev_data_included"] = self.data["configuration"] & 0b0100_0000_0000_0000 > 0
            elif field_name == "num_beams_and_coordinate_system_and_num_cells":
                if self.data["echo_sounder_data_included"]:
                    self.data["num_echo_sounder_cells"] = self.data["num_beams_and_coordinate_system_and_num_cells"]
                else:
                    self.data["num_cells"] = self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b0000_0011_1111_1111
                    self.data["coordinate_system"] = (
                        self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b0000_1100_0000_0000) >> 10
                    self.data["num_beams"] = (
                        self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b1111_0000_0000_0000) >> 12
            elif field_name == "ambiguity_velocity_or_echo_sounder_frequency":
                if self.data["echo_sounder_data_included"]:
                    # This is specified as "echo sounder frequency", but the description technically
                    # says "number of echo sounder cells". It is probably the frequency and not the number of cells
                    # because the number of cells already replaces the data in "num_beams_and_coordinate_system_and_num_cells"
                    # when an echo sounder is present
                    self.data["echo_sounder_frequency"] = self.data["ambiguity_velocity_or_echo_sounder_frequency"]
                else:
                    self.data["ambiguity_velocity"] = self.data["ambiguity_velocity_or_echo_sounder_frequency"]
        elif self.data_record_format == self.BOTTOM_TRACK_DATA_RECORD_FORMAT:
            if field_name == "configuration":
                self.data["pressure_sensor_valid"] = self.data["data"]["configuration"] & 0b0000_0000_0000_0001
                self.data["temperature_sensor_valid"] = self.data["data"]["configuration"] & 0b0000_0000_0000_0010
                self.data["compass_sensor_valid"] = self.data["data"]["configuration"] & 0b0000_0000_0000_0100
                self.data["tilt_sensor_valid"] = self.data["data"]["configuration"] & 0b0000_0000_0000_1000
                self.data["velocity_data_included"] = self.data["data"]["configuration"] & 0b0000_0000_0010_0000
                self.data["distance_data_included"] = self.data["data"]["configuration"] & 0b0000_0001_0000_0000
                self.data["figure_of_merit_data_included"] = self.data["data"]["configuration"] & 0b0000_0010_0000_0000
                # self.data["pressure_sensor_valid"] = self.data["data"]["configuration"] & 0b0000_0000_0000_0001 > 0
                # self.data["temperature_sensor_valid"] = self.data["data"]["configuration"] & 0b0000_0000_0000_0010 > 0
                # self.data["compass_sensor_valid"] = self.data["data"]["configuration"] & 0b0000_0000_0000_0100 > 0
                # self.data["tilt_sensor_valid"] = self.data["data"]["configuration"] & 0b0000_0000_0000_1000 > 0
                # self.data["velocity_data_included"] = self.data["data"]["configuration"] & 0b0000_0000_0010_0000 > 0
                # self.data["distance_data_included"] = self.data["data"]["configuration"] & 0b0000_0001_0000_0000 > 0
                # self.data["figure_of_merit_data_included"] = self.data["data"]["configuration"] & 0b0000_0010_0000_0000 > 0
            elif field_name == "num_beams_and_coordinate_system_and_num_cells":
                self.data["num_cells"] = self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b0000_0011_1111_1111
                self.data["coordinate_system"] = (
                    self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b0000_1100_0000_0000) >> 10
                self.data["num_beams"] = (
                    self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b1111_0000_0000_0000) >> 12

    @staticmethod
    def checksum(data: bytes) -> int:
        """
        Computes the checksum for the given data
        """

        checksum = 0xb58c
        for i in range(0, len(data), 2):
            checksum += int.from_bytes(data[i: i + 2], byteorder="little")
            checksum %= 2 ** 16
        if len(data) % 2 == 1:
            checksum += data[-1] << 8
            checksum %= 2 ** 16
        return checksum

    HEADER_FORMAT: List[Field] = [
        F("sync", 1, UNSIGNED_INTEGER),
        F("header_size", 1, UNSIGNED_INTEGER),
        F("id", 1, UNSIGNED_INTEGER),
        F("family", 1, UNSIGNED_INTEGER),
        F("data_record_size", 2, UNSIGNED_INTEGER),
        F("data_record_checksum", 2, UNSIGNED_INTEGER),
        F("header_checksum", 2, UNSIGNED_INTEGER),
    ]
    STRING_DATA_RECORD_FORMAT: List[Field] = [
        F("string_data_id", 1, UNSIGNED_INTEGER),
        F("string_data",
          lambda self: self.data["data_record_size"] - 1, STRING),
    ]
    BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT: List[Field] = [
        F("version", 1, UNSIGNED_INTEGER),
        F("offset_of_data", 1, UNSIGNED_INTEGER),
        F("serial_number", 4, UNSIGNED_INTEGER),
        F("configuration", 2, UNSIGNED_INTEGER),
        F("year", 1, UNSIGNED_INTEGER),
        F("month", 1, UNSIGNED_INTEGER),
        F("day", 1, UNSIGNED_INTEGER),
        F("hour", 1, UNSIGNED_INTEGER),
        F("minute", 1, UNSIGNED_INTEGER),
        F("seconds", 1, UNSIGNED_INTEGER),
        F("microsec100", 2, UNSIGNED_INTEGER),
        F("speed_of_sound", 2, UNSIGNED_INTEGER),
        F("temperature", 2, SIGNED_INTEGER),
        F("pressure", 4, UNSIGNED_INTEGER),
        F("heading", 2, UNSIGNED_INTEGER),
        F("pitch", 2, SIGNED_INTEGER),
        F("roll", 2, SIGNED_INTEGER),
        F("error", 2, UNSIGNED_INTEGER),
        F("status", 2, UNSIGNED_INTEGER),
        F("num_beams_and_coordinate_system_and_num_cells", 2, UNSIGNED_INTEGER),
        F("cell_size", 2, UNSIGNED_INTEGER),
        F("blanking", 2, UNSIGNED_INTEGER),
        F("velocity_range", 2, UNSIGNED_INTEGER),
        F("battery_voltage", 2, UNSIGNED_INTEGER),
        F("magnetometer_raw_x", 2, SIGNED_INTEGER),
        F("magnetometer_raw_y", 2, SIGNED_INTEGER),
        F("magnetometer_raw_z", 2, SIGNED_INTEGER),
        F("accelerometer_raw_x_axis", 2, SIGNED_INTEGER),
        F("accelerometer_raw_y_axis", 2, SIGNED_INTEGER),
        F("accelerometer_raw_z_axis", 2, SIGNED_INTEGER),
        F("ambiguity_velocity", 2, UNSIGNED_INTEGER),
        F("dataset_description", 2, UNSIGNED_INTEGER),
        F("transmit_energy", 2, UNSIGNED_INTEGER),
        F("velocity_scaling", 1, SIGNED_INTEGER),
        F("power_level", 1, SIGNED_INTEGER),
        F(None, 4, UNSIGNED_INTEGER),
        F(
            "velocity_data",
            2,
            SIGNED_INTEGER,
            field_shape=lambda self: [
                self.data.get("num_beams", 0), self.data.get("num_cells", 0)],
            field_exists_predicate=lambda self: self.data["velocity_data_included"]
        ),
        F(
            "amplitude_data",
            1,
            UNSIGNED_INTEGER,
            field_shape=lambda self: [
                self.data.get("num_beams", 0), self.data.get("num_cells", 0)],
            field_exists_predicate=lambda self: self.data["amplitude_data_included"]
        ),
        F(
            "correlation_data",
            1,
            UNSIGNED_INTEGER,
            field_shape=lambda self: [
                self.data.get("num_beams", 0), self.data.get("num_cells", 0)],
            field_exists_predicate=lambda self: self.data["correlation_data_included"]
        )
    ]
    BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT: List[Field] = [
        F("version", 1, UNSIGNED_INTEGER),
        F("offset_of_data", 1, UNSIGNED_INTEGER),
        F("configuration", 2, UNSIGNED_INTEGER),
        F("serial_number", 4, UNSIGNED_INTEGER),
        F("year", 1, UNSIGNED_INTEGER),
        F("month", 1, UNSIGNED_INTEGER),
        F("day", 1, UNSIGNED_INTEGER),
        F("hour", 1, UNSIGNED_INTEGER),
        F("minute", 1, UNSIGNED_INTEGER),
        F("seconds", 1, UNSIGNED_INTEGER),
        F("microsec100", 2, UNSIGNED_INTEGER),
        F("speed_of_sound", 2, UNSIGNED_INTEGER),
        F("temperature", 2, SIGNED_INTEGER),
        F("pressure", 4, UNSIGNED_INTEGER),
        F("heading", 2, UNSIGNED_INTEGER),
        F("pitch", 2, SIGNED_INTEGER),
        F("roll", 2, SIGNED_INTEGER),
        F("num_beams_and_coordinate_system_and_num_cells", 2, UNSIGNED_INTEGER),
        F("cell_size", 2, UNSIGNED_INTEGER),
        F("blanking", 2, UNSIGNED_INTEGER),
        F("nominal_correlation", 1, UNSIGNED_INTEGER),
        F("temperature_from_pressure_sensor", 1, UNSIGNED_INTEGER),
        F("battery_voltage", 2, UNSIGNED_INTEGER),
        F("magnetometer_raw_x", 2, SIGNED_INTEGER),
        F("magnetometer_raw_y", 2, SIGNED_INTEGER),
        F("magnetometer_raw_z", 2, SIGNED_INTEGER),
        F("accelerometer_raw_x_axis", 2, SIGNED_INTEGER),
        F("accelerometer_raw_y_axis", 2, SIGNED_INTEGER),
        F("accelerometer_raw_z_axis", 2, SIGNED_INTEGER),
        F("ambiguity_velocity_or_echo_sounder_frequency", 2, UNSIGNED_INTEGER),
        F("dataset_description", 2, UNSIGNED_INTEGER),
        F("transmit_energy", 2, UNSIGNED_INTEGER),
        F("velocity_scaling", 1, SIGNED_INTEGER),
        F("power_level", 1, SIGNED_INTEGER),
        F("magnetometer_temperature", 2, SIGNED_INTEGER),
        F("real_time_clock_temperature", 2, SIGNED_INTEGER),
        F("error", 2, UNSIGNED_INTEGER),
        F("status0", 2, UNSIGNED_INTEGER),
        F("status", 4, UNSIGNED_INTEGER),
        F("ensemble_counter", 4, UNSIGNED_INTEGER),
        F(
            "velocity_data",
            2,
            SIGNED_INTEGER,
            field_shape=lambda self: [
                self.data.get("num_beams", 0), self.data.get("num_cells", 0)],
            field_exists_predicate=lambda self: self.data["velocity_data_included"]
        ),
        F(
            "amplitude_data",
            1,
            UNSIGNED_INTEGER,
            field_shape=lambda self: [
                self.data.get("num_beams", 0), self.data.get("num_cells", 0)],
            field_exists_predicate=lambda self: self.data["amplitude_data_included"]
        ),
        F(
            "correlation_data",
            1,
            UNSIGNED_INTEGER,
            field_shape=lambda self: [
                self.data.get("num_beams", 0), self.data.get("num_cells", 0)],
            field_exists_predicate=lambda self: self.data["correlation_data_included"]
        ),
        F("altimeter_distance", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["altimeter_data_included"]),
        F("altimeter_quality", 2, UNSIGNED_INTEGER,
          field_exists_predicate=lambda self: self.data["altimeter_data_included"]),
        F("ast_distance", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ast_data_included"]),
        F("ast_quality", 2, UNSIGNED_INTEGER,
          field_exists_predicate=lambda self: self.data["ast_data_included"]),
        F("ast_offset_10us", 2, SIGNED_INTEGER,
          field_exists_predicate=lambda self: self.data["ast_data_included"]),
        F("ast_pressure", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ast_data_included"]),
        F("altimeter_spare", 1, RAW_BYTES, field_shape=[8], field_exists_predicate=lambda self: self.data["ast_data_included"]
          ),
        F(
            "altimeter_raw_data_num_samples",
            # The field size of this field is technically specified as number of samples * 2,
            # but seeing as the field is called "num samples," and the field which is supposed
            # to contain the samples is specified as having a constant size of 2, these fields
            # sizes were likely incorrectly swapped.
            2,
            UNSIGNED_INTEGER,
            field_exists_predicate=lambda self: self.data["altimeter_raw_data_included"]
        ),
        F("altimeter_raw_data_sample_distance", 2, UNSIGNED_INTEGER,
          field_exists_predicate=lambda self: self.data["altimeter_raw_data_included"]),
        F(
            "altimeter_raw_data_samples",
            2,
            SIGNED_FRACTION,
            field_shape=lambda self: [
                self.data["altimeter_raw_data_num_samples"]],
            field_exists_predicate=lambda self: self.data["altimeter_raw_data_included"],
        ),
        F(
            "echo_sounder_data",
            2,
            UNSIGNED_INTEGER,
            # field_shape=lambda self: [self.data.get("num_cells", 0)],
            field_shape=lambda self: [self.data.get("num_echo_sounder_cells", 0)],
            field_exists_predicate=lambda self: self.data["echo_sounder_data_included"]
        ),
        F("ahrs_rotation_matrix_m11", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_rotation_matrix_m12", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_rotation_matrix_m13", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_rotation_matrix_m21", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_rotation_matrix_m22", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_rotation_matrix_m23", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_rotation_matrix_m31", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_rotation_matrix_m32", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_rotation_matrix_m33", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_quaternions_w", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_quaternions_x", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_quaternions_y", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_quaternions_z", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_gyro_x", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_gyro_y", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_gyro_z", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        # ("ahrs_gyro", 4, FLOAT, [3], lambda self: self.data["ahrs_data_included"]),
        F(
            "percentage_good_data",
            1,
            UNSIGNED_INTEGER,
            field_shape=lambda self: [self.data.get("num_cells", 0)],
            field_exists_predicate=lambda self: self.data["percentage_good_data_included"]
        ),
        # only the pitch field is labeled as included when the "std dev data included"
        # bit is set, but this is likely a mistake
        F("std_dev_pitch", 2, SIGNED_INTEGER,
          field_exists_predicate=lambda self: self.data["std_dev_data_included"]),
        F("std_dev_roll", 2, SIGNED_INTEGER,
          field_exists_predicate=lambda self: self.data["std_dev_data_included"]),
        F("std_dev_heading", 2, SIGNED_INTEGER,
          field_exists_predicate=lambda self: self.data["std_dev_data_included"]),
        F("std_dev_pressure", 2, SIGNED_INTEGER,
          field_exists_predicate=lambda self: self.data["std_dev_data_included"]),
        F(None, 24, RAW_BYTES,
          field_exists_predicate=lambda self: self.data["std_dev_data_included"])
    ]
    BOTTOM_TRACK_DATA_RECORD_FORMAT: List[Field] = [
        F("version", 1, UNSIGNED_INTEGER),
        F("offset_of_data", 1, UNSIGNED_INTEGER),
        F("configuration", 2, UNSIGNED_INTEGER),
        F("serial_number", 4, UNSIGNED_INTEGER),
        F("year", 1, UNSIGNED_INTEGER),
        F("month", 1, UNSIGNED_INTEGER),
        F("day", 1, UNSIGNED_INTEGER),
        F("hour", 1, UNSIGNED_INTEGER),
        F("minute", 1, UNSIGNED_INTEGER),
        F("seconds", 1, UNSIGNED_INTEGER),
        F("microsec100", 2, UNSIGNED_INTEGER),
        F("speed_of_sound", 2, UNSIGNED_INTEGER),
        F("temperature", 2, SIGNED_INTEGER),
        F("pressure", 4, UNSIGNED_INTEGER),
        F("heading", 2, UNSIGNED_INTEGER),
        F("pitch", 2, SIGNED_INTEGER),
        F("roll", 2, SIGNED_INTEGER),
        F("num_beams_and_coordinate_system_and_num_cells", 2, UNSIGNED_INTEGER),
        F("cell_size", 2, UNSIGNED_INTEGER),
        F("blanking", 2, UNSIGNED_INTEGER),
        F("nominal_correlation", 1, UNSIGNED_INTEGER),
        F(None, 1, RAW_BYTES),
        F("battery_voltage", 2, UNSIGNED_INTEGER),
        F("magnetometer_raw_x", 2, SIGNED_INTEGER),
        F("magnetometer_raw_y", 2, SIGNED_INTEGER),
        F("magnetometer_raw_z", 2, SIGNED_INTEGER),
        F("accelerometer_raw_x_axis", 2, SIGNED_INTEGER),
        F("accelerometer_raw_y_axis", 2, SIGNED_INTEGER),
        F("accelerometer_raw_z_axis", 2, SIGNED_INTEGER),
        F("ambiguity_velocity", 4, UNSIGNED_INTEGER),
        F("dataset_description", 2, UNSIGNED_INTEGER),
        F("transmit_energy", 2, UNSIGNED_INTEGER),
        F("velocity_scaling", 1, SIGNED_INTEGER),
        F("power_level", 1, SIGNED_INTEGER),
        F("magnetometer_temperature", 2, SIGNED_INTEGER),
        F("real_time_clock_temperature", 2, SIGNED_INTEGER),
        F("error", 4, UNSIGNED_INTEGER),
        F("status", 4, UNSIGNED_INTEGER),
        F("ensemble_counter", 4, UNSIGNED_INTEGER),
        F(
            "velocity_data",
            4,
            SIGNED_INTEGER,
            field_shape=lambda self: [self.data.get("num_beams", 0)],
            field_exists_predicate=lambda self: self.data["velocity_data_included"]
        ),
        F(
            "distance_data",
            4,
            SIGNED_INTEGER,
            field_shape=lambda self: [self.data.get("num_beams", 0)],
            field_exists_predicate=lambda self: self.data["distance_data_included"]
        ),
        F(
            "figure_of_merit_data",
            2,
            UNSIGNED_INTEGER,
            field_shape=lambda self: [self.data.get("num_beams", 0)],
            field_exists_predicate=lambda self: self.data["figure_of_merit_data_included"]
        )
    ]
