from collections import OrderedDict
from enum import Enum, auto, unique
from typing import Any, BinaryIO, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from typing_extensions import Literal

from .parse_base import ParseBase


@unique
class BurstAverageDataRecordVersion(Enum):
    """
    Determines the version of the burst/average data record
    """

    VERSION2 = auto()  # Burst/Average Data Record Definition (DF2)
    VERSION3 = auto()  # Burst/Average Data Record Definition (DF3)


@unique
class DataRecordType(Enum):
    """
    Determines the type of data record
    """

    BURST_VERSION2 = auto()
    BURST_VERSION3 = auto()
    AVERAGE_VERSION2 = auto()
    AVERAGE_VERSION3 = auto()
    ECHOSOUNDER = auto()
    ECHOSOUNDER_RAW = auto()
    ECHOSOUNDER_RAW_TRANSMIT = auto()
    BOTTOM_TRACK = auto()
    STRING = auto()


@unique
class DataType(Enum):
    """
    Determines the data type of raw bytes
    """

    RAW_BYTES = auto()
    STRING = auto()
    SIGNED_INTEGER = auto()
    UNSIGNED_INTEGER = auto()
    # UNSIGNED_LONG = auto()
    FLOAT = auto()
    SIGNED_FRACTION = auto()

    def dtype(self, size_bytes: int) -> np.dtype:
        if self in (SIGNED_INTEGER, UNSIGNED_INTEGER, FLOAT):
            return np.dtype(DTYPES[(self, size_bytes)])  # type: ignore
        elif self == RAW_BYTES:
            return np.dtype("<u1")
        elif self == STRING:
            return np.dtype("U")
        elif self == SIGNED_FRACTION:
            return np.dtype("<f8")
        else:
            raise ValueError("unrecognized data type")

    @staticmethod
    def default_dtype() -> np.dtype:
        return np.dtype("<u8")


RAW_BYTES = DataType.RAW_BYTES
STRING = DataType.STRING
SIGNED_INTEGER = DataType.SIGNED_INTEGER
UNSIGNED_INTEGER = DataType.UNSIGNED_INTEGER
# UNSIGNED_LONG = DataType.UNSIGNED_LONG
FLOAT = DataType.FLOAT
SIGNED_FRACTION = DataType.SIGNED_FRACTION

DtypesHint = Literal["<i1", "<i2", "<i4", "<i8", "<u1", "<u2", "<u4", "<u8", "<f2", "<f4", "<f8"]
DTYPES: Dict[Tuple[DataType, int], DtypesHint] = {
    (SIGNED_INTEGER, 1): "<i1",
    (SIGNED_INTEGER, 2): "<i2",
    (SIGNED_INTEGER, 4): "<i4",
    (SIGNED_INTEGER, 8): "<i8",
    (UNSIGNED_INTEGER, 1): "<u1",
    (UNSIGNED_INTEGER, 2): "<u2",
    (UNSIGNED_INTEGER, 4): "<u4",
    (UNSIGNED_INTEGER, 8): "<u8",
    (FLOAT, 2): "<f2",
    (FLOAT, 4): "<f4",
    (FLOAT, 8): "<f8",
    (SIGNED_FRACTION, 1): "<i1",
    (SIGNED_FRACTION, 2): "<i2",
    (SIGNED_FRACTION, 4): "<i4",
    (SIGNED_FRACTION, 8): "<i8",
}


@unique
class Dimension(Enum):
    """
    Determines the dimensions of the data in the output dataset
    """

    PING_TIME = auto()
    PING_TIME_AVERAGE = auto()
    PING_TIME_BURST = auto()
    PING_TIME_ECHOSOUNDER = auto()
    PING_TIME_ECHOSOUNDER_RAW = auto()
    PING_TIME_ECHOSOUNDER_RAW_TRANSMIT = auto()
    BEAM = auto()
    RANGE_SAMPLE_BURST = auto()
    RANGE_SAMPLE_AVERAGE = auto()
    RANGE_SAMPLE_ECHOSOUNDER = auto()
    NUM_ALTIMETER_SAMPLES = auto()
    SAMPLE = auto()
    SAMPLE_TRANSMIT = auto()
    MIJ = auto()
    XYZ = auto()
    WXYZ = auto()

    def dimension_name(self) -> str:
        return DIMENSION_NAMES[self]


DIMENSION_NAMES = {
    Dimension.PING_TIME: "time1",
    Dimension.PING_TIME_AVERAGE: "ping_time",
    Dimension.PING_TIME_BURST: "ping_time",
    Dimension.PING_TIME_ECHOSOUNDER: "ping_time",
    Dimension.PING_TIME_ECHOSOUNDER_RAW: "ping_time",
    Dimension.PING_TIME_ECHOSOUNDER_RAW_TRANSMIT: "ping_time_transmit",
    Dimension.BEAM: "beam",
    Dimension.RANGE_SAMPLE_BURST: "range_sample",
    Dimension.RANGE_SAMPLE_AVERAGE: "range_sample",
    Dimension.RANGE_SAMPLE_ECHOSOUNDER: "range_sample",
    Dimension.NUM_ALTIMETER_SAMPLES: "num_altimeter_samples",
    Dimension.SAMPLE: "range_sample",
    Dimension.SAMPLE_TRANSMIT: "transmit_sample",
    Dimension.MIJ: "mij",
    Dimension.XYZ: "xyz",
    Dimension.WXYZ: "wxyz",
}


class Field:
    """
    Represents a single field within a data record and controls the way
    the field will be parsed
    """

    def __init__(
        self,
        field_name: Optional[str],
        field_entry_size_bytes: Union[int, Callable[["Ad2cpDataPacket"], int]],
        field_entry_data_type: DataType,
        # field_entry_data_type: Union[DataType, Callable[["Ad2cpDataPacket"], DataType]],
        *,
        field_shape: Union[List[int], Callable[["Ad2cpDataPacket"], List[int]]] = [],
        field_dimensions: Union[List[Dimension], Callable[[DataRecordType], List[Dimension]]] = [
            Dimension.PING_TIME
        ],
        field_unit_conversion: Callable[
            ["Ad2cpDataPacket", np.ndarray], np.ndarray
        ] = lambda _, x: x,
        field_exists_predicate: Callable[["Ad2cpDataPacket"], bool] = lambda _: True,
    ):
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
        field_dimensions: Dimensions of the field in the output dataset
        field_unit_conversion: Unit conversion function on field
        field_exists_predicate: Tests to see whether the field should be parsed at all
        """

        self.field_name = field_name
        self.field_entry_size_bytes = field_entry_size_bytes
        self.field_entry_data_type = field_entry_data_type
        self.field_shape = field_shape
        self.field_dimensions = field_dimensions
        self.field_unit_conversion = field_unit_conversion
        self.field_exists_predicate = field_exists_predicate

    def dimensions(self, data_record_type: DataRecordType) -> List[Dimension]:
        """
        Returns the dimensions of the field given the data record type
        """

        dims = self.field_dimensions
        if callable(dims):
            dims = dims(data_record_type)
        return dims

    @staticmethod
    def default_dimensions() -> List[Dimension]:
        """
        Returns the default dimensions for fields
        """

        return [Dimension.PING_TIME]


F = Field  # use F instead of Field to make the repeated fields easier to read


class NoMorePackets(Exception):
    """
    Indicates that there are no more packets to be parsed from the file
    """

    pass


class ParseAd2cp(ParseBase):
    def __init__(
        self,
        file,
        storage_options={},
        sonar_model="AD2CP",
        **kwargs,
    ):
        super().__init__(file, storage_options, sonar_model)
        self.config = None
        self.packets: List[Ad2cpDataPacket] = []

    def parse_raw(self):
        """
        Parses the source file into AD2CP packets
        """

        with open(self.source_file, "rb") as f:
            while True:
                try:
                    packet = Ad2cpDataPacket(f, self)
                    self.packets.append(packet)
                except NoMorePackets:
                    break
                else:
                    if self.config is None and packet.is_string():
                        self.config = self.parse_config(packet.data["string_data"])

        if self.config is not None and "GETCLOCKSTR" in self.config:
            self.ping_time.append(np.datetime64(self.config["GETCLOCKSTR"]["TIME"]))
        else:
            self.ping_time.append(np.datetime64())

    @staticmethod
    def parse_config(data: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Parses the configuration string for the ADCP, which will be the first string data record.
        The data is in the form:

        HEADING1,KEY1=VALUE1,KEY2=VALUE2
        HEADING2,KEY3=VALUE3,KEY4=VALUE4,KEY5=VALUE5
        ...

        where VALUEs can be
        strings: "foo"
        integers: 123
        floats: 123.456
        """

        result = dict()
        for line in data[()].splitlines():
            tokens = line.split(",")
            line_dict = dict()
            for token in tokens[1:]:
                k, v = token.split("=")
                if v.startswith('"'):
                    v = v.strip('"')
                else:
                    try:
                        v = int(v)
                    except ValueError:
                        try:
                            v = float(v)
                        except ValueError:
                            v = str(v)
                line_dict[k] = v
            result[tokens[0]] = line_dict
        return result

    def get_firmware_version(self) -> Optional[Dict[str, Any]]:
        return self.config.get("GETHW")  # type: ignore

    def get_pulse_compressed(self) -> int:
        for i in range(1, 3 + 1):
            if "GETECHO" in self.config and self.config["GETECHO"][f"PULSECOMP{i}"] > 0:  # type: ignore # noqa
                return i
        return 0


class Ad2cpDataPacket:
    """
    Represents a single data packet. Each data packet consists of a header data record followed by a
    """

    def __init__(
        self,
        f: BinaryIO,
        parser: ParseAd2cp,
    ):
        self.parser = parser
        self.data_record_type: Optional[DataRecordType] = None
        self.data = dict()
        self._read_header(f)
        self._read_data_record(f)

    @property
    def timestamp(self) -> np.datetime64:
        """
        Calculates and returns the timestamp of the packet
        """

        year = self.data["year"] + 1900
        month = self.data["month"] + 1
        day = self.data["day"]
        hour = self.data["hour"]
        minute = self.data["minute"]
        seconds = self.data["seconds"]
        microsec100 = self.data["microsec100"]
        try:
            return np.datetime64(
                f"{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{seconds:02}.{microsec100:04}"
            )  # type: ignore
        except ValueError:
            return np.datetime64("NaT")  # type: ignore

    def is_burst(self) -> bool:
        """
        Returns whether the current packet is a burst packet
        """

        return self.data["id"] in (0x15, 0x18)

    def is_average(self) -> bool:
        """
        Returns whether the current packet is an average packet
        """

        return self.data["id"] == 0x16

    def is_bottom_track(self) -> bool:
        """
        Returns whether the current packet is a bottom track packet
        """

        return self.data["id"] in (0x17, 0x1B)

    def is_echosounder(self) -> bool:
        """
        Returns whether the current packet is an echosounder packet
        """

        return self.data["id"] == 0x1C

    def is_echosounder_raw(self) -> bool:
        """
        Returns whether the current packet is a raw echosounder packet
        """

        return self.data["id"] == 0x23

    def is_echosounder_raw_transmit(self) -> bool:
        """
        Returns whether the current packet is a raw echosounder transmit packet
        """

        return self.data["id"] == 0x24

    def is_burst_altimeter(self) -> bool:
        return self.data["id"] == 0x1A

    def is_dvl_water_track(self) -> bool:
        return self.data["id"] == 0x1D

    def is_altimeter(self) -> bool:
        return self.data["id"] == 0x1E

    def is_average_altimeter(self) -> bool:
        return self.data["id"] == 0x1F

    def is_string(self) -> bool:
        """
        Returns whether the current packet is a string packet
        """

        return self.data["id"] == 0xA0

    def has_timestamp(self) -> bool:
        """
        Returns whether the packet has a timestamp (.timestamp can be called)
        """

        return not self.is_string()

    def _read_header(self, f: BinaryIO):
        """
        Reads the header part of the AD2CP packet from the given stream
        """

        self.data_record_format = HeaderOrDataRecordFormats.HEADER_FORMAT
        raw_header = self._read_data(f, self.data_record_format)
        # don't include the last 2 bytes, which is the header checksum itself
        calculated_checksum = self.checksum(raw_header[:-2])
        expected_checksum = self.data["header_checksum"]
        assert (
            calculated_checksum == expected_checksum
        ), f"invalid header checksum: found {calculated_checksum}, expected {expected_checksum}"

    def _read_data_record(self, f: BinaryIO):
        """
        Reads the data record part of the AD2CP packet from the stream
        """

        if self.is_burst():  # burst
            self.data_record_type = DataRecordType.BURST_VERSION3
        elif self.is_average():  # average
            self.data_record_type = DataRecordType.AVERAGE_VERSION3
        elif self.is_bottom_track():  # bottom track
            self.data_record_type = DataRecordType.BOTTOM_TRACK
        elif self.is_echosounder_raw():  # echosounder raw
            self.data_record_type = DataRecordType.ECHOSOUNDER_RAW
        elif self.is_echosounder_raw_transmit():  # echosounder raw transmit
            self.data_record_type = DataRecordType.ECHOSOUNDER_RAW_TRANSMIT
        elif self.is_burst_altimeter():  # burst altimeter
            # altimeter is only supported by burst/average version 3
            self.data_record_type = DataRecordType.BURST_VERSION3
        elif self.is_echosounder():  # echosounder
            # echosounder is only supported by burst/average version 3
            self.data_record_type = DataRecordType.ECHOSOUNDER
        elif self.is_dvl_water_track():  # dvl water track record
            # TODO: is this correct?
            self.data_record_type = DataRecordType.AVERAGE_VERSION3
        elif self.is_altimeter():  # altimeter
            # altimeter is only supported by burst/average version 3
            self.data_record_type = DataRecordType.AVERAGE_VERSION3
        elif self.is_average_altimeter():  # average altimeter
            self.data_record_type = DataRecordType.AVERAGE_VERSION3
        elif self.is_string():  # string data
            self.data_record_type = DataRecordType.STRING
        else:
            raise ValueError("invalid data record type id: 0x{:02x}".format(self.data["id"]))

        self.data_record_format = HeaderOrDataRecordFormats.data_record_format(
            self.data_record_type
        )

        raw_data_record = self._read_data(f, self.data_record_format)
        calculated_checksum = self.checksum(raw_data_record)
        expected_checksum = self.data["data_record_checksum"]
        assert (
            calculated_checksum == expected_checksum
        ), f"invalid data record checksum: found {calculated_checksum}, expected {expected_checksum}"  # noqa

    def _read_data(self, f: BinaryIO, data_format: "HeaderOrDataRecordFormat") -> bytes:
        """
        Reads data from the stream, interpreting the data using the given format
        """

        raw_bytes = bytes()  # combination of all raw fields
        for field_format in data_format.fields_iter():
            field_name = field_format.field_name
            field_entry_size_bytes = field_format.field_entry_size_bytes
            field_entry_data_type = field_format.field_entry_data_type
            field_shape = field_format.field_shape
            field_unit_conversion = field_format.field_unit_conversion
            field_exists_predicate = field_format.field_exists_predicate
            if not field_exists_predicate(self):
                continue
            if callable(field_entry_size_bytes):
                field_entry_size_bytes = field_entry_size_bytes(self)
            # if callable(field_entry_data_type):
            #     field_entry_data_type = field_entry_data_type(self)
            if callable(field_shape):
                field_shape = field_shape(self)

            raw_field = self._read_exact(f, field_entry_size_bytes * int(np.prod(field_shape)))
            raw_bytes += raw_field
            # we cannot check for this before reading because some fields are placeholder fields
            # which, if not read in the correct order with other fields,
            # will offset the rest of the data
            if field_name is not None:
                parsed_field = self._parse(raw_field, field_entry_data_type, field_entry_size_bytes)
                parsed_field = np.reshape(parsed_field, field_shape)
                parsed_field = field_unit_conversion(self, parsed_field)
                self.data[field_name] = parsed_field
                self._postprocess(field_name)

        return raw_bytes

    @staticmethod
    def _parse(value: bytes, data_type: DataType, size_bytes: int) -> np.ndarray:
        """
        Parses raw bytes into a value given its data type
        """

        if data_type in (SIGNED_INTEGER, UNSIGNED_INTEGER, FLOAT):
            dtype = np.dtype(DTYPES[(data_type, size_bytes)])  # type: ignore
            return np.frombuffer(value, dtype=dtype)
        elif data_type == RAW_BYTES:
            return np.frombuffer(value, dtype="<u1")
        elif data_type == STRING:
            return np.array(value.decode("utf-8"))
        elif data_type == SIGNED_FRACTION:
            # Although the specification states that the data is represented in a
            # signed-magnitude format, an email exchange with Nortek revealed that it is
            # actually in 2's complement form.
            dtype = np.dtype(DTYPES[(SIGNED_FRACTION, size_bytes)])  # type: ignore
            return (np.frombuffer(value, dtype=dtype) / (np.iinfo(dtype).max + 1)).astype("<f8")
        else:
            raise ValueError("unrecognized data type")

    @staticmethod
    def _read_exact(f: BinaryIO, total_num_bytes_to_read: int) -> bytes:
        """
        Drives a stream until an exact amount of bytes is read from it.
        This is necessary because a single read may not return the correct number of bytes
            (see https://github.com/python/cpython/blob/5e437fb872279960992c9a07f1a4c051b4948c53/Python/fileutils.c#L1599-L1661
            and https://github.com/python/cpython/blob/63298930fb531ba2bb4f23bc3b915dbf1e17e9e1/Modules/_io/fileio.c#L778-L835,
            note "Only makes one system call, so less data may be returned than requested")
            (see https://man7.org/linux/man-pages/man2/read.2.html#RETURN_VALUE,
            note "It is not an error if this number is smaller than the number of bytes requested")
            (see https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/read?view=msvc-160#return-value,
            note "_read returns the number of bytes read,
                which might be less than buffer_size...if the file was opened in text mode")
        """  # noqa

        all_bytes_read = bytes()
        if total_num_bytes_to_read <= 0:
            return all_bytes_read
        last_bytes_read = None
        while last_bytes_read is None or (
            len(last_bytes_read) > 0 and len(all_bytes_read) < total_num_bytes_to_read
        ):
            last_bytes_read = f.read(total_num_bytes_to_read - len(all_bytes_read))
            if len(last_bytes_read) == 0:
                # 0 bytes read with non-0 bytes requested means eof
                raise NoMorePackets
            else:
                all_bytes_read += last_bytes_read
        return all_bytes_read

    def _postprocess_bitfield(
        self,
        field_value: np.ndarray,
        bitfield_format: List[Tuple[str, int, int]],
    ):
        """
        _postprocess helper; postprocesses a bitfield

        bitfield_format:
        [
            (bit sequence name, start bit, end bit)
        ]

        e.g., with mask 0b00111100, start bit is 5 and end bit is 2
        """
        for bit_sequence_name, start_bit, end_bit in bitfield_format:
            self.data[bit_sequence_name] = np.array(
                (field_value >> end_bit) & ((1 << (start_bit - end_bit + 1)) - 1),
                dtype="<u8",
            )

    def _postprocess_beams(
        self,
        field_value: np.ndarray,
        beams_format: List[Tuple[int, int]],
    ):
        """
        _postprocess helper; postprocesses beams

        beams_format:
        [
            (start bit, end bit)
        ]
        """

        beams = []
        for start_bit, end_bit in beams_format:
            beam = (field_value >> end_bit) & ((1 << (start_bit - end_bit + 1)) - 1)
            if beam > 0:
                beams.append(beam)
        self.data["beams"] = np.array(beams, dtype="<u8")

    def _postprocess(self, field_name):
        """
        Calculates values based on parsed data. This should be called immediately after
        parsing each field in a data record.
        """
        if (
            self.data_record_format
            == HeaderOrDataRecordFormats.BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT
        ):
            if field_name == "version":
                if self.data["version"] == 3:
                    self.data_record_format = (
                        HeaderOrDataRecordFormats.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
                    )
            elif field_name == "configuration":
                self._postprocess_bitfield(
                    self.data["configuration"],
                    [
                        ("pressure_sensor_valid", 0, 0),
                        ("temperature_sensor_valid", 1, 1),
                        ("compass_sensor_valid", 2, 2),
                        ("tilt_sensor_valid", 3, 3),
                        ("velocity_data_included", 5, 5),
                        ("amplitude_data_included", 6, 6),
                        ("correlation_data_included", 7, 7),
                    ],
                )
            elif field_name == "num_beams_and_coordinate_system_and_num_cells":
                self._postprocess_bitfield(
                    self.data["num_beams_and_coordinate_system_and_num_cells"],
                    [
                        ("num_cells", 9, 0),
                        ("coordinate_system", 11, 10),
                        ("num_beams", 15, 12),
                    ],
                )
            elif field_name == "dataset_description":
                self._postprocess_beams(
                    self.data["dataset_description"],
                    [(2, 0), (5, 3), (8, 6), (11, 9), (14, 12)],
                )
        elif (
            self.data_record_format
            == HeaderOrDataRecordFormats.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
        ):
            if field_name == "version":
                if self.data["version"] == 2:
                    self.data_record_format = (
                        HeaderOrDataRecordFormats.BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT
                    )
            elif field_name == "configuration":
                self._postprocess_bitfield(
                    self.data["configuration"],
                    [
                        ("pressure_sensor_valid", 0, 0),
                        ("temperature_sensor_valid", 1, 1),
                        ("compass_sensor_valid", 2, 2),
                        ("tilt_sensor_valid", 3, 3),
                        ("velocity_data_included", 5, 5),
                        ("amplitude_data_included", 6, 6),
                        ("correlation_data_included", 7, 7),
                        ("altimeter_data_included", 8, 8),
                        ("altimeter_raw_data_included", 9, 9),
                        ("ast_data_included", 10, 10),
                        ("echosounder_data_included", 11, 11),
                        ("ahrs_data_included", 12, 12),
                        ("percentage_good_data_included", 13, 13),
                        ("std_dev_data_included", 14, 14),
                    ],
                )
            elif field_name == "num_beams_and_coordinate_system_and_num_cells":
                if self.data["echosounder_data_included"]:
                    self.data["num_echosounder_cells"] = self.data[
                        "num_beams_and_coordinate_system_and_num_cells"
                    ]
                else:
                    self._postprocess_bitfield(
                        self.data["num_beams_and_coordinate_system_and_num_cells"],
                        [
                            ("num_cells", 9, 0),
                            ("coordinate_system", 11, 10),
                            ("num_beams", 15, 12),
                        ],
                    )
            elif field_name == "ambiguity_velocity_or_echosounder_frequency":
                if self.data["echosounder_data_included"]:
                    # This is specified as "echo sounder frequency", but the description technically
                    # says "number of echo sounder cells".
                    # It is probably the frequency and not the number of cells
                    # because the number of cells already replaces the data in
                    # "num_beams_and_coordinate_system_and_num_cells"
                    # when an echo sounder is present
                    self.data["echosounder_frequency"] = self.data[
                        "ambiguity_velocity_or_echosounder_frequency"
                    ]
                else:
                    self.data["ambiguity_velocity"] = self.data[
                        "ambiguity_velocity_or_echosounder_frequency"
                    ]
            elif field_name == "velocity_scaling":
                if not self.data["echosounder_data_included"]:
                    # The unit conversion for ambiguity velocity is done here because it
                    # requires the velocity_scaling, which is not known
                    # when ambiguity velocity field is parsed
                    self.data["ambiguity_velocity"] = self.data["ambiguity_velocity"] * (
                        10.0 ** self.data["velocity_scaling"]
                    )
            elif field_name == "dataset_description":
                self._postprocess_beams(
                    self.data["dataset_description"],
                    [(3, 0), (7, 4), (11, 8), (16, 12)],
                )
                if (
                    self.parser.packets[-1].is_echosounder_raw()
                    or self.parser.packets[-1].is_echosounder_raw_transmit()
                ):
                    self.parser.packets[-1].data["echosounder_raw_beam"] = self.data["beams"][0]
            elif field_name == "status0":
                if self.data["status0"] & 0b1000_0000_0000_0000:
                    self._postprocess_bitfield(
                        self.data["status0"],
                        [
                            ("procidle3", 0, 0),
                            ("procidle6", 1, 1),
                            ("procidle12", 2, 2),
                        ],
                    )
            elif field_name == "status":
                self._postprocess_bitfield(
                    self.data["status"],
                    [
                        ("wakeup_state", 31, 28),
                        ("orientation", 27, 25),
                        ("autoorientation", 24, 22),
                        ("previous_wakeup_state", 21, 18),
                        ("last_measurement_low_voltage_skip", 17, 17),
                        ("active_configuration", 16, 16),
                        ("echosounder_index", 15, 12),
                        ("telemetry_data", 11, 11),
                        ("boost_running", 10, 10),
                        ("echosounder_frequency_bin", 9, 5),
                        ("bd_scaling", 1, 1),
                    ],
                )
        elif self.data_record_format == HeaderOrDataRecordFormats.BOTTOM_TRACK_DATA_RECORD_FORMAT:
            if field_name == "configuration":
                self._postprocess_bitfield(
                    self.data["configuration"],
                    [
                        ("pressure_sensor_valid", 0, 0),
                        ("temperature_sensor_valid", 1, 1),
                        ("compass_sensor_valid", 2, 2),
                        ("tilt_sensor_valid", 3, 3),
                        ("velocity_data_included", 5, 5),
                        ("distance_data_included", 8, 8),
                        ("figure_of_merit_data_included", 9, 9),
                        ("ahrs_data_included", 10, 10),
                    ],
                )
            elif field_name == "num_beams_and_coordinate_system_and_num_cells":
                self._postprocess_bitfield(
                    self.data["num_beams_and_coordinate_system_and_num_cells"],
                    [
                        ("num_cells", 9, 0),
                        ("coordinate_system", 11, 10),
                        ("num_beams", 15, 12),
                    ],
                )
            elif field_name == "dataset_description":
                self._postprocess_beams(
                    self.data["dataset_description"],
                    [(16, 12), (11, 8), (7, 4), (3, 0)],
                )

            elif field_name == "velocity_scaling":
                # The unit conversion for ambiguity velocity is done here because it
                # requires the velocity_scaling,
                # which is not known when ambiguity velocity field is parsed
                self.data["ambiguity_velocity"] = self.data["ambiguity_velocity"] * (
                    10.0 ** self.data["velocity_scaling"]
                )
        elif (
            self.data_record_format == HeaderOrDataRecordFormats.ECHOSOUNDER_RAW_DATA_RECORD_FORMAT
        ):
            if field_name == "echosounder_raw_samples":
                self.data["echosounder_raw_samples_i"] = self.data["echosounder_raw_samples"][:, 0]
                self.data["echosounder_raw_samples_q"] = self.data["echosounder_raw_samples"][:, 1]
            elif field_name == "echosounder_raw_transmit_samples":
                self.data["echosounder_raw_transmit_samples_i"] = self.data[
                    "echosounder_raw_transmit_samples"
                ][:, 0]
                self.data["echosounder_raw_transmit_samples_q"] = self.data[
                    "echosounder_raw_transmit_samples"
                ][:, 1]
            elif field_name == "status":
                self._postprocess_bitfield(
                    self.data["status"],
                    [
                        ("wakeup_state", 31, 28),
                        ("orientation", 27, 25),
                        ("autoorientation", 24, 22),
                        ("previous_wakeup_state", 21, 18),
                        ("last_measurement_low_voltage_skip", 17, 17),
                        ("active_configuration", 16, 16),
                        ("echosounder_index", 15, 12),
                        ("telemetry_data", 11, 11),
                        ("boost_running", 10, 10),
                        ("echosounder_frequency_bin", 9, 5),
                        ("bd_scaling", 1, 1),
                    ],
                )

    @staticmethod
    def checksum(data: bytes) -> int:
        """
        Computes the checksum for the given data
        """

        checksum = 0xB58C
        for i in range(0, len(data), 2):
            checksum += int.from_bytes(data[i : i + 2], byteorder="little")
            checksum %= 2**16
        if len(data) % 2 == 1:
            checksum += data[-1] << 8
            checksum %= 2**16
        return checksum


RANGE_SAMPLES = {
    DataRecordType.AVERAGE_VERSION2: Dimension.RANGE_SAMPLE_AVERAGE,
    DataRecordType.AVERAGE_VERSION3: Dimension.RANGE_SAMPLE_AVERAGE,
    DataRecordType.BURST_VERSION2: Dimension.RANGE_SAMPLE_BURST,
    DataRecordType.BURST_VERSION3: Dimension.RANGE_SAMPLE_BURST,
    DataRecordType.ECHOSOUNDER: Dimension.RANGE_SAMPLE_ECHOSOUNDER,
}


class HeaderOrDataRecordFormat:
    """
    A collection of fields which represents the header format or a data record format
    """

    def __init__(self, name: str, fields: List[Field]):
        self.name = name
        self.fields = OrderedDict([(f.field_name, f) for f in fields])

    def get_field(self, field_name: str) -> Optional[Field]:
        """
        Gets a field from the current packet based on its name.
        Since the field could also be in the packet's header, the header
            is searched in addition to this data record.
        """

        if field_name in HeaderOrDataRecordFormats.HEADER_FORMAT.fields:
            return HeaderOrDataRecordFormats.HEADER_FORMAT.fields.get(field_name)
        return self.fields.get(field_name)

    def fields_iter(self) -> Iterable[Field]:
        """
        Returns an iterable over the fields in this header or data record format
        """

        return self.fields.values()


class HeaderOrDataRecordFormats:
    @classmethod
    def data_record_format(cls, data_record_type: DataRecordType) -> HeaderOrDataRecordFormat:
        """
        Returns data record format that should be used to parse the given data record type
        """

        return cls.DATA_RECORD_FORMATS[data_record_type]

    HEADER_FORMAT: HeaderOrDataRecordFormat = HeaderOrDataRecordFormat(
        "HEADER_FORMAT",
        [
            F("sync", 1, UNSIGNED_INTEGER),
            F("header_size", 1, UNSIGNED_INTEGER),
            F("id", 1, UNSIGNED_INTEGER),
            F("family", 1, UNSIGNED_INTEGER),
            F(
                "data_record_size",
                lambda packet: 4 if packet.data["id"] in (0x23, 0x24) else 2,
                UNSIGNED_INTEGER,
            ),
            # F("data_record_size", lambda packet: 4 if packet.raw_fields["id"] in (
            #     0x23, 0x24) else 2, lambda packet: UNSIGNED_LONG if packet.raw_fields["id"]
            #     in (0x23, 0x24) else UNSIGNED_INTEGER),
            F("data_record_checksum", 2, UNSIGNED_INTEGER),
            F("header_checksum", 2, UNSIGNED_INTEGER),
        ],
    )
    STRING_DATA_RECORD_FORMAT: HeaderOrDataRecordFormat = HeaderOrDataRecordFormat(
        "STRING_DATA_RECORD_FORMAT",
        [
            F("string_data_id", 1, UNSIGNED_INTEGER),
            F(
                "string_data",
                lambda packet: packet.data["data_record_size"] - 1,
                STRING,
            ),
        ],
    )
    BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT: HeaderOrDataRecordFormat = HeaderOrDataRecordFormat(
        "BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT",
        [
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
            F(
                "speed_of_sound",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 10,
            ),
            F(
                "temperature",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
            ),
            F(
                "pressure",
                4,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 1000,
            ),
            F(
                "heading",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
            ),
            F(
                "pitch",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
            ),
            F(
                "roll",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
            ),
            F("error", 2, UNSIGNED_INTEGER),
            F("status", 2, UNSIGNED_INTEGER),
            F("num_beams_and_coordinate_system_and_num_cells", 2, UNSIGNED_INTEGER),
            F(
                "cell_size",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 1000,
            ),
            F(
                "blanking",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 1000,
            ),
            F(
                "velocity_range",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 1000,
            ),
            F(
                "battery_voltage",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 10,
            ),
            F(
                "magnetometer_raw",
                2,
                SIGNED_INTEGER,
                field_shape=[3],
                field_dimensions=[Dimension.PING_TIME, Dimension.XYZ],
            ),
            F(
                "accelerometer_raw_x_axis",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 16384 * 9.819,
            ),
            F(
                "accelerometer_raw_y_axis",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 16384 * 9.819,
            ),
            F(
                "accelerometer_raw_z_axis",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 16384 * 9.819,
            ),
            F(
                "ambiguity_velocity",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 10000,
            ),
            F("dataset_description", 2, UNSIGNED_INTEGER),
            F("transmit_energy", 2, UNSIGNED_INTEGER),
            F("velocity_scaling", 1, SIGNED_INTEGER),
            F("power_level", 1, SIGNED_INTEGER),
            F(None, 4, UNSIGNED_INTEGER),
            F(  # used when burst
                "velocity_data_burst",
                2,
                SIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_BURST,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_BURST,
                ],
                field_unit_conversion=lambda packet, x: x
                * (10.0 ** packet.data["velocity_scaling"]),
                field_exists_predicate=lambda packet: packet.is_burst()
                and packet.data["velocity_data_included"],
            ),
            F(  # used when average
                "velocity_data_average",
                2,
                SIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_AVERAGE,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_AVERAGE,
                ],
                field_unit_conversion=lambda packet, x: x
                * (10.0 ** packet.data["velocity_scaling"]),
                field_exists_predicate=lambda packet: packet.is_average()
                and packet.data["velocity_data_included"],
            ),
            F(  # used when echosounder
                "velocity_data_echosounder",
                2,
                SIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_ECHOSOUNDER,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_ECHOSOUNDER,
                ],
                field_unit_conversion=lambda packet, x: x
                * (10.0 ** packet.data["velocity_scaling"]),
                field_exists_predicate=lambda packet: packet.is_echosounder()
                and packet.data["velocity_data_included"],
            ),
            F(
                "amplitude_data_burst",
                1,
                UNSIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_BURST,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_BURST,
                ],
                field_unit_conversion=lambda packet, x: x / 2,
                field_exists_predicate=lambda packet: packet.is_burst()
                and packet.data["amplitude_data_included"],
            ),
            F(
                "amplitude_data_average",
                1,
                UNSIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_AVERAGE,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_AVERAGE,
                ],
                field_unit_conversion=lambda packet, x: x / 2,
                field_exists_predicate=lambda packet: packet.is_average()
                and packet.data["amplitude_data_included"],
            ),
            F(
                "amplitude_data_echosounder",
                1,
                UNSIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_ECHOSOUNDER,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_ECHOSOUNDER,
                ],
                field_unit_conversion=lambda packet, x: x / 2,
                field_exists_predicate=lambda packet: packet.is_echosounder()
                and packet.data["amplitude_data_included"],
            ),
            F(
                "correlation_data_burst",
                1,
                UNSIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_BURST,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_BURST,
                ],
                field_exists_predicate=lambda packet: packet.is_burst()
                and packet.data["correlation_data_included"],
            ),
            F(
                "correlation_data_average",
                1,
                UNSIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_AVERAGE,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_AVERAGE,
                ],
                field_exists_predicate=lambda packet: packet.is_average()
                and packet.data["correlation_data_included"],
            ),
            F(
                "correlation_data_echosounder",
                1,
                UNSIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_ECHOSOUNDER,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_ECHOSOUNDER,
                ],
                field_exists_predicate=lambda packet: packet.is_echosounder()
                and packet.data["correlation_data_included"],
            ),
        ],
    )
    BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT: HeaderOrDataRecordFormat = HeaderOrDataRecordFormat(
        "BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT",
        [
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
            F(
                "speed_of_sound",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 10,
            ),
            F(
                "temperature",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
            ),
            F(
                "pressure",
                4,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 1000,
            ),
            F(
                "heading",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
            ),
            F(
                "pitch",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
            ),
            F(
                "roll",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
            ),
            F("num_beams_and_coordinate_system_and_num_cells", 2, UNSIGNED_INTEGER),
            F(
                "cell_size",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 1000,
            ),
            # This field is listed to be in cm, but testing has shown that it is actually in mm.
            # Being in mm would be consistent with the "blanking" field units in all other formats.
            F(
                "blanking",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 1000,
            ),
            F("nominal_correlation", 1, UNSIGNED_INTEGER),
            F(
                "temperature_from_pressure_sensor",
                1,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x * 5,
            ),
            F(
                "battery_voltage",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 10,
            ),
            F(
                "magnetometer_raw",
                2,
                SIGNED_INTEGER,
                field_shape=[3],
                field_dimensions=[Dimension.PING_TIME, Dimension.XYZ],
            ),
            F(
                "accelerometer_raw_x_axis",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 16384 * 9.819,
            ),
            F(
                "accelerometer_raw_y_axis",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 16384 * 9.819,
            ),
            F(
                "accelerometer_raw_z_axis",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 16384 * 9.819,
            ),
            # Unit conversions for this field are done in Ad2cpDataPacket._postprocess
            # because the ambiguity velocity unit conversion requires the velocity_scaling field,
            # which is not known when this field is parsed
            F("ambiguity_velocity_or_echosounder_frequency", 2, UNSIGNED_INTEGER),
            F("dataset_description", 2, UNSIGNED_INTEGER),
            F("transmit_energy", 2, UNSIGNED_INTEGER),
            F("velocity_scaling", 1, SIGNED_INTEGER),
            F("power_level", 1, SIGNED_INTEGER),
            F(
                "magnetometer_temperature",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x * 1000,
            ),
            F(
                "real_time_clock_temperature",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
            ),
            F("error", 2, UNSIGNED_INTEGER),
            F("status0", 2, UNSIGNED_INTEGER),
            F("status", 4, UNSIGNED_INTEGER),
            F("ensemble_counter", 4, UNSIGNED_INTEGER),
            F(
                "velocity_data_burst",
                2,
                SIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_BURST,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_BURST,
                ],
                field_unit_conversion=lambda packet, x: x
                * (10.0 ** packet.data["velocity_scaling"]),
                field_exists_predicate=lambda packet: packet.is_burst()
                and packet.data["velocity_data_included"],
            ),
            F(
                "velocity_data_average",
                2,
                SIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_AVERAGE,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_AVERAGE,
                ],
                field_unit_conversion=lambda packet, x: x
                * (10.0 ** packet.data["velocity_scaling"]),
                field_exists_predicate=lambda packet: packet.is_average()
                and packet.data["velocity_data_included"],
            ),
            F(
                "velocity_data_echosounder",
                2,
                SIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_ECHOSOUNDER,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_ECHOSOUNDER,
                ],
                field_unit_conversion=lambda packet, x: x
                * (10.0 ** packet.data["velocity_scaling"]),
                field_exists_predicate=lambda packet: packet.is_echosounder()
                and packet.data["velocity_data_included"],
            ),
            F(
                "amplitude_data_burst",
                1,
                UNSIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_BURST,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_BURST,
                ],
                field_unit_conversion=lambda packet, x: x / 2,
                field_exists_predicate=lambda packet: packet.is_burst()
                and packet.data["amplitude_data_included"],
            ),
            F(
                "amplitude_data_average",
                1,
                UNSIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_AVERAGE,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_AVERAGE,
                ],
                field_unit_conversion=lambda packet, x: x / 2,
                field_exists_predicate=lambda packet: packet.is_average()
                and packet.data["amplitude_data_included"],
            ),
            F(
                "amplitude_data_echosounder",
                1,
                UNSIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_ECHOSOUNDER,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_ECHOSOUNDER,
                ],
                field_unit_conversion=lambda packet, x: x / 2,
                field_exists_predicate=lambda packet: packet.is_echosounder()
                and packet.data["amplitude_data_included"],
            ),
            F(
                "correlation_data_burst",
                1,
                UNSIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_BURST,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_BURST,
                ],
                field_exists_predicate=lambda packet: packet.is_burst()
                and packet.data["correlation_data_included"],
            ),
            F(
                "correlation_data_average",
                1,
                UNSIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_AVERAGE,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_AVERAGE,
                ],
                field_exists_predicate=lambda packet: packet.is_average()
                and packet.data["correlation_data_included"],
            ),
            F(
                "correlation_data_echosounder",
                1,
                UNSIGNED_INTEGER,
                field_shape=lambda packet: [
                    packet.data.get("num_beams", 0),
                    packet.data.get("num_cells", 0),
                ],
                field_dimensions=[
                    Dimension.PING_TIME_ECHOSOUNDER,
                    Dimension.BEAM,
                    Dimension.RANGE_SAMPLE_ECHOSOUNDER,
                ],
                field_exists_predicate=lambda packet: packet.is_echosounder()
                and packet.data["correlation_data_included"],
            ),
            F(
                "altimeter_distance",
                4,
                FLOAT,
                field_exists_predicate=lambda packet: packet.data["altimeter_data_included"],
            ),
            F(
                "altimeter_quality",
                2,
                UNSIGNED_INTEGER,
                field_exists_predicate=lambda packet: packet.data["altimeter_data_included"],
            ),
            F(
                "ast_distance",
                4,
                FLOAT,
                field_exists_predicate=lambda packet: packet.data["ast_data_included"],
            ),
            F(
                "ast_quality",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
                field_exists_predicate=lambda packet: packet.data["ast_data_included"],
            ),
            F(
                "ast_offset_100us",
                2,
                SIGNED_INTEGER,
                field_exists_predicate=lambda packet: packet.data["ast_data_included"],
            ),
            F(
                "ast_pressure",
                4,
                FLOAT,
                field_exists_predicate=lambda packet: packet.data["ast_data_included"],
            ),
            F(
                "altimeter_spare",
                1,
                RAW_BYTES,
                field_shape=[8],
                field_exists_predicate=lambda packet: packet.data["ast_data_included"],
            ),
            F(
                "altimeter_raw_data_num_samples",
                # The field size of this field is technically specified as number of samples * 2,
                # but seeing as the field is called "num samples," and the field which is supposed
                # to contain the samples is specified as having a constant size of 2, these fields
                # sizes were likely incorrectly swapped.
                2,
                UNSIGNED_INTEGER,
                field_exists_predicate=lambda packet: packet.data["altimeter_raw_data_included"],
            ),
            F(
                "altimeter_raw_data_sample_distance",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 10000,
                field_exists_predicate=lambda packet: packet.data["altimeter_raw_data_included"],
            ),
            F(
                "altimeter_raw_data_samples",
                2,
                SIGNED_FRACTION,
                field_shape=lambda packet: [packet.data["altimeter_raw_data_num_samples"]],
                field_dimensions=[Dimension.PING_TIME, Dimension.NUM_ALTIMETER_SAMPLES],
                field_exists_predicate=lambda packet: packet.data["altimeter_raw_data_included"],
            ),
            F(
                "echosounder_data",
                2,
                # Although the specification says that this should be an unsigned integer,
                # testing has shown that it should be a signed integer
                SIGNED_INTEGER,
                field_shape=lambda packet: [packet.data.get("num_echosounder_cells", 0)],
                field_dimensions=[
                    Dimension.PING_TIME_ECHOSOUNDER,
                    Dimension.RANGE_SAMPLE_ECHOSOUNDER,
                ],
                field_unit_conversion=lambda packet, x: x / 100,
                field_exists_predicate=lambda packet: packet.data["echosounder_data_included"],
            ),
            F(
                "ahrs_rotation_matrix",
                4,
                FLOAT,
                field_shape=[9],
                field_dimensions=[Dimension.PING_TIME, Dimension.MIJ],
                field_exists_predicate=lambda packet: packet.data["ahrs_data_included"],
            ),
            F(
                "ahrs_quaternions",
                4,
                FLOAT,
                field_shape=[4],
                field_dimensions=[Dimension.PING_TIME, Dimension.WXYZ],
                field_exists_predicate=lambda packet: packet.data["ahrs_data_included"],
            ),
            F(
                "ahrs_gyro",
                4,
                FLOAT,
                field_shape=[3],
                field_dimensions=[Dimension.PING_TIME, Dimension.XYZ],
                field_exists_predicate=lambda packet: packet.data["ahrs_data_included"],
            ),
            F(
                "percentage_good_data",
                1,
                UNSIGNED_INTEGER,
                field_shape=lambda packet: [packet.data.get("num_cells", 0)],
                field_dimensions=lambda data_record_type: [
                    Dimension.PING_TIME,
                    RANGE_SAMPLES[data_record_type],
                ],
                field_exists_predicate=lambda packet: packet.data["percentage_good_data_included"],
            ),
            # Only the pitch field is labeled as included when the "std dev data included"
            # bit is set, but this is likely a mistake
            F(
                "std_dev_pitch",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
                field_exists_predicate=lambda packet: packet.data["std_dev_data_included"],
            ),
            F(
                "std_dev_roll",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
                field_exists_predicate=lambda packet: packet.data["std_dev_data_included"],
            ),
            F(
                "std_dev_heading",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
                field_exists_predicate=lambda packet: packet.data["std_dev_data_included"],
            ),
            F(
                "std_dev_pressure",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
                field_exists_predicate=lambda packet: packet.data["std_dev_data_included"],
            ),
            F(
                None,
                24,
                RAW_BYTES,
                field_exists_predicate=lambda packet: packet.data["std_dev_data_included"],
            ),
        ],
    )
    BOTTOM_TRACK_DATA_RECORD_FORMAT: HeaderOrDataRecordFormat = HeaderOrDataRecordFormat(
        "BOTTOM_TRACK_DATA_RECORD_FORMAT",
        [
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
            F(
                "speed_of_sound",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 10,
            ),
            F(
                "temperature",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
            ),
            F(
                "pressure",
                4,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 1000,
            ),
            F(
                "heading",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
            ),
            F(
                "pitch",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
            ),
            F(
                "roll",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
            ),
            F("num_beams_and_coordinate_system_and_num_cells", 2, UNSIGNED_INTEGER),
            F(
                "cell_size",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 1000,
            ),
            F(
                "blanking",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 1000,
            ),
            F("nominal_correlation", 1, UNSIGNED_INTEGER),
            F(None, 1, RAW_BYTES),
            F(
                "battery_voltage",
                2,
                UNSIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 10,
            ),
            F(
                "magnetometer_raw",
                2,
                SIGNED_INTEGER,
                field_shape=[3],
                field_dimensions=[Dimension.PING_TIME, Dimension.XYZ],
            ),
            F(
                "accelerometer_raw_x_axis",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 16384 * 9.819,
            ),
            F(
                "accelerometer_raw_y_axis",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 16384 * 9.819,
            ),
            F(
                "accelerometer_raw_z_axis",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 16384 * 9.819,
            ),
            # Unit conversions for this field are done in Ad2cpDataPacket._postprocess
            # because the ambiguity velocity unit conversion requires the velocity_scaling field,
            # which is not known when this field is parsed
            F("ambiguity_velocity", 4, UNSIGNED_INTEGER),
            F("dataset_description", 2, UNSIGNED_INTEGER),
            F("transmit_energy", 2, UNSIGNED_INTEGER),
            F("velocity_scaling", 1, SIGNED_INTEGER),
            F("power_level", 1, SIGNED_INTEGER),
            F(
                "magnetometer_temperature",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x * 1000,
            ),
            F(
                "real_time_clock_temperature",
                2,
                SIGNED_INTEGER,
                field_unit_conversion=lambda packet, x: x / 100,
            ),
            F("error", 4, UNSIGNED_INTEGER),
            F("status", 4, UNSIGNED_INTEGER),
            F("ensemble_counter", 4, UNSIGNED_INTEGER),
            F(
                "velocity_data",
                4,
                SIGNED_INTEGER,
                field_shape=lambda packet: [packet.data.get("num_beams", 0)],
                field_dimensions=[Dimension.PING_TIME, Dimension.BEAM],
                field_unit_conversion=lambda packet, x: x
                * (10.0 ** packet.data["velocity_scaling"]),
                field_exists_predicate=lambda packet: packet.data["velocity_data_included"],
            ),
            F(
                "distance_data",
                4,
                SIGNED_INTEGER,
                field_shape=lambda packet: [packet.data.get("num_beams", 0)],
                field_dimensions=[Dimension.PING_TIME, Dimension.BEAM],
                field_unit_conversion=lambda packet, x: x / 1000,
                field_exists_predicate=lambda packet: packet.data["distance_data_included"],
            ),
            F(
                "figure_of_merit_data",
                2,
                UNSIGNED_INTEGER,
                field_shape=lambda packet: [packet.data.get("num_beams", 0)],
                field_dimensions=[Dimension.PING_TIME, Dimension.BEAM],
                field_exists_predicate=lambda packet: packet.data["figure_of_merit_data_included"],
            ),
        ],
    )
    ECHOSOUNDER_RAW_DATA_RECORD_FORMAT: HeaderOrDataRecordFormat = HeaderOrDataRecordFormat(
        "ECHOSOUNDER_RAW_DATA_RECORD_FORMAT",
        [
            F("version", 1, UNSIGNED_INTEGER),
            F("offset_of_data", 1, UNSIGNED_INTEGER),
            F("year", 1, UNSIGNED_INTEGER),
            F("month", 1, UNSIGNED_INTEGER),
            F("day", 1, UNSIGNED_INTEGER),
            F("hour", 1, UNSIGNED_INTEGER),
            F("minute", 1, UNSIGNED_INTEGER),
            F("seconds", 1, UNSIGNED_INTEGER),
            F("microsec100", 2, UNSIGNED_INTEGER),
            F("error", 2, UNSIGNED_INTEGER),
            F("status", 4, UNSIGNED_INTEGER),
            F("serial_number", 4, UNSIGNED_INTEGER),
            F("num_complex_samples", 4, UNSIGNED_INTEGER),
            F("ind_start_samples", 4, UNSIGNED_INTEGER),
            F("freq_raw_sample_data", 4, FLOAT),
            F(None, 208, RAW_BYTES),
            F(
                "echosounder_raw_samples",
                4,
                SIGNED_FRACTION,
                field_shape=lambda packet: [
                    packet.data["num_complex_samples"],
                    2,
                ],
                field_dimensions=[
                    Dimension.PING_TIME_ECHOSOUNDER_RAW,
                    Dimension.SAMPLE,
                ],
                field_exists_predicate=lambda packet: packet.is_echosounder_raw(),
            ),
            # These next 2 fields are included so that the dimensions for these fields
            #   can be determined based on the field name.
            #   They are actually constructed in _postprocess.
            F(
                "echosounder_raw_samples_i",
                0,
                RAW_BYTES,
                field_dimensions=[
                    Dimension.PING_TIME_ECHOSOUNDER_RAW,
                    Dimension.SAMPLE,
                ],
                field_exists_predicate=lambda packet: False,
            ),
            F(
                "echosounder_raw_samples_q",
                0,
                RAW_BYTES,
                field_dimensions=[
                    Dimension.PING_TIME_ECHOSOUNDER_RAW,
                    Dimension.SAMPLE,
                ],
                field_exists_predicate=lambda packet: False,
            ),
            F(
                "echosounder_raw_transmit_samples",
                4,
                SIGNED_FRACTION,
                field_shape=lambda packet: [
                    packet.data["num_complex_samples"],
                    2,
                ],
                field_dimensions=[
                    Dimension.PING_TIME_ECHOSOUNDER_RAW_TRANSMIT,
                    Dimension.SAMPLE_TRANSMIT,
                ],
                field_exists_predicate=lambda packet: packet.is_echosounder_raw_transmit(),
            ),
            # These next 2 fields are included so that the dimensions for these fields
            #   can be determined based on the field name.
            #   They are actually constructed in _postprocess.
            F(
                "echosounder_raw_transmit_samples_i",
                0,
                RAW_BYTES,
                field_dimensions=[
                    Dimension.PING_TIME_ECHOSOUNDER_RAW_TRANSMIT,
                    Dimension.SAMPLE_TRANSMIT,
                ],
                field_exists_predicate=lambda packet: False,
            ),
            F(
                "echosounder_raw_transmit_samples_q",
                0,
                RAW_BYTES,
                field_dimensions=[
                    Dimension.PING_TIME_ECHOSOUNDER_RAW_TRANSMIT,
                    Dimension.SAMPLE_TRANSMIT,
                ],
                field_exists_predicate=lambda packet: False,
            ),
        ],
    )

    DATA_RECORD_FORMATS = {
        DataRecordType.BURST_VERSION2: BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT,
        DataRecordType.BURST_VERSION3: BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT,
        DataRecordType.AVERAGE_VERSION2: BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT,
        DataRecordType.AVERAGE_VERSION3: BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT,
        DataRecordType.BOTTOM_TRACK: BOTTOM_TRACK_DATA_RECORD_FORMAT,
        DataRecordType.ECHOSOUNDER: BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT,
        DataRecordType.ECHOSOUNDER_RAW: ECHOSOUNDER_RAW_DATA_RECORD_FORMAT,
        DataRecordType.ECHOSOUNDER_RAW_TRANSMIT: ECHOSOUNDER_RAW_DATA_RECORD_FORMAT,
        DataRecordType.STRING: STRING_DATA_RECORD_FORMAT,
    }
