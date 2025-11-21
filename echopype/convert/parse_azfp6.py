import os
import xml.etree.ElementTree as ET
from datetime import datetime as dt
from io import BytesIO
from struct import unpack

import numpy as np

from ..utils.log import _init_logger
from ..utils.misc import camelcase2snakecase
from .parse_azfp import ParseAZFP

FILENAME_DATETIME_AZFP = "\\w+_\\w+.[azfp|aps6]"

MAXIMUM_CODES = 127
HEADER_MAXIMUM_BYTES = (MAXIMUM_CODES * 2) + 512

# Latest header field codes
HEADER_CODES = dict(
    START_FLAG=0xBCD0,  # Start of header block
    END_FLAG=0xABC1,  # End of header block
    FIRST_HEADER_RECORD=0xAA20,  # Always the first record written
    LAST_HEADER_RECORD=0xFE20,  # Always the last record written
    HEADER_BYTES=0xBB20,  # The number of bytes of the header    # always second
    HEADER_NUM_RECORDS=0xCC20,  # The number of variables              # always first
    BURST_NUMBER=0x8060,  # Burst number
    PROFILE_NUMBER=0x8060,  # Backwards naming convention with ULS5
    SERIAL_NUMBER=0x8160,  # Instrument Serial Number 2024 01 30 changed from 2 bytes to 4 bytes
    DATE_TIME=0x8226,  # date and time
    ACQ_STATUS=0x8320,  # Acquisition status
    BURST_INT=0x8460,  # Burst interval
    BASE_TIME=0x85C0,  # Base time
    PING_PERIOD=0x86C0,  # Ping period
    PING_PERIOD_COUNTS=0x8720,  # Ping period counts
    PING_PER_BURST=0x8820,  # Pings per burst
    PING_PER_PROFILE=0x8820,  # Backwards naming compatibility with ULS5
    AVERAGE_BURST_PINGS=0x8920,  # Average burst pings
    AVG_PINGS=0x8920,  # Backwards naming compatibility with ULS5
    NUM_ACQUIRED_BURST_PINGS=0x8A20,  # Number of acquired burst pings
    NUM_ACQ_PINGS=0x8A20,  # Backwards naming compatibility with ULS5
    FIRST_PING=0x8B60,  # First ping number
    LAST_PING=0x8C60,  # Last acquired ping
    DATA_ERROR=0x8D20,  # Data error flag
    OVER_RUN=0x8E20,  # Overrun flag
    PHASE=0x8F20,  # The phase the data was collected from
    NUM_CHAN=0x9020,  # Number of stored frequencies
    DIG_RATE=0x9120,  # Digitization rate of stored frequencies
    LOCK_OUT_INDEX=0x9220,  # Lockout index
    NUM_BINS=0x9320,  # Number of bins
    RANGE_SAMPLES_PER_BIN=0x9420,  # Ranged
    DATA_TYPE=0x9520,  # The data type for each frequency
    PULSE_LEN=0x9620,  # The Pulse Length each frequency
    BOARD_NUM=0x9720,  # The Board number frequency
    FREQUENCY=0x9820,  # The frequency for each channel
    NUM_SENSORS=0x9920,  # The number of analog to digital sensor records summed
    SENSOR_STATUS=0x9A20,  # The Sensor status indicates available sensors
    SENSOR_DATA=0x9B26,  # The sensor data this is the analog sensor data
    ANCILLARY=0x9B26,  # Backwards naming compatibility with ULS5
    GPS_DATE_TIME=0x2026,  # GPS date
    GPS_LAT_LONG=0x21C1,  # GPS Latitude longitude
    PAROS_DATE_TIME=0x2226,  # PAROS DATE
    PAROS_PRESS_TEMP_RAW=0x2361,  # PAROS PRESSURE & TEMPERATURE Raw Reading
    PAROS_PRESS_TEMP_ENG=0x24C1,  # PAROS PRESSURE & TEMPERATURE Engineering units
    CUSTOM=0x5000,  # Custom value set by user. values 50 to 5F can be custom values
)
HEADER_LOOKUP = {v: k for k, v in HEADER_CODES.items()}

# Some earlier versions of ULS6 AZFP files have different codes
OLDER_CODES = dict(
    SERIAL_NUMBER=0x8120,  # Instrument Serial Number 2024 01 30 changed from 2 bytes to 4 bytes
    DIG_RATE=0x9123,  # Digitization rate of stored frequencies
    LOCK_OUT_INDEX=0x9223,  # Lockout index
    NUM_BINS=0x9323,  # Number of bins
    RANGE_SAMPLES_PER_BIN=0x9423,  # Ranged
    DATA_TYPE=0x9523,  # The data type for each frequency
    PULSE_LEN=0x9623,  # The Pulse Length each frequency
    BOARD_NUM=0x9723,  # The Board number frequency
    FREQUENCY=0x9823,  # The frequency for each channel
)
OLDER_HEADER_LOOKUP = {v: k for k, v in OLDER_CODES.items()}
HEADER_LOOKUP = {**HEADER_LOOKUP, **OLDER_HEADER_LOOKUP}


logger = _init_logger(__name__)


class ParseAZFP6(ParseAZFP):
    """Class for converting data from ASL Environmental Sciences AZFP echosounder."""

    # Instrument specific constants
    XML_FILE_TYPE = 0xF044CC11  # Also the start flag
    XML_END_FLAG = 0xE088DD66
    DATA_START_FLAG = 0xFF01AA00
    DATA_END_FLAG = 0xEF02BB66

    RECORD_DATA_TYPE_MASK = 0x00E0
    ARRAY_BITS_MASK = 0x001F
    CODE_BITS_MASK = 0x7F00
    TYPE_BITS_MASK = 0x00E0
    REQUIRED_BITS_MASK = 0x8000

    def __init__(
        self,
        file,
        file_meta,
        storage_options={},
        sonar_model="AZFP6",
        **kwargs,
    ):
        super().__init__(file, file_meta, storage_options, sonar_model, **kwargs)

        self.sonar_model = "AZFP6"
        self.sonar_firmware = "ULS6"

    def load_AZFP_xml(self, raw):
        """
        Parses the AZFP XML file embedded in the AZFP file.

        Updates self.parameters

        """
        xml_byte_size = unpack("<I", raw.read(4))[0]
        xml_string = raw.read(xml_byte_size)
        self.unpacked_data["num_prev_xml_bytes"] = xml_byte_size

        if int.from_bytes(raw.read(4), "little") != self.XML_END_FLAG:
            logger.error("Error reading xml string")
            raise ValueError("Error reading xml string")

        xml_prev_byte_size = unpack("<I", raw.read(4))[0]  # read num bytes for prev record
        self.unpacked_data["num_prev_xml_bytes"] = xml_prev_byte_size

        parser = ET.XMLParser(encoding="iso-8859-5")
        phase_number = None
        for event, child in ET.iterparse(
            BytesIO(xml_string), events=("start", "end"), parser=parser
        ):
            if event == "end" and child.tag == "Phases":
                phase_number = None
            if event == "start":

                if len(child.tag) > 3 and not child.tag.startswith("VTX"):
                    camel_case_tag = camelcase2snakecase(child.tag)
                else:
                    camel_case_tag = child.tag

                if len(child.attrib) > 0:
                    for key, val in child.attrib.items():
                        attrib_tag = camel_case_tag + "_" + camelcase2snakecase(key)
                        if phase_number is not None and camel_case_tag != "phase":
                            attrib_tag += f"_phase{phase_number}"
                        self.parameters[attrib_tag].append(val)
                        if child.tag == "Phase":
                            phase_number = val

                if child.text is None or all(char == "\n" for char in child.text):
                    continue

                try:
                    val = int(child.text)
                except ValueError:
                    try:
                        val = float(child.text)
                    except:
                        val = child.text

                if phase_number is not None and camel_case_tag != "phase":
                    camel_case_tag += f"_phase{phase_number}"

                self.parameters[camel_case_tag].append(val)

        # Handling the case where there is only one value for each parameter
        for key, val in self.parameters.items():
            if len(val) == 1 and key != "phase_number":
                self.parameters[key] = val[0]

        self.parameters["phase_number"] = [str(n + 1) for n in range(self.parameters["num_phases"])]
        # Gain was removed, for backward compatibility adding in a Gain=1 field
        for phase in range(self.parameters["num_phases"]):
            self.parameters[f"gain_phase{phase + 1}"] = [1] * self.parameters["num_freq"]

    def _parse_header(self, file):
        """Reads the first bytes of the header to get the header flag and number of data bytes
            Calls _split_header where the header block is parsed.

        Modifies self.unpacked_data

        Parameters
        ----------
        raw
            open binary file

        Returns
        -------
            True or False depending on whether the unpacking was successful
        """
        try:
            header_flag, num_data_bytes = unpack("<II", file.read(8))
        except:
            return False

        if header_flag == self.DATA_START_FLAG:
            # Reading will stop if the file contains an unexpected flag
            self.unpacked_data["num_data_bytes"].append(num_data_bytes)
            return self._split_header(file)

        return False

    def _print_status(self):
        """Prints message to console giving information about the raw file being parsed."""
        filename = os.path.basename(self.source_file)
        date_vals = self.unpacked_data["date_time"][0]
        timestamp = dt(
            date_vals[0],
            date_vals[1],
            date_vals[2],
            date_vals[3],
            date_vals[4],
            int(date_vals[5] + date_vals[6] / 100),
        )

        timestr = timestamp.strftime("%Y-%b-%d %H:%M:%S")
        logger.info(f"parsing file {filename}, " f"time of first ping: {timestr}")

    def _get_masked_data(self, rc):
        """
        Determine the datatype and size of the data

        Parameters
        ----------
        rc
            address byte code


        Returns
        ----------
            field_no
                hex value indicates field number/order
            byte_code
                struct parse code
            byte_size
                number of bytes for each data block element
            array_size
                number of items in data block

        """
        dt = rc & self.RECORD_DATA_TYPE_MASK
        field_no = rc & self.CODE_BITS_MASK
        array_size = (rc & self.ARRAY_BITS_MASK) + 1

        if dt == 0x00:  # int16
            byte_code = "h"
            byte_size = 2
        elif dt == 0x20:  # uint16
            byte_code = "H"
            byte_size = 2
        elif dt == 0x40:  # int32
            byte_code = "i"
            byte_size = 4
        elif dt == 0x60:  # int32
            byte_code = "I"
            byte_size = 4
        elif dt == 0x80:  # int64
            byte_code = "q"
            byte_size = 8
        elif dt == 0xA0:  # uint64
            byte_code = "Q"
            byte_size = 8
        elif dt == 0xC0:  # double
            byte_code = "d"
            byte_size = 8
        elif dt == 0xE0:  # uint8
            byte_code = "c"
            byte_size = 1
        return field_no, byte_code, byte_size, array_size

    def _split_header(self, raw):
        """Splits the header information into a dictionary.

        Modifies self.unpacked_data

        Parameters
        ----------
        raw
            open binary file

        Returns
        -------
            True or False depending on whether the unpacking was successful
        """

        header_byte_cnt = 4

        # Read first 4 bytes which contain the first header record
        rc, val = unpack("<HH", raw.read(4))
        if rc != HEADER_CODES["FIRST_HEADER_RECORD"] or val != HEADER_CODES["START_FLAG"]:
            logger.error(f"Invalid header block, is this an {self.sonar_type} file?")
            return False
        self.unpacked_data["first_header_record"].append(val)

        while header_byte_cnt < HEADER_MAXIMUM_BYTES:
            field_code = unpack("<H", raw.read(2))[0]
            _, byte_code, byte_size, array_size = self._get_masked_data(field_code)
            val = unpack("<" + byte_code * array_size, raw.read(byte_size * array_size))
            header_byte_cnt += 2 + byte_size * array_size
            try:
                field = HEADER_LOOKUP[field_code].lower()
            except:  # Unknown field
                field = f"code_{hex(field_code)}"
                logger.warning(
                    f"Unknown code found in file: {hex(field_code)}, field stored as {field}"
                )

            self.unpacked_data[field].append(*val if len(val) == 1 else [val])  # list(val)

            if field_code == HEADER_CODES["LAST_HEADER_RECORD"]:
                break

        if header_byte_cnt != self.unpacked_data["header_bytes"][0]:
            logger.error(
                "Error reading header: {} != {}".format(
                    header_byte_cnt, self.unpacked_data["header_bytes"][0]
                )
            )
            return False

        return True

    def _get_paros_time(self):
        """get date time from Paros sensor, if installed
        This is an optional parameter, the values will be 0 if no Paros sensor was
        attached during data collection
        """
        return self._get_time(key="paros_date_time")

    def _get_gps_time(self):
        """get date time from GPS sensor, if installed
        This is an optional parameter, the values will be 0 if no GPS was
        attached during data collection
        """
        return self._get_time(key="gps_date_time")

    def _get_ping_time(self):
        """get ping time"""
        self.ping_time = self._get_time(key="date_time")

    def _get_time(self, key="date_time"):
        """Assemble data time from parsed values using the specified data field."""
        if not self.unpacked_data:
            self.parse_raw()

        np_time = []
        for year, month, day, hour, min, sec, nsec in self.unpacked_data[key]:  #:
            try:
                np_time.append(
                    np.datetime64(
                        dt(year, month, day, hour, min, sec, int(sec + nsec / 100.0)).replace(
                            tzinfo=None
                        ),
                        "[ns]",
                    )
                )
            except:
                np_time.append(np.int64(0))
        return np_time
