import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime as dt
from io import BytesIO
from struct import unpack

import fsspec
import numpy as np

from ..utils.log import _init_logger
from ..utils.misc import camelcase2snakecase
from .parse_base import ParseBase

FILENAME_DATETIME_AZFP = "\\w+_\\w+.azfp"

# NOTE: These values may change once the new AZFP details are finalized

# Common Sv_offset values for frequency > 38 kHz
SV_OFFSET_HF = {
    300: 1.1,
    500: 0.8,
    700: 0.5,
    900: 0.3,
    1000: 0.3,
}
SV_OFFSET_LF = {
    500: 1.1,
    1000: 0.7,
}
SV_OFFSET = {
    38000.0: {**SV_OFFSET_LF},
    67000.0: {
        500: 1.1,
        **SV_OFFSET_HF,
    },
    125000.0: {
        150: 1.4,
        250: 1.3,
        **SV_OFFSET_HF,
    },
    200000.0: {
        150: 1.4,
        250: 1.3,
        **SV_OFFSET_HF,
    },
    417000.0: {
        **SV_OFFSET_HF,
        68: 0,  # NOTE: Not official offset, Matlab code defaults to 0 in this scenario
    },
    455000.0: {
        250: 1.3,
        **SV_OFFSET_HF,
    },
    769000.0: {
        150: 1.4,
        **SV_OFFSET_HF,
    },
}

HEADER_FIELDS = [
    "FirstHeaderRecord",
    "HeaderBytes",
    "HeaderNumRecords",
    "ProfileNumber",
    "SerialNumber",
    "Date",
    "AcqStatus",
    "BurstInt",
    "BaseTime",
    "PingPeriod",
    "PingPeriodCounts",
    "PingPerProfile",
    "AvgPings",
    "NumAcqPings",
    "FirstPing",
    "LastPing",
    "DataError",
    "OverRun",
    "Phase",
    "NumChan",
    "DigRate",
    "LockOutIndex",
    "NumBins",
    "RangeSamplesPerBin",
    "DataType",
    "PulseLen",
    "BoardNum",
    "Frequency",
    "NumSensors",
    "SensorStatus",
    "Ancillary",
    "GpsDateTime",
    "GpsLatLon",
    # 0X70 to 0x7F CUSTOM 0 Set by user Set by user Variable values
    "Custom",
    "LastHeaderRecord",
]

OPTIONAL_HEADER_FIELDS = ["gps_date_time", "gps_lat_lon", "custom"]

logger = _init_logger(__name__)


class ParseAZFP6(ParseBase):
    """Class for converting data from ASL Environmental Sciences AZFP echosounder."""

    # Instrument specific constants
    XML_FILE_TYPE = 0xF044CC11  # Also the start flag
    XML_END_FLAG = 0xE088DD66
    DATA_START_FLAG = 0xFF01AA00
    HEADER_START_FLAG = 0xBCD0
    HEADER_END_FLAG = 0xABC1
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
        super().__init__(file, storage_options, sonar_model)
        # Parent class attributes
        #  regex pattern used to grab datetime embedded in filename
        self.timestamp_pattern = FILENAME_DATETIME_AZFP

        # Class attributes
        self.parameters = defaultdict(list)
        self.unpacked_data = defaultdict(list)
        self.sonar_type = "AZFP6"

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

                if all(char == "\n" for char in child.text):
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

                # print(camel_case_tag, val)
                self.parameters[camel_case_tag].append(val)

        # Handling the case where there is only one value for each parameter
        for key, val in self.parameters.items():
            if len(val) == 1 and key != "phase_number":
                self.parameters[key] = val[0]

        self.parameters["phase_number"] = [str(n + 1) for n in range(self.parameters["num_phases"])]
        # Gain was removed, for backward compatibility adding in a Gain=1 field
        for phase in range(self.parameters["num_phases"]):
            self.parameters[f"gain_phase{phase + 1}"] = [1] * self.parameters["num_freq"]

        # from pprint import pprint as pp
        # pp(self.parameters)

    def _compute_temperature(self, ping_num, is_valid):
        """
        Compute temperature in celsius.

        Parameters
        ----------
        ping_num
            ping number
        is_valid
            whether the associated parameters have valid values
        """
        if not is_valid:
            return np.nan

        counts = self.unpacked_data["ancillary"][ping_num][4]
        v_in = 2.5 * (counts / 65535)
        R = (self.parameters["ka"] + self.parameters["kb"] * v_in) / (self.parameters["kc"] - v_in)

        # fmt: off
        T = 1 / (
            self.parameters["A"]
            + self.parameters["B"] * (np.log(R))
            + self.parameters["C"] * (np.log(R) ** 3)
        ) - 273
        # fmt: on
        return T

    def _compute_tilt(self, ping_num, xy, is_valid):
        """
        Compute instrument tilt.

        Parameters
        ----------
        ping_num
            ping number
        xy
            either "X" or "Y"
        is_valid
            whether the associated parameters have valid values
        """
        if not is_valid:
            return np.nan
        else:
            idx = 0 if xy == "X" else 1
            N = self.unpacked_data["ancillary"][ping_num][idx]
            a = self.parameters[f"{xy}_a"]
            b = self.parameters[f"{xy}_b"]
            c = self.parameters[f"{xy}_c"]
            d = self.parameters[f"{xy}_d"]
            return a + b * N + c * N**2 + d * N**3

    def _compute_battery(self, ping_num, battery_type):
        """
        Compute battery voltage.

        Parameters
        ----------
        ping_num
            ping number
        type
            either "main" or "tx"
        """
        USL6_BAT_CONSTANT = (2.5 / 65535.0) * (86.6 + 475.0) / 86.6

        if battery_type == "main":
            N = self.unpacked_data["ancillary"][ping_num][2]
        elif battery_type == "tx":
            N = self.unpacked_data["ancillary"][ping_num][-2]

        return N * USL6_BAT_CONSTANT

    def _compute_pressure(self, ping_num, is_valid):
        """
        Compute pressure in decibar

        Parameters
        ----------
        ping_num
            ping number
        is_valid
            whether the associated parameters have valid values
        """
        if not is_valid or self.parameters["sensors_flag_pressure_sensor_installed"] == "no":
            return np.nan

        counts = self.unpacked_data["ancillary"][ping_num][3]
        v_in = 2.5 * (counts / 65535)
        P = v_in * self.parameters["a1"] + self.parameters["a0"]  # - 10.125
        return P

    def parse_raw(self):
        """
        Parse raw data file from AZFP echosounder.
        """

        # Read xml file into dict
        fmap = fsspec.get_mapper(self.source_file, **self.storage_options)

        # Set flags for presence of valid parameters for temperature and tilt
        def _test_valid_params(params):
            if all([np.isclose(self.parameters[p], 0) for p in params]):
                return False
            else:
                return True

        temperature_is_valid = _test_valid_params(["ka", "kb", "kc"])
        pressure_is_valid = _test_valid_params(["a0", "a1"])
        tilt_x_is_valid = _test_valid_params(["X_a", "X_b", "X_c"])
        tilt_y_is_valid = _test_valid_params(["Y_a", "Y_b", "Y_c"])

        with fmap.fs.open(fmap.root, "rb") as file:

            if (
                unpack("<I", file.read(4))[0] == self.XML_FILE_TYPE
            ):  # first field should match hard-coded FILE_TYPE from manufacturer
                self.load_AZFP_xml(file)
            else:
                raise ValueError("Unknown file type")

            ping_num = 0
            eof = False
            while not eof:
                try:
                    header_flag, num_data_bytes = unpack("<II", file.read(8))
                except:
                    break

                if header_flag == self.DATA_START_FLAG:
                    # Reading will stop if the file contains an unexpected flag
                    self.unpacked_data["num_data_bytes"].append(num_data_bytes)
                    if self._split_header(file):
                        # Appends the actual 'data values' to unpacked_data
                        self._add_counts(file, ping_num)
                        if ping_num == 0:
                            # Display information about the file that was loaded in
                            self._print_status()
                        # Compute temperature from unpacked_data[ii]['ancillary][4]
                        self.unpacked_data["temperature"].append(
                            self._compute_temperature(ping_num, temperature_is_valid)
                        )
                        # Compute pressure from unpacked_data[ii]['ancillary'][3]
                        self.unpacked_data["pressure"].append(
                            self._compute_pressure(ping_num, pressure_is_valid)
                        )
                        # compute x tilt from unpacked_data[ii]['ancillary][0]
                        self.unpacked_data["tilt_x"].append(
                            self._compute_tilt(ping_num, "X", tilt_x_is_valid)
                        )
                        # Compute y tilt from unpacked_data[ii]['ancillary][1]
                        self.unpacked_data["tilt_y"].append(
                            self._compute_tilt(ping_num, "Y", tilt_y_is_valid)
                        )
                        # Compute cos tilt magnitude from tilt x and y values
                        self.unpacked_data["cos_tilt_mag"].append(
                            np.cos(
                                (
                                    np.sqrt(
                                        self.unpacked_data["tilt_x"][ping_num] ** 2
                                        + self.unpacked_data["tilt_y"][ping_num] ** 2
                                    )
                                )
                                * np.pi
                                / 180
                            )
                        )
                        # Calculate voltage of main battery pack
                        self.unpacked_data["battery_main"].append(
                            self._compute_battery(ping_num, battery_type="main")
                        )
                        # If there is a Tx battery pack
                        self.unpacked_data["battery_tx"].append(
                            self._compute_battery(ping_num, battery_type="tx")
                        )

                        header_flag, num_data_bytes = unpack("<II", file.read(8))
                        if header_flag != self.DATA_END_FLAG:
                            logger.error("Invalid flag detected, possibly corrupted data file.")
                            break
                        if num_data_bytes != self.unpacked_data["num_data_bytes"][ping_num]:
                            logger.error("Invalid data block size, possibly corrupted data file.")
                            break
                    else:
                        break
                else:
                    # End of file
                    eof = True
                ping_num += 1

        self._check_uniqueness()
        self._get_ping_time()

        # Explicitly cast frequency to a float in accordance with the SONAR-netCDF4 convention
        self.unpacked_data["frequency"] = self.unpacked_data["frequency"].astype(np.float64)

        # cast unpacked_data values to np arrays, so they are easier to reference
        for key, val in self.unpacked_data.items():
            # if it is not a nested list, make the value into a ndarray
            if isinstance(val, (list, tuple)) and (not isinstance(val[0], (list, tuple))):
                self.unpacked_data[key] = np.asarray(val)

        # cast all list parameter values to np array, so they are easier to reference
        for key, val in self.parameters.items():
            if isinstance(val, (list, tuple)):
                self.parameters[key] = np.asarray(val)

        # Get frequency values
        freq_old = self.unpacked_data["frequency"]

        # Obtain sorted frequency indices
        self.freq_ind_sorted = freq_old.argsort()

        # Obtain sorted frequencies
        self.freq_sorted = freq_old[self.freq_ind_sorted] * 1000.0

        # Build Sv offset
        self.Sv_offset = np.zeros_like(self.freq_sorted)
        for ind, ich in enumerate(self.freq_ind_sorted):
            self.Sv_offset[ind] = self._calc_Sv_offset(
                self.freq_sorted[ind], self.unpacked_data["pulse_len"][ich]
            )

    def _print_status(self):
        """Prints message to console giving information about the raw file being parsed."""
        filename = os.path.basename(self.source_file)
        date_vals = self.unpacked_data["date"][0]
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
        -----------
            byte_code
                struct parse code
            byte_size
                number of bytes for each data block element
            array_size
                number of items in data block

        """
        dt = rc & self.RECORD_DATA_TYPE_MASK
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
        return byte_code, byte_size, array_size

    def _split_header(self, raw):
        """Splits the header information into a dictionary.
        Modifies self.unpacked_data

        Parameters
        ----------
        raw
            open binary file
        header_unpacked
            output of struct unpack of raw file

        Returns
        -------
            True or False depending on whether the unpacking was successful
        """

        header_byte_cnt = 4

        # Read first 4 bytes which contain the first header record
        rc, val = unpack("<HH", raw.read(4))
        if val != self.HEADER_START_FLAG:
            logger.error(f"Invalid header block, is this an {self.sonar_type} file?")
            return False

        self.unpacked_data[camelcase2snakecase(HEADER_FIELDS[0])].append(val)
        for field in HEADER_FIELDS[1:]:
            field = camelcase2snakecase(field)
            rc = unpack("<H", raw.read(2))[0]
            byte_code, byte_size, array_size = self._get_masked_data(rc)
            val = unpack("<" + byte_code * array_size, raw.read(byte_size * array_size))
            header_byte_cnt += 2 + byte_size * array_size

            if val[0] == self.HEADER_END_FLAG:
                self.unpacked_data[camelcase2snakecase(HEADER_FIELDS[-1])].append(val[0])
                break

            self.unpacked_data[field].append(*val if len(val) == 1 else [val])  # list(val)

        if header_byte_cnt != self.unpacked_data["header_bytes"][0]:
            logger.error(
                "Error reading header: {} != {}".format(
                    header_byte_cnt, self.unpacked_data["header_bytes"][0]
                )
            )
            return False

        return True

    def _add_counts(self, raw, ping_num):
        """Unpacks the echosounder raw data. Modifies self.unpacked_data."""
        vv_tmp = [[]] * self.unpacked_data["num_chan"][ping_num]
        for freq_ch in range(self.unpacked_data["num_chan"][ping_num]):
            counts_byte_size = self.unpacked_data["num_bins"][ping_num][freq_ch]
            if self.unpacked_data["data_type"][ping_num][freq_ch]:
                if self.unpacked_data["avg_pings"][ping_num]:  # if pings are averaged over time
                    divisor = (
                        self.unpacked_data["ping_per_profile"][ping_num]
                        * self.unpacked_data["range_samples_per_bin"][ping_num][freq_ch]
                    )
                else:
                    divisor = self.unpacked_data["range_samples_per_bin"][ping_num][freq_ch]
                ls = unpack(
                    "<" + "I" * counts_byte_size, raw.read(counts_byte_size * 4)
                )  # Linear sum
                lso = unpack(
                    "<" + "B" * counts_byte_size, raw.read(counts_byte_size * 1)
                )  # linear sum overflow
                v = (np.array(ls) + np.array(lso) * 4294967295) / divisor
                v = (np.log10(v) - 2.5) * (8 * 65535) * self.parameters["DS"][freq_ch]
                v[np.isinf(v)] = 0
                vv_tmp[freq_ch] = v
            else:
                counts_chunk = raw.read(counts_byte_size * 2)
                counts_unpacked = unpack("<" + "H" * counts_byte_size, counts_chunk)
                vv_tmp[freq_ch] = counts_unpacked
        self.unpacked_data["counts"].append(vv_tmp)

        return True

    def _check_uniqueness(self):
        """Check for ping-by-ping consistency of sampling parameters and reduce if identical."""
        if not self.unpacked_data:
            self.parse_raw()

        if (
            np.array(self.unpacked_data["first_header_record"]).size != 1
        ):  # profile_flag # Only check uniqueness once.
            # fields with num_freq data
            field_w_freq = (
                "dig_rate",
                "lock_out_index",
                "num_bins",
                "range_samples_per_bin",
                "data_type",
                # "gain", #Gain was removed from sensor in ULS6
                "pulse_len",
                "board_num",
                "frequency",
            )
            # fields to reduce size if the same for all pings
            field_include = (
                "base_time",
                "ping_period_counts",
                "serial_number",
                "burst_int",
                "ping_per_profile",
                "avg_pings",
                "ping_period",
                "phase",
                "num_chan",
                # "spare_chan",
                "custom",
            )
            for field in field_w_freq:
                uniq = np.unique(self.unpacked_data[field], axis=0)
                if uniq.shape[0] == 1:
                    self.unpacked_data[field] = uniq.squeeze()
                else:
                    raise ValueError(f"Header value {field} is not constant for each ping")
            for field in field_include:
                uniq = np.unique(self.unpacked_data[field])
                if uniq.shape[0] == 1:
                    self.unpacked_data[field] = uniq.squeeze()
                elif uniq.shape[0] == 0 and field in OPTIONAL_HEADER_FIELDS:
                    self.unpacked_data[field] = (
                        None  # TODO: This may break sonar-netcdf4 conventions
                    )
                else:
                    raise ValueError(f"Header value {field} is not constant for each ping")

    def _get_gps_time(self):
        """Assemble gps time from parsed values. This is an optional parameter, the values will be
        0 if no GPS was attached during data collection"""
        if not self.unpacked_data:
            self.parse_raw()

        np_time = []
        for year, month, day, hour, min, sec, nsec in self.unpacked_data["gps_date_time"]:  #:
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

    def _get_ping_time(self):
        """Assemble ping time from parsed values."""

        if not self.unpacked_data:
            self.parse_raw()

        ping_time = []
        for year, month, day, hour, min, sec, nsec in self.unpacked_data["date"]:
            ping_time.append(
                np.datetime64(
                    dt(year, month, day, hour, min, int(sec + nsec / 100.0)).replace(tzinfo=None),
                    "[ns]",
                )  # + np.timedelta64(nsec, 'ns')
            )
        self.ping_time = ping_time

    @staticmethod
    def _calc_Sv_offset(freq, pulse_len):
        """
        Calculate the compensation factor for Sv calculation.

        Parameters
        ----------
        freq : number
            transmit frequency
        pulse_len : number
            pulse length
        """
        # Check if the specified freq is in the allowable Sv_offset dict
        if freq not in SV_OFFSET.keys():
            raise ValueError(
                f"Frequency {freq} Hz is not in the Sv offset dictionary! "
                "Please contact AZFP Environmental Sciences "
                "and raise an issue in the echopype repository."
            )

        # Check if the specified freq-pulse length combination is in the allowable Sv_offset dict
        if pulse_len not in SV_OFFSET[freq]:
            raise ValueError(
                f"Pulse length {pulse_len} us is not in the Sv offset dictionary "
                f"for the {freq} Hz channel! "  # add freq info
                "Please contact AZFP Environmental Sciences "
                "and raise an issue in the echopype repository."
            )

        return SV_OFFSET[freq][pulse_len]
