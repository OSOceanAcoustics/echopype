import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime as dt
from struct import unpack

import fsspec
import numpy as np

from ..utils.log import _init_logger
from ..utils.misc import camelcase2snakecase
from .parse_base import ParseBase

FILENAME_DATETIME_AZFP = "\\w+.01A"

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
    120000.0: {
        150: 1.4,
        250: 1.3,
        **SV_OFFSET_HF,
    },
    125000.0: {
        150: 1.4,
        250: 1.3,
        **SV_OFFSET_HF,
    },
    130000.0: {
        150: 1.4,
        250: 1.3,
        **SV_OFFSET_HF,
    },
    200000.0: {
        150: 1.4,
        250: 1.3,
        **SV_OFFSET_HF,
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

HEADER_FIELDS = (
    ("profile_flag", "u2"),
    ("profile_number", "u2"),
    ("serial_number", "u2"),
    ("ping_status", "u2"),
    ("burst_int", "u4"),
    ("year", "u2"),  # Year
    ("month", "u2"),  # Month
    ("day", "u2"),  # Day
    ("hour", "u2"),  # Hour
    ("minute", "u2"),  # Minute
    ("second", "u2"),  # Second
    ("hundredths", "u2"),  # Hundredths of a second
    ("dig_rate", "u2", 4),  # Digitalization rate for each channel
    ("lock_out_index", "u2", 4),  # Lockout index for each channel
    ("num_bins", "u2", 4),  # Number of bins for each channel
    (
        "range_samples_per_bin",
        "u2",
        4,
    ),  # Range samples per bin for each channel
    ("ping_per_profile", "u2"),  # Number of pings per profile
    ("avg_pings", "u2"),  # Flag indicating whether the pings average in time
    ("num_acq_pings", "u2"),  # Pings acquired in the burst
    ("ping_period", "u2"),  # Ping period in seconds
    ("first_ping", "u2"),
    ("last_ping", "u2"),
    (
        "data_type",
        "u1",
        4,
    ),  # Datatype for each channel 1=Avg unpacked_data (5bytes), 0=raw (2bytes)
    ("data_error", "u2"),  # Error number is an error occurred
    ("phase", "u1"),  # Phase number used to acquire this profile
    ("overrun", "u1"),  # 1 if an overrun occurred
    ("num_chan", "u1"),  # 1, 2, 3, or 4
    ("gain", "u1", 4),  # gain channel 1-4
    ("spare_chan", "u1"),  # spare channel
    ("pulse_len", "u2", 4),  # Pulse length chan 1-4 uS
    ("board_num", "u2", 4),  # The board the data came from channel 1-4
    ("frequency", "u2", 4),  # frequency for channel 1-4 in kHz
    (
        "sensor_flag",
        "u2",
    ),  # Flag indicating if pressure sensor or temperature sensor is available
    ("ancillary", "u2", 5),  # Tilt-X, Y, Battery, Pressure, Temperature
    ("ad", "u2", 2),  # AD channel 6 and 7
)

logger = _init_logger(__name__)


class ParseAZFP(ParseBase):
    """Class for converting data from ASL Environmental Sciences AZFP echosounder."""

    # Instrument specific constants
    HEADER_SIZE = 124
    HEADER_FORMAT = ">HHHHIHHHHHHHHHHHHHHHHHHHHHHHHHHHHHBBBBHBBBBBBBBHHHHHHHHHHHHHHHHHHHH"
    FILE_TYPE = 64770

    def __init__(
        self,
        file,
        file_meta,
        storage_options={},
        sonar_model="AZFP",
        **kwargs,
    ):
        super().__init__(file, storage_options, sonar_model)
        # Parent class attributes
        #  regex pattern used to grab datetime embedded in filename
        self.timestamp_pattern = FILENAME_DATETIME_AZFP
        self.xml_path = file_meta

        # Class attributes
        self.parameters = defaultdict(list)
        self.unpacked_data = defaultdict(list)
        self.sonar_type = "AZFP"

    def load_AZFP_xml(self):
        """
        Parses the AZFP XML file.
        """

        xmlmap = fsspec.get_mapper(self.xml_path, **self.storage_options)
        phase_number = None
        for event, child in ET.iterparse(xmlmap.fs.open(xmlmap.root), events=("start", "end")):
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
                else:
                    try:
                        val = int(child.text)
                    except ValueError:
                        val = float(child.text)
                    if phase_number is not None and camel_case_tag != "phase":
                        camel_case_tag += f"_phase{phase_number}"
                    self.parameters[camel_case_tag].append(val)

        # Handling the case where there is only one value for each parameter
        for key, val in self.parameters.items():
            if len(val) == 1:
                self.parameters[key] = val[0]

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
        USL5_BAT_CONSTANT = (2.5 / 65536.0) * (86.6 + 475.0) / 86.6

        if battery_type == "main":
            N = self.unpacked_data["ancillary"][ping_num][2]
        elif battery_type == "tx":
            N = self.unpacked_data["ad"][ping_num][0]

        return N * USL5_BAT_CONSTANT

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
        P = v_in * self.parameters["a1"] + self.parameters["a0"] - 10.125
        return P

    def parse_raw(self):
        """
        Parse raw data file from AZFP echosounder.
        """

        # Read xml file into dict
        self.load_AZFP_xml()
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
            ping_num = 0
            eof = False
            while not eof:
                header_chunk = file.read(self.HEADER_SIZE)
                if header_chunk:
                    header_unpacked = unpack(self.HEADER_FORMAT, header_chunk)
                    # Reading will stop if the file contains an unexpected flag
                    if self._split_header(file, header_unpacked):
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
            if isinstance(val, list) and (not isinstance(val[0], list)):
                self.unpacked_data[key] = np.asarray(val)

        # cast all list parameter values to np array, so they are easier to reference
        for key, val in self.parameters.items():
            if isinstance(val, list):
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
        timestamp = dt(
            self.unpacked_data["year"][0],
            self.unpacked_data["month"][0],
            self.unpacked_data["day"][0],
            self.unpacked_data["hour"][0],
            self.unpacked_data["minute"][0],
            int(self.unpacked_data["second"][0] + self.unpacked_data["hundredths"][0] / 100),
        )
        timestr = timestamp.strftime("%Y-%b-%d %H:%M:%S")
        pathstr, xml_name = os.path.split(self.xml_path)
        logger.info(f"parsing file {filename} with {xml_name}, " f"time of first ping: {timestr}")

    def _split_header(self, raw, header_unpacked):
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
        if (
            header_unpacked[0] != self.FILE_TYPE
        ):  # first field should match hard-coded FILE_TYPE from manufacturer
            check_eof = raw.read(1)
            if check_eof:
                logger.error("Unknown file type")
                return False
        header_byte_cnt = 0

        # fields with num_freq data still takes 4 bytes,
        # the extra bytes contain random numbers
        firmware_freq_len = 4

        field_w_freq = (
            "dig_rate",
            "lock_out_index",
            "num_bins",
            "range_samples_per_bin",  # fields with num_freq data
            "data_type",
            "gain",
            "pulse_len",
            "board_num",
            "frequency",
        )
        for field in HEADER_FIELDS:
            if field[0] in field_w_freq:  # fields with num_freq data
                self.unpacked_data[field[0]].append(
                    header_unpacked[header_byte_cnt : header_byte_cnt + self.parameters["num_freq"]]
                )
                header_byte_cnt += firmware_freq_len
            elif len(field) == 3:  # other longer fields ('ancillary' and 'ad')
                self.unpacked_data[field[0]].append(
                    header_unpacked[header_byte_cnt : header_byte_cnt + field[2]]
                )
                header_byte_cnt += field[2]
            else:
                self.unpacked_data[field[0]].append(header_unpacked[header_byte_cnt])
                header_byte_cnt += 1
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
                    ">" + "I" * counts_byte_size, raw.read(counts_byte_size * 4)
                )  # Linear sum
                lso = unpack(
                    ">" + "B" * counts_byte_size, raw.read(counts_byte_size * 1)
                )  # linear sum overflow
                v = (np.array(ls) + np.array(lso) * 4294967295) / divisor
                v = (np.log10(v) - 2.5) * (8 * 65535) * self.parameters["DS"][freq_ch]
                v[np.isinf(v)] = 0
                vv_tmp[freq_ch] = v
            else:
                counts_chunk = raw.read(counts_byte_size * 2)
                counts_unpacked = unpack(">" + "H" * counts_byte_size, counts_chunk)
                vv_tmp[freq_ch] = counts_unpacked
        self.unpacked_data["counts"].append(vv_tmp)

    def _check_uniqueness(self):
        """Check for ping-by-ping consistency of sampling parameters and reduce if identical."""
        if not self.unpacked_data:
            self.parse_raw()

        if np.array(self.unpacked_data["profile_flag"]).size != 1:  # Only check uniqueness once.
            # fields with num_freq data
            field_w_freq = (
                "dig_rate",
                "lock_out_index",
                "num_bins",
                "range_samples_per_bin",
                "data_type",
                "gain",
                "pulse_len",
                "board_num",
                "frequency",
            )
            # fields to reduce size if the same for all pings
            field_include = (
                "profile_flag",
                "serial_number",
                "burst_int",
                "ping_per_profile",
                "avg_pings",
                "ping_period",
                "phase",
                "num_chan",
                "spare_chan",
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
                else:
                    raise ValueError(f"Header value {field} is not constant for each ping")

    def _get_ping_time(self):
        """Assemble ping time from parsed values."""

        if not self.unpacked_data:
            self.parse_raw()

        ping_time = []
        for ping_num, year in enumerate(self.unpacked_data["year"]):
            ping_time.append(
                np.datetime64(
                    dt(
                        year,
                        self.unpacked_data["month"][ping_num],
                        self.unpacked_data["day"][ping_num],
                        self.unpacked_data["hour"][ping_num],
                        self.unpacked_data["minute"][ping_num],
                        int(
                            self.unpacked_data["second"][ping_num]
                            + self.unpacked_data["hundredths"][ping_num] / 100
                        ),
                    ).replace(tzinfo=None),
                    "[ns]",
                )
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
