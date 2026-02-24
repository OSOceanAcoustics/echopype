import os
import xml.etree.ElementTree as ET
from datetime import datetime as dt
from struct import unpack

import fsspec
import numpy as np

from ..utils.log import _init_logger
from ..utils.misc import camelcase2snakecase
from .parse_azfp import ParseAZFP

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


class ParseULS5(ParseAZFP):
    """Class for converting data from ASL Environmental Sciences AZFP echosounder."""

    # Instrument specific constants
    HEADER_SIZE = 124
    HEADER_FORMAT = ">HHHHIHHHHHHHHHHHHHHHHHHHHHHHHHHHHHBBBBHBBBBBBBBHHHHHHHHHHHHHHHHHHHH"
    FILE_TYPE = 64770

    BAT_CONSTANT = (2.5 / 65536.0) * (86.6 + 475.0) / 86.6

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
        "BP",  # for single channel data, these are stored as scalars
        "DS",
        "EL",
        "TVR",
        "VTX0",
        "VTX1",
        "VTX2",
        "VTX3",
    )
    # fields to reduce size if the same for all pings
    field_reduce = (
        "profile_flag",
        "ping_period",
        "serial_number",
        "burst_int",
        "ping_per_profile",
        "avg_pings",
        "phase",
        "num_chan",
        "spare_chan",
    )

    def __init__(
        self,
        file,
        file_meta,
        storage_options={},
        sonar_model="AZFP",
        **kwargs,
    ):
        super().__init__(file, storage_options, storage_options, sonar_model)
        # Parent class attributes
        #  regex pattern used to grab datetime embedded in filename
        self.xml_path = file_meta
        # Class attributes
        self.sonar_type = "AZFP"
        self.sonar_firmware = "ULS5"

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

        # The last phase can be a Repeat phase, which switches back to the first phase
        phase_number = self.parameters["phase_number"][-1]
        if self.parameters[f"phase_type_svalue_phase{phase_number}"] == "Repeat":
            items = list(self.parameters.items())
            for k, v in items:
                if "_phase1" in k:
                    new_key = k.replace("_phase1", f"_phase{phase_number}")
                    if new_key not in self.parameters.keys():
                        self.parameters[new_key] = v

    def _parse_header(self, file):
        header_chunk = file.read(self.HEADER_SIZE)
        if header_chunk:
            header_unpacked = unpack(self.HEADER_FORMAT, header_chunk)

            # Reading will stop if the file contains an unexpected flag
            return self._split_header(file, header_unpacked)
        return False

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

        with fmap.fs.open(fmap.root, "rb") as file:

            temperature_is_valid = _test_valid_params(["ka", "kb", "kc"])
            pressure_is_valid = _test_valid_params(["a0", "a1"])
            tilt_x_is_valid = _test_valid_params(["X_a", "X_b", "X_c"])
            tilt_y_is_valid = _test_valid_params(["Y_a", "Y_b", "Y_c"])

            ping_num = 0
            eof = False
            while not eof:
                if self._parse_header(file):
                    # Appends the actual 'data values' to unpacked_data
                    self._add_counts(file, ping_num, endian=">")

                    if ping_num == 0:
                        # Display information about the file that was loaded in
                        self._print_status()

                    # Compute temperature from unpacked_data[ii]['ancillary][4] or using Paros
                    self.unpacked_data["temperature"].append(
                        self._compute_analog_temperature(ping_num, temperature_is_valid)
                    )
                    # Compute pressure from unpacked_data[ii]['ancillary'][3]
                    self.unpacked_data["pressure"].append(
                        self._compute_analog_pressure(ping_num, pressure_is_valid)
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

                    # print(f"ping {ping_num} ends at", file.tell())
                else:
                    # End of file
                    eof = True
                ping_num += 1
        self._check_uniqueness("profile_flag")
        self._get_ping_time()

        # Explicitly cast frequency to a float in accordance with the SONAR-netCDF4 convention
        self.unpacked_data["frequency"] = self.unpacked_data["frequency"].astype(np.float64)

        # cast unpacked_data values to np arrays, so they are easier to reference
        for key, val in self.unpacked_data.items():
            # if it is not a nested list, make the value into a ndarray
            if isinstance(val, (list, tuple)) and (not isinstance(val[0], (tuple, list))):
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
        """
        self.Sv_offset = np.zeros_like(self.freq_sorted)
        for ind, ich in enumerate(self.freq_ind_sorted):
            self.Sv_offset[ind] = self._calc_Sv_offset(
                self.freq_sorted[ind], self.unpacked_data["pulse_len"][ich]
            )
        """

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
        if battery_type == "main":
            N = self.unpacked_data["ancillary"][ping_num][2]
        elif battery_type == "tx":
            N = self.unpacked_data["ad"][ping_num][0]

        return N * self.BAT_CONSTANT

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

    def _get_ping_time(self):
        """Assemble ping time from parsed values."""

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
