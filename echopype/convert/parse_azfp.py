import abc
from collections import defaultdict
from struct import unpack

import numpy as np

from ..utils.log import _init_logger
from .parse_base import ParseBase

logger = _init_logger(__name__)

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


class ParseAZFP(ParseBase, abc.ABC):
    """Base class for converting data from ASL Environmental Sciences AZFP echosounder."""

    def __init__(
        self,
        file,
        file_meta,
        storage_options={},
        sonar_model="AZFP",
        **kwargs,
    ):
        super().__init__(file, storage_options, sonar_model)
        # Class attributes
        self.parameters = defaultdict(list)
        self.unpacked_data = defaultdict(list)

    @abc.abstractmethod
    def load_AZFP_xml(self):
        """
        Parses the AZFP XML file.
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def _parse_header(self, file):
        """Parse header of raw data file."""

    @abc.abstractmethod
    def parse_raw(self):
        """
        Parse raw data file from AZFP echosounder.
        """

    @abc.abstractmethod
    def _print_status(self):
        """Prints message to console giving information about the raw file being parsed."""

    @abc.abstractmethod
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

    def _add_counts(self, raw, ping_num, endian):
        """Unpacks the echosounder raw data. Modifies self.unpacked_data."""
        vv_tmp = [[]] * self.unpacked_data["num_chan"][ping_num]

        # TODO: this is a bit hacky, convert the parameters to a numpy array and make a extra dim?
        if self.unpacked_data["num_chan"][ping_num] == 1:
            self.unpacked_data["num_bins"][ping_num] = [self.unpacked_data["num_bins"][ping_num]]
            self.unpacked_data["data_type"][ping_num] = [self.unpacked_data["data_type"][ping_num]]
            self.unpacked_data["range_samples_per_bin"][ping_num] = [
                self.unpacked_data["range_samples_per_bin"][ping_num]
            ]

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
                    endian + "I" * counts_byte_size, raw.read(counts_byte_size * 4)
                )  # Linear sum
                lso = unpack(
                    endian + "B" * counts_byte_size, raw.read(counts_byte_size * 1)
                )  # linear sum overflow
                v = (np.array(ls) + np.array(lso) * 4294967295) / divisor
                v = (np.log10(v) - 2.5) * (8 * 65535) * self.parameters["DS"][freq_ch]
                v[np.isinf(v)] = 0
                vv_tmp[freq_ch] = v
            else:
                counts_chunk = raw.read(counts_byte_size * 2)
                counts_unpacked = unpack(endian + "H" * counts_byte_size, counts_chunk)
                vv_tmp[freq_ch] = counts_unpacked
        self.unpacked_data["counts"].append(vv_tmp)

    def _check_uniqueness(self, profile_flag):
        """Check for ping-by-ping consistency of sampling parameters and reduce if identical."""

        if not self.unpacked_data:
            raise ValueError("Possibly corrupted AZFP file.")

        if np.array(self.unpacked_data[profile_flag]).size != 1:  # Only check uniqueness once.
            # fields with num_freq data

            for field in self.field_w_freq:
                if field not in self.unpacked_data:
                    if field not in self.parameters:
                        continue
                    if not isinstance(self.parameters[field], (list, tuple)):
                        self.parameters[field] = [self.parameters[field]]
                    continue

                uniq = np.unique(self.unpacked_data[field], axis=0)
                if uniq.shape[0] == 1:
                    uniq = uniq.squeeze()
                    if len(uniq.shape) == 0:
                        self.unpacked_data[field] = np.asarray([uniq])
                    else:
                        self.unpacked_data[field] = uniq
                else:
                    raise ValueError(f"Header value {field} is not constant for each ping")

            for field in self.field_reduce:
                if field not in self.unpacked_data:
                    continue

                uniq = np.unique(self.unpacked_data[field])
                if uniq.shape[0] == 1:
                    self.unpacked_data[field] = uniq.squeeze()
                # else:
                #    raise ValueError(f"Header value {field} is not constant for each ping")

    @abc.abstractmethod
    def _get_ping_time(self):
        """Assemble ping time from parsed values."""

    def _compute_analog_temperature(self, ping_num, is_valid):
        """
        Compute temperature in celsius from analog sensor.

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

        # Sept 2007, use linear equation if ka < -98 for use with linear sensors
        if self.parameters["ka"] < -98:
            return self.parameters["A"] * v_in + self.parameters["B"]

        R = (self.parameters["ka"] + self.parameters["kb"] * v_in) / (self.parameters["kc"] - v_in)
        if R <= 0:
            return -99.0

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

    def _compute_analog_pressure(self, ping_num, is_valid):
        """
        Compute pressure in decibar from analog sensor

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
