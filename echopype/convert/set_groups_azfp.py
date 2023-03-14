"""
Class to save unpacked echosounder data to appropriate groups in netcdf or zarr.
"""
from typing import List

import numpy as np
import xarray as xr

from ..utils.coding import set_time_encodings
from .set_groups_base import SetGroupsBase


class SetGroupsAZFP(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from AZFP data files."""

    # The sets beam_only_names, ping_time_only_names, and
    # beam_ping_time_names are used in set_groups_base and
    # in converting from v0.5.x to v0.6.0. The values within
    # these sets are applied to all Sonar/Beam_groupX groups.

    # Variables that need only the beam dimension added to them.
    beam_only_names = {"backscatter_r"}

    # Variables that need only the ping_time dimension added to them.
    ping_time_only_names = {"sample_interval", "transmit_duration_nominal"}

    # Variables that need beam and ping_time dimensions added to them.
    beam_ping_time_names = {"equivalent_beam_angle", "gain_correction"}

    beamgroups_possible = [
        {
            "name": "Beam_group1",
            "descr": "contains backscatter power (uncalibrated) and other beam or channel-specific data.",  # noqa
        }
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # get frequency values
        freq_old = list(self.parser_obj.unpacked_data["frequency"])

        # sort the frequencies in ascending order
        freq_new = freq_old[:]
        freq_new.sort(reverse=False)

        # obtain sorted frequency indices
        self.freq_ind_sorted = [freq_new.index(ch) for ch in freq_old]

        # obtain sorted frequencies
        self.freq_sorted = self.parser_obj.unpacked_data["frequency"][self.freq_ind_sorted]

        # obtain channel_ids
        self.channel_ids_sorted = self._create_unique_channel_name()

        # Put Frequency in Hz (this should be done after create_unique_channel_name)
        self.freq_sorted = self.freq_sorted * 1000  # Frequency in Hz

    def _create_unique_channel_name(self):
        """
        Creates a unique channel name for AZFP sensor
        using the variable unpacked_data created by
        the AZFP parser
        """

        serial_number = self.parser_obj.unpacked_data["serial_number"]

        if serial_number.size == 1:
            freq_as_str = self.freq_sorted.astype(int).astype(str)

            # TODO: replace str(i+1) with Frequency Number from XML
            channel_id = [
                str(serial_number) + "-" + freq + "-" + str(i + 1)
                for i, freq in enumerate(freq_as_str)
            ]

            return channel_id
        else:
            raise NotImplementedError(
                "Creating a channel name for more than"
                + " one serial number has not been implemented."
            )

    def set_env(self) -> xr.Dataset:
        """Set the Environment group."""
        # TODO Look at why this cannot be encoded without the modifications
        # @ngkavin: what modification?
        ping_time = self.parser_obj.ping_time
        ds = xr.Dataset(
            {
                "temperature": (
                    ["time1"],
                    self.parser_obj.unpacked_data["temperature"],
                )
            },
            coords={
                "time1": (
                    ["time1"],
                    ping_time,
                    {
                        "axis": "T",
                        "long_name": "Timestamp of each ping",
                        "standard_name": "time",
                        "comment": "Time coordinate corresponding to environmental variables.",
                    },
                )
            },
            attrs={"long_name": "Water temperature", "units": "C"},
        )

        return set_time_encodings(ds)

    def set_sonar(self) -> xr.Dataset:
        """Set the Sonar group."""

        # Add beam_group and beam_group_descr variables sharing a common dimension
        # (beam_group), using the information from self._beamgroups
        self._beamgroups = self.beamgroups_possible
        beam_groups_vars, beam_groups_coord = self._beam_groups_vars()
        ds = xr.Dataset(beam_groups_vars, coords=beam_groups_coord)

        # Assemble sonar group global attribute dictionary
        sonar_attr_dict = {
            "sonar_manufacturer": "ASL Environmental Sciences",
            "sonar_model": self.sonar_model,
            "sonar_serial_number": int(self.parser_obj.unpacked_data["serial_number"]),
            "sonar_software_name": "Based on AZFP Matlab Toolbox",
            "sonar_software_version": "1.4",
            "sonar_type": "echosounder",
        }
        ds = ds.assign_attrs(sonar_attr_dict)

        return ds

    def set_platform(self) -> xr.Dataset:
        """Set the Platform group."""
        platform_dict = {
            "platform_name": self.ui_param["platform_name"],
            "platform_type": self.ui_param["platform_type"],
            "platform_code_ICES": self.ui_param["platform_code_ICES"],
        }
        unpacked_data = self.parser_obj.unpacked_data
        time2 = self.parser_obj.ping_time

        ds = xr.Dataset(
            {
                "tilt_x": (["time2"], unpacked_data["tilt_x"]),
                "tilt_y": (["time2"], unpacked_data["tilt_y"]),
                **{
                    var: ([], np.nan, self._varattrs["platform_var_default"][var])
                    for var in [
                        "MRU_offset_x",
                        "MRU_offset_y",
                        "MRU_offset_z",
                        "MRU_rotation_x",
                        "MRU_rotation_y",
                        "MRU_rotation_z",
                        "position_offset_x",
                        "position_offset_y",
                        "position_offset_z",
                        "transducer_offset_x",
                        "transducer_offset_y",
                        "transducer_offset_z",
                        "vertical_offset",
                        "water_level",
                    ]
                },
            },
            coords={
                "time2": (
                    ["time2"],
                    time2,
                    {
                        "axis": "T",
                        "long_name": "Timestamps for platform motion and orientation data",
                        "standard_name": "time",
                        "comment": "Time coordinate corresponding to platform motion and "
                        "orientation data.",
                    },
                ),
            },
        )
        ds = ds.assign_attrs(platform_dict)
        return set_time_encodings(ds)

    def set_beam(self) -> List[xr.Dataset]:
        """Set the Beam group."""
        unpacked_data = self.parser_obj.unpacked_data
        parameters = self.parser_obj.parameters
        dig_rate = unpacked_data["dig_rate"][self.freq_ind_sorted]  # dim: freq
        ping_time = self.parser_obj.ping_time

        # Build variables in the output xarray Dataset
        N = []  # for storing backscatter_r values for each frequency
        for ich in self.freq_ind_sorted:
            N.append(
                np.array(
                    [unpacked_data["counts"][p][ich] for p in range(len(unpacked_data["year"]))]
                )
            )

        # Largest number of counts along the range dimension among the different channels
        longest_range_sample = np.max(unpacked_data["num_bins"])
        range_sample = np.arange(longest_range_sample)

        # Pad power data
        if any(unpacked_data["num_bins"] != longest_range_sample):
            N_tmp = np.full((len(N), len(ping_time), longest_range_sample), np.nan)
            for i, n in enumerate(N):
                N_tmp[i, :, : n.shape[1]] = n
            N = N_tmp
            del N_tmp

        tdn = (
            unpacked_data["pulse_length"][self.freq_ind_sorted] / 1e6
        )  # Convert microseconds to seconds
        range_samples_per_bin = unpacked_data["range_samples_per_bin"][
            self.freq_ind_sorted
        ]  # from data header

        # Calculate sample interval in seconds
        if len(dig_rate) == len(range_samples_per_bin):
            # TODO: below only correct if range_samples_per_bin=1,
            #  need to implement p.86 for the case when averaging is used
            sample_int = range_samples_per_bin / dig_rate
        else:
            # TODO: not sure what this error means
            raise ValueError("dig_rate and range_samples not unique across frequencies")

        ds = xr.Dataset(
            {
                "frequency_nominal": (
                    ["channel"],
                    self.freq_sorted,
                    {
                        "units": "Hz",
                        "long_name": "Transducer frequency",
                        "valid_min": 0.0,
                        "standard_name": "sound_frequency",
                    },
                ),
                "backscatter_r": (["channel", "ping_time", "range_sample"], N),
                "equivalent_beam_angle": (["channel"], parameters["BP"][self.freq_ind_sorted]),
                "gain_correction": (["channel"], unpacked_data["gain"][self.freq_ind_sorted]),
                "sample_interval": (["channel"], sample_int, {"units": "s"}),
                "transmit_duration_nominal": (
                    ["channel"],
                    tdn,
                    {
                        "long_name": "Nominal bandwidth of transmitted pulse",
                        "units": "s",
                        "valid_min": 0.0,
                    },
                ),
            },
            coords={
                "channel": (
                    ["channel"],
                    self.channel_ids_sorted,
                    self._varattrs["beam_coord_default"]["channel"],
                ),
                "ping_time": (
                    ["ping_time"],
                    ping_time,
                    self._varattrs["beam_coord_default"]["ping_time"],
                ),
                "range_sample": (
                    ["range_sample"],
                    range_sample,
                    self._varattrs["beam_coord_default"]["range_sample"],
                ),
            },
            attrs={
                "beam_mode": "",
                "conversion_equation_t": "type_4",
            },
        )

        # Manipulate some Dataset dimensions to adhere to convention
        self.beam_groups_to_convention(
            ds, self.beam_only_names, self.beam_ping_time_names, self.ping_time_only_names
        )

        return [set_time_encodings(ds)]

    def set_vendor(self) -> xr.Dataset:
        """Set the Vendor_specific group."""
        unpacked_data = self.parser_obj.unpacked_data
        parameters = self.parser_obj.parameters
        ping_time = self.parser_obj.ping_time
        tdn = parameters["pulse_length"][self.freq_ind_sorted] / 1e6
        anc = np.array(unpacked_data["ancillary"])  # convert to np array for easy slicing

        # Build variables in the output xarray Dataset
        Sv_offset = np.zeros_like(self.freq_sorted)
        for ind, ich in enumerate(self.freq_ind_sorted):
            # TODO: should not access the private function, better to compute Sv_offset in parser
            Sv_offset[ind] = self.parser_obj._calc_Sv_offset(
                self.freq_sorted[ind], unpacked_data["pulse_length"][ich]
            )

        ds = xr.Dataset(
            {
                "frequency_nominal": (
                    ["channel"],
                    self.freq_sorted,
                    {
                        "units": "Hz",
                        "long_name": "Transducer frequency",
                        "valid_min": 0.0,
                        "standard_name": "sound_frequency",
                    },
                ),
                "XML_transmit_duration_nominal": (["channel"], tdn),
                "XML_gain_correction": (["channel"], parameters["gain"][self.freq_ind_sorted]),
                "XML_digitization_rate": (
                    ["channel"],
                    parameters["dig_rate"][self.freq_ind_sorted],
                ),
                "XML_lockout_index": (
                    ["channel"],
                    parameters["lockout_index"][self.freq_ind_sorted],
                ),
                "digitization_rate": (["channel"], unpacked_data["dig_rate"][self.freq_ind_sorted]),
                "lockout_index": (
                    ["channel"],
                    unpacked_data["lockout_index"][self.freq_ind_sorted],
                ),
                "number_of_bins_per_channel": (
                    ["channel"],
                    unpacked_data["num_bins"][self.freq_ind_sorted],
                ),
                "number_of_samples_per_average_bin": (
                    ["channel"],
                    unpacked_data["range_samples_per_bin"][self.freq_ind_sorted],
                ),
                "board_number": (["channel"], unpacked_data["board_num"][self.freq_ind_sorted]),
                "data_type": (["channel"], unpacked_data["data_type"][self.freq_ind_sorted]),
                "ping_status": (["ping_time"], unpacked_data["ping_status"]),
                "number_of_acquired_pings": (
                    ["ping_time"],
                    unpacked_data["num_acq_pings"],
                ),
                "first_ping": (["ping_time"], unpacked_data["first_ping"]),
                "last_ping": (["ping_time"], unpacked_data["last_ping"]),
                "data_error": (["ping_time"], unpacked_data["data_error"]),
                "sensors_flag": (["ping_time"], unpacked_data["sensor_flag"]),
                "ancillary": (
                    ["ping_time", "ancillary_len"],
                    unpacked_data["ancillary"],
                ),
                "ad_channels": (["ping_time", "ad_len"], unpacked_data["ad"]),
                "battery_main": (["ping_time"], unpacked_data["battery_main"]),
                "battery_tx": (["ping_time"], unpacked_data["battery_tx"]),
                "profile_number": (["ping_time"], unpacked_data["profile_number"]),
                "temperature_counts": (["ping_time"], anc[:, 4]),
                "tilt_x_count": (["ping_time"], anc[:, 0]),
                "tilt_y_count": (["ping_time"], anc[:, 1]),
                "DS": (["channel"], parameters["DS"][self.freq_ind_sorted]),
                "EL": (["channel"], parameters["EL"][self.freq_ind_sorted]),
                "TVR": (["channel"], parameters["TVR"][self.freq_ind_sorted]),
                "VTX": (["channel"], parameters["VTX"][self.freq_ind_sorted]),
                "Sv_offset": (["channel"], Sv_offset),
                "number_of_samples_digitized_per_pings": (
                    ["channel"],
                    parameters["range_samples"][self.freq_ind_sorted],
                ),
                "number_of_digitized_samples_averaged_per_pings": (
                    ["channel"],
                    parameters["range_averaging_samples"][self.freq_ind_sorted],
                ),
            },
            coords={
                "channel": (
                    ["channel"],
                    self.channel_ids_sorted,
                    self._varattrs["beam_coord_default"]["channel"],
                ),
                "ping_time": (
                    ["ping_time"],
                    ping_time,
                    {
                        "axis": "T",
                        "long_name": "Timestamp of each ping",
                        "standard_name": "time",
                    },
                ),
                "ancillary_len": (
                    ["ancillary_len"],
                    list(range(len(unpacked_data["ancillary"][0]))),
                ),
                "ad_len": (["ad_len"], list(range(len(unpacked_data["ad"][0])))),
            },
            attrs={
                "XML_sensors_flag": parameters["sensors_flag"],
                "XML_burst_interval": parameters["burst_interval"],
                "XML_sonar_serial_number": parameters["serial_number"],
                "profile_flag": unpacked_data["profile_flag"],
                "burst_interval": unpacked_data["burst_int"],
                "ping_per_profile": unpacked_data["ping_per_profile"],
                "average_pings_flag": unpacked_data["avg_pings"],
                "spare_channel": unpacked_data["spare_chan"],
                "ping_period": unpacked_data["ping_period"],
                "phase": unpacked_data["phase"],
                "number_of_channels": unpacked_data["num_chan"],
                "number_of_frequency": parameters["num_freq"],
                "number_of_pings_per_burst": parameters["pings_per_burst"],
                "average_burst_pings_flag": parameters["average_burst_pings"],
                # Temperature coefficients
                "temperature_ka": parameters["ka"],
                "temperature_kb": parameters["kb"],
                "temperature_kc": parameters["kc"],
                "temperature_A": parameters["A"],
                "temperature_B": parameters["B"],
                "temperature_C": parameters["C"],
                # Tilt coefficients
                "tilt_X_a": parameters["X_a"],
                "tilt_X_b": parameters["X_b"],
                "tilt_X_c": parameters["X_c"],
                "tilt_X_d": parameters["X_d"],
                "tilt_Y_a": parameters["Y_a"],
                "tilt_Y_b": parameters["Y_b"],
                "tilt_Y_c": parameters["Y_c"],
                "tilt_Y_d": parameters["Y_d"],
            },
        )
        return set_time_encodings(ds)
