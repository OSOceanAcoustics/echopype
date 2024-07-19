"""
Class to save unpacked echosounder data to appropriate groups in netcdf or zarr.
"""

from typing import List

import numpy as np
import xarray as xr

from ..utils.coding import set_time_encodings

# from .set_groups_base import SetGroupsBase
from .set_groups_azfp import SetGroupsAZFP


class SetGroupsAZFP6(SetGroupsAZFP):
    """Class for saving groups to netcdf or zarr from AZFP6 data files."""

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
            "sonar_software_name": "AZFP6",
            "sonar_software_version": "alpha version based on 2.0.07 version",
            "sonar_type": "echosounder",
        }
        ds = ds.assign_attrs(sonar_attr_dict)

        return ds

    def set_platform(self) -> xr.Dataset:
        """Set the Platform group."""
        platform_dict = {"platform_name": "", "platform_type": "", "platform_code_ICES": ""}
        unpacked_data = self.parser_obj.unpacked_data

        # If tilt_x and/or tilt_y are all nan, create single-value time2 dimension
        # and single-value (np.nan) tilt_x and tilt_y
        tilt_x = [np.nan] if np.isnan(unpacked_data["tilt_x"]).all() else unpacked_data["tilt_x"]
        tilt_y = [np.nan] if np.isnan(unpacked_data["tilt_y"]).all() else unpacked_data["tilt_y"]
        if (len(tilt_x) == 1 and np.isnan(tilt_x)) and (len(tilt_y) == 1 and np.isnan(tilt_y)):
            time2 = [self.parser_obj.ping_time[0]]
        else:
            time2 = self.parser_obj.ping_time

        gps_latlon = np.array(unpacked_data["gps_lat_lon"])
        lat = (
            [np.nan]
            if np.isnan(gps_latlon[:, 0]).all() or not np.any(gps_latlon[:, 0])
            else gps_latlon[:, 0]
        )
        lon = (
            [np.nan]
            if np.isnan(gps_latlon[:, 1]).all() or not np.any(gps_latlon[:, 1])
            else gps_latlon[:, 1]
        )
        # Create nan time coordinate for lat/lon (lat/lon do not exist in AZFP 01A data)
        time1 = self.parser_obj._get_gps_time()
        # If there is an issue with the GPS timestamps, use ping time?
        # time1 = time2 if not np.any(time1) else time1
        time1 = [np.nan] if len(lat) != len(time1) else time1

        # Handle potential nan timestamp for time1 and time2
        time1 = self._nan_timestamp_handler(time1)
        time2 = self._nan_timestamp_handler(time2)  # should not be nan; but add for completeness

        ds = xr.Dataset(
            {
                "latitude": (
                    ["time1"],
                    lat,
                    self._varattrs["platform_var_default"]["latitude"],
                ),
                "longitude": (
                    ["time1"],
                    lon,
                    self._varattrs["platform_var_default"]["longitude"],
                ),
                "pitch": (
                    ["time2"],
                    [np.nan] * len(time2),
                    self._varattrs["platform_var_default"]["pitch"],
                ),
                "roll": (
                    ["time2"],
                    [np.nan] * len(time2),
                    self._varattrs["platform_var_default"]["roll"],
                ),
                "vertical_offset": (
                    ["time2"],
                    [np.nan] * len(time2),
                    self._varattrs["platform_var_default"]["vertical_offset"],
                ),
                "water_level": (
                    [],
                    np.nan,
                    self._varattrs["platform_var_default"]["water_level"],
                ),
                "tilt_x": (
                    ["time2"],
                    tilt_x,
                    {
                        "long_name": "Tilt X",
                        "units": "arc_degree",
                    },
                ),
                "tilt_y": (
                    ["time2"],
                    tilt_y,
                    {
                        "long_name": "Tilt Y",
                        "units": "arc_degree",
                    },
                ),
                **{
                    var: (
                        ["channel"],
                        [np.nan] * len(self.channel_ids_sorted),
                        self._varattrs["platform_var_default"][var],
                    )
                    for var in [
                        "transducer_offset_x",
                        "transducer_offset_y",
                        "transducer_offset_z",
                    ]
                },
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
                    ]
                },
                "frequency_nominal": (
                    ["channel"],
                    self.parser_obj.freq_sorted,
                    {
                        "units": "Hz",
                        "long_name": "Transducer frequency",
                        "valid_min": 0.0,
                        "standard_name": "sound_frequency",
                    },
                ),
            },
            coords={
                "channel": (
                    ["channel"],
                    self.channel_ids_sorted,
                    self._varattrs["beam_coord_default"]["channel"],
                ),
                "time1": (
                    ["time1"],
                    # xarray and probably CF don't accept time coordinate variable with Nan values
                    time1,
                    {
                        **self._varattrs["platform_coord_default"]["time1"],
                        "comment": "Time coordinate corresponding to GPS position data.",
                    },
                ),
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
        dig_rate = unpacked_data["dig_rate"][self.parser_obj.freq_ind_sorted]  # dim: freq
        ping_time = self.parser_obj.ping_time

        # Build variables in the output xarray Dataset
        N = []  # for storing backscatter_r values for each frequency
        for ich in self.parser_obj.freq_ind_sorted:
            N.append(
                np.array(
                    [
                        unpacked_data["counts"][p][ich] for p in range(len(unpacked_data["date"]))
                    ]  # year
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
            unpacked_data["pulse_len"][self.parser_obj.freq_ind_sorted] / 1e6
        )  # Convert microseconds to seconds
        range_samples_per_bin = unpacked_data["range_samples_per_bin"][
            self.parser_obj.freq_ind_sorted
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
                    self.parser_obj.freq_sorted,
                    {
                        "units": "Hz",
                        "long_name": "Transducer frequency",
                        "valid_min": 0.0,
                        "standard_name": "sound_frequency",
                    },
                ),
                "beam_type": (
                    ["channel"],
                    [0] * len(self.channel_ids_sorted),
                    {
                        "long_name": "Beam type",
                        "flag_values": [0, 1],
                        "flag_meanings": [
                            "Single beam",
                            "Split aperture beam",
                        ],
                    },
                ),
                **{
                    f"beam_direction_{var}": (
                        ["channel"],
                        [np.nan] * len(self.channel_ids_sorted),
                        {
                            "long_name": f"{var}-component of the vector that gives the pointing "
                            "direction of the beam, in sonar beam coordinate "
                            "system",
                            "units": "1",
                            "valid_range": (-1.0, 1.0),
                        },
                    )
                    for var in ["x", "y", "z"]
                },
                "backscatter_r": (
                    ["channel", "ping_time", "range_sample"],
                    np.array(N, dtype=np.float32),
                    {
                        "long_name": self._varattrs["beam_var_default"]["backscatter_r"][
                            "long_name"
                        ],
                        "units": "count",
                    },
                ),
                "equivalent_beam_angle": (
                    ["channel"],
                    parameters["BP"][self.parser_obj.freq_ind_sorted],
                    {
                        "long_name": "Equivalent beam angle",
                        "units": "sr",
                        "valid_range": (0.0, 4 * np.pi),
                    },
                ),
                # "gain_correction": (
                #    ["channel"],
                #    np.array(
                #        unpacked_data["gain"][self.parser_obj.freq_ind_sorted], dtype=np.float64
                #    ),
                #    {"long_name": "Gain correction", "units": "dB"},
                # ),
                "sample_interval": (
                    ["channel"],
                    sample_int,
                    {
                        "long_name": "Interval between recorded raw data samples",
                        "units": "s",
                        "valid_min": 0.0,
                    },
                ),
                "transmit_duration_nominal": (
                    ["channel"],
                    tdn,
                    {
                        "long_name": "Nominal bandwidth of transmitted pulse",
                        "units": "s",
                        "valid_min": 0.0,
                    },
                ),
                "transmit_frequency_start": (
                    ["channel"],
                    self.parser_obj.freq_sorted,
                    self._varattrs["beam_var_default"]["transmit_frequency_start"],
                ),
                "transmit_frequency_stop": (
                    ["channel"],
                    self.parser_obj.freq_sorted,
                    self._varattrs["beam_var_default"]["transmit_frequency_stop"],
                ),
                "transmit_type": (
                    [],
                    "CW",
                    {
                        "long_name": "Type of transmitted pulse",
                        "flag_values": ["CW"],
                        "flag_meanings": [
                            "Continuous Wave – a pulse nominally of one frequency",
                        ],
                    },
                ),
                "beam_stabilisation": (
                    [],
                    np.array(0, np.byte),
                    {
                        "long_name": "Beam stabilisation applied (or not)",
                        "flag_values": [0, 1],
                        "flag_meanings": ["not stabilised", "stabilised"],
                    },
                ),
                "non_quantitative_processing": (
                    [],
                    np.array(0, np.int16),
                    {
                        "long_name": "Presence or not of non-quantitative processing applied"
                        " to the backscattering data (sonar specific)",
                        "flag_values": [0],
                        "flag_meanings": ["None"],
                    },
                ),
                "sample_time_offset": (
                    [],
                    0.0,
                    {
                        "long_name": "Time offset that is subtracted from the timestamp"
                        " of each sample",
                        "units": "s",
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
        Sv_offset = self.parser_obj.Sv_offset
        phase_params = [
            "burst_interval",
            "pings_per_burst",
            "average_burst_pings",
            "base_time",
            "ping_period_counts",
        ]
        phase_freq_params = [
            "dig_rate",
            "range_samples",
            "range_averaging_samples",
            "lock_out_index",
            "gain",
            "storage_format",
        ]

        tdn = []
        for num in parameters["phase_number"]:
            try:
                tdn.append(
                    parameters[f"pulse_len_phase{num}"][self.parser_obj.freq_ind_sorted] / 1e6
                )
            except:
                tdn.append([np.nan] * len(self.parser_obj.freq_ind_sorted))
        tdn = np.array(tdn)
        for param in phase_freq_params:
            for num in parameters["phase_number"]:
                try:
                    parameters[param].append(
                        parameters[f"{param}_phase{num}"][self.parser_obj.freq_ind_sorted]
                    )
                except:
                    parameters[param].append([np.nan] * len(self.parser_obj.freq_ind_sorted))

        for param in phase_params:
            for num in parameters["phase_number"]:
                p = parameters[f"{param}_phase{num}"]
                parameters[param].append(np.nan if isinstance(p, list) else p)
        anc = np.array(unpacked_data["ancillary"])  # convert to np array for easy slicing

        ds = xr.Dataset(
            {
                "frequency_nominal": (
                    ["channel"],
                    self.parser_obj.freq_sorted,
                    {
                        "units": "Hz",
                        "long_name": "Transducer frequency",
                        "valid_min": 0.0,
                        "standard_name": "sound_frequency",
                    },
                ),
                # unpacked ping by ping data from 01A file
                "digitization_rate": (
                    ["channel"],
                    unpacked_data["dig_rate"][self.parser_obj.freq_ind_sorted],
                    {
                        "long_name": "Number of samples per second in kHz that is processed by the "
                        "A/D converter when digitizing the returned acoustic signal"
                    },
                ),
                "lock_out_index": (
                    ["channel"],
                    unpacked_data["lock_out_index"][self.parser_obj.freq_ind_sorted],
                    {
                        "long_name": "The distance, rounded to the nearest Bin Size after the "
                        "pulse is transmitted that over which AZFP will ignore echoes"
                    },
                ),
                "number_of_bins_per_channel": (
                    ["channel"],
                    unpacked_data["num_bins"][self.parser_obj.freq_ind_sorted],
                    {"long_name": "Number of bins per channel"},
                ),
                "number_of_samples_per_average_bin": (
                    ["channel"],
                    unpacked_data["range_samples_per_bin"][self.parser_obj.freq_ind_sorted],
                    {"long_name": "Range samples per bin for each channel"},
                ),
                "board_number": (
                    ["channel"],
                    unpacked_data["board_num"][self.parser_obj.freq_ind_sorted],
                    {"long_name": "The board the data came from channel 1-4"},
                ),
                "data_type": (
                    ["channel"],
                    unpacked_data["data_type"][self.parser_obj.freq_ind_sorted],
                    {
                        "long_name": "Datatype for each channel 1=Avg unpacked_data (5bytes), "
                        "0=raw (2bytes)"
                    },
                ),
                "ping_status": (["ping_time"], unpacked_data["acq_status"]),
                "number_of_acquired_pings": (
                    ["ping_time"],
                    unpacked_data["num_acq_pings"],
                    {"long_name": "Pings acquired in the burst"},
                ),
                "first_ping": (["ping_time"], unpacked_data["first_ping"]),
                "last_ping": (["ping_time"], unpacked_data["last_ping"]),
                "data_error": (
                    ["ping_time"],
                    unpacked_data["data_error"],
                    {"long_name": "Error number if an error occurred"},
                ),
                "sensors_flag": (["ping_time"], unpacked_data["sensor_status"]),
                "ancillary": (
                    ["ping_time", "ancillary_len"],
                    unpacked_data["ancillary"],
                    {"long_name": "Tilt-X, Y, Battery, Pressure, Temperature"},
                ),
                "ad_channels": (
                    ["ping_time", "ad_len"],
                    anc[:, -2:],  # compatibility with <uls5
                    {"long_name": "AD channel 6 and 7"},
                ),
                "battery_main": (["ping_time"], unpacked_data["battery_main"]),
                "battery_tx": (["ping_time"], unpacked_data["battery_tx"]),
                "profile_number": (["ping_time"], unpacked_data["profile_number"]),
                # unpacked ping by ping ancillary data from 01A file
                "temperature_counts": (
                    ["ping_time"],
                    anc[:, 4],
                    {"long_name": "Raw counts for temperature"},
                ),
                "tilt_x_count": (["ping_time"], anc[:, 0], {"long_name": "Raw counts for Tilt-X"}),
                "tilt_y_count": (["ping_time"], anc[:, 1], {"long_name": "Raw counts for Tilt-Y"}),
                # unpacked data with dim len=0 from 01A file
                "profile_flag": unpacked_data["profile_flag"],
                "burst_interval": (
                    [],
                    unpacked_data["burst_int"],
                    {
                        "long_name": "Time in seconds between bursts or between pings if the burst"
                        " interval has been set equal to the ping period"
                    },
                ),
                "ping_per_profile": (
                    [],
                    unpacked_data["ping_per_profile"],
                    {
                        "long_name": "Number of pings in a profile if ping averaging has been "
                        "selected"
                    },  # noqa
                ),
                "average_pings_flag": (
                    [],
                    unpacked_data["avg_pings"],
                    {"long_name": "Flag indicating whether the pings average in time"},
                ),
                "spare_channel": ([], unpacked_data["custom"], {"long_name": "Spare channel"}),
                "ping_period": (
                    [],
                    unpacked_data["ping_period"],
                    {"long_name": "Time between pings in a profile set"},
                ),
                "phase": (
                    [],
                    unpacked_data["phase"],
                    {"long_name": "Phase number used to acquire the profile"},
                ),
                "number_of_channels": (
                    [],
                    unpacked_data["num_chan"],
                    {"long_name": "Number of channels (1, 2, 3, or 4)"},
                ),
                "base_time": (
                    [],
                    unpacked_data["base_time"],
                    {"long_name": "Base time driving ping interval, min 0.2 sec and max 1.5 sec."},
                ),
                "ping_period_counts": (
                    [],
                    unpacked_data["ping_period_counts"],
                    {"long_name": "Ping period counts. The number of BaseTimer counts."},
                ),
                # parameters with channel dimension from XML file
                "XML_transmit_duration_nominal": (
                    ["phase_number", "channel"],
                    tdn,
                    {"long_name": "(From XML file) Nominal bandwidth of transmitted pulse"},
                ),  # tdn comes from parameters
                "XML_gain_correction": (
                    ["phase_number", "channel"],
                    parameters["gain"],
                    {"long_name": "(From XML file) Gain correction"},
                ),
                "instrument_type": parameters["instrument_type"][0],
                "minor": parameters["minor"],
                "major": parameters["major"],
                "date": parameters["date"],
                "program": parameters["program"],
                "cpu": parameters["cpu"],
                "serial_number": parameters["serial_number"],
                "board_version": parameters["board_version"],
                "file_version": parameters["file_version"],
                "parameter_version": parameters["parameter_version"],
                "configuration_version": parameters["configuration_version"],
                "XML_digitization_rate": (
                    ["phase_number", "channel"],
                    parameters["dig_rate"],
                    {
                        "long_name": "(From XML file) Number of samples per second in kHz that is "
                        "processed by the A/D converter when digitizing the returned acoustic "
                        "signal"
                    },
                ),
                "XML_lockout_index": (
                    ["phase_number", "channel"],
                    parameters["lock_out_index"],
                    {
                        "long_name": "(From XML file) The distance, rounded to the nearest "
                        "Bin Size after the pulse is transmitted that over which AZFP will "
                        "ignore echoes"
                    },
                ),
                "DS": (["channel"], parameters["DS"][self.parser_obj.freq_ind_sorted]),
                "EL": (
                    ["channel"],
                    parameters["EL"][self.parser_obj.freq_ind_sorted],
                    {"long_name": "Sound pressure at the transducer", "units": "dB"},
                ),
                "TVR": (
                    ["channel"],
                    parameters["TVR"][self.parser_obj.freq_ind_sorted],
                    {
                        "long_name": "Transmit voltage response of the transducer",
                        "units": "dB re 1uPa/V at 1m",
                    },
                ),
                "VTX0": (
                    ["channel"],
                    parameters["VTX0"][self.parser_obj.freq_ind_sorted],
                    {"long_name": "Amplified voltage 0 sent to the transducer"},
                ),
                "VTX1": (
                    ["channel"],
                    parameters["VTX1"][self.parser_obj.freq_ind_sorted],
                    {"long_name": "Amplified voltage 1 sent to the transducer"},
                ),
                "VTX2": (
                    ["channel"],
                    parameters["VTX2"][self.parser_obj.freq_ind_sorted],
                    {"long_name": "Amplified voltage 2 sent to the transducer"},
                ),
                "VTX3": (
                    ["channel"],
                    parameters["VTX3"][self.parser_obj.freq_ind_sorted],
                    {"long_name": "Amplified voltage 3 sent to the transducer"},
                ),
                "Sv_offset": (["channel"], Sv_offset),
                "number_of_samples_digitized_per_pings": (
                    ["phase_number", "channel"],
                    parameters["range_samples"],
                ),
                "number_of_digitized_samples_averaged_per_pings": (
                    ["phase_number", "channel"],
                    parameters["range_averaging_samples"],
                ),
                # parameters with dim len=0 from XML file
                "XML_sensors_flag": parameters["sensors_flag"],
                "XML_burst_interval": (
                    ["phase_number"],
                    parameters["burst_interval"],
                    {
                        "long_name": "Time in seconds between bursts or between pings if the burst "
                        "interval has been set equal to the ping period"
                    },
                ),
                "XML_sonar_serial_number": parameters["serial_number"],
                "number_of_frequency": parameters["num_freq"],
                "number_of_pings_per_burst": (
                    ["phase_number"],
                    parameters["pings_per_burst"],
                ),
                "average_burst_pings_flag": (
                    ["phase_number"],
                    parameters["average_burst_pings"],
                ),
                # temperature coefficients from XML file
                **{
                    f"temperature_k{var}": (
                        [],
                        parameters[f"k{var}"],
                        {"long_name": f"Thermistor bridge coefficient {var}"},
                    )
                    for var in ["a", "b", "c"]
                },
                **{
                    f"temperature_{var}": (
                        [],
                        parameters[var],
                        {"long_name": f"Thermistor calibration coefficient {var}"},
                    )
                    for var in ["A", "B", "C"]
                },
                # tilt coefficients from XML file
                **{
                    f"tilt_X_{var}": (
                        [],
                        parameters[f"X_{var}"],
                        {"long_name": f"Calibration coefficient {var} for Tilt-X"},
                    )
                    for var in ["a", "b", "c", "d"]
                },
                **{
                    f"tilt_Y_{var}": (
                        [],
                        parameters[f"Y_{var}"],
                        {"long_name": f"Calibration coefficient {var} for Tilt-Y"},
                    )
                    for var in ["a", "b", "c", "d"]
                },
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
                    list(range(anc.shape[-1])),
                ),
                "ad_len": (["ad_len"], list(range(anc[:, -2:].shape[-1]))),
                "phase_number": (
                    ["phase_number"],
                    sorted([int(num) for num in parameters["phase_number"]]),
                ),
            },
        )
        return set_time_encodings(ds)
