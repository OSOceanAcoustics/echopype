from collections import defaultdict
from typing import List

import numpy as np
import xarray as xr

from ..utils.coding import set_time_encodings
from ..utils.log import _init_logger
from .set_groups_base import SetGroupsBase

logger = _init_logger(__name__)


class SetGroupsEK80(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from EK80 data files."""

    # The sets beam_only_names, ping_time_only_names, and
    # beam_ping_time_names are used in set_groups_base and
    # in converting from v0.5.x to v0.6.0. The values within
    # these sets are applied to all Sonar/Beam_groupX groups.

    # Variables that need only the beam dimension added to them.
    beam_only_names = {
        "backscatter_r",
        "backscatter_i",
        "angle_athwartship",
        "angle_alongship",
        "frequency_start",
        "frequency_end",
    }

    # Variables that need only the ping_time dimension added to them.
    ping_time_only_names = {"beam_type"}

    # Variables that need beam and ping_time dimensions added to them.
    beam_ping_time_names = {
        "beam_direction_x",
        "beam_direction_y",
        "beam_direction_z",
        "angle_offset_alongship",
        "angle_offset_athwartship",
        "angle_sensitivity_alongship",
        "angle_sensitivity_athwartship",
        "equivalent_beam_angle",
        "beamwidth_twoway_alongship",
        "beamwidth_twoway_athwartship",
    }

    beamgroups_possible = [
        {
            "name": "Beam_group1",
            "descr": {
                "power": "contains backscatter power (uncalibrated) and "
                "other beam or channel-specific data,"
                " including split-beam angle data when they exist.",
                "complex": "contains complex backscatter data and other "
                "beam or channel-specific data.",
            },
        },
        {
            "name": "Beam_group2",
            "descr": (
                "contains backscatter power (uncalibrated) and other beam or channel-specific data,"  # noqa
                " including split-beam angle data when they exist."
            ),
        },
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # if we have zarr files, create parser_obj.ch_ids
        if self.parsed2zarr_obj.temp_zarr_dir:
            for k, v in self.parsed2zarr_obj.p2z_ch_ids.items():
                self.parser_obj.ch_ids[k] = self._get_channel_ids(v)

        # obtain sorted channel dict in ascending order for each usage scenario
        self.sorted_channel = {
            "all": self._sort_list(list(self.parser_obj.config_datagram["configuration"].keys())),
            "power": self._sort_list(self.parser_obj.ch_ids["power"]),
            "complex": self._sort_list(self.parser_obj.ch_ids["complex"]),
            "power_complex": self._sort_list(
                self.parser_obj.ch_ids["power"] + self.parser_obj.ch_ids["complex"]
            ),
            "angle": self._sort_list(self.parser_obj.ch_ids["angle"]),
        }

    @staticmethod
    def _sort_list(list_in: List[str]) -> List[str]:
        """
        Sorts a list in ascending order and then returns
        the sorted list.

        Parameters
        ----------
        list_in: List[str]
            List to be sorted

        Returns
        -------
        List[str]
            A copy of the input list in ascending order
        """

        # make copy so we don't directly modify input list
        list_in_copy = list_in.copy()

        # sort list in ascending order
        list_in_copy.sort(reverse=False)

        return list_in_copy

    def set_env(self) -> xr.Dataset:
        """Set the Environment group."""

        # set time1 if it exists
        if "timestamp" in self.parser_obj.environment:
            time1 = np.array([self.parser_obj.environment["timestamp"]])
        else:
            time1 = np.array([np.datetime64("NaT")])

        # Collect variables
        dict_env = dict()
        for k, v in self.parser_obj.environment.items():
            if k in ["temperature", "depth", "acidity", "salinity", "sound_speed"]:
                dict_env[k] = (["time1"], [v])

        # Rename to conform with those defined in convention
        if "sound_speed" in dict_env:
            dict_env["sound_speed_indicative"] = dict_env.pop("sound_speed")
        for k in [
            "sound_absorption",
            "absorption",
        ]:  # add possible variation until having example
            if k in dict_env:
                dict_env["absorption_indicative"] = dict_env.pop(k)

        if "sound_velocity_profile" in self.parser_obj.environment:
            dict_env["sound_velocity_profile"] = (
                ["time1", "sound_velocity_profile_depth"],
                [self.parser_obj.environment["sound_velocity_profile"][1::2]],
                {
                    "long_name": "sound velocity profile",
                    "standard_name": "speed_of_sound_in_sea_water",
                    "units": "m/s",
                    "valid_min": 0.0,
                    "comment": "parsed from raw data files as (depth, sound_speed) value pairs",
                },
            )

        varnames = ["sound_velocity_source", "transducer_name", "transducer_sound_speed"]
        for vn in varnames:
            if vn in self.parser_obj.environment:
                dict_env[vn] = (
                    ["time1"],
                    [self.parser_obj.environment[vn]],
                )

        ds = xr.Dataset(
            dict_env,
            coords={
                "time1": (
                    ["time1"],
                    time1,
                    {
                        "axis": "T",
                        "long_name": "Timestamps for NMEA position datagrams",
                        "standard_name": "time",
                        "comment": "Time coordinate corresponding to environmental "
                        "variables. Note that Platform.time3 is the same "
                        "as Environment.time1.",
                    },
                ),
                "sound_velocity_profile_depth": (
                    ["sound_velocity_profile_depth"],
                    self.parser_obj.environment["sound_velocity_profile"][::2]
                    if "sound_velocity_profile" in self.parser_obj.environment
                    else [],
                    {
                        "standard_name": "depth",
                        "units": "m",
                        "axis": "Z",
                        "positive": "down",
                        "valid_min": 0.0,
                    },
                ),
            },
        )
        return set_time_encodings(ds)

    def set_sonar(self, beam_group_type: list = ["power", None]) -> xr.Dataset:
        # Collect unique variables
        params = [
            "transducer_frequency",
            "serial_number",
            "transducer_name",
            "transducer_serial_number",
            "application_name",
            "application_version",
            "channel_id_short",
        ]
        var = defaultdict(list)

        # collect all variables in params
        for ch_id in self.sorted_channel["all"]:
            data = self.parser_obj.config_datagram["configuration"][ch_id]
            for param in params:
                var[param].append(data[param])

        # obtain the correct beam_group and corresponding description from beamgroups_possible
        for idx, beam in enumerate(beam_group_type):
            if beam is None:
                # obtain values from an element where the key 'descr' does not have keys
                self._beamgroups.append(self.beamgroups_possible[idx])
            else:
                # obtain values from an element where the key 'descr' DOES have keys
                self._beamgroups.append(
                    {
                        "name": self.beamgroups_possible[idx]["name"],
                        "descr": self.beamgroups_possible[idx]["descr"][beam],
                    }
                )

        # Add beam_group and beam_group_descr variables sharing a common dimension
        # (beam_group), using the information from self._beamgroups
        beam_groups_vars, beam_groups_coord = self._beam_groups_vars()

        # Create dataset
        sonar_vars = {
            "frequency_nominal": (
                ["channel"],
                var["transducer_frequency"],
                {
                    "units": "Hz",
                    "long_name": "Transducer frequency",
                    "valid_min": 0.0,
                    "standard_name": "sound_frequency",
                },
            ),
            "transceiver_serial_number": (
                ["channel"],
                var["serial_number"],
                {
                    "long_name": "Transceiver serial number",
                },
            ),
            "transducer_name": (
                ["channel"],
                var["transducer_name"],
                {
                    "long_name": "Transducer name",
                },
            ),
            "transducer_serial_number": (
                ["channel"],
                var["transducer_serial_number"],
                {
                    "long_name": "Transducer serial number",
                },
            ),
        }
        ds = xr.Dataset(
            {**sonar_vars, **beam_groups_vars},
            coords={
                "channel": (
                    ["channel"],
                    self.sorted_channel["all"],
                    self._varattrs["beam_coord_default"]["channel"],
                ),
                **beam_groups_coord,
            },
        )

        # Assemble sonar group global attribute dictionary
        sonar_attr_dict = {
            "sonar_manufacturer": "Simrad",
            "sonar_model": self.sonar_model,
            # transducer (sonar) serial number is not reliably stored in the EK80 raw
            # data file and would be channel-dependent. For consistency with EK60,
            # will not try to populate sonar_serial_number from the raw datagrams
            "sonar_serial_number": "",
            "sonar_software_name": var["application_name"][0],
            "sonar_software_version": var["application_version"][0],
            "sonar_type": "echosounder",
        }
        ds = ds.assign_attrs(sonar_attr_dict)

        return ds

    def set_platform(self) -> xr.Dataset:
        """Set the Platform group."""

        freq = np.array(
            [
                self.parser_obj.config_datagram["configuration"][ch]["transducer_frequency"]
                for ch in self.sorted_channel["power_complex"]
            ]
        )

        # Collect variables
        if self.ui_param["water_level"] is not None:
            water_level = self.ui_param["water_level"]
        elif "water_level_draft" in self.parser_obj.environment:
            water_level = self.parser_obj.environment["water_level_draft"]
        else:
            water_level = np.nan
            logger.info("WARNING: The water_level_draft was not in the file. Value set to NaN.")

        time1, msg_type, lat, lon = self._extract_NMEA_latlon()
        time2 = self.parser_obj.mru.get("timestamp", None)
        time2 = np.array(time2) if time2 is not None else [np.nan]

        # Assemble variables into a dataset: variables filled with nan if do not exist
        ds = xr.Dataset(
            {
                "frequency_nominal": (
                    ["channel"],
                    freq,
                    {
                        "units": "Hz",
                        "long_name": "Transducer frequency",
                        "valid_min": 0.0,
                        "standard_name": "sound_frequency",
                    },
                ),
                "pitch": (
                    ["time2"],
                    np.array(self.parser_obj.mru.get("pitch", [np.nan])),
                    self._varattrs["platform_var_default"]["pitch"],
                ),
                "roll": (
                    ["time2"],
                    np.array(self.parser_obj.mru.get("roll", [np.nan])),
                    self._varattrs["platform_var_default"]["roll"],
                ),
                "vertical_offset": (
                    ["time2"],
                    np.array(self.parser_obj.mru.get("heave", [np.nan])),
                    self._varattrs["platform_var_default"]["vertical_offset"],
                ),
                "latitude": (["time1"], lat, self._varattrs["platform_var_default"]["latitude"]),
                "longitude": (["time1"], lon, self._varattrs["platform_var_default"]["longitude"]),
                "sentence_type": (["time1"], msg_type),
                "drop_keel_offset": (
                    ["time3"],
                    [self.parser_obj.environment["drop_keel_offset"]]
                    if hasattr(self.parser_obj.environment, "drop_keel_offset")
                    else [np.nan],
                ),
                "drop_keel_offset_is_manual": (
                    ["time3"],
                    [self.parser_obj.environment["drop_keel_offset_is_manual"]]
                    if "drop_keel_offset_is_manual" in self.parser_obj.environment
                    else [np.nan],
                ),
                "transducer_offset_x": (
                    ["channel"],
                    [
                        self.parser_obj.config_datagram["configuration"][ch].get(
                            "transducer_offset_x", np.nan
                        )
                        for ch in self.sorted_channel["power_complex"]
                    ],
                    self._varattrs["platform_var_default"]["transducer_offset_x"],
                ),
                "transducer_offset_y": (
                    ["channel"],
                    [
                        self.parser_obj.config_datagram["configuration"][ch].get(
                            "transducer_offset_y", np.nan
                        )
                        for ch in self.sorted_channel["power_complex"]
                    ],
                    self._varattrs["platform_var_default"]["transducer_offset_y"],
                ),
                "transducer_offset_z": (
                    ["channel"],
                    [
                        self.parser_obj.config_datagram["configuration"][ch].get(
                            "transducer_offset_z", np.nan
                        )
                        for ch in self.sorted_channel["power_complex"]
                    ],
                    self._varattrs["platform_var_default"]["transducer_offset_z"],
                ),
                "water_level": (
                    ["time3"],
                    [water_level],
                    {
                        "long_name": "z-axis distance from the platform coordinate system "
                        "origin to the sonar transducer",
                        "units": "m",
                    },
                ),
                "water_level_draft_is_manual": (
                    ["time3"],
                    [self.parser_obj.environment["water_level_draft_is_manual"]]
                    if "water_level_draft_is_manual" in self.parser_obj.environment
                    else [np.nan],
                ),
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
            },
            coords={
                "channel": (
                    ["channel"],
                    self.sorted_channel["power_complex"],
                    self._varattrs["beam_coord_default"]["channel"],
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
                "time3": (
                    ["time3"],
                    [self.parser_obj.environment["timestamp"]]
                    if "timestamp" in self.parser_obj.environment
                    else np.datetime64("NaT"),
                    {
                        "axis": "T",
                        "long_name": "Timestamps for platform-related sampling environment",
                        "standard_name": "time",
                        "comment": "Time coordinate corresponding to platform-related "
                        "sampling environment. Note that Platform.time3 is "
                        "the same as Environment.time1.",
                    },
                ),
                "time1": (
                    ["time1"],
                    time1,
                    {
                        **self._varattrs["platform_coord_default"]["time1"],
                        "comment": "Time coordinate corresponding to NMEA position data.",
                    },
                ),
            },
            attrs={
                "platform_code_ICES": self.ui_param["platform_code_ICES"],
                "platform_name": self.ui_param["platform_name"],
                "platform_type": self.ui_param["platform_type"],
                # TODO: check what this 'drop_keel_offset' is
            },
        )
        return set_time_encodings(ds)

    def _assemble_ds_ping_invariant(self, params, data_type):
        """Assemble dataset for ping-invariant params in the /Sonar/Beam_group1 group.

        Parameters
        ----------
        data_type : str
            'complex' or 'power'
        params : dict
            beam parameters that do not change across ping
        """

        freq = np.array(
            [
                self.parser_obj.config_datagram["configuration"][ch]["transducer_frequency"]
                for ch in self.sorted_channel[data_type]
            ]
        )
        beam_params = defaultdict()
        for param in params:
            beam_params[param] = [
                self.parser_obj.config_datagram["configuration"][ch].get(param, np.nan)
                for ch in self.sorted_channel[data_type]
            ]
        ds = xr.Dataset(
            {
                "frequency_nominal": (
                    ["channel"],
                    freq,
                    {
                        "units": "Hz",
                        "long_name": "Transducer frequency",
                        "valid_min": 0.0,
                        "standard_name": "sound_frequency",
                    },
                ),
                "beam_type": (["channel"], beam_params["transducer_beam_type"]),
                "beamwidth_twoway_alongship": (
                    ["channel"],
                    beam_params["beam_width_alongship"],
                    {
                        "long_name": "Half power two-way beam width along alongship axis of beam",  # noqa
                        "units": "arc_degree",
                        "valid_range": (0.0, 360.0),
                        "comment": (
                            "Introduced in echopype for Simrad echosounders to avoid potential confusion with convention definitions. "  # noqa
                            "The alongship angle corresponds to the minor angle in SONAR-netCDF4 vers 2. "  # noqa
                            "The convention defines one-way transmit or receive beamwidth (beamwidth_receive_minor and beamwidth_transmit_minor), but Simrad echosounders record two-way beamwidth in the data."  # noqa
                        ),
                    },
                ),
                "beamwidth_twoway_athwartship": (
                    ["channel"],
                    beam_params["beam_width_athwartship"],
                    {
                        "long_name": "Half power two-way beam width along athwartship axis of beam",  # noqa
                        "units": "arc_degree",
                        "valid_range": (0.0, 360.0),
                        "comment": (
                            "Introduced in echopype for Simrad echosounders to avoid potential confusion with convention definitions. "  # noqa
                            "The athwartship angle corresponds to the major angle in SONAR-netCDF4 vers 2. "  # noqa
                            "The convention defines one-way transmit or receive beamwidth (beamwidth_receive_major and beamwidth_transmit_major), but Simrad echosounders record two-way beamwidth in the data."  # noqa
                        ),
                    },
                ),
                "beam_direction_x": (
                    ["channel"],
                    beam_params["transducer_alpha_x"],
                    {
                        "long_name": "x-component of the vector that gives the pointing "
                        "direction of the beam, in sonar beam coordinate "
                        "system",
                        "units": "1",
                        "valid_range": (-1.0, 1.0),
                    },
                ),
                "beam_direction_y": (
                    ["channel"],
                    beam_params["transducer_alpha_y"],
                    {
                        "long_name": "y-component of the vector that gives the pointing "
                        "direction of the beam, in sonar beam coordinate "
                        "system",
                        "units": "1",
                        "valid_range": (-1.0, 1.0),
                    },
                ),
                "beam_direction_z": (
                    ["channel"],
                    beam_params["transducer_alpha_z"],
                    {
                        "long_name": "z-component of the vector that gives the pointing "
                        "direction of the beam, in sonar beam coordinate "
                        "system",
                        "units": "1",
                        "valid_range": (-1.0, 1.0),
                    },
                ),
                "angle_offset_alongship": (
                    ["channel"],
                    beam_params["angle_offset_alongship"],
                    {
                        "long_name": "electrical alongship angle offset of the transducer",
                        "comment": (
                            "Introduced in echopype for Simrad echosounders. "  # noqa
                            "The alongship angle corresponds to the minor angle in SONAR-netCDF4 vers 2. "  # noqa
                        ),
                    },
                ),
                "angle_offset_athwartship": (
                    ["channel"],
                    beam_params["angle_offset_athwartship"],
                    {
                        "long_name": "electrical athwartship angle offset of the transducer",
                        "comment": (
                            "Introduced in echopype for Simrad echosounders. "  # noqa
                            "The athwartship angle corresponds to the major angle in SONAR-netCDF4 vers 2. "  # noqa
                        ),
                    },
                ),
                "angle_sensitivity_alongship": (
                    ["channel"],
                    beam_params["angle_sensitivity_alongship"],
                    {
                        "long_name": "alongship angle sensitivity of the transducer",
                        "comment": (
                            "Introduced in echopype for Simrad echosounders. "  # noqa
                            "The alongship angle corresponds to the minor angle in SONAR-netCDF4 vers 2. "  # noqa
                        ),
                    },
                ),
                "angle_sensitivity_athwartship": (
                    ["channel"],
                    beam_params["angle_sensitivity_athwartship"],
                    {
                        "long_name": "athwartship angle sensitivity of the transducer",
                        "comment": (
                            "Introduced in echopype for Simrad echosounders. "  # noqa
                            "The athwartship angle corresponds to the major angle in SONAR-netCDF4 vers 2. "  # noqa
                        ),
                    },
                ),
                "equivalent_beam_angle": (
                    ["channel"],
                    beam_params["equivalent_beam_angle"],
                    {
                        "long_name": "Equivalent beam angle",
                        "units": "sr",
                        "valid_range": (0.0, 4 * np.pi),
                    },
                ),
                "transceiver_software_version": (
                    ["channel"],
                    beam_params["transceiver_software_version"],
                ),
            },
            coords={
                "channel": (
                    ["channel"],
                    self.sorted_channel[data_type],
                    self._varattrs["beam_coord_default"]["channel"],
                ),
            },
            attrs={"beam_mode": "vertical", "conversion_equation_t": "type_3"},
        )

        return ds

    def _add_freq_start_end_ds(self, ds_tmp: xr.Dataset, ch: str) -> xr.Dataset:
        """
        Returns a Dataset with variables
        ``frequency_start`` and ``frequency_end``
        added to ``ds_tmp`` for a specific channel,
        if ``frequency_start`` is in ping_data_dict.

        Parameters
        ----------
        ds_tmp: xr.Dataset
            Dataset containing the complex data
        ch: str
            Channel id
        """

        # CW data encoded as complex samples do NOT have frequency_start and frequency_end
        # TODO: use PulseForm instead of checking for the existence
        #   of FrequencyStart and FrequencyEnd
        if (
            "frequency_start" in self.parser_obj.ping_data_dict.keys()
            and self.parser_obj.ping_data_dict["frequency_start"][ch]
        ):
            ds_f_start_end = xr.Dataset(
                {
                    "frequency_start": (
                        ["ping_time"],
                        np.array(
                            self.parser_obj.ping_data_dict["frequency_start"][ch],
                            dtype=int,
                        ),
                        {
                            "long_name": "Starting frequency of the transducer",
                            "units": "Hz",
                        },
                    ),
                    "frequency_end": (
                        ["ping_time"],
                        np.array(
                            self.parser_obj.ping_data_dict["frequency_end"][ch],
                            dtype=int,
                        ),
                        {
                            "long_name": "Ending frequency of the transducer",
                            "units": "Hz",
                        },
                    ),
                },
                coords={
                    "ping_time": (
                        ["ping_time"],
                        self.parser_obj.ping_time[ch],
                        {
                            "axis": "T",
                            "long_name": "Timestamp of each ping",
                            "standard_name": "time",
                        },
                    ),
                },
            )

            ds_tmp = xr.merge(
                [ds_tmp, ds_f_start_end], combine_attrs="override"
            )  # override keeps the Dataset attributes

        return ds_tmp

    def _assemble_ds_complex(self, ch):
        num_transducer_sectors = np.unique(
            np.array(self.parser_obj.ping_data_dict["n_complex"][ch])
        )
        if num_transducer_sectors.size > 1:  # this is not supposed to happen
            raise ValueError("Transducer sector number changes in the middle of the file!")
        else:
            num_transducer_sectors = num_transducer_sectors[0]

        data_shape = self.parser_obj.ping_data_dict["complex"][ch].shape
        data_shape = (
            data_shape[0],
            int(data_shape[1] / num_transducer_sectors),
            num_transducer_sectors,
        )
        data = self.parser_obj.ping_data_dict["complex"][ch].reshape(data_shape)

        ds_tmp = xr.Dataset(
            {
                "backscatter_r": (
                    ["ping_time", "range_sample", "beam"],
                    np.real(data),
                    {"long_name": "Real part of backscatter power", "units": "V"},
                ),
                "backscatter_i": (
                    ["ping_time", "range_sample", "beam"],
                    np.imag(data),
                    {"long_name": "Imaginary part of backscatter power", "units": "V"},
                ),
            },
            coords={
                "ping_time": (
                    ["ping_time"],
                    self.parser_obj.ping_time[ch],
                    self._varattrs["beam_coord_default"]["ping_time"],
                ),
                "range_sample": (
                    ["range_sample"],
                    np.arange(data_shape[1]),
                    self._varattrs["beam_coord_default"]["range_sample"],
                ),
                "beam": (
                    ["beam"],
                    np.arange(start=1, stop=num_transducer_sectors + 1).astype(str),
                    self._varattrs["beam_coord_default"]["beam"],
                ),
            },
        )

        ds_tmp = self._add_freq_start_end_ds(ds_tmp, ch)

        return set_time_encodings(ds_tmp)

    def _add_trasmit_pulse_complex(self, ds_tmp: xr.Dataset, ch: str) -> xr.Dataset:
        """
        Adds RAW4 datagram values (transmit pulse recorded in
        complex samples), if it exists, to the power and angle
        data.

        Parameters
        ----------
        ds_tmp : xr.Dataset
            Dataset to add the transmit data to
        ch : str
            Name of channel key to grab the data from

        Returns
        -------
        ds_tmp : xr.Dataset
            The input Dataset with transmit data added to it.
        """

        # If RAW4 datagram (transmit pulse recorded in complex samples) exists
        if (len(self.parser_obj.ping_data_dict_tx["complex"]) != 0) and (
            ch in self.parser_obj.ping_data_dict_tx["complex"].keys()
        ):
            # Add coordinate transmit_sample
            ds_tmp = ds_tmp.assign_coords(
                {
                    "transmit_sample": (
                        ["transmit_sample"],
                        np.arange(self.parser_obj.ping_data_dict_tx["complex"][ch].shape[1]),
                        {
                            "long_name": "Transmit pulse sample number, base 0",
                            "comment": "Only exist for Simrad EK80 file with RAW4 datagrams",
                        },
                    ),
                },
            )
            # Add data variables transmit_pulse_r/i
            ds_tmp = ds_tmp.assign(
                {
                    "transmit_pulse_r": (
                        ["ping_time", "transmit_sample"],
                        np.real(self.parser_obj.ping_data_dict_tx["complex"][ch]),
                        {
                            "long_name": "Real part of the transmit pulse",
                            "units": "V",
                            "comment": "Only exist for Simrad EK80 file with RAW4 datagrams",
                        },
                    ),
                    "transmit_pulse_i": (
                        ["ping_time", "transmit_sample"],
                        np.imag(self.parser_obj.ping_data_dict_tx["complex"][ch]),
                        {
                            "long_name": "Imaginary part of the transmit pulse",
                            "units": "V",
                            "comment": "Only exist for Simrad EK80 file with RAW4 datagrams",
                        },
                    ),
                },
            )

        return ds_tmp

    def _assemble_ds_power(self, ch):
        ds_tmp = xr.Dataset(
            {
                "backscatter_r": (
                    ["ping_time", "range_sample"],
                    self.parser_obj.ping_data_dict["power"][ch],
                    {"long_name": "Backscatter power", "units": "dB"},
                ),
            },
            coords={
                "ping_time": (
                    ["ping_time"],
                    self.parser_obj.ping_time[ch],
                    self._varattrs["beam_coord_default"]["ping_time"],
                ),
                "range_sample": (
                    ["range_sample"],
                    np.arange(self.parser_obj.ping_data_dict["power"][ch].shape[1]),
                    self._varattrs["beam_coord_default"]["range_sample"],
                ),
            },
        )

        ds_tmp = self._add_trasmit_pulse_complex(ds_tmp, ch)

        # If angle data exist
        if ch in self.sorted_channel["angle"]:
            ds_tmp = ds_tmp.assign(
                {
                    "angle_athwartship": (
                        ["ping_time", "range_sample"],
                        self.parser_obj.ping_data_dict["angle"][ch][:, :, 0],
                        {
                            "long_name": "electrical athwartship angle",
                            "comment": (
                                "Introduced in echopype for Simrad echosounders. "  # noqa
                                + "The athwartship angle corresponds to the major angle in SONAR-netCDF4 vers 2. "  # noqa
                            ),
                        },
                    ),
                    "angle_alongship": (
                        ["ping_time", "range_sample"],
                        self.parser_obj.ping_data_dict["angle"][ch][:, :, 1],
                        {
                            "long_name": "electrical alongship angle",
                            "comment": (
                                "Introduced in echopype for Simrad echosounders. "  # noqa
                                + "The alongship angle corresponds to the minor angle in SONAR-netCDF4 vers 2. "  # noqa
                            ),
                        },
                    ),
                }
            )

        return set_time_encodings(ds_tmp)

    def _assemble_ds_common(self, ch, range_sample_size):
        """Variables common to complex and power/angle data."""
        # pulse duration may have different names
        if "pulse_length" in self.parser_obj.ping_data_dict:
            pulse_length = np.array(
                self.parser_obj.ping_data_dict["pulse_length"][ch], dtype="float32"
            )
        else:
            pulse_length = np.array(
                self.parser_obj.ping_data_dict["pulse_duration"][ch], dtype="float32"
            )

        ds_common = xr.Dataset(
            {
                "sample_interval": (
                    ["ping_time"],
                    self.parser_obj.ping_data_dict["sample_interval"][ch],
                    {
                        "long_name": "Interval between recorded raw data samples",
                        "units": "s",
                        "valid_min": 0.0,
                    },
                ),
                "transmit_power": (
                    ["ping_time"],
                    self.parser_obj.ping_data_dict["transmit_power"][ch],
                    {
                        "long_name": "Nominal transmit power",
                        "units": "W",
                        "valid_min": 0.0,
                    },
                ),
                "transmit_duration_nominal": (
                    ["ping_time"],
                    pulse_length,
                    {
                        "long_name": "Nominal bandwidth of transmitted pulse",
                        "units": "s",
                        "valid_min": 0.0,
                    },
                ),
                "slope": (
                    ["ping_time"],
                    self.parser_obj.ping_data_dict["slope"][ch],
                ),
            },
            coords={
                "ping_time": (
                    ["ping_time"],
                    self.parser_obj.ping_time[ch],
                    self._varattrs["beam_coord_default"]["ping_time"],
                ),
                "range_sample": (
                    ["range_sample"],
                    np.arange(range_sample_size),
                    self._varattrs["beam_coord_default"]["range_sample"],
                ),
            },
        )
        return set_time_encodings(ds_common)

    @staticmethod
    def merge_save(ds_combine: List[xr.Dataset], ds_invariant: xr.Dataset) -> xr.Dataset:
        """Merge data from all complex or all power/angle channels"""
        ds_combine = xr.merge(ds_combine)

        ds_combine = xr.merge(
            [ds_invariant, ds_combine], combine_attrs="override"
        )  # override keeps the Dataset attributes
        return set_time_encodings(ds_combine)

    def _attach_vars_to_ds_data(self, ds_data: xr.Dataset, ch: str, rs_size: int) -> xr.Dataset:
        """
        Attaches common variables and the channel dimension.

        Parameters
        ----------
        ds_data : xr.Dataset
            Data set to add variables to
        ch: str
            Channel string associated with variables
        rs_size: int
            The size of the range sample dimension
            i.e. ``range_sample.size``

        Returns
        -------
        ``ds_data`` with the variables added to it.
        """

        ds_common = self._assemble_ds_common(ch, rs_size)

        ds_data = xr.merge([ds_data, ds_common], combine_attrs="override")

        # Attach channel dimension/coordinate
        ds_data = ds_data.expand_dims(
            {"channel": [self.parser_obj.config_datagram["configuration"][ch]["channel_id"]]}
        )
        ds_data["channel"] = ds_data["channel"].assign_attrs(
            **self._varattrs["beam_coord_default"]["channel"]
        )

        return ds_data

    def _get_ds_beam_power_zarr(self, ds_invariant_power: xr.Dataset) -> xr.Dataset:
        """
        Constructs the data set `ds_beam_power` when
        there are zarr variables present.

        Parameters
        ----------
        ds_invariant_power : xr.Dataset
            Dataset for ping-invariant params associated with power

        Returns
        -------
        A Dataset representing `ds_beam_power`.
        """

        # TODO: In the future it would be nice to have a dictionary of
        #  attributes stored in one place for all of the variables.
        #  This would reduce unnecessary code duplication in the
        #  functions below.

        # obtain DataArrays using zarr variables
        zarr_path = self.parsed2zarr_obj.zarr_file_name
        backscatter_r = self._get_power_dataarray(zarr_path)
        angle_athwartship, angle_alongship = self._get_angle_dataarrays(zarr_path)

        # create power related ds using DataArrays created from zarr file
        ds_power = xr.merge([backscatter_r, angle_athwartship, angle_alongship])
        ds_power = set_time_encodings(ds_power)

        # obtain additional variables that need to be added to ds_power
        ds_tmp = []
        for ch in self.sorted_channel["power"]:
            ds_data = self._add_trasmit_pulse_complex(ds_tmp=xr.Dataset(), ch=ch)
            ds_data = set_time_encodings(ds_data)

            ds_data = self._attach_vars_to_ds_data(ds_data, ch, rs_size=ds_power.range_sample.size)
            ds_tmp.append(ds_data)

        ds_tmp = self.merge_save(ds_tmp, ds_invariant_power)

        return xr.merge([ds_tmp, ds_power], combine_attrs="override")

    def _get_ds_complex_zarr(self, ds_invariant_complex: xr.Dataset) -> xr.Dataset:
        """
        Constructs the data set `ds_complex` when
        there are zarr variables present.

        Parameters
        ----------
        ds_invariant_complex : xr.Dataset
            Dataset for ping-invariant params associated with complex data

        Returns
        -------
        A Dataset representing `ds_complex`.
        """

        # TODO: In the future it would be nice to have a dictionary of
        #  attributes stored in one place for all of the variables.
        #  This would reduce unnecessary code duplication in the
        #  functions below.

        # obtain DataArrays using zarr variables
        zarr_path = self.parsed2zarr_obj.zarr_file_name
        backscatter_r, backscatter_i = self._get_complex_dataarrays(zarr_path)

        # create power related ds using DataArrays created from zarr file
        ds_complex = xr.merge([backscatter_r, backscatter_i])
        ds_complex = set_time_encodings(ds_complex)

        # obtain additional variables that need to be added to ds_complex
        ds_tmp = []
        for ch in self.sorted_channel["complex"]:
            ds_data = self._add_trasmit_pulse_complex(ds_tmp=xr.Dataset(), ch=ch)
            ds_data = self._add_freq_start_end_ds(ds_data, ch)

            ds_data = set_time_encodings(ds_data)

            ds_data = self._attach_vars_to_ds_data(
                ds_data, ch, rs_size=ds_complex.range_sample.size
            )
            ds_tmp.append(ds_data)

        ds_tmp = self.merge_save(ds_tmp, ds_invariant_complex)

        return xr.merge([ds_tmp, ds_complex], combine_attrs="override")

    def set_beam(self) -> List[xr.Dataset]:
        """Set the /Sonar/Beam_group1 group."""

        # Assemble ping-invariant beam data variables
        params = [
            "transducer_beam_type",
            "beam_width_alongship",
            "beam_width_athwartship",
            "transducer_alpha_x",
            "transducer_alpha_y",
            "transducer_alpha_z",
            "angle_offset_alongship",
            "angle_offset_athwartship",
            "angle_sensitivity_alongship",
            "angle_sensitivity_athwartship",
            "transducer_offset_x",
            "transducer_offset_y",
            "transducer_offset_z",
            "equivalent_beam_angle",
            "transceiver_software_version",
        ]

        # Assemble dataset for ping-invariant params
        if self.sorted_channel["complex"]:
            ds_invariant_complex = self._assemble_ds_ping_invariant(params, "complex")
        if self.sorted_channel["power"]:
            ds_invariant_power = self._assemble_ds_ping_invariant(params, "power")

        if not self.parsed2zarr_obj.temp_zarr_dir:
            # Assemble dataset for backscatter data and other ping-by-ping data
            ds_complex = []
            ds_power = []
            for ch in self.sorted_channel["all"]:
                if ch in self.sorted_channel["complex"]:
                    ds_data = self._assemble_ds_complex(ch)
                elif ch in self.sorted_channel["power"]:
                    ds_data = self._assemble_ds_power(ch)
                else:  # skip for channels containing no data
                    continue

                ds_data = self._attach_vars_to_ds_data(
                    ds_data, ch, rs_size=ds_data.range_sample.size
                )

                if ch in self.sorted_channel["complex"]:
                    ds_complex.append(ds_data)
                else:
                    ds_power.append(ds_data)

            # Merge and save group:
            #  if both complex and power data exist: complex data in /Sonar/Beam_group1 group
            #   and power data in /Sonar/Beam_group2
            #  if only one type of data exist: data in /Sonar/Beam_group1 group
            ds_beam_power = None
            if len(ds_complex) > 0:
                ds_beam = self.merge_save(ds_complex, ds_invariant_complex)
                if len(ds_power) > 0:
                    ds_beam_power = self.merge_save(ds_power, ds_invariant_power)
            else:
                ds_beam = self.merge_save(ds_power, ds_invariant_power)
        else:
            if self.sorted_channel["power"]:
                ds_power = self._get_ds_beam_power_zarr(ds_invariant_power)
            else:
                ds_power = None

            if self.sorted_channel["complex"]:
                ds_complex = self._get_ds_complex_zarr(ds_invariant_complex)
            else:
                ds_complex = None

            # correctly assign the beam groups
            ds_beam_power = None
            if ds_complex:
                ds_beam = ds_complex
                if ds_power:
                    ds_beam_power = ds_power
            else:
                ds_beam = ds_power

        # Manipulate some Dataset dimensions to adhere to convention
        if isinstance(ds_beam_power, xr.Dataset):
            self.beam_groups_to_convention(
                ds_beam_power,
                self.beam_only_names,
                self.beam_ping_time_names,
                self.ping_time_only_names,
            )

        self.beam_groups_to_convention(
            ds_beam, self.beam_only_names, self.beam_ping_time_names, self.ping_time_only_names
        )

        return [ds_beam, ds_beam_power]

    def set_vendor(self) -> xr.Dataset:
        """Set the Vendor_specific group."""
        config = self.parser_obj.config_datagram["configuration"]

        # Channel-specific parameters
        # exist for all channels:
        #   - sa_correction
        #   - gain (indexed by pulse_length)
        # may not exist for data from earlier EK80 software:
        #   - impedance
        #   - receiver sampling frequency
        #   - transceiver type
        table_params = [
            "transducer_frequency",
            "impedance",  # receive impedance (z_er), different from transmit impedance (z_et)
            "rx_sample_frequency",  # receiver sampling frequency
            "transceiver_type",
            "pulse_duration",
            "sa_correction",
            "gain",
        ]

        # grab all variables in table_params
        param_dict = defaultdict(list)
        for ch in self.sorted_channel["all"]:
            v = self.parser_obj.config_datagram["configuration"][ch]
            for p in table_params:
                if p in v:  # only for parameter that exist in configuration dict
                    param_dict[p].append(v[p])

        # make values into numpy arrays
        for p in param_dict.keys():
            param_dict[p] = np.array(param_dict[p])

        # Param size check
        if (
            not param_dict["pulse_duration"].shape
            == param_dict["sa_correction"].shape
            == param_dict["gain"].shape
        ):
            raise ValueError("Narrowband calibration parameters dimension mismatch!")

        ds_table = xr.Dataset(
            {
                "frequency_nominal": (
                    ["channel"],
                    param_dict["transducer_frequency"],
                    {
                        "units": "Hz",
                        "long_name": "Transducer frequency",
                        "valid_min": 0.0,
                        "standard_name": "sound_frequency",
                    },
                ),
                "sa_correction": (
                    ["channel", "pulse_length_bin"],
                    np.array(param_dict["sa_correction"]),
                ),
                "gain_correction": (
                    ["channel", "pulse_length_bin"],
                    np.array(param_dict["gain"]),
                ),
                "pulse_length": (
                    ["channel", "pulse_length_bin"],
                    np.array(param_dict["pulse_duration"]),
                ),
            },
            coords={
                "channel": (
                    ["channel"],
                    self.sorted_channel["all"],
                    self._varattrs["beam_coord_default"]["channel"],
                ),
                "pulse_length_bin": (
                    ["pulse_length_bin"],
                    np.arange(param_dict["pulse_duration"].shape[1]),
                ),
            },
        )

        # Parameters that may or may not exist (due to EK80 software version)
        if "impedance" in param_dict:
            ds_table["impedance_receive"] = xr.DataArray(
                param_dict["impedance"],
                dims=["channel"],
                coords={"channel": ds_table["channel"]},
                attrs={
                    "units": "ohm",
                    "long_name": "Receiver impedance",
                },
            )
        if "rx_sample_frequency" in param_dict:
            ds_table["fs_receiver"] = xr.DataArray(
                param_dict["rx_sample_frequency"].astype(float),
                dims=["channel"],
                coords={"channel": ds_table["channel"]},
                attrs={
                    "units": "Hz",
                    "long_name": "Receiver sampling frequency",
                },
            )
        if "transceiver_type" in param_dict:
            ds_table["transceiver_type"] = xr.DataArray(
                param_dict["transceiver_type"],
                dims=["channel"],
                coords={"channel": ds_table["channel"]},
                attrs={
                    "long_name": "Transceiver type",
                },
            )

        # Broadband calibration parameters: use the zero padding approach
        cal_ch_ids = [
            ch for ch in self.sorted_channel["all"] if "calibration" in config[ch]
        ]  # channels with cal params
        ds_cal = []
        for ch_id in cal_ch_ids:
            # TODO: consider using the full_ch_name below in place of channel id (ch_id)
            # full_ch_name = (f"{config[ch]['transceiver_type']} " +
            #                 f"{config[ch]['serial_number']}-" +
            #                 f"{config[ch]['hw_channel_configuration']} " +
            #                 f"{config[ch]['channel_id_short']}")
            cal_params = [
                "gain",
                "impedance",  # transmit impedance (z_et), different from receive impedance (z_er)
                "phase",
                "beamwidth_alongship",
                "beamwidth_athwartship",
                "angle_offset_alongship",
                "angle_offset_athwartship",
            ]
            param_dict = {}
            for p in cal_params:
                if p in config[ch_id]["calibration"]:  # only for parameters that exist in dict
                    param_dict[p] = (["cal_frequency"], config[ch_id]["calibration"][p])
            ds_ch = xr.Dataset(
                data_vars=param_dict,
                coords={
                    "cal_frequency": (
                        ["cal_frequency"],
                        config[ch_id]["calibration"]["frequency"],
                        {
                            "long_name": "Frequency of calibration parameter",
                            "units": "Hz",
                        },
                    )
                },
            )
            ds_ch = ds_ch.expand_dims({"cal_channel_id": [ch_id]})
            ds_ch["cal_channel_id"].attrs[
                "long_name"
            ] = "ID of channels containing broadband calibration information"
            ds_cal.append(ds_ch)
        ds_cal = xr.merge(ds_cal)

        if "impedance" in ds_cal:
            ds_cal = ds_cal.rename_vars({"impedance": "impedance_transmit"})

        #  Save decimation factors and filter coefficients
        coeffs = dict()
        decimation_factors = dict()
        for ch in self.sorted_channel["all"]:
            # filter coeffs and decimation factor for wide band transceiver (WBT)
            if self.parser_obj.fil_coeffs and (ch in self.parser_obj.fil_coeffs.keys()):
                coeffs[f"{ch} WBT filter"] = self.parser_obj.fil_coeffs[ch][1]
                decimation_factors[f"{ch} WBT decimation"] = self.parser_obj.fil_df[ch][1]
            # filter coeffs and decimation factor for pulse compression (PC)
            if self.parser_obj.fil_df and (ch in self.parser_obj.fil_coeffs.keys()):
                coeffs[f"{ch} PC filter"] = self.parser_obj.fil_coeffs[ch][2]
                decimation_factors[f"{ch} PC decimation"] = self.parser_obj.fil_df[ch][2]

        # Assemble everything into a Dataset
        ds = xr.merge([ds_table, ds_cal])

        # Save filter coefficients as real and imaginary parts as attributes
        for k, v in coeffs.items():
            ds.attrs[k + "_r"] = np.real(v)
            ds.attrs[k + "_i"] = np.imag(v)

        # Save decimation factors as attributes
        for k, v in decimation_factors.items():
            ds.attrs[k] = v

        # Save the entire config XML in vendor group in case of info loss
        ds.attrs["config_xml"] = self.parser_obj.config_datagram["xml"]

        return ds
