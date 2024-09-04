from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..utils.coding import set_time_encodings
from ..utils.log import _init_logger
from .set_groups_base import SetGroupsBase

logger = _init_logger(__name__)

WIDE_BAND_TRANS = "WBT"
PULSE_COMPRESS = "PC"
FILTER_IMAG = "filter_i"
FILTER_REAL = "filter_r"
DECIMATION = "decimation"


class SetGroupsEK80(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from EK80 data files."""

    # The sets beam_only_names, ping_time_only_names, and
    # beam_ping_time_names are used in set_groups_base and
    # in converting from v0.5.x to v0.6.0. The values within
    # these sets are applied to all Sonar/Beam_groupX groups.

    # 2023-07-24:
    #   PRs:
    #     - https://github.com/OSOceanAcoustics/echopype/pull/1056
    #     - https://github.com/OSOceanAcoustics/echopype/pull/1083
    #   The artificially added beam and ping_time dimensions at v0.6.0
    #   were reverted at v0.8.0, due to concerns with efficiency and code clarity
    #   (see https://github.com/OSOceanAcoustics/echopype/issues/684 and
    #        https://github.com/OSOceanAcoustics/echopype/issues/978).
    #   However, the mechanisms to expand these dimensions were preserved for
    #   flexibility and potential later use.
    #   Note such expansion is still applied on AZFP data for 2 variables
    #   (see set_groups_azfp.py).

    # Variables that need only the beam dimension added to them.
    beam_only_names = set()

    # Variables that need only the ping_time dimension added to them.
    ping_time_only_names = set()

    # Variables that need beam and ping_time dimensions added to them.
    beam_ping_time_names = set()

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
                    (
                        self.parser_obj.environment["sound_velocity_profile"][::2]
                        if "sound_velocity_profile" in self.parser_obj.environment
                        else [np.nan]
                    ),
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
        if "water_level_draft" in self.parser_obj.environment:
            water_level = self.parser_obj.environment["water_level_draft"]
        else:
            water_level = np.nan
            logger.info("WARNING: The water_level_draft was not in the file. Value set to NaN.")

        time1, msg_type, lat_nmea, lon_nmea = self._extract_NMEA_latlon()
        time2 = self.parser_obj.mru0.get("timestamp", None)
        time2 = np.array(time2) if time2 is not None else [np.nan]
        time3 = self.parser_obj.mru1.get("timestamp", None)
        time3 = np.array(time3) if time3 is not None else [np.nan]

        # Handle potential nan timestamp for time1, time2, and time3
        time1 = self._nan_timestamp_handler(time1)
        time2 = self._nan_timestamp_handler(time2)
        time3 = self._nan_timestamp_handler(time3)

        # Set MRU1 lat lon attributes
        latitude_mru1_attrs = self._varattrs["platform_var_default"]["latitude"].copy()
        latitude_mru1_attrs.update(
            {
                "comment": "Derived from the Simrad MRU1 Datagrams which are "
                "a wrapper of the Kongsberg Maritime Binary Datagrams."
            }
        )
        longitude_mru1_attrs = self._varattrs["platform_var_default"]["longitude"].copy()
        longitude_mru1_attrs.update(
            {
                "comment": "Derived from the Simrad MRU1 Datagrams which are "
                "a wrapper of the Kongsberg Maritime Binary Datagrams."
            }
        ),

        # Assemble variables into a dataset: variables filled with nan if do not exist
        platform_dict = {"platform_name": "", "platform_type": "", "platform_code_ICES": ""}
        ds = xr.Dataset(
            {
                "latitude": (
                    ["time1"],
                    lat_nmea,
                    self._varattrs["platform_var_default"]["latitude"],
                ),
                "longitude": (
                    ["time1"],
                    lon_nmea,
                    self._varattrs["platform_var_default"]["longitude"],
                ),
                "sentence_type": (
                    ["time1"],
                    msg_type,
                    self._varattrs["platform_var_default"]["sentence_type"],
                ),
                "pitch": (
                    ["time2"],
                    np.array(self.parser_obj.mru0.get("pitch", [np.nan])),
                    self._varattrs["platform_var_default"]["pitch"],
                ),
                "roll": (
                    ["time2"],
                    np.array(self.parser_obj.mru0.get("roll", [np.nan])),
                    self._varattrs["platform_var_default"]["roll"],
                ),
                "vertical_offset": (
                    ["time2"],
                    np.array(self.parser_obj.mru0.get("heave", [np.nan])),
                    self._varattrs["platform_var_default"]["vertical_offset"],
                ),
                "water_level": (
                    [],
                    water_level,
                    self._varattrs["platform_var_default"]["water_level"],
                ),
                "drop_keel_offset": (
                    [],
                    self.parser_obj.environment.get("drop_keel_offset", np.nan),
                ),
                "drop_keel_offset_is_manual": (
                    [],
                    self.parser_obj.environment.get("drop_keel_offset_is_manual", np.nan),
                ),
                "water_level_draft_is_manual": (
                    [],
                    self.parser_obj.environment.get("water_level_draft_is_manual", np.nan),
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
                    freq,
                    {
                        "units": "Hz",
                        "long_name": "Transducer frequency",
                        "valid_min": 0.0,
                        "standard_name": "sound_frequency",
                    },
                ),
                "heading": (
                    ["time2"],
                    np.array(self.parser_obj.mru0.get("heading", [np.nan])),
                    {
                        "long_name": "Platform heading (true)",
                        "standard_name": "platform_orientation",
                        "units": "degrees_north",
                        "valid_min": 0.0,
                        "valid_max": 360.0,
                    },
                ),
                "latitude_mru1": (
                    ["time3"],
                    np.array(self.parser_obj.mru1.get("latitude", [np.nan])),
                    latitude_mru1_attrs,
                ),
                "longitude_mru1": (
                    ["time3"],
                    np.array(self.parser_obj.mru1.get("longitude", [np.nan])),
                    longitude_mru1_attrs,
                ),
            },
            coords={
                "channel": (
                    ["channel"],
                    self.sorted_channel["power_complex"],
                    self._varattrs["beam_coord_default"]["channel"],
                ),
                "time1": (
                    ["time1"],
                    time1,
                    {
                        **self._varattrs["platform_coord_default"]["time1"],
                        "comment": "Time coordinate corresponding to NMEA position data.",
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
                "time3": (
                    ["time3"],
                    time3,
                    {
                        "axis": "T",
                        "long_name": "Timestamps for platform motion and orientation data "
                        "from the Kongsberg Maritime Binary Datagram",
                        "standard_name": "time",
                        "comment": "Time coordinate corresponding to platform motion and "
                        "orientation data from the Kongsberg Maritime Binary Datagram.",
                    },
                ),
            },
        )
        ds = ds.assign_attrs(platform_dict)

        # If `.IDX` file exists and `.IDX` data is parsed
        if (
            (self.parser_obj.idx_file != "")
            and self.parser_obj.idx["ping_number"]
            and self.parser_obj.idx["file_offset"]
            and self.parser_obj.idx["vessel_distance"]
            and self.parser_obj.idx["latitude"]
            and self.parser_obj.idx["longitude"]
            and self.parser_obj.idx["timestamp"]
        ):
            ds = self._add_index_data_to_platform_ds(ds)

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

        for i, ch in enumerate(self.sorted_channel[data_type]):
            if (
                np.isclose(beam_params["transducer_alpha_x"][i], 0.00)
                and np.isclose(beam_params["transducer_alpha_y"][i], 0.00)
                and np.isclose(beam_params["transducer_alpha_z"][i], 0.00)
            ):
                beam_params["transducer_alpha_x"][i] = np.nan
                beam_params["transducer_alpha_y"][i] = np.nan
                beam_params["transducer_alpha_z"][i] = np.nan

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
                "beam_type": (
                    ["channel"],
                    beam_params["transducer_beam_type"],
                    {"long_name": "type of transducer (0-single, 1-split)"},
                ),
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

        if data_type == "power":
            ds = ds.assign(
                {
                    "transmit_frequency_start": (
                        ["channel"],
                        freq,
                        self._varattrs["beam_var_default"]["transmit_frequency_start"],
                    ),
                    "transmit_frequency_stop": (
                        ["channel"],
                        freq,
                        self._varattrs["beam_var_default"]["transmit_frequency_stop"],
                    ),
                }
            )

        return ds

    def _add_freq_start_end_ds(self, ds_tmp: xr.Dataset, ch: str) -> xr.Dataset:
        """
        Returns a Dataset with variables
        ``transmit_frequency_start`` and ``transmit_frequency_stop``
        added to ``ds_tmp`` for a specific channel.
        Parameters
        ----------
        ds_tmp: xr.Dataset
            Dataset containing the complex data
        ch: str
            Channel id
        """

        # Process if it's a BB channel (not all pings are CW, where pulse_form encodes CW as 0)
        # CW data encoded as complex samples do NOT have frequency_start and frequency_end
        if not np.all(np.array(self.parser_obj.ping_data_dict["pulse_form"][ch]) == 0):
            freq_start = np.array(self.parser_obj.ping_data_dict["frequency_start"][ch])
            freq_stop = np.array(self.parser_obj.ping_data_dict["frequency_end"][ch])
        elif not self.sorted_channel["power"]:
            freq = self.parser_obj.config_datagram["configuration"][ch]["transducer_frequency"]
            freq_start = np.full(len(self.parser_obj.ping_time[ch]), freq)
            freq_stop = freq_start
        else:
            return ds_tmp

        ds_f_start_end = xr.Dataset(
            {
                "transmit_frequency_start": (
                    ["ping_time"],
                    freq_start.astype(float),
                    self._varattrs["beam_var_default"]["transmit_frequency_start"],
                ),
                "transmit_frequency_stop": (
                    ["ping_time"],
                    freq_stop.astype(float),
                    self._varattrs["beam_var_default"]["transmit_frequency_stop"],
                ),
            },
            coords={
                "ping_time": (
                    ["ping_time"],
                    self.parser_obj.ping_time[ch],
                    self._varattrs["beam_coord_default"]["ping_time"],
                ),
            },
        )

        ds_tmp = xr.merge(
            [ds_tmp, ds_f_start_end], combine_attrs="override"
        )  # override keeps the Dataset attributes

        return ds_tmp

    def _assemble_ds_complex(self, ch):
        data_shape = self.parser_obj.ping_data_dict["complex"][ch]["real"].shape
        ds_tmp = xr.Dataset(
            {
                "backscatter_r": (
                    ["ping_time", "range_sample", "beam"],
                    self.parser_obj.ping_data_dict["complex"][ch]["real"],
                    {
                        "long_name": self._varattrs["beam_var_default"]["backscatter_r"][
                            "long_name"
                        ],
                        "units": "dB",
                    },
                ),
                "backscatter_i": (
                    ["ping_time", "range_sample", "beam"],
                    self.parser_obj.ping_data_dict["complex"][ch]["imag"],
                    {
                        "long_name": self._varattrs["beam_var_default"]["backscatter_i"][
                            "long_name"
                        ],
                        "units": "dB",
                    },
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
                    np.arange(start=1, stop=data_shape[2] + 1).astype(str),
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
                        np.arange(
                            self.parser_obj.ping_data_dict_tx["complex"][ch]["real"].shape[1]
                        ),
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
                        self.parser_obj.ping_data_dict_tx["complex"][ch]["real"],
                        {
                            "long_name": "Real part of the transmit pulse",
                            "units": "V",
                            "comment": "Only exist for Simrad EK80 file with RAW4 datagrams",
                        },
                    ),
                    "transmit_pulse_i": (
                        ["ping_time", "transmit_sample"],
                        self.parser_obj.ping_data_dict_tx["complex"][ch]["imag"],
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
                    {
                        "long_name": self._varattrs["beam_var_default"]["backscatter_r"][
                            "long_name"
                        ],
                        "units": "dB",
                    },
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

        ds_tmp = self._add_freq_start_end_ds(ds_tmp, ch)

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

        def pulse_form_map(pulse_form):
            str_map = np.array(["CW", "LFM", "", "", "", "FMD"])
            return str_map[pulse_form]

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
                    {"long_name": "Hann window slope parameter for transmit signal"},
                ),
                "channel_mode": (
                    ["ping_time"],
                    np.array(self.parser_obj.ping_data_dict["channel_mode"][ch], dtype=np.byte),
                    {
                        "long_name": "Transceiver mode",
                        "flag_values": [0, 1],
                        "flag_meanings": ["Active", "Unknown"],
                    },
                ),
                "transmit_type": (
                    ["ping_time"],
                    pulse_form_map(np.array(self.parser_obj.ping_data_dict["pulse_form"][ch])),
                    {
                        "long_name": "Type of transmitted pulse",
                        "flag_values": ["CW", "LFM", "FMD"],
                        "flag_meanings": [
                            "Continuous Wave – a pulse nominally of one frequency",
                            "Linear Frequency Modulation – a pulse which varies from "
                            "transmit_frequency_start to transmit_frequency_stop in a linear "
                            "manner over the nominal pulse duration (transmit_duration_nominal)",
                            "Frequency Modulated 'D' - An EK80-specific FM type that is not "
                            "clearly described",
                        ],
                    },
                ),
                "sample_time_offset": (
                    ["ping_time"],
                    (
                        np.array(self.parser_obj.ping_data_dict["offset"][ch])
                        * np.array(self.parser_obj.ping_data_dict["sample_interval"][ch])
                    ),
                    {
                        "long_name": "Time offset that is subtracted from the timestamp"
                        " of each sample",
                        "units": "s",
                    },
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
        # Combine all channels into one Dataset
        ds_combine = xr.concat(ds_combine, dim="channel")

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

            ds_data = self._attach_vars_to_ds_data(ds_data, ch, rs_size=ds_data.range_sample.size)

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
            "impedance",  # transceiver impedance (z_er), different from transducer impedance (z_et)
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
            ds_table["impedance_transceiver"] = xr.DataArray(
                param_dict["impedance"],
                dims=["channel"],
                coords={"channel": ds_table["channel"]},
                attrs={
                    "units": "ohm",
                    "long_name": "Transceiver impedance",
                },
            )
        if "rx_sample_frequency" in param_dict:
            ds_table["receiver_sampling_frequency"] = xr.DataArray(
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
                "impedance",  # transducer impedance (z_et), different from transceiver impedance (z_er)  # noqa
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
            ds_cal = ds_cal.rename_vars({"impedance": "impedance_transducer"})

        # Save decimation factors and filter coefficients
        # Param map values
        # 1: wide band transceiver (WBT)
        # 2: pulse compression (PC)
        param_map = {1: WIDE_BAND_TRANS, 2: PULSE_COMPRESS}
        coeffs_and_decimation = {
            t: {FILTER_IMAG: [], FILTER_REAL: [], DECIMATION: []} for t in list(param_map.values())
        }

        for ch in self.sorted_channel["all"]:
            fil_coeffs = self.parser_obj.fil_coeffs.get(ch, None)
            fil_df = self.parser_obj.fil_df.get(ch, None)

            if fil_coeffs and fil_df:
                # get filter coefficient values
                for type_num, values in fil_coeffs.items():
                    param = param_map[type_num]
                    coeffs_and_decimation[param][FILTER_IMAG].append(np.imag(values))
                    coeffs_and_decimation[param][FILTER_REAL].append(np.real(values))

                # get decimation factor values
                for type_num, value in fil_df.items():
                    param = param_map[type_num]
                    coeffs_and_decimation[param][DECIMATION].append(value)

        # Assemble everything into a Dataset
        ds = xr.merge([ds_table, ds_cal])

        # Add the coeffs and decimation
        ds = ds.pipe(self._add_filter_params, coeffs_and_decimation)

        # Save the entire config XML in vendor group in case of info loss
        ds["config_xml"] = self.parser_obj.config_datagram["xml"]

        # If `.BOT` file exists and `.BOT` data is parsed
        if (
            (self.parser_obj.bot_file != "")
            and self.parser_obj.bot["depth"]
            and self.parser_obj.bot["timestamp"]
        ):
            ds = self._add_seafloor_detection_data_to_vendor_ds(ds)

        return ds

    @staticmethod
    def _add_filter_params(
        dataset: xr.Dataset, coeffs_and_decimation: Dict[str, Dict[str, List[Union[int, NDArray]]]]
    ) -> xr.Dataset:
        """
        Assembles filter coefficient and decimation factors and add to the dataset

        Parameters
        ----------
        dataset : xr.Dataset
            xarray dataset where the filter coefficient and decimation factors will be added
        coeffs_and_decimation : dict
            dictionary holding the filter coefficient and decimation factors

        Returns
        -------
        xr.Dataset
            The modified dataset with filter coefficient and decimation factors included
        """
        attribute_values = {
            FILTER_IMAG: "filter coefficients (imaginary part)",
            FILTER_REAL: "filter coefficients (real part)",
            DECIMATION: "decimation factor",
            WIDE_BAND_TRANS: "Wideband transceiver",
            PULSE_COMPRESS: "Pulse compression",
        }

        coeffs_xr_data = {}
        for cd_type, values in coeffs_and_decimation.items():
            for key, data in values.items():
                if data:
                    if "filter" in key:
                        attrs = {
                            "long_name": f"{attribute_values[cd_type]} {attribute_values[key]}"
                        }
                        # filter_i and filter_r
                        max_len = np.max([len(a) for a in data])
                        # Pad arrays
                        data = np.asarray(
                            [
                                np.pad(a, (0, max_len - len(a)), "constant", constant_values=np.nan)
                                for a in data
                            ]
                        )
                        dims = ["channel", f"{cd_type}_filter_n"]
                    else:
                        attrs = {
                            "long_name": f"{attribute_values[cd_type]} {attribute_values[DECIMATION]}"  # noqa
                        }
                        dims = ["channel"]
                    # Set the xarray data dictionary
                    coeffs_xr_data[f"{cd_type}_{key}"] = (dims, data, attrs)

        return dataset.assign(coeffs_xr_data)
