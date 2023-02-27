from collections import defaultdict
from typing import List

import numpy as np
import xarray as xr

from ..utils.coding import set_time_encodings
from ..utils.log import _init_logger
from ..utils.prov import echopype_prov_attrs, source_files_vars

# fmt: off
from .set_groups_base import SetGroupsBase

# fmt: on

logger = _init_logger(__name__)


class SetGroupsEK60(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from EK60 data files."""

    # The sets beam_only_names, ping_time_only_names, and
    # beam_ping_time_names are used in set_groups_base and
    # in converting from v0.5.x to v0.6.0. The values within
    # these sets are applied to all Sonar/Beam_groupX groups.

    # Variables that need only the beam dimension added to them.
    beam_only_names = {"backscatter_r", "angle_athwartship", "angle_alongship"}

    # Variables that need only the ping_time dimension added to them.
    ping_time_only_names = {"beam_type"}

    # Variables that need beam and ping_time dimensions added to them.
    beam_ping_time_names = {
        "beam_direction_x",
        "beam_direction_y",
        "beam_direction_z",
        "beamwidth_twoway_alongship",
        "beamwidth_twoway_athwartship",
        "angle_offset_alongship",
        "angle_offset_athwartship",
        "angle_sensitivity_alongship",
        "angle_sensitivity_athwartship",
        "equivalent_beam_angle",
        "gain_correction",
    }

    beamgroups_possible = [
        {
            "name": "Beam_group1",
            "descr": (
                "contains backscatter power (uncalibrated) and other beam or"
                " channel-specific data, including split-beam angle data when they exist."
            ),
        }
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.old_ping_time = None

        # obtain sorted channel dict in ascending order
        channels = list(self.parser_obj.config_datagram["transceivers"].keys())
        channel_ids = {
            ch: self.parser_obj.config_datagram["transceivers"][ch]["channel_id"] for ch in channels
        }
        # example sorted_channel from a 5-channel data file for future reference:
        # 1: 'GPT  18 kHz 009072034d45 1-1 ES18-11'
        # 2: 'GPT  38 kHz 009072033fa2 2-1 ES38B'
        # 3: 'GPT  70 kHz 009072058c6c 3-1 ES70-7C'
        # 4: 'GPT 120 kHz 00907205794e 4-1 ES120-7C'
        # 5: 'GPT 200 kHz 0090720346a8 5-1 ES200-7C'
        # In some examples the channels may not be ordered, thus sorting is required
        self.sorted_channel = dict(sorted(channel_ids.items(), key=lambda item: item[1]))

        # obtain corresponding frequency dict from sorted channels
        self.freq = [
            self.parser_obj.config_datagram["transceivers"][ch]["frequency"]
            for ch in self.sorted_channel.keys()
        ]

        # correct duplicate ping_time
        for ch in self.sorted_channel.keys():
            ping_time = self.parser_obj.ping_time[ch]
            _, unique_idx = np.unique(ping_time, return_index=True)
            duplicates = np.invert(np.isin(np.arange(len(ping_time)), unique_idx))
            if duplicates.any():
                if self.old_ping_time is None:
                    if (
                        len({arr.shape for arr in self.parser_obj.ping_time.values()}) == 1
                        and np.unique(np.stack(self.parser_obj.ping_time.values()), axis=0).shape[0]
                        == 1
                    ):
                        self.old_ping_time = self.parser_obj.ping_time[ch]
                    else:
                        ping_times = [
                            xr.DataArray(arr, dims="ping_time")
                            for arr in self.parser_obj.ping_time.values()
                        ]
                        self.old_ping_time = xr.concat(ping_times, dim="ping_time")

                backscatter_r = self.parser_obj.ping_data_dict["power"][ch]
                # indexes of duplicates including the originals
                # (if there are 2 times that are the same, both will be included)
                (all_duplicates_idx,) = np.where(np.isin(ping_time, ping_time[duplicates][0]))
                if np.array_equal(
                    backscatter_r[all_duplicates_idx[0]],
                    backscatter_r[all_duplicates_idx[1]],
                ):
                    logger.warning(
                        "duplicate pings with identical values detected; the duplicate pings will be removed"  # noqa
                    )
                    for v in self.parser_obj.ping_data_dict.values():
                        if v[ch] is None or len(v[ch]) == 0:
                            continue
                        if isinstance(v[ch], np.ndarray):
                            v[ch] = v[ch][unique_idx]
                        else:
                            v[ch] = [v[ch][i] for i in unique_idx]
                    self.parser_obj.ping_time[ch] = self.parser_obj.ping_time[ch][unique_idx]
                else:
                    logger.warning(
                        "duplicate ping times detected; the duplicate times will be incremented by 1 nanosecond and remain in the ping_time coordinate. The original ping times will be preserved in the Provenance group"  # noqa
                    )

                    deltas = duplicates * np.timedelta64(1, "ns")
                    new_ping_time = ping_time + deltas
                    self.parser_obj.ping_time[ch] = new_ping_time

    def set_provenance(self) -> xr.Dataset:
        """Set the Provenance group."""
        prov_dict = echopype_prov_attrs(process_type="conversion")
        prov_dict["duplicate_ping_times"] = 1 if self.old_ping_time is not None else 0
        files_vars = source_files_vars(self.input_file)
        if self.old_ping_time is not None:
            source_vars = {"old_ping_time": self.old_ping_time, **files_vars["source_files_var"]}
        else:
            source_vars = files_vars["source_files_var"]

        ds = xr.Dataset(
            data_vars=source_vars, coords=files_vars["source_files_coord"], attrs=prov_dict
        )

        return ds

    def set_env(self) -> xr.Dataset:
        """Set the Environment group."""

        # Loop over channels
        ds_env = []
        for ch in self.sorted_channel.keys():
            ds_tmp = xr.Dataset(
                {
                    "absorption_indicative": (
                        ["time1"],
                        self.parser_obj.ping_data_dict["absorption_coefficient"][ch],
                        {
                            "long_name": "Indicative acoustic absorption",
                            "units": "dB/m",
                            "valid_min": 0.0,
                        },
                    ),
                    "sound_speed_indicative": (
                        ["time1"],
                        self.parser_obj.ping_data_dict["sound_velocity"][ch],
                        {
                            "long_name": "Indicative sound speed",
                            "standard_name": "speed_of_sound_in_sea_water",
                            "units": "m/s",
                            "valid_min": 0.0,
                        },
                    ),
                },
                coords={
                    "time1": (
                        ["time1"],
                        self.parser_obj.ping_time[ch],
                        {
                            "axis": "T",
                            "long_name": "Timestamps for NMEA position datagrams",
                            "standard_name": "time",
                            "comment": "Time coordinate corresponding to environmental variables.",
                        },
                    )
                },
            )
            # Attach channel dimension/coordinate
            ds_tmp = ds_tmp.expand_dims({"channel": [self.sorted_channel[ch]]})
            ds_tmp["channel"] = ds_tmp["channel"].assign_attrs(
                self._varattrs["beam_coord_default"]["channel"]
            )

            ds_tmp["frequency_nominal"] = (
                ["channel"],
                [self.parser_obj.config_datagram["transceivers"][ch]["frequency"]],
                {
                    "units": "Hz",
                    "long_name": "Transducer frequency",
                    "valid_min": 0.0,
                    "standard_name": "sound_frequency",
                },
            )

            ds_env.append(ds_tmp)

        # Merge data from all channels
        ds = xr.merge(ds_env)

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
            "sonar_manufacturer": "Simrad",
            "sonar_model": self.sonar_model,
            # transducer (sonar) serial number is not stored in the EK60 raw data file,
            # so sonar_serial_number can't be populated from the raw datagrams
            "sonar_serial_number": "",
            "sonar_software_name": self.parser_obj.config_datagram["sounder_name"],
            "sonar_software_version": self.parser_obj.config_datagram["version"],
            "sonar_type": "echosounder",
        }
        ds = ds.assign_attrs(sonar_attr_dict)

        return ds

    def set_platform(self) -> xr.Dataset:
        """Set the Platform group."""

        # Collect variables
        # Read lat/long from NMEA datagram
        time1, msg_type, lat, lon = self._extract_NMEA_latlon()

        # NMEA dataset: variables filled with nan if do not exist
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
                "sentence_type": (["time1"], msg_type),
            },
            coords={
                "time1": (
                    ["time1"],
                    time1,
                    {
                        **self._varattrs["platform_coord_default"]["time1"],
                        "comment": "Time coordinate corresponding to NMEA position data.",
                    },
                )
            },
        )

        # TODO: consider allow users to set water_level like in EK80?
        # if self.ui_param['water_level'] is not None:
        #     water_level = self.ui_param['water_level']
        # else:
        #     water_level = np.nan
        #     print('WARNING: The water_level_draft was not in the file. Value '
        #           'set to None.')

        # Loop over channels and merge all
        ds_plat = []
        for ch in self.sorted_channel.keys():
            ds_tmp = xr.Dataset(
                {
                    "pitch": (
                        ["time2"],
                        self.parser_obj.ping_data_dict["pitch"][ch],
                        self._varattrs["platform_var_default"]["pitch"],
                    ),
                    "roll": (
                        ["time2"],
                        self.parser_obj.ping_data_dict["roll"][ch],
                        self._varattrs["platform_var_default"]["roll"],
                    ),
                    "vertical_offset": (
                        ["time2"],
                        self.parser_obj.ping_data_dict["heave"][ch],
                        self._varattrs["platform_var_default"]["vertical_offset"],
                    ),
                    "water_level": (
                        ["time3"],
                        self.parser_obj.ping_data_dict["transducer_depth"][ch],
                        self._varattrs["platform_var_default"]["water_level"],
                    ),
                    "transducer_offset_x": (
                        [],
                        self.parser_obj.config_datagram["transceivers"][ch].get("pos_x", np.nan),
                        self._varattrs["platform_var_default"]["transducer_offset_x"],
                    ),
                    "transducer_offset_y": (
                        [],
                        self.parser_obj.config_datagram["transceivers"][ch].get("pos_y", np.nan),
                        self._varattrs["platform_var_default"]["transducer_offset_y"],
                    ),
                    "transducer_offset_z": (
                        [],
                        self.parser_obj.config_datagram["transceivers"][ch].get("pos_z", np.nan),
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
                },
                coords={
                    "time2": (
                        ["time2"],
                        self.parser_obj.ping_time[ch],
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
                        self.parser_obj.ping_time[ch],
                        {
                            "axis": "T",
                            "long_name": "Timestamps for platform-related sampling environment",
                            "standard_name": "time",
                            "comment": "Time coordinate corresponding to platform-related "
                            "sampling environment.",
                        },
                    ),
                },
                attrs={
                    "platform_code_ICES": self.ui_param["platform_code_ICES"],
                    "platform_name": self.ui_param["platform_name"],
                    "platform_type": self.ui_param["platform_type"],
                },
            )

            # Attach channel dimension/coordinate
            ds_tmp = ds_tmp.expand_dims({"channel": [self.sorted_channel[ch]]})
            ds_tmp["channel"] = ds_tmp["channel"].assign_attrs(
                self._varattrs["beam_coord_default"]["channel"]
            )

            ds_tmp["frequency_nominal"] = (
                ["channel"],
                [self.parser_obj.config_datagram["transceivers"][ch]["frequency"]],
                {
                    "units": "Hz",
                    "long_name": "Transducer frequency",
                    "valid_min": 0.0,
                    "standard_name": "sound_frequency",
                },
            )

            ds_plat.append(ds_tmp)

        # Merge data from all channels
        # TODO: for current test data we see all
        #  pitch/roll/heave are the same for all freq channels
        #  consider only saving those from the first channel
        ds_plat = xr.merge(ds_plat)

        # Merge with NMEA data
        ds = xr.merge([ds, ds_plat], combine_attrs="override")

        return set_time_encodings(ds)

    def _set_beam_group1_zarr_vars(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Modifies ds by setting all variables associated with
        ``Beam_group1``, that were directly written to a
        temporary zarr file.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset representing ``Beam_group1`` filled with
            all variables, besides those written to zarr

        Returns
        -------
        A modified version of ``ds`` with the zarr variables
        added to it.
        """

        # TODO: In the future it would be nice to have a dictionary of
        #  attributes stored in one place for all of the variables.
        #  This would reduce unnecessary code duplication in the
        #  functions below.

        # obtain DataArrays using zarr variables
        zarr_path = self.parsed2zarr_obj.zarr_file_name
        backscatter_r = self._get_power_dataarray(zarr_path)
        angle_athwartship, angle_alongship = self._get_angle_dataarrays(zarr_path)

        # append DataArrays created from zarr file
        ds = ds.assign(
            backscatter_r=backscatter_r,
            angle_athwartship=angle_athwartship,
            angle_alongship=angle_alongship,
        )
        return ds

    def set_beam(self) -> List[xr.Dataset]:
        """Set the /Sonar/Beam_group1 group."""

        # Channel-specific variables
        params = [
            "beam_type",
            "beamwidth_alongship",
            "beamwidth_athwartship",
            "dir_x",
            "dir_y",
            "dir_z",
            "angle_offset_alongship",
            "angle_offset_athwartship",
            "angle_sensitivity_alongship",
            "angle_sensitivity_athwartship",
            "pos_x",
            "pos_y",
            "pos_z",
            "equivalent_beam_angle",
            "gpt_software_version",
            "gain",
        ]
        beam_params = defaultdict()
        for param in params:
            beam_params[param] = [
                self.parser_obj.config_datagram["transceivers"][ch_seq].get(param, np.nan)
                for ch_seq in self.sorted_channel.keys()
            ]

        # TODO: Need to discuss if to remove INDEX2POWER factor from the backscatter_r
        #  currently this factor is multiplied to the raw data before backscatter_r is saved.
        #  This is if we are encoding only raw data to the .nc/zarr file.
        #  Need discussion since then the units won't match
        #  with convention (though it didn't match already...).
        # Assemble variables into a dataset
        ds = xr.Dataset(
            {
                "frequency_nominal": (
                    ["channel"],
                    self.freq,
                    {
                        "units": "Hz",
                        "long_name": "Transducer frequency",
                        "valid_min": 0.0,
                        "standard_name": "sound_frequency",
                    },
                ),
                "beam_type": (
                    "channel",
                    beam_params["beam_type"],
                    {"long_name": "type of transducer (0-single, 1-split)"},
                ),
                "beamwidth_twoway_alongship": (
                    ["channel"],
                    beam_params["beamwidth_alongship"],
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
                    beam_params["beamwidth_athwartship"],
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
                    beam_params["dir_x"],
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
                    beam_params["dir_y"],
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
                    beam_params["dir_z"],
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
                "gain_correction": (
                    ["channel"],
                    beam_params["gain"],
                    {"long_name": "Gain correction", "units": "dB"},
                ),
                "gpt_software_version": (
                    ["channel"],
                    beam_params["gpt_software_version"],
                ),
            },
            coords={
                "channel": (
                    ["channel"],
                    list(self.sorted_channel.values()),
                    self._varattrs["beam_coord_default"]["channel"],
                ),
            },
            attrs={"beam_mode": "vertical", "conversion_equation_t": "type_3"},
        )

        # Construct Dataset with ping-by-ping data from all channels
        ds_backscatter = []
        for ch in self.sorted_channel.keys():
            var_dict = {
                "sample_interval": (
                    ["ping_time"],
                    self.parser_obj.ping_data_dict["sample_interval"][ch],
                    {
                        "long_name": "Interval between recorded raw data samples",
                        "units": "s",
                        "valid_min": 0.0,
                    },
                ),
                "transmit_bandwidth": (
                    ["ping_time"],
                    self.parser_obj.ping_data_dict["bandwidth"][ch],
                    {
                        "long_name": "Nominal bandwidth of transmitted pulse",
                        "units": "Hz",
                        "valid_min": 0.0,
                    },
                ),
                "transmit_duration_nominal": (
                    ["ping_time"],
                    self.parser_obj.ping_data_dict["pulse_length"][ch],
                    {
                        "long_name": "Nominal bandwidth of transmitted pulse",
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
                "data_type": (
                    ["ping_time"],
                    self.parser_obj.ping_data_dict["mode"][ch],
                    {
                        "long_name": "recorded data type (1-power only, 2-angle only 3-power and angle)"  # noqa
                    },
                ),
                "count": (
                    ["ping_time"],
                    self.parser_obj.ping_data_dict["count"][ch],
                    {"long_name": "Number of samples "},
                ),
                "offset": (
                    ["ping_time"],
                    self.parser_obj.ping_data_dict["offset"][ch],
                    {"long_name": "Offset of first sample"},
                ),
                "transmit_mode": (
                    ["ping_time"],
                    self.parser_obj.ping_data_dict["transmit_mode"][ch],
                    {"long_name": "0 = Active, 1 = Passive, 2 = Test, -1 = Unknown"},
                ),
            }

            if not self.parsed2zarr_obj.temp_zarr_dir:
                var_dict["backscatter_r"] = (
                    ["ping_time", "range_sample"],
                    self.parser_obj.ping_data_dict["power"][ch],
                    {"long_name": "Backscatter power", "units": "dB"},
                )

                ds_tmp = xr.Dataset(
                    var_dict,
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
            else:
                ds_tmp = xr.Dataset(
                    var_dict,
                    coords={
                        "ping_time": (
                            ["ping_time"],
                            self.parser_obj.ping_time[ch],
                            self._varattrs["beam_coord_default"]["ping_time"],
                        ),
                    },
                )

            if not self.parsed2zarr_obj.temp_zarr_dir:
                # Save angle data if exist based on values in
                # self.parser_obj.ping_data_dict['mode'][ch]
                # Assume the mode of all pings are identical
                # 1 = Power only, 2 = Angle only 3 = Power & Angle
                if np.all(np.array(self.parser_obj.ping_data_dict["mode"][ch]) != 1):
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

            # Attach frequency dimension/coordinate
            ds_tmp = ds_tmp.expand_dims({"channel": [self.sorted_channel[ch]]})
            ds_tmp["channel"] = ds_tmp["channel"].assign_attrs(
                self._varattrs["beam_coord_default"]["channel"]
            )
            ds_backscatter.append(ds_tmp)

        # Merge data from all channels
        ds = xr.merge(
            [ds, xr.merge(ds_backscatter)], combine_attrs="override"
        )  # override keeps the Dataset attributes

        if self.parsed2zarr_obj.temp_zarr_dir:
            ds = self._set_beam_group1_zarr_vars(ds)

        # Manipulate some Dataset dimensions to adhere to convention
        self.beam_groups_to_convention(
            ds, self.beam_only_names, self.beam_ping_time_names, self.ping_time_only_names
        )

        return [set_time_encodings(ds)]

    def set_vendor(self) -> xr.Dataset:
        # Retrieve pulse length, gain, and sa correction
        pulse_length = np.array(
            [
                self.parser_obj.config_datagram["transceivers"][ch]["pulse_length_table"]
                for ch in self.sorted_channel.keys()
            ]
        )

        gain = np.array(
            [
                self.parser_obj.config_datagram["transceivers"][ch]["gain_table"]
                for ch in self.sorted_channel.keys()
            ]
        )

        sa_correction = [
            self.parser_obj.config_datagram["transceivers"][ch]["sa_correction_table"]
            for ch in self.sorted_channel.keys()
        ]

        # Save pulse length and sa correction
        ds = xr.Dataset(
            {
                "frequency_nominal": (
                    ["channel"],
                    self.freq,
                    {
                        "units": "Hz",
                        "long_name": "Transducer frequency",
                        "valid_min": 0.0,
                        "standard_name": "sound_frequency",
                    },
                ),
                "sa_correction": (["channel", "pulse_length_bin"], sa_correction),
                "gain_correction": (["channel", "pulse_length_bin"], gain),
                "pulse_length": (["channel", "pulse_length_bin"], pulse_length),
            },
            coords={
                "channel": (
                    ["channel"],
                    list(self.sorted_channel.values()),
                    self._varattrs["beam_coord_default"]["channel"],
                ),
                "pulse_length_bin": (
                    ["pulse_length_bin"],
                    np.arange(pulse_length.shape[1]),
                ),
            },
        )
        return ds
