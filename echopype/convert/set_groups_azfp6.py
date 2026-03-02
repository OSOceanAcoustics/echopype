"""
Class to save unpacked echosounder data to appropriate groups in netcdf or zarr.
"""

import numpy as np
import xarray as xr

from .set_groups_azfp import SetGroupsAZFP


class SetGroupsAZFP6(SetGroupsAZFP):
    """Class for saving groups to netcdf or zarr from AZFP6 data files."""

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
            "sonar_software_name": "ULS6",
            "sonar_software_version": "beta version",
            "sonar_type": "echosounder",
        }
        ds = ds.assign_attrs(sonar_attr_dict)

        return ds

    def _firmware_specific_platform_fields(self, unpacked_data):
        gps_latlon = np.array(unpacked_data["gps_lat_long"])

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

        variable_fields = {
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
        }
        return variable_fields, {"time1": time1}

    def _firmware_specific_vendor_fields(self, unpacked_data, anc):
        """Adjust for different naming conventions and fields between firmware versions"""
        ad_len = list(range(anc[:, -2:].shape[-1]))
        ad_channels = anc[:, -2:]
        ancillary_len = list(range(anc.shape[-1]))
        if len(unpacked_data["custom"]) == 0:
            unpacked_data["custom"] = 0

        variable_fields = {
            "acq_status": (["ping_time"], unpacked_data["acq_status"]),
            "sensor_status": (["ping_time"], unpacked_data["sensor_status"]),
            "ad_channels": (
                ["ping_time", "ad_len"],
                ad_channels,
                {"long_name": "AD channel 6 and 7"},
            ),
            "custom": ([], unpacked_data["custom"], {"long_name": "Spare/custom channel"}),
        }
        coords = {"ancillary_len": ancillary_len, "ad_len": ad_len}

        paros_time = self.parser_obj._get_paros_time()
        paros_raw = np.asarray(unpacked_data["paros_press_temp_raw"])
        if len(paros_raw) > 0:
            paros_fields = {
                "paros_pressure_counts": (
                    ["paros_time"],
                    paros_raw[:, 0],
                    {"long_name": "Raw counts for Paros pressure"},
                ),
                "paros_temperature_counts": (
                    ["paros_time"],
                    paros_raw[:, 1],
                    {"long_name": "Raw counts for Paros temperature"},
                ),
            }
            coords["paros_time"] = paros_time
            variable_fields = {**variable_fields, **paros_fields}

        return variable_fields, coords

    def _firmware_specific_env_fields(self):
        # Additional variables, if present
        temp_press = dict()
        if not np.isnan(self.parser_obj.unpacked_data["paros_temperature"]).all():
            temp_press["temperature"] = (
                ["paros_time"],
                self.parser_obj.unpacked_data["paros_temperature"],
                {
                    "long_name": "Water temperature",
                    "standard_name": "sea_water_temperature",
                    "units": "deg_C",
                },
            )
        if not np.isnan(self.parser_obj.unpacked_data["paros_pressure"]).all():
            temp_press["pressure"] = (
                ["paros_time"],
                self.parser_obj.unpacked_data["paros_pressure"],
                {
                    "long_name": "Sea water pressure",
                    "standard_name": "sea_water_pressure_due_to_sea_water",
                    "units": "dbar",
                },
            )

        if len(temp_press) > 0:
            coords = {
                "paros_time": (
                    ["paros_time"],
                    self.parser_obj._get_paros_time(),
                    {
                        "axis": "T",
                        "long_name": "Timestamp of each Paros measurement",
                        "standard_name": "time",
                        "comment": "Time coordinate corresponding to Paros sensor variables.",
                    },
                )
            }
            return temp_press, coords

        if not np.isnan(self.parser_obj.unpacked_data["temperature"]).all():
            temp_press["temperature"] = (
                ["time1"],
                self.parser_obj.unpacked_data["temperature"],
                {
                    "long_name": "Water temperature",
                    "standard_name": "sea_water_temperature",
                    "units": "deg_C",
                },
            )
        if not np.isnan(self.parser_obj.unpacked_data["pressure"]).all():
            temp_press["pressure"] = (
                ["time1"],
                self.parser_obj.unpacked_data["pressure"],
                {
                    "long_name": "Sea water pressure",
                    "standard_name": "sea_water_pressure_due_to_sea_water",
                    "units": "dbar",
                },
            )

        coords = {
            "time1": (
                ["time1"],
                self.parser_obj.ping_time,
                {
                    "axis": "T",
                    "long_name": "Timestamp of each ping",
                    "standard_name": "time",
                    "comment": "Time coordinate corresponding to environmental variables.",
                },
            )
        }

        return temp_press, coords
