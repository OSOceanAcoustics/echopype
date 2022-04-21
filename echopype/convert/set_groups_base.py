import abc
from datetime import datetime as dt

import numpy as np
import pynmea2
import xarray as xr
from _echopype_version import version as ECHOPYPE_VERSION

from ..echodata.convention import sonarnetcdf_1
from ..utils.coding import COMPRESSION_SETTINGS, set_encodings

DEFAULT_CHUNK_SIZE = {"range_sample": 25000, "ping_time": 2500}

# Variables that need only the beam dimension added to them.
# These lists are applied to all Sonar/Beam_groupX groups.
beam_only_names = {
    "EK60": {"backscatter_r", "angle_athwartship", "angle_alongship"},
    "EK80": {
        "backscatter_r",
        "backscatter_i",
        "angle_athwartship",
        "angle_alongship",
        "frequency_start",
        "frequency_end",
    },
    "AZFP": {},  # {"backscatter_r"}
}

# Variables that need only the ping_time dimension added to them.
# These lists are applied to all Sonar/Beam_groupX groups.
ping_time_only_names = {"EK60": {"beam_type"}, "EK80": {"beam_type"}, "AZFP": {}}

# Variables that need beam and ping_time dimensions added to them.
# These lists are applied to all Sonar/Beam_groupX groups.
beam_ping_time_names = {
    "EK60": {
        "beam_direction_x",
        "beam_direction_y",
        "beam_direction_z",
        "beamwidth_receive_alongship",
        "beamwidth_receive_athwartship",
        "beamwidth_transmit_alongship",
        "beamwidth_transmit_athwartship",
        "angle_offset_alongship",
        "angle_offset_athwartship",
        "angle_sensitivity_alongship",
        "angle_sensitivity_athwartship",
        "equivalent_beam_angle",
        "gain_correction",
    },
    "EK80": {
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
    },
    "AZFP": {},  # {"equivalent_beam_angle", "gain_correction"}
}


class SetGroupsBase(abc.ABC):
    """Base class for saving groups to netcdf or zarr from echosounder data files."""

    def __init__(
        self,
        parser_obj,
        input_file,
        output_path,
        sonar_model=None,
        engine="zarr",
        compress=True,
        overwrite=True,
        params=None,
    ):
        # parser object ParseEK60/ParseAZFP/etc...
        self.parser_obj = parser_obj

        # Used for when a sonar that is not AZFP/EK60/EK80 can still be saved
        self.sonar_model = sonar_model

        self.input_file = input_file
        self.output_path = output_path
        self.engine = engine
        self.compress = compress
        self.overwrite = overwrite
        self.ui_param = params

        if not self.compress:
            self.compression_settings = None
        else:
            self.compression_settings = COMPRESSION_SETTINGS[self.engine]

        self._varattrs = sonarnetcdf_1.yaml_dict["variable_and_varattributes"]
        # self._beamgroups must be a list of dicts, eg:
        # [{"name":"Beam_group1", "descr":"contains complex backscatter data
        # and other beam or channel-specific data."}]
        self._beamgroups = []

    # TODO: change the set_XXX methods to return a dataset to be saved
    #  in the overarching save method
    def set_toplevel(self, sonar_model, date_created=None) -> xr.Dataset:
        """Set the top-level group."""
        # Collect variables
        tl_dict = {
            "conventions": "CF-1.7, SONAR-netCDF4-1.0, ACDD-1.3",
            "keywords": sonar_model,
            "sonar_convention_authority": "ICES",
            "sonar_convention_name": "SONAR-netCDF4",
            "sonar_convention_version": "1.0",
            "summary": "",
            "title": "",
            "date_created": np.datetime_as_string(date_created, "s") + "Z",
            "survey_name": self.ui_param["survey_name"],
        }
        # Save
        ds = xr.Dataset()
        ds = ds.assign_attrs(tl_dict)
        return ds

    def set_provenance(self) -> xr.Dataset:
        """Set the Provenance group."""
        # Collect variables
        prov_dict = {
            "conversion_software_name": "echopype",
            "conversion_software_version": ECHOPYPE_VERSION,
            "conversion_time": dt.utcnow().isoformat(timespec="seconds") + "Z",  # use UTC time
            "src_filenames": self.input_file,
        }
        # Save
        ds = xr.Dataset()
        ds = ds.assign_attrs(prov_dict)
        return ds

    @abc.abstractmethod
    def set_env(self) -> xr.Dataset:
        """Set the Environment group."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_sonar(self) -> xr.Dataset:
        """Set the Sonar group."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_beam(self) -> xr.Dataset:
        """Set the /Sonar/Beam group."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_platform(self) -> xr.Dataset:
        """Set the Platform group."""
        raise NotImplementedError

    def set_nmea(self) -> xr.Dataset:
        """Set the Platform/NMEA group."""
        # Save nan if nmea data is not encoded in the raw file
        if len(self.parser_obj.nmea["nmea_string"]) != 0:
            # Convert np.datetime64 numbers to seconds since 1900-01-01 00:00:00Z
            # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
            time = (
                self.parser_obj.nmea["timestamp"] - np.datetime64("1900-01-01T00:00:00")
            ) / np.timedelta64(1, "s")
            raw_nmea = self.parser_obj.nmea["nmea_string"]
        else:
            time = [np.nan]
            raw_nmea = [np.nan]

        ds = xr.Dataset(
            {
                "NMEA_datagram": (
                    ["location_time"],
                    raw_nmea,
                    {"long_name": "NMEA datagram"},
                )
            },
            coords={
                "location_time": (
                    ["location_time"],
                    time,
                    {
                        "axis": "T",
                        "long_name": "Timestamps for NMEA datagrams",
                        "standard_name": "time",
                    },
                )
            },
            attrs={"description": "All NMEA sensor datagrams"},
        )

        return set_encodings(ds)

    @abc.abstractmethod
    def set_vendor(self) -> xr.Dataset:
        """Set the Vendor group."""
        raise NotImplementedError

    # TODO: move this to be part of parser as it is not a "set" operation
    def _parse_NMEA(self):
        """Get the lat and lon values from the raw nmea data"""
        messages = [string[3:6] for string in self.parser_obj.nmea["nmea_string"]]
        idx_loc = np.argwhere(np.isin(messages, self.ui_param["nmea_gps_sentence"])).squeeze()
        if idx_loc.size == 1:  # in case of only 1 matching message
            idx_loc = np.expand_dims(idx_loc, axis=0)
        nmea_msg = []
        for x in idx_loc:
            try:
                nmea_msg.append(pynmea2.parse(self.parser_obj.nmea["nmea_string"][x]))
            except (
                pynmea2.ChecksumError,
                pynmea2.SentenceTypeError,
                AttributeError,
                pynmea2.ParseError,
            ):
                nmea_msg.append(None)
        lat = (
            np.array([x.latitude if hasattr(x, "latitude") else np.nan for x in nmea_msg])
            if nmea_msg
            else [np.nan]
        )
        lon = (
            np.array([x.longitude if hasattr(x, "longitude") else np.nan for x in nmea_msg])
            if nmea_msg
            else [np.nan]
        )
        msg_type = (
            np.array([x.sentence_type if hasattr(x, "sentence_type") else np.nan for x in nmea_msg])
            if nmea_msg
            else [np.nan]
        )
        location_time = (
            (
                np.array(self.parser_obj.nmea["timestamp"])[idx_loc]
                - np.datetime64("1900-01-01T00:00:00")
            )
            / np.timedelta64(1, "s")
            if nmea_msg
            else [np.nan]
        )

        return location_time, msg_type, lat, lon

    def _beam_groups_vars(self):
        """Stage beam_group_name and beam_group_descr variables sharing a common dimension,
        beam_group, to be inserted in the Sonar group"""
        beam_groups_vars = {
            "beam_group_name": (
                ["beam_group"],
                [di["name"] for di in self._beamgroups],
                {"long_name": "Beam group name"},
            ),
            "beam_group_descr": (
                ["beam_group"],
                [di["descr"] for di in self._beamgroups],
                {"long_name": "Beam group description"},
            ),
        }

        return beam_groups_vars

    @staticmethod
    def _add_beam_dim(ds: xr.Dataset, sonar_model: str):
        """
        Adds ``beam`` as the last dimension to the appropriate
        variables in ``Sonar/Beam_groupX`` groups when necessary.

        Notes
        -----
        When expanding the dimension of a Dataarray, it is necessary
        to copy the array (hence the .copy()). This allows the array
        to be writable downstream (i.e. we can assign values to
        certain indices).

        To retain the attributes and encoding of ``beam``
        it is necessary to use .assign_coords() with ``beam``
        from ds.
        """

        # variables to add beam to
        add_beam_names = set(ds.variables).intersection(
            beam_only_names[sonar_model].union(beam_ping_time_names[sonar_model])
        )

        for var_name in add_beam_names:
            if "beam" in ds.dims:
                if "beam" not in ds[var_name].dims:
                    ds[var_name] = (
                        ds[var_name]
                        .expand_dims(dim={"beam": ds.beam}, axis=ds[var_name].ndim)
                        .assign_coords(beam=ds.beam)
                        .copy()
                    )
            else:
                # TODO: right now there is no attr or encoding for the beam dimension
                #  if this changes in the future, we need to add them here.
                # Add a single-value beam dimension
                ds[var_name] = (
                    ds[var_name]
                    .expand_dims(dim={"beam": np.array(["1"], dtype=str)}, axis=ds[var_name].ndim)
                    .copy()
                )

    @staticmethod
    def _add_ping_time_dim(ds: xr.Dataset, sonar_model: str):
        """
        Adds ``ping_time`` as the last dimension to the appropriate
        variables in ``Sonar/Beam_group1`` and ``Sonar/Beam_group2``
        (when necessary).

        Notes
        -----
        When expanding the dimension of a Dataarray, it is necessary
        to copy the array (hence the .copy()). This allows the array
        to be writable downstream (i.e. we can assign values to
        certain indices).

        To retain the attributes and encoding of ``ping_time``
        it is necessary to use .assign_coords() with ``ping_time``
        from ds.
        """

        # variables to add ping_time to
        add_ping_time_names = set(ds.variables).intersection(beam_ping_time_names[sonar_model])
        add_ping_time_names = add_ping_time_names.union(ping_time_only_names[sonar_model])

        ping_time = ds.ping_time.values

        for var_name in add_ping_time_names:

            ds[var_name] = (
                ds[var_name]
                .expand_dims(dim={"ping_time": ping_time}, axis=ds[var_name].ndim)
                .assign_coords(ping_time=ds.ping_time)
                .copy()
            )

    def beamgroups_to_convention(self, ds: xr.Dataset, sonar_model: str):
        """
        Manipulates variables in ``Sonar/Beam_groupX``
        to adhere to SONAR-netCDF4 vers. 1 with respect
        to the use of ``ping_time`` and ``beam`` dimensions.

        This does several things:
        1. Creates ``beam`` dimension and coordinate variable
        when not present
        2. Adds ``beam`` dimension to several variables
        when missing
        3. Adds ``ping_time`` dimension to several variables
        when missing

        Parameters
        ----------
        ds : xr.Dataset
            Dataset corresponding to ``Beam_groupX``.
        sonar_model : str
            Specifies the sensor that ds comes from.
        """

        # account for sensors that are not EK60, EK80, AD2CP, AZFP
        if sonar_model in ["EK60", "ES70"]:
            sensor = "EK60"
        elif sonar_model in ["EK80", "ES80", "EA640"]:
            sensor = "EK80"
        else:
            sensor = sonar_model

        self._add_ping_time_dim(ds, sensor)
        self._add_beam_dim(ds, sensor)
