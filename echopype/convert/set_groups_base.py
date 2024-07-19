import abc
import warnings
from typing import List, Set

import numpy as np
import pynmea2
import xarray as xr

from ..echodata.convention import sonarnetcdf_1
from ..utils.coding import COMPRESSION_SETTINGS, DEFAULT_TIME_ENCODING, set_time_encodings
from ..utils.prov import echopype_prov_attrs, source_files_vars

NMEA_SENTENCE_DEFAULT = ["GGA", "GLL", "RMC"]


class SetGroupsBase(abc.ABC):
    """Base class for saving groups to netcdf or zarr from echosounder data files."""

    def __init__(
        self,
        parser_obj,
        input_file,
        xml_path,
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
        self.xml_path = xml_path
        self.output_path = output_path
        self.engine = engine
        self.compress = compress
        self.overwrite = overwrite

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
        }
        # Save
        ds = xr.Dataset()
        ds = ds.assign_attrs(tl_dict)
        return ds

    def set_provenance(self) -> xr.Dataset:
        """Set the Provenance group."""
        prov_dict = echopype_prov_attrs(process_type="conversion")
        files_vars = source_files_vars(self.input_file, self.xml_path)
        if files_vars["meta_source_files_var"] is None:
            source_vars = files_vars["source_files_var"]
        else:
            source_vars = {**files_vars["source_files_var"], **files_vars["meta_source_files_var"]}

        ds = xr.Dataset(
            data_vars=source_vars, coords=files_vars["source_files_coord"], attrs=prov_dict
        )

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

    def _nan_timestamp_handler(self, time_val) -> List:
        """
        Replace nan in time coordinate to avoid xarray warning.
        """
        if len(time_val) == 1 and np.isnan(time_val[0]):
            # set time_val to earliest ping_time among all channels
            if self.sonar_model in ["EK60", "ES70", "EK80", "ES80", "EA640"]:
                return [np.array([v[0] for v in self.parser_obj.ping_time.values()]).min()]
            elif self.sonar_model in ["AZFP", "AZFP6"]:
                return [self.parser_obj.ping_time[0]]
            else:
                return NotImplementedError(
                    f"Nan timestamp handling has not been implemented for {self.sonar_model}!"
                )
        else:
            return time_val

    def set_nmea(self) -> xr.Dataset:
        """Set the Platform/NMEA group."""
        # Save nan if nmea data is not encoded in the raw file
        if len(self.parser_obj.nmea["nmea_string"]) != 0:
            # Convert np.datetime64 numbers to nanoseconds since 1970-01-01 00:00:00Z
            # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
            # print(np.array(self.parser_obj.nmea["timestamp"])[idx_loc].shape)
            time, _, _ = xr.coding.times.encode_cf_datetime(
                self.parser_obj.nmea["timestamp"],
                **{
                    "units": DEFAULT_TIME_ENCODING["units"],
                    "calendar": DEFAULT_TIME_ENCODING["calendar"],
                },
            )
            raw_nmea = self.parser_obj.nmea["nmea_string"]
        else:
            time = [np.nan]
            raw_nmea = [np.nan]

        # Handle potential nan timestamp for time
        time = self._nan_timestamp_handler(time)

        ds = xr.Dataset(
            {
                "NMEA_datagram": (
                    ["time1"],
                    raw_nmea,
                    {"long_name": "NMEA datagram"},
                )
            },
            coords={
                "time1": (
                    ["time1"],
                    time,
                    {
                        "axis": "T",
                        "long_name": "Timestamps for NMEA datagrams",
                        "standard_name": "time",
                        "comment": "Time coordinate corresponding to NMEA sensor data.",
                    },
                )
            },
            attrs={"description": "All NMEA sensor datagrams"},
        )

        return set_time_encodings(ds)

    @abc.abstractmethod
    def set_vendor(self) -> xr.Dataset:
        """Set the Vendor_specific group."""
        raise NotImplementedError

    # TODO: move this to be part of parser as it is not a "set" operation
    def _extract_NMEA_latlon(self):
        """Get the lat and lon values from the raw nmea data"""
        messages = [string[3:6] for string in self.parser_obj.nmea["nmea_string"]]
        idx_loc = np.argwhere(np.isin(messages, NMEA_SENTENCE_DEFAULT)).squeeze()
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
        if nmea_msg:
            lat, lon = [], []
            for x in nmea_msg:
                try:
                    lat.append(x.latitude if hasattr(x, "latitude") else np.nan)
                except ValueError as ve:
                    lat.append(np.nan)
                    warnings.warn(
                        "At least one latitude entry is problematic and "
                        f"are assigned None in the converted data: {str(ve)}"
                    )
                try:
                    lon.append(x.longitude if hasattr(x, "longitude") else np.nan)
                except ValueError as ve:
                    lon.append(np.nan)
                    warnings.warn(
                        f"At least one longitude entry is problematic and "
                        f"are assigned None in the converted data: {str(ve)}"
                    )
        else:
            lat, lon = [np.nan], [np.nan]
        msg_type = (
            [x.sentence_type if hasattr(x, "sentence_type") else np.nan for x in nmea_msg]
            if nmea_msg
            else [np.nan]
        )
        if nmea_msg:
            time1, _, _ = xr.coding.times.encode_cf_datetime(
                np.array(self.parser_obj.nmea["timestamp"])[idx_loc],
                **{
                    "units": DEFAULT_TIME_ENCODING["units"],
                    "calendar": DEFAULT_TIME_ENCODING["calendar"],
                },
            )
        else:
            time1 = [np.nan]

        return time1, msg_type, lat, lon

    def _beam_groups_vars(self):
        """Stage beam_group coordinate and beam_group_descr variables sharing
        a common dimension, beam_group, to be inserted in the Sonar group"""
        beam_groups_vars = {
            "beam_group_descr": (
                ["beam_group"],
                [di["descr"] for di in self._beamgroups],
                {"long_name": "Beam group description"},
            ),
        }
        beam_groups_coord = {
            "beam_group": (
                ["beam_group"],
                [di["name"] for di in self._beamgroups],
                {"long_name": "Beam group name"},
            ),
        }

        return beam_groups_vars, beam_groups_coord

    @staticmethod
    def _add_beam_dim(ds: xr.Dataset, beam_only_names: Set[str], beam_ping_time_names: Set[str]):
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
        add_beam_names = set(ds.variables).intersection(beam_only_names.union(beam_ping_time_names))

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
                # Add a single-value beam dimension and its attributes
                ds[var_name] = (
                    ds[var_name]
                    .expand_dims(dim={"beam": np.array(["1"], dtype=str)}, axis=ds[var_name].ndim)
                    .copy()
                )
                ds[var_name].beam.attrs = sonarnetcdf_1.yaml_dict["variable_and_varattributes"][
                    "beam_coord_default"
                ]["beam"]

    @staticmethod
    def _add_ping_time_dim(
        ds: xr.Dataset, beam_ping_time_names: Set[str], ping_time_only_names: Set[str]
    ):
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
        add_ping_time_names = (
            set(ds.variables).intersection(beam_ping_time_names).union(ping_time_only_names)
        )

        for var_name in add_ping_time_names:
            ds[var_name] = (
                ds[var_name]
                .expand_dims(dim={"ping_time": ds.ping_time}, axis=ds[var_name].ndim)
                .assign_coords(ping_time=ds.ping_time)
                .copy()
            )

    def beam_groups_to_convention(
        self,
        ds: xr.Dataset,
        beam_only_names: Set[str],
        beam_ping_time_names: Set[str],
        ping_time_only_names: Set[str],
    ):
        """
        Manipulates variables in ``Sonar/Beam_groupX``
        to adhere to SONAR-netCDF4 vers. 1 with respect
        to the use of ``ping_time`` and ``beam`` dimensions.

        This does several things:
        1. Creates ``beam`` dimension and coordinate variable
        when not present.
        2. Adds ``beam`` dimension to several variables
        when missing.
        3. Adds ``ping_time`` dimension to several variables
        when missing.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset corresponding to ``Beam_groupX``.
        beam_only_names : Set[str]
            Variables that need only the beam dimension added to them.
        beam_ping_time_names : Set[str]
            Variables that need beam and ping_time dimensions added to them.
        ping_time_only_names : Set[str]
            Variables that need only the ping_time dimension added to them.
        """

        self._add_ping_time_dim(ds, beam_ping_time_names, ping_time_only_names)
        self._add_beam_dim(ds, beam_only_names, beam_ping_time_names)

    def _add_index_data_to_platform_ds(
        self,
        platform_ds: xr.Dataset,
    ) -> xr.Dataset:
        """
        Append index data from IDX file to the `Platform` dataset.
        Index file data contains latitude, longitude, and vessel distance traveled.

        Parameters
        ----------
        platform_ds : xr.Dataset
            `Platform` dataset without IDX data.

        Returns
        -------
        platform_ds : xr.Dataset
            `Platform` dataset with IDX data.
            Contains new `time4` dimension to correspond with IDX timestamps that
            align with `vessel_distance`, `idx_latitude`, and `idx_longitude`.

        Notes
        -----
        This function is only called for EK60/EK80 conversion.
        """
        timestamp_array, _, _ = xr.coding.times.encode_cf_datetime(
            np.array(self.parser_obj.idx["timestamp"]),
            **{
                "units": DEFAULT_TIME_ENCODING["units"],
                "calendar": DEFAULT_TIME_ENCODING["calendar"],
            },
        )
        # TODO: Add attributes for `ping_number` and `file_offset`
        platform_ds = platform_ds.assign(
            {
                "ping_number_idx": xr.DataArray(
                    np.array(self.parser_obj.idx["ping_number"]),
                    dims=("time4"),
                    coords={"time4": timestamp_array},
                ),
                "file_offset_idx": xr.DataArray(
                    np.array(self.parser_obj.idx["file_offset"]),
                    dims=("time4"),
                    coords={"time4": timestamp_array},
                ),
                "vessel_distance_idx": xr.DataArray(
                    np.array(self.parser_obj.idx["vessel_distance"]),
                    dims=("time4"),
                    coords={"time4": timestamp_array},
                    attrs={
                        "long_name": "Vessel distance in nautical miles (nmi) from start of "
                        + "recording.",
                        "comment": "Data from the IDX datagrams. Aligns time-wise with this "
                        + "dataset's `time4` dimension.",
                    },
                ),
                "latitude_idx": xr.DataArray(
                    np.array(self.parser_obj.idx["latitude"]),
                    dims=("time4"),
                    coords={"time4": timestamp_array},
                    attrs={
                        "long_name": "Index File Derived Platform Latitude",
                        "comment": "Data from the IDX datagrams. Aligns time-wise with this "
                        + "dataset's `time4` dimension. "
                        + "This is different from latitude stored in the NMEA datagram.",
                    },
                ),
                "longitude_idx": xr.DataArray(
                    np.array(self.parser_obj.idx["longitude"]),
                    dims=("time4"),
                    coords={"time4": timestamp_array},
                    attrs={
                        "long_name": "Index File Derived Platform Longitude",
                        "comment": "Data from the IDX datagrams. Aligns time-wise with this "
                        + "dataset's `time4` dimension. "
                        + "This is different from longitude from the NMEA datagram.",
                    },
                ),
            }
        )
        platform_ds["time4"] = platform_ds["time4"].assign_attrs(
            {
                "axis": "T",
                "long_name": "Timestamps from the IDX datagrams",
                "standard_name": "time",
                "comment": "Time coordinate corresponding to index file vessel "
                + "distance and latitude/longitude data.",
            }
        )

        return platform_ds.transpose(
            "channel", "time1", "time2", "time3", "time4", missing_dims="ignore"
        )

    def _add_seafloor_detection_data_to_vendor_ds(
        self,
        vendor_ds: xr.Dataset,
    ) -> xr.Dataset:
        """
        Append seafloor detection data from `.BOT` file to the `Vendor_specific` dataset.

        Parameters
        ----------
        vendor_ds : xr.Dataset
            `Vendor_specific` dataset without `.BOT` data.

        Returns
        -------
        vendor_ds : xr.Dataset
            `Vendor_specific` dataset with `.BOT` data.
            Contains new `ping_time` dimension to correspond with `detected_seafloor_depth`.
            Note that `detected_seafloor_depth` values corresponding to the same `ping_time`
            may have differing values along `channel`.

        Notes
        -----
        This function is only called for EK60/EK80 conversion.
        """
        timestamp_array, _, _ = xr.coding.times.encode_cf_datetime(
            np.array(self.parser_obj.bot["timestamp"]),
            **{
                "units": DEFAULT_TIME_ENCODING["units"],
                "calendar": DEFAULT_TIME_ENCODING["calendar"],
            },
        )
        vendor_ds = vendor_ds.assign(
            {
                "detected_seafloor_depth": xr.DataArray(
                    np.array(self.parser_obj.bot["depth"]).T,
                    dims=("channel", "ping_time"),
                    coords={"ping_time": timestamp_array},
                    attrs={
                        "long_name": "Echosounder detected seafloor depth from the BOT datagrams."
                    },
                )
            }
        )
        vendor_ds["ping_time"] = vendor_ds["ping_time"].assign_attrs(
            {
                "long_name": "Timestamps from the BOT datagrams",
                "standard_name": "time",
                "axis": "T",
                "comment": "Time coordinate corresponding to seafloor detection data.",
            }
        )
        vendor_ds = set_time_encodings(vendor_ds)

        return vendor_ds
