import datetime
import uuid
import warnings
from collections import OrderedDict
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import fsspec
import numpy as np
import xarray as xr
from zarr.errors import GroupNotFoundError, PathNotFoundError

if TYPE_CHECKING:
    from ..core import EngineHint, FileFormatHint, PathHint, SonarModelsHint

from ..calibrate.calibrate_base import EnvParams
from ..utils.coding import set_encodings
from ..utils.io import check_file_existence, sanitize_file_path
from ..utils.repr import HtmlTemplate
from ..utils.uwa import calc_sound_speed
from .convention import _get_convention
from .convention.attrs import DEFAULT_PLATFORM_COORD_ATTRS

XARRAY_ENGINE_MAP: Dict["FileFormatHint", "EngineHint"] = {
    ".nc": "netcdf4",
    ".zarr": "zarr",
}

TVG_CORRECTION_FACTOR = {
    "EK60": 2,
    "EK80": 0,
}


class EchoData:
    """Echo data model class for handling raw converted data,
    including multiple files associated with the same data set.
    """

    group_map = OrderedDict(_get_convention()["groups"])

    def __init__(
        self,
        converted_raw_path: Optional["PathHint"] = None,
        storage_options: Dict[str, str] = None,
        source_file: Optional["PathHint"] = None,
        xml_path: "PathHint" = None,
        sonar_model: "SonarModelsHint" = None,
        open_kwargs: Dict[str, str] = None,
    ):

        # TODO: consider if should open datasets in init
        #  or within each function call when echodata is used. Need to benchmark.

        self.storage_options: Dict[str, str] = (
            storage_options if storage_options is not None else {}
        )
        self.open_kwargs: Dict[str, str] = (
            open_kwargs if open_kwargs is not None else {}
        )
        self.source_file: Optional["PathHint"] = source_file
        self.xml_path: Optional["PathHint"] = xml_path
        self.sonar_model: Optional["SonarModelsHint"] = sonar_model
        self.converted_raw_path: Optional["PathHint"] = None

        self.__setup_groups()
        self.__read_converted(converted_raw_path)

    def __repr__(self) -> str:
        """Make string representation of InferenceData object."""
        existing_groups = [
            f"{group}: ({self.group_map[group]['name']}) {self.group_map[group]['description']}"  # noqa
            for group in self.group_map.keys()
            if isinstance(getattr(self, group), xr.Dataset)
        ]
        fpath = "Internal Memory"
        if self.converted_raw_path:
            fpath = self.converted_raw_path
        msg = "EchoData: standardized raw data from {file_path}\n  > {options}".format(
            options="\n  > ".join(existing_groups),
            file_path=fpath,
        )
        return msg

    def _repr_html_(self) -> str:
        """Make html representation of InferenceData object."""
        try:
            from xarray.core.options import OPTIONS

            display_style = OPTIONS["display_style"]
            if display_style == "text":
                html_repr = f"<pre>{escape(repr(self))}</pre>"
            else:
                xr_collections = []
                for group in self.group_map.keys():
                    if isinstance(getattr(self, group), xr.Dataset):
                        xr_data = getattr(self, group)._repr_html_()
                        xr_collections.append(
                            HtmlTemplate.element_template.format(  # noqa
                                group_id=group + str(uuid.uuid4()),
                                group=group,
                                group_name=self.group_map[group]["name"],
                                group_description=self.group_map[group]["description"],
                                xr_data=xr_data,
                            )
                        )
                elements = "".join(xr_collections)
                fpath = "Internal Memory"
                if self.converted_raw_path:
                    fpath = self.converted_raw_path
                formatted_html_template = HtmlTemplate.html_template.format(
                    elements, file_path=str(fpath)
                )  # noqa
                css_template = HtmlTemplate.css_template  # noqa
                html_repr = "%(formatted_html_template)s%(css_template)s" % locals()
        except:  # noqa
            html_repr = f"<pre>{escape(repr(self))}</pre>"
        return html_repr

    def __setup_groups(self):
        for group in self.group_map.keys():
            setattr(self, group, None)

    def __read_converted(self, converted_raw_path: Optional["PathHint"]):
        if converted_raw_path is not None:
            self._check_path(converted_raw_path)
            converted_raw_path = self._sanitize_path(converted_raw_path)
            self._load_file(converted_raw_path)

            if isinstance(converted_raw_path, fsspec.FSMap):
                # Convert fsmap to Path so it can be used
                # for retrieving the path strings
                converted_raw_path = Path(converted_raw_path.root)

        self.converted_raw_path = converted_raw_path

    def compute_range(
        self,
        env_params=None,
        azfp_cal_type=None,
        ek_waveform_mode=None,
        ek_encode_mode="complex",
    ):
        """
        Computes the range of the data contained in this `EchoData` object, in meters.

        Currently this operation is supported for the following ``sonar_model``:
        EK60, AZFP, EK80 (see Notes below for detail).

        Parameters
        ----------
        env_params : dict
            Environmental parameters needed for computing sonar range.
            Users can supply `"sound speed"` directly,
            or specify other variables that can be used to compute them,
            including `"temperature"`, `"salinity"`, and `"pressure"`.

            For EK60 and EK80 echosounders, by default echopype uses
            environmental variables stored in the data files.
            For AZFP echosounder, all environmental parameters need to be supplied.
            AZFP echosounders typically are equipped with an internal temperature
            sensor, and some are equipped with a pressure sensor, but automatically
            using these pressure data is not currently supported.

        azfp_cal_type : {"Sv", "Sp"}, optional

            - `"Sv"` for calculating volume backscattering strength
            - `"Sp"` for calculating point backscattering strength.

            This parameter needs to be specified for data from the AZFP echosounder,
            due to a difference in computing range for Sv and Sp.

        ek_waveform_mode : {"CW", "BB"}, optional
            Type of transmit waveform.
            Required only for data from the EK80 echosounder.

            - `"CW"` for narrowband transmission,
              returned echoes recorded either as complex or power/angle samples
            - `"BB"` for broadband transmission,
              returned echoes recorded as complex samples

        ek_encode_mode : {"complex", "power"}, optional
            Type of encoded return echo data.
            Required only for data from the EK80 echosounder.

            - `"complex"` for complex samples
            - `"power"` for power/angle samples, only allowed when
              the echosounder is configured for narrowband transmission

        Returns
        -------
        xr.DataArray
            The range of the data in meters.

        Raises
        ------
        ValueError
            - When `sonar_model` is `"AZFP"` but `azfp_cal_type` is not specified or is `None`.
            - When `sonar_model` is `"EK80"` but `ek_waveform_mode` is not specified or is `None`.
            - When `sonar_model` is `"EK60"` but `waveform_mode` is `"BB"`
            - When `sonar_model` is `"AZFP"` and `env_params` does not contain
              either `"sound_speed"` or all of `"temperature"`, `"salinity"`, and `"pressure"`.
            - When `sonar_model` is `"EK60"` or `"EK80"`,
              EchoData.environment.sound_speed_indicative does not exist,
              and `env_params` does not contain either `"sound_speed"`
              or all of `"temperature"`, `"salinity"`, and `"pressure"`.
            - When `sonar_model` is not `"AZFP"`, `"EK60"`, or `"EK80"`.

        Notes
        -----
        The EK80 echosounder can be configured to transmit
        either broadband (``waveform_mode="BB"``)
        or narrowband (``waveform_mode="CW"``) signals.
        When transmitting in broadband mode, the returned echoes are
        encoded as complex samples (``encode_mode="complex"``).
        When transmitting in narrowband mode, the returned echoes can be encoded
        either as complex samples (``encode_mode="complex"``)
        or as power/angle combinations (``encode_mode="power"``) in a format
        similar to those recorded by EK60 echosounders.

        For AZFP echosounder, the returned range is duplicated along `ping_time`
        to conform with outputs from other echosounders, even though within each data
        file the range is held constant.
        """

        if isinstance(env_params, EnvParams):
            env_params = env_params._apply(self)

        def squeeze_non_scalar(n):
            if not np.isscalar(n):
                n = n.squeeze()
            return n

        if "sound_speed" in env_params:
            sound_speed = squeeze_non_scalar(env_params["sound_speed"])
        elif all(
            [param in env_params for param in ("temperature", "salinity", "pressure")]
        ):
            sound_speed = calc_sound_speed(
                squeeze_non_scalar(env_params["temperature"]),
                squeeze_non_scalar(env_params["salinity"]),
                squeeze_non_scalar(env_params["pressure"]),
                formula_source="AZFP" if self.sonar_model == "AZFP" else "Mackenzie",
            )
        elif (
            self.sonar_model in ("EK60", "EK80")
            and "sound_speed_indicative" in self.environment
        ):
            sound_speed = squeeze_non_scalar(self.environment["sound_speed_indicative"])
        else:
            raise ValueError(
                "sound speed must be specified in env_params, "
                "with temperature/salinity/pressure in env_params to be calculated, "
                "or in EchoData.environment.sound_speed_indicative for EK60 and EK80 sonar models"
            )

        # AZFP
        if self.sonar_model == "AZFP":
            cal_type = azfp_cal_type
            if cal_type is None:
                raise ValueError(
                    "azfp_cal_type must be specified when sonar_model is AZFP"
                )

            # Notation below follows p.86 of user manual
            N = self.vendor["number_of_samples_per_average_bin"]  # samples per bin
            f = self.vendor["digitization_rate"]  # digitization rate
            L = self.vendor["lockout_index"]  # number of lockout samples

            # keep this in ref of AZFP matlab code,
            # set to 1 since we want to calculate from raw data
            bins_to_avg = 1

            # Calculate range using parameters for each freq
            # This is "the range to the centre of the sampling volume
            # for bin m" from p.86 of user manual
            if cal_type == "Sv":
                range_offset = 0
            else:
                range_offset = (
                    sound_speed * self.beam["transmit_duration_nominal"] / 4
                )  # from matlab code
            range_meter = (
                sound_speed * L / (2 * f)
                + (sound_speed / 4)
                * (
                    ((2 * (self.beam.range_bin + 1) - 1) * N * bins_to_avg - 1) / f
                    + self.beam["transmit_duration_nominal"]
                )
                - range_offset
            )
            range_meter.name = "range"  # add name to facilitate xr.merge

            # duplicate range for all ping_times for consistency with EK case
            # if it is not indexed by ping_time then echopype.preprocess.compute_MVBS will fail
            #   because it expects the range included in the Sv/Sp dataset
            #   to be indexed by ping_time
            range_meter = range_meter.expand_dims(
                {"ping_time": self.beam["ping_time"]}, axis=1
            )

            return range_meter

        # EK
        elif self.sonar_model in ("EK60", "EK80"):
            waveform_mode = ek_waveform_mode
            encode_mode = ek_encode_mode

            # EK60 can only be CW mode
            if self.sonar_model == "EK60":
                if waveform_mode is None:
                    waveform_mode = "CW"  # default to CW mode
                elif waveform_mode != "CW":
                    raise ValueError("EK60 must have CW samples")

            # EK80 needs waveform_mode specification
            if self.sonar_model == "EK80" and waveform_mode is None:
                raise ValueError(
                    "ek_waveform_mode must be specified when sonar_model is EK80"
                )

            # TVG correction factor changes depending when the echo recording starts
            # wrt when the transmit signal is sent out.
            # This implementation is different for EK60 and EK80.
            tvg_correction_factor = TVG_CORRECTION_FACTOR[self.sonar_model]

            if waveform_mode == "CW":
                if (
                    self.sonar_model == "EK80"
                    and encode_mode == "power"
                    and self.beam_power is not None
                ):
                    # if both CW and BB exist and beam_power group is not empty
                    # this means that CW is recorded in power/angle mode
                    beam = self.beam_power
                else:
                    beam = self.beam

                sample_thickness = beam["sample_interval"] * sound_speed / 2
                # TODO: Check with the AFSC about the half sample difference
                range_meter = (
                    beam.range_bin - tvg_correction_factor
                ) * sample_thickness  # [frequency x range_bin]
            elif waveform_mode == "BB":
                beam = self.beam  # always use the Beam group
                # TODO: bug: right now only first ping_time has non-nan range
                shift = beam[
                    "transmit_duration_nominal"
                ]  # based on Lar Anderson's Matlab code
                # TODO: once we allow putting in arbitrary sound_speed,
                # change below to use linearly-interpolated values
                range_meter = (
                    (beam.range_bin * beam["sample_interval"] - shift) * sound_speed / 2
                )
                # TODO: Lar Anderson's code include a slicing by minRange with a default of 0.02 m,
                #  need to ask why and see if necessary here
            else:
                raise ValueError("Input waveform_mode not recognized!")

            # make order of dims conform with the order of backscatter data
            range_meter = range_meter.transpose("frequency", "ping_time", "range_bin")
            range_meter = range_meter.where(
                range_meter > 0, 0
            )  # set negative ranges to 0

            # set entries with NaN backscatter data to NaN
            if "quadrant" in beam["backscatter_r"].dims:
                valid_idx = (
                    ~beam["backscatter_r"].isel(quadrant=0).drop("quadrant").isnull()
                )
            else:
                valid_idx = ~beam["backscatter_r"].isnull()
            range_meter = range_meter.where(valid_idx)

            # add name to facilitate xr.merge
            range_meter.name = "range"

            return range_meter

        # OTHERS
        else:
            raise ValueError(
                "this method only supports the following sonar_model: AZFP, EK60, and EK80"
            )

    def update_platform(
        self,
        extra_platform_data: xr.Dataset,
        time_dim="time",
        extra_platform_data_file_name=None,
    ):
        """
        Updates the `EchoData.platform` group with additional external platform data.

        `extra_platform_data` must be an xarray Dataset.
        The name of the time dimension in `extra_platform_data` is specified by the
        `time_dim` parameter.
        Data is extracted from `extra_platform_data` by variable name; only the data
        in `extra_platform_data` with the following variable names will be used:
            - `"pitch"`
            - `"roll"`
            - `"heave"`
            - `"latitude"`
            - `"longitude"`
            - `"water_level"`
        The data inserted into the Platform group will be indexed by a dimension named
        `"location_time"`.

        Parameters
        ----------
        extra_platform_data : xr.Dataset
            An `xr.Dataset` containing the additional platform data to be added
            to the `EchoData.platform` group.
        time_dim: str, default="time"
            The name of the time dimension in `extra_platform_data`; used for extracting
            data from `extra_platform_data`.
        extra_platform_data_file_name: str, default=None
            File name for source of extra platform data, if read from a file

        Examples
        --------
        >>> ed = echopype.open_raw(raw_file, "EK60")
        >>> extra_platform_data = xr.open_dataset(extra_platform_data_file)
        >>> ed.update_platform(extra_platform_data,
        >>>         extra_platform_data_file_name=extra_platform_data_file)
        """

        # Handle data stored as a CF Trajectory Discrete Sampling Geometry
        # http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#trajectory-data
        # The Saildrone sample data file follows this convention
        if (
            "featureType" in extra_platform_data.attrs
            and extra_platform_data.attrs["featureType"].lower() == "trajectory"
        ):
            for coordvar in extra_platform_data.coords:
                if (
                    "cf_role" in extra_platform_data[coordvar].attrs
                    and extra_platform_data[coordvar].attrs["cf_role"]
                    == "trajectory_id"
                ):
                    trajectory_var = coordvar

            # assumes there's only one trajectory in the dataset (index 0)
            extra_platform_data = extra_platform_data.sel(
                {trajectory_var: extra_platform_data[trajectory_var][0]}
            )
            extra_platform_data = extra_platform_data.drop_vars(trajectory_var)
            extra_platform_data = extra_platform_data.swap_dims({"obs": time_dim})

        # clip incoming time to 1 less than min of EchoData.beam["ping_time"] and
        #   1 greater than max of EchoData.beam["ping_time"]
        # account for unsorted external time by checking whether each time value is between
        #   min and max ping_time instead of finding the 2 external times corresponding to the
        #   min and max ping_time and taking all the times between those indices
        sorted_external_time = extra_platform_data[time_dim].data
        sorted_external_time.sort()
        # fmt: off
        min_index = max(
            np.searchsorted(
                sorted_external_time, self.beam["ping_time"].min(), side="left"
            ) - 1,
            0,
        )
        # fmt: on
        max_index = min(
            np.searchsorted(
                sorted_external_time, self.beam["ping_time"].max(), side="right"
            ),
            len(sorted_external_time) - 1,
        )
        extra_platform_data = extra_platform_data.sel(
            {
                time_dim: np.logical_and(
                    sorted_external_time[min_index] <= extra_platform_data[time_dim],
                    extra_platform_data[time_dim] <= sorted_external_time[max_index],
                )
            }
        )

        platform = self.platform
        platform_vars_attrs = {var: platform[var].attrs for var in platform.variables}
        platform = platform.drop_dims(["location_time"], errors="ignore")
        # drop_dims is also dropping latitude, longitude and sentence_type why?
        platform = platform.assign_coords(
            location_time=extra_platform_data[time_dim].values
        )
        history_attr = (
            f"{datetime.datetime.utcnow()} +00:00. Added from external platform data"
        )
        if extra_platform_data_file_name:
            history_attr += ", from file " + extra_platform_data_file_name
        location_time_attrs = {
            **DEFAULT_PLATFORM_COORD_ATTRS["location_time"],
            **{"history": history_attr},
        }
        platform["location_time"] = platform["location_time"].assign_attrs(
            **location_time_attrs
        )

        dropped_vars_target = [
            "pitch",
            "roll",
            "heave",
            "latitude",
            "longitude",
            "water_level",
        ]
        dropped_vars = []
        for var in dropped_vars_target:
            if var in platform and (~platform[var].isnull()).all():
                dropped_vars.append(var)
        if len(dropped_vars) > 0:
            warnings.warn(
                f"Some variables in the original Platform group will be overwritten: {', '.join(dropped_vars)}"  # noqa
            )
        platform = platform.drop_vars(
            dropped_vars_target,
            errors="ignore",
        )

        num_obs = len(extra_platform_data[time_dim])

        def mapping_search_variable(mapping, keys, default=None):
            for key in keys:
                if key in mapping:
                    return mapping[key].data
            return default

        platform = platform.update(
            {
                "pitch": (
                    "location_time",
                    mapping_search_variable(
                        extra_platform_data,
                        ["pitch", "PITCH"],
                        platform.get("pitch", np.full(num_obs, np.nan)),
                    ),
                ),
                "roll": (
                    "location_time",
                    mapping_search_variable(
                        extra_platform_data,
                        ["roll", "ROLL"],
                        platform.get("roll", np.full(num_obs, np.nan)),
                    ),
                ),
                "heave": (
                    "location_time",
                    mapping_search_variable(
                        extra_platform_data,
                        ["heave", "HEAVE"],
                        platform.get("heave", np.full(num_obs, np.nan)),
                    ),
                ),
                "latitude": (
                    "location_time",
                    mapping_search_variable(
                        extra_platform_data,
                        ["lat", "latitude", "LATITUDE"],
                        default=platform.get("latitude", np.full(num_obs, np.nan)),
                    ),
                ),
                "longitude": (
                    "location_time",
                    mapping_search_variable(
                        extra_platform_data,
                        ["lon", "longitude", "LONGITUDE"],
                        default=platform.get("longitude", np.full(num_obs, np.nan)),
                    ),
                ),
                "water_level": (
                    "location_time",
                    mapping_search_variable(
                        extra_platform_data,
                        ["water_level", "WATER_LEVEL"],
                        default=platform.get("water_level", np.zeros(num_obs)),
                    ),
                ),
            }
        )
        for var in dropped_vars_target:
            platform[var] = platform[var].assign_attrs(**platform_vars_attrs[var])

        self.platform = set_encodings(platform)

    @classmethod
    def _load_convert(cls, convert_obj):
        new_cls = cls()
        for group in new_cls.group_map.keys():
            if hasattr(convert_obj, group):
                setattr(new_cls, group, getattr(convert_obj, group))

        setattr(new_cls, "sonar_model", getattr(convert_obj, "sonar_model"))
        setattr(new_cls, "source_file", getattr(convert_obj, "source_file"))
        return new_cls

    def _load_file(self, raw_path: "PathHint"):
        """Lazy load Top-level, Beam, Environment, and Vendor groups from raw file."""
        for group, value in self.group_map.items():
            # EK80 data may have a Beam_power group if both complex and power data exist.
            ds = None
            try:
                ds = self._load_group(
                    raw_path,
                    group=value["ep_group"],
                )
            except (OSError, GroupNotFoundError, PathNotFoundError):
                # Skips group not found errors for EK80 and ADCP
                ...
            if group == "top" and hasattr(ds, "keywords"):
                self.sonar_model = ds.keywords.upper()  # type: ignore

            if isinstance(ds, xr.Dataset):
                setattr(self, group, ds)

    def _check_path(self, filepath: "PathHint"):
        """Check if converted_raw_path exists"""
        file_exists = check_file_existence(filepath, self.storage_options)
        if not file_exists:
            raise FileNotFoundError(f"There is no file named {filepath}")

    def _sanitize_path(self, filepath: "PathHint") -> "PathHint":
        filepath = sanitize_file_path(filepath, self.storage_options)
        return filepath

    def _check_suffix(self, filepath: "PathHint") -> "FileFormatHint":
        """Check if file type is supported."""
        # TODO: handle multiple files through the same set of checks for combining files
        if isinstance(filepath, fsspec.FSMap):
            suffix = Path(filepath.root).suffix
        else:
            suffix = Path(filepath).suffix

        if suffix not in XARRAY_ENGINE_MAP:
            raise ValueError("Input file type not supported!")

        return suffix  # type: ignore

    def _load_group(self, filepath: "PathHint", group: Optional[str] = None):
        """Loads each echodata group"""
        suffix = self._check_suffix(filepath)
        return xr.open_dataset(
            filepath, group=group, engine=XARRAY_ENGINE_MAP[suffix], **self.open_kwargs
        )

    def to_netcdf(self, save_path: Optional["PathHint"] = None, **kwargs):
        """Save content of EchoData to netCDF.

        Parameters
        ----------
        save_path : str
            path that converted .nc file will be saved
        compress : bool
            whether or not to perform compression on data variables
            Defaults to ``True``
        overwrite : bool
            whether or not to overwrite existing files
            Defaults to ``False``
        parallel : bool
            whether or not to use parallel processing. (Not yet implemented)
        output_storage_options : dict
            Additional keywords to pass to the filesystem class.
        """
        from ..convert.api import to_file

        return to_file(self, "netcdf4", save_path=save_path, **kwargs)

    def to_zarr(self, save_path: Optional["PathHint"] = None, **kwargs):
        """Save content of EchoData to zarr.

        Parameters
        ----------
        save_path : str
            path that converted .nc file will be saved
        compress : bool
            whether or not to perform compression on data variables
            Defaults to ``True``
        overwrite : bool
            whether or not to overwrite existing files
            Defaults to ``False``
        parallel : bool
            whether or not to use parallel processing. (Not yet implemented)
        output_storage_options : dict
            Additional keywords to pass to the filesystem class.
        """
        from ..convert.api import to_file

        return to_file(self, "zarr", save_path=save_path, **kwargs)

    # TODO: Remove below in future versions. They are for supporting old API calls.
    @property
    def nc_path(self) -> Optional["PathHint"]:
        warnings.warn(
            "`nc_path` is deprecated, Use `converted_raw_path` instead.",
            DeprecationWarning,
            2,
        )
        if self.converted_raw_path.endswith(".nc"):
            return self.converted_raw_path
        else:
            path = Path(self.converted_raw_path)
            return str(path.parent / (path.stem + ".nc"))

    @property
    def zarr_path(self) -> Optional["PathHint"]:
        warnings.warn(
            "`zarr_path` is deprecated, Use `converted_raw_path` instead.",
            DeprecationWarning,
            2,
        )
        if self.converted_raw_path.endswith(".zarr"):
            return self.converted_raw_path
        else:
            path = Path(self.converted_raw_path)
            return str(path.parent / (path.stem + ".zarr"))

    def raw2nc(
        self,
        save_path: "PathHint" = None,
        combine_opt: bool = False,
        overwrite: bool = False,
        compress: bool = True,
    ):
        warnings.warn(
            "`raw2nc` is deprecated, use `to_netcdf` instead.",
            DeprecationWarning,
            2,
        )
        return self.to_netcdf(
            save_path=save_path,
            compress=compress,
            combine=combine_opt,
            overwrite=overwrite,
        )

    def raw2zarr(
        self,
        save_path: "PathHint" = None,
        combine_opt: bool = False,
        overwrite: bool = False,
        compress: bool = True,
    ):
        warnings.warn(
            "`raw2zarr` is deprecated, use `to_zarr` instead.",
            DeprecationWarning,
            2,
        )
        return self.to_zarr(
            save_path=save_path,
            compress=compress,
            combine=combine_opt,
            overwrite=overwrite,
        )
