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

from ..utils.io import check_file_existence, sanitize_file_path
from ..utils.repr import HtmlTemplate
from ..utils.uwa import calc_sound_speed
from .convention import _get_convention

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
    ):

        # TODO: consider if should open datasets in init
        #  or within each function call when echodata is used. Need to benchmark.

        self.storage_options: Dict[str, str] = (
            storage_options if storage_options is not None else {}
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

        This method only applies to `sonar_model`s of `"AZFP"`, `"EK60"`, and `"EK80"`.
        If the `sonar_model` is not `"AZFP"`, `"EK60"`, or `"EK80"`, an error is raised.

        Parameters
        ----------
        env_params: dict
            This dictionary should contain either:
            - `"sound_speed"`: `float`
            - `"temperature"`, `"salinity"`, and `"pressure"`: `float`s,
            in which case the sound speed will be calculated.
            If the `sonar_model` is `"EK60"` or `"EK80"`, and
            `EchoData.environment.sound_speed_indicative` exists, then this parameter
            does not need to be specified.
        azfp_cal_type : {"Sv", "Sp"}, optional
            - `"Sv"` for calculating volume backscattering strength
            - `"Sp"` for calculating point backscattering strength.
            This parameter is only used if `sonar_model` is `"AZFP"`,
            and in that case it must be specified.
        ek_waveform_mode : {"CW", "BB"}, optional
            - `"CW"` for CW-mode samples, either recorded as complex or power samples
            - `"BB"` for BB-mode samples, recorded as complex samples
            This parameter is only used if `sonar_model` is `"EK60"` or `"EK80"`,
            and in those cases it must be specified.
        ek_encode_mode : {"complex", "power"}, optional
            For EK80 data, range can be computed from complex or power samples.
            The type of sample used can be specified with this parameter.
            - `"complex"` to use complex samples
            - `"power"` to use power samples
            This parameter is only used if `sonar_model` is `"EK80"`.

        Returns
        -------
        xr.DataArray
            The range of the data in meters.

        Raises
        ------
        ValueError
            - When `sonar_model` is `"AZFP"` but `azfp_cal_type` is not specified or is `None`.
            - When `sonar_model` is `"EK60"` or `"EK80"` but `ek_waveform_mode`
            is not specified or is `None`.
            - When `sonar_model` is `"EK60"` but `waveform_mode` is `"BB"` (EK60 cannot have
            broadband samples).
            - When `sonar_model` is `"AZFP"` and `env_params` does not contain
            either `"sound_speed"` or all of `"temperature"`, `"salinity"`, and `"pressure"`.
            - When `sonar_model` is `"EK60"` or `"EK80"`,
            EchoData.environment.sound_speed_indicative does not exist,
            and `env_params` does not contain either `"sound_speed"` or all of `"temperature"`,
            `"salinity"`, and `"pressure"`.
            - When `sonar_model` is not `"AZFP"`, `"EK60"`, or `"EK80"`.
        """

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

            return range_meter
        elif self.sonar_model in ("EK60", "EK80"):
            waveform_mode = ek_waveform_mode
            encode_mode = ek_encode_mode

            if self.sonar_model == "EK60" and waveform_mode == "BB":
                raise ValueError("EK60 cannot have BB samples")

            if waveform_mode is None:
                raise ValueError(
                    "ek_waveform_mode must be specified when sonar_model is EK60 or EK80"
                )
            tvg_correction_factor = TVG_CORRECTION_FACTOR[self.sonar_model]

            if waveform_mode == "CW":
                if (
                    self.sonar_model == "EK80"
                    and encode_mode == "power"
                    and self.beam_power is not None
                ):
                    beam = self.beam_power
                else:
                    beam = self.beam

                sample_thickness = beam["sample_interval"] * sound_speed / 2
                # TODO: Check with the AFSC about the half sample difference
                range_meter = (
                    beam.range_bin - tvg_correction_factor
                ) * sample_thickness  # [frequency x range_bin]
            elif waveform_mode == "BB":
                # TODO: bug: right now only first ping_time has non-nan range
                shift = self.beam[
                    "transmit_duration_nominal"
                ]  # based on Lar Anderson's Matlab code
                # TODO: once we allow putting in arbitrary sound_speed,
                # change below to use linearly-interpolated values
                range_meter = (
                    (self.beam.range_bin * self.beam["sample_interval"] - shift)
                    * sound_speed
                    / 2
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
            range_meter.name = "range"  # add name to facilitate xr.merge

            return range_meter
        else:
            raise ValueError(
                "this method only supports sonar_model values of AZFP, EK60, and EK80"
            )

    def update_platform(self, extra_platform_data: xr.Dataset, time_dim="time"):
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
        The data inserted into the Platform group will be indexed by a dimension named `"time2"`.

        Parameters
        ----------
        extra_platform_data : xr.Dataset
            An `xr.Dataset` containing the additional platform data to be added
            to the `EchoData.platform` group.
        time_dim: str, default="time"
            The name of the time dimension in `extra_platform_data`; used for extracting
            data from `extra_platform_data`.

        Examples
        --------
        >>> ed = echopype.open_raw(raw_file, "EK60")
        >>> extra_platform_data = xr.open_dataset(extra_platform_data_file)
        >>> ed.update_platform(extra_platform_data)
        """

        # # only take data during ping times
        # start_time, end_time = min(self.beam["ping_time"]), max(self.beam["ping_time"])

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

        platform = self.platform.assign_coords(
            time2=extra_platform_data[time_dim].values
        )
        platform["time2"].attrs[
            "long_name"
        ] = "time dimension for external platform data"

        dropped_vars = []
        for var in ["pitch", "roll", "heave", "latitude", "longitude", "water_level"]:
            if var in platform and (~platform[var].isnull()).all():
                dropped_vars.append(var)
        if len(dropped_vars) > 0:
            warnings.warn(
                f"some variables in the original Platform group will be overwritten: {', '.join(dropped_vars)}"  # noqa
            )
        platform = platform.drop_vars(
            ["pitch", "roll", "heave", "latitude", "longitude", "water_level"],
            errors="ignore",
        )

        num_obs = len(extra_platform_data[time_dim])

        def mapping_search_variable(mapping, keys, default=None):
            for key in keys:
                if key in mapping:
                    return mapping[key].data
            return default

        self.platform = platform.update(
            {
                "pitch": (
                    "time2",
                    mapping_search_variable(
                        extra_platform_data,
                        ["pitch", "PITCH"],
                        platform.get("pitch", np.full(num_obs, np.nan)),
                    ),
                ),
                "roll": (
                    "time2",
                    mapping_search_variable(
                        extra_platform_data,
                        ["roll", "ROLL"],
                        platform.get("roll", np.full(num_obs, np.nan)),
                    ),
                ),
                "heave": (
                    "time2",
                    mapping_search_variable(
                        extra_platform_data,
                        ["heave", "HEAVE"],
                        platform.get("heave", np.full(num_obs, np.nan)),
                    ),
                ),
                "latitude": (
                    "time2",
                    mapping_search_variable(
                        extra_platform_data,
                        ["lat", "latitude", "LATITUDE"],
                        default=platform.get("latitude", np.full(num_obs, np.nan)),
                    ),
                ),
                "longitude": (
                    "time2",
                    mapping_search_variable(
                        extra_platform_data,
                        ["lon", "longitude", "LONGITUDE"],
                        default=platform.get("longitude", np.full(num_obs, np.nan)),
                    ),
                ),
                "water_level": (
                    "time2",
                    mapping_search_variable(
                        extra_platform_data,
                        ["water_level", "WATER_LEVEL"],
                        default=platform.get("water_level", np.zeros(num_obs)),
                    ),
                ),
            }
        )

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
        return xr.open_dataset(filepath, group=group, engine=XARRAY_ENGINE_MAP[suffix])

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
