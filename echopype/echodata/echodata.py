import datetime
import shutil
import warnings
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple, Union

import fsspec
import numpy as np
import xarray as xr
from datatree import DataTree, open_datatree
from zarr.errors import GroupNotFoundError, PathNotFoundError

if TYPE_CHECKING:
    from ..core import EngineHint, FileFormatHint, PathHint, SonarModelsHint

from ..utils.coding import sanitize_dtypes, set_time_encodings
from ..utils.io import check_file_existence, sanitize_file_path
from ..utils.log import _init_logger
from ..utils.prov import add_processing_level
from .convention import sonarnetcdf_1
from .widgets.utils import tree_repr
from .widgets.widgets import _load_static_files, get_template

XARRAY_ENGINE_MAP: Dict["FileFormatHint", "EngineHint"] = {
    ".nc": "netcdf4",
    ".zarr": "zarr",
}

TVG_CORRECTION_FACTOR = {
    "EK60": 2,
    "ES70": 2,
    "EK80": 0,
    "ES80": 0,
    "EA640": 0,
}

logger = _init_logger(__name__)


class EchoData:
    """Echo data model class for handling raw converted data,
    including multiple files associated with the same data set.
    """

    group_map: Dict[str, Any] = sonarnetcdf_1.yaml_dict["groups"]

    def __init__(
        self,
        converted_raw_path: Optional["PathHint"] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        source_file: Optional["PathHint"] = None,
        xml_path: Optional["PathHint"] = None,
        sonar_model: Optional["SonarModelsHint"] = None,
        open_kwargs: Optional[Dict[str, Any]] = None,
        parsed2zarr_obj=None,
    ):
        # TODO: consider if should open datasets in init
        #  or within each function call when echodata is used. Need to benchmark.

        self.storage_options: Dict[str, Any] = (
            storage_options if storage_options is not None else {}
        )
        self.open_kwargs: Dict[str, Any] = open_kwargs if open_kwargs is not None else {}
        self.source_file: Optional["PathHint"] = source_file
        self.xml_path: Optional["PathHint"] = xml_path
        self.sonar_model: Optional["SonarModelsHint"] = sonar_model
        self.converted_raw_path: Optional["PathHint"] = converted_raw_path
        self._tree: Optional["DataTree"] = None

        # object associated with directly writing to a zarr file
        self.parsed2zarr_obj = parsed2zarr_obj

        self.__setup_groups()
        # self.__read_converted(converted_raw_path)

        self._varattrs = sonarnetcdf_1.yaml_dict["variable_and_varattributes"]

    def __del__(self):
        # TODO: this destructor seems to not work in Jupyter Lab if restart or
        #  even clear all outputs is used. It will work if you explicitly delete the object

        if (self.parsed2zarr_obj is not None) and (self.parsed2zarr_obj.zarr_file_name is not None):
            # get Path object of temporary zarr file created by Parsed2Zarr
            p2z_temp_file = Path(self.parsed2zarr_obj.zarr_file_name)

            # remove temporary directory created by Parsed2Zarr, if it exists
            if p2z_temp_file.exists():
                # TODO: do we need to check file permissions here?
                shutil.rmtree(p2z_temp_file)

    def __str__(self) -> str:
        fpath = "Internal Memory"
        if self.converted_raw_path:
            fpath = self.converted_raw_path
        repr_str = "No data found."
        if self._tree is not None:
            repr_str = tree_repr(self._tree)
        return f"<EchoData: standardized raw data from {fpath}>\n{repr_str}"

    def __repr__(self) -> str:
        return str(self)

    def _repr_html_(self) -> str:
        """Make html representation of InferenceData object."""
        _, css_style = _load_static_files()
        try:
            from xarray.core.options import OPTIONS

            display_style = OPTIONS["display_style"]
            if display_style == "text":
                html_repr = f"<pre>{escape(repr(self))}</pre>"
            else:
                return get_template("echodata.html.j2").render(echodata=self, css_style=css_style)
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

    def _set_tree(self, tree: DataTree):
        self._tree = tree

    @classmethod
    def from_file(
        cls,
        converted_raw_path: str,
        storage_options: Optional[Dict[str, Any]] = None,
        open_kwargs: Dict[str, Any] = {},
    ) -> "EchoData":
        echodata = cls(
            converted_raw_path=converted_raw_path,
            storage_options=storage_options,
            open_kwargs=open_kwargs,
        )
        echodata._check_path(converted_raw_path)
        converted_raw_path = echodata._sanitize_path(converted_raw_path)
        suffix = echodata._check_suffix(converted_raw_path)
        tree = open_datatree(
            converted_raw_path,
            engine=XARRAY_ENGINE_MAP[suffix],
            **echodata.open_kwargs,
        )
        tree.name = "root"

        echodata._set_tree(tree)

        if isinstance(converted_raw_path, fsspec.FSMap):
            # Convert fsmap to Path so it can be used
            # for retrieving the path strings
            converted_raw_path = Path(converted_raw_path.root)
        echodata.converted_raw_path = converted_raw_path
        echodata._load_tree()
        return echodata

    def _load_tree(self) -> None:
        if self._tree is None:
            raise ValueError("Datatree not found!")

        for group, value in self.group_map.items():
            # EK80 data may have a Beam_power group if both complex and power data exist.
            ds = None
            try:
                if value["ep_group"] is None:
                    node = self._tree
                else:
                    node = self._tree[value["ep_group"]]

                ds = self.__get_dataset(node)
            except KeyError:
                # Skips group not found errors for EK80 and ADCP
                ...
            if group == "top" and hasattr(ds, "keywords"):
                self.sonar_model = ds.keywords.upper()  # type: ignore

            if isinstance(ds, xr.Dataset):
                setattr(self, group, node)

    @property
    def version_info(self) -> Union[Tuple[int], None]:
        def _get_version_tuple(provenance_type):
            """
            Parameters
            ----------
            provenance_type : str
                Either conversion or combination
            """
            version_str = self["Provenance"].attrs.get(f"{provenance_type}_software_version", None)
            if version_str is not None:
                if version_str.startswith("v"):
                    # Removes v in case of v0.4.x or less
                    version_str = version_str.strip("v")
                version_num = version_str.split(".")[:3]
                return tuple([int(i) for i in version_num])

        if self["Provenance"].attrs.get("combination_software_name", None) == "echopype":
            return _get_version_tuple("combination")
        elif self["Provenance"].attrs.get("conversion_software_name", None) == "echopype":
            return _get_version_tuple("conversion")
        else:
            return None

    @property
    def nbytes(self) -> float:
        return float(sum(self[p].nbytes for p in self.group_paths))

    @property
    def group_paths(self) -> Set[str]:
        return tuple(i[1:] if i != "/" else "Top-level" for i in self._tree.groups)

    @staticmethod
    def __get_dataset(node: DataTree) -> Optional[xr.Dataset]:
        if node.has_data or node.has_attrs:
            # validate and clean dtypes
            return sanitize_dtypes(node.ds)
        return None

    def __get_node(self, key: Optional[str]) -> DataTree:
        if key in ["Top-level", "/"]:
            # Access to root
            return self._tree
        return self._tree[key]

    def __getitem__(self, __key: Optional[str]) -> Optional[xr.Dataset]:
        if self._tree:
            try:
                node = self.__get_node(__key)
                return self.__get_dataset(node)
            except KeyError:
                return None
        else:
            raise ValueError("Datatree not found!")

    def __setitem__(self, __key: Optional[str], __newvalue: Any) -> Optional[xr.Dataset]:
        if self._tree:
            try:
                node = self.__get_node(__key)
                node.ds = __newvalue
                return self.__get_dataset(node)
            except KeyError:
                raise GroupNotFoundError(__key)
        else:
            raise ValueError("Datatree not found!")

    def __setattr__(self, __name: str, __value: Any) -> None:
        attr_value = __value
        if isinstance(__value, DataTree) and __name != "_tree":
            attr_value = self.__get_dataset(__value)
        elif isinstance(__value, xr.Dataset):
            group_map = sonarnetcdf_1.yaml_dict["groups"]
            if __name in group_map:
                group = group_map.get(__name)
                group_path = group["ep_group"]
                if self._tree:
                    if __name == "top":
                        self._tree.ds = __value
                    else:
                        self._tree[group_path].ds = __value
        super().__setattr__(__name, attr_value)

    @add_processing_level("L1A")
    def update_platform(
        self,
        extra_platform_data: xr.Dataset,
        variable_mappings=Dict[str, str],
        extra_platform_data_file_name=None,
    ):
        """
        Updates the `EchoData["Platform"]` group with additional external platform data.

        `extra_platform_data` must be an xarray Dataset. Data is extracted from
        `extra_platform_data` by variable name. Only data assigned to a pre-existing
        (but possibly all-nan) variable name in Platform will be processed. These
        Platform variables include latitude, longitude, pitch, roll, vertical_offset, etc.
        See the variables present in the EchoData object's Platform group to obtain a
        complete list of possible variables.
        Different external variables may be dependent on different time dimensions, but
        latitude and longitude (if specified) must share the same time dimension. New
        time dimensions will be added as needed. For example, if variables to be added
        from the external data use two time dimensions and the Platform group has time
        dimensions time2 and time2, new dimensions time3 and time4 will be created.

        Parameters
        ----------
        extra_platform_data : xr.Dataset
            An `xr.Dataset` containing the additional platform data to be added
            to the `EchoData["Platform"]` group.
        variable_mappings: Dict[str,str]
            A dictionary mapping Platform variable names (dict key) to the
            external-data variable name (dict value).
        extra_platform_data_file_name: str, default=None
            File name for source of extra platform data, if read from a file

        Examples
        --------
        >>> ed = echopype.open_raw(raw_file, "EK60")
        >>> extra_platform_data = xr.open_dataset(extra_platform_data_file_name)
        >>> ed.update_platform(
        >>>     extra_platform_data,
        >>>     variable_mappings={"longitude": "lon", "latitude": "lat", "roll": "ROLL"},
        >>>     extra_platform_data_file_name=extra_platform_data_file_name
        >>> )
        """

        # Handle data stored as a CF Trajectory Discrete Sampling Geometry
        # http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#trajectory-data
        # The Saildrone sample data file follows this convention
        if (
            "featureType" in extra_platform_data.attrs
            and extra_platform_data.attrs["featureType"].lower() == "trajectory"
        ):
            for coordvar in extra_platform_data.coords:
                coordvar_attrs = extra_platform_data[coordvar].attrs
                if "cf_role" in coordvar_attrs and coordvar_attrs["cf_role"] == "trajectory_id":
                    trajectory_var = coordvar

                if "standard_name" in coordvar_attrs and coordvar_attrs["standard_name"] == "time":
                    time_dim = coordvar

            # assumes there's only one trajectory in the dataset (index 0)
            extra_platform_data = extra_platform_data.sel(
                {trajectory_var: extra_platform_data[trajectory_var][0]}
            )
            extra_platform_data = extra_platform_data.drop_vars(trajectory_var)
            obs_dim = list(extra_platform_data[time_dim].dims)[0]
            extra_platform_data = extra_platform_data.swap_dims({obs_dim: time_dim})

        def _extvar_properties(ds, name):
            """Test the external variable for presence and all-nan values,
            and extract its time dimension name.
            Returns <presence>, <valid values>, <time dim name>
            """
            if name in ds:
                # Assumes the only dimension in the variable is a time dimension
                time_dim_name = ds[name].dims[0] if len(ds[name].dims) > 0 else "scalar"
                if not ds[name].isnull().all():
                    return True, True, time_dim_name
                else:
                    return True, False, time_dim_name
            else:
                return False, False, None

        # clip incoming time to 1 less than min of EchoData["Sonar/Beam_group1"]["ping_time"] and
        #   1 greater than max of EchoData["Sonar/Beam_group1"]["ping_time"]
        # account for unsorted external time by checking whether each time value is between
        #   min and max ping_time instead of finding the 2 external times corresponding to the
        #   min and max ping_time and taking all the times between those indices
        def _clip_by_time_dim(external_ds, ext_time_dim_name):
            # external_ds is extra_platform_data[vars-list-on-time_dim_name]
            sorted_external_time = external_ds[ext_time_dim_name].data
            sorted_external_time.sort()
            # fmt: off
            min_index = max(
                np.searchsorted(
                    sorted_external_time,
                    self["Sonar/Beam_group1"]["ping_time"].min(),
                    side="left"
                ) - 1,
                0,
            )
            # fmt: on
            max_index = min(
                np.searchsorted(
                    sorted_external_time,
                    self["Sonar/Beam_group1"]["ping_time"].max(),
                    side="right",
                ),
                len(sorted_external_time) - 1,
            )
            # TODO: this element-wise comparison is expensive and seems an ad-hoc patch
            # to deal with potentially reversed timestamps in the external dataset.
            # Review at workflow stage to see if to clean up timestamp reversals
            # and just find start/end timestamp for slicing.
            return external_ds.sel(
                {
                    ext_time_dim_name: np.logical_and(
                        sorted_external_time[min_index] <= external_ds[ext_time_dim_name],
                        external_ds[ext_time_dim_name] <= sorted_external_time[max_index],
                    )
                }
            )

        # History attribute to be included in each updated variable
        history_attr = f"{datetime.datetime.utcnow()} +00:00. Added from external platform data"
        if extra_platform_data_file_name:
            history_attr += ", from file " + extra_platform_data_file_name

        platform = self["Platform"]

        # Retain only variable_mappings items where
        # either the Platform group or extra_platform_data
        # contain the corresponding variables or contain valid (not all nan) data
        mappings_expanded = {}
        for platform_var, external_var in variable_mappings.items():
            # TODO: instead of using existing Platform group variables, a better practice is to
            # define a set of allowable Platform variables (sonar_model dependent) for this check.
            # This set can be dynamically generated from an external source like a CDL or yaml.
            if platform_var in platform:
                platform_validvalues = not platform[platform_var].isnull().all()
                ext_present, ext_validvalues, ext_time_dim_name = _extvar_properties(
                    extra_platform_data, external_var
                )
                if ext_present and ext_validvalues:
                    mappings_expanded[platform_var] = dict(
                        external_var=external_var,
                        ext_time_dim_name=ext_time_dim_name,
                        platform_validvalues=platform_validvalues,
                    )

        # Generate warning if mappings_expanded is empty
        if not mappings_expanded:
            logger.warning(
                "No variables will be updated, "
                "check variable_mappings to ensure variable names are correctly specified!"
            )

        # If longitude or latitude are requested, verify that both are present
        # and they share the same external time dimension
        if "longitude" in mappings_expanded or "latitude" in mappings_expanded:
            if "longitude" not in mappings_expanded or "latitude" not in mappings_expanded:
                raise ValueError(
                    "Only one of latitude and longitude are specified. Please include both, or neither."  # noqa
                )
            if (
                mappings_expanded["longitude"]["ext_time_dim_name"]
                != mappings_expanded["latitude"]["ext_time_dim_name"]
            ):
                raise ValueError(
                    "The external latitude and longitude use different time dimensions. "
                    "They must share the same time dimension."
                )

        # Generate warnings regarding variables that will be updated
        vars_not_handled = set(variable_mappings.keys()).difference(mappings_expanded.keys())
        if len(vars_not_handled) > 0:
            logger.warning(
                f"The following requested variables will not be updated: {', '.join(vars_not_handled)}"  # noqa
            )

        vars_notnan_replaced = [
            platform_var
            for platform_var, v in mappings_expanded.items()
            if v["platform_validvalues"]
        ]
        if len(vars_notnan_replaced) > 0:
            logger.warning(
                f"Some variables with valid data in the original Platform group will be overwritten: {', '.join(vars_notnan_replaced)}"  # noqa
            )

        # Create names for required new time dimensions
        ext_time_dims = list(
            {
                v["ext_time_dim_name"]
                for v in mappings_expanded.values()
                if v["ext_time_dim_name"] != "scalar"
            }
        )
        time_dims_max = max([int(dim[-1]) for dim in platform.dims if dim.startswith("time")])
        new_time_dims = [f"time{time_dims_max+i+1}" for i in range(len(ext_time_dims))]
        # Map each new time dim name to the external time dim name:
        new_time_dims_mappings = {new: ext for new, ext in zip(new_time_dims, ext_time_dims)}

        # Process variable updates by corresponding new time dimensions
        for time_dim in new_time_dims:
            ext_time_dim = new_time_dims_mappings[time_dim]
            mappings_selected = {
                k: v for k, v in mappings_expanded.items() if v["ext_time_dim_name"] == ext_time_dim
            }
            ext_vars = [v["external_var"] for v in mappings_selected.values()]
            ext_ds = _clip_by_time_dim(extra_platform_data[ext_vars], ext_time_dim)

            # Create new time coordinate and dimension
            platform = platform.assign_coords(**{time_dim: ext_ds[ext_time_dim].values})
            time_attrs = {
                "axis": "T",
                "standard_name": "time",
                "long_name": "Timestamps from an external dataset",
                "comment": "Time coordinate originated from a dataset "
                "external to the sonar data files.",
                "history": f"{history_attr}. From external {ext_time_dim} variable.",
            }
            platform[time_dim] = platform[time_dim].assign_attrs(**time_attrs)

            # Process each platform variable that will be replaced
            for platform_var in mappings_selected.keys():
                ext_var = mappings_expanded[platform_var]["external_var"]
                platform_var_attrs = platform[platform_var].attrs.copy()

                # Create new (replaced) variable using dataset "update"
                # With update, dropping the variable first is not needed
                platform = platform.update({platform_var: (time_dim, ext_ds[ext_var].data)})

                # Assign attributes to newly created (replaced) variables
                var_attrs = platform_var_attrs
                var_attrs["history"] = f"{history_attr}. From external {ext_var} variable."
                platform[platform_var] = platform[platform_var].assign_attrs(**var_attrs)

        # Update scalar variables, if any
        scalar_vars = [
            platform_var
            for platform_var, v in mappings_expanded.items()
            if v["ext_time_dim_name"] == "scalar"
        ]
        for platform_var in scalar_vars:
            ext_var = mappings_expanded[platform_var]["external_var"]
            # Replace the scalar value and add a history attribute
            platform[platform_var].data = float(extra_platform_data[ext_var].data)
            platform[platform_var] = platform[platform_var].assign_attrs(
                **{"history": f"{history_attr}. From external {ext_var} variable."}
            )

        # Drop pre-existing time dimensions that are no longer being used
        used_dims = {
            platform[platform_var].dims[0]
            for platform_var in platform.data_vars
            if len(platform[platform_var].dims) > 0
        }
        platform = platform.drop_dims(set(platform.dims).difference(used_dims), errors="ignore")

        self["Platform"] = set_time_encodings(platform)

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
        """Lazy load all groups and subgroups from raw file."""
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
            filepath,
            group=group,
            engine=XARRAY_ENGINE_MAP[suffix],
            **self.open_kwargs,
        )

    def to_netcdf(
        self,
        save_path: Optional["PathHint"] = None,
        compress: bool = True,
        overwrite: bool = False,
        parallel: bool = False,
        output_storage_options: Dict[str, str] = {},
        **kwargs,
    ):
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
        **kwargs : dict, optional
            Extra arguments to `xr.Dataset.to_netcdf`: refer to
            xarray's documentation for a list of all possible arguments.
        """
        from ..convert.api import to_file

        return to_file(
            echodata=self,
            engine="netcdf4",
            save_path=save_path,
            compress=compress,
            overwrite=overwrite,
            parallel=parallel,
            output_storage_options=output_storage_options,
            **kwargs,
        )

    def to_zarr(
        self,
        save_path: Optional["PathHint"] = None,
        compress: bool = True,
        overwrite: bool = False,
        parallel: bool = False,
        output_storage_options: Dict[str, str] = {},
        consolidated: bool = True,
        **kwargs,
    ):
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
        consolidated : bool
            Flag to consolidate zarr metadata.
            Defaults to ``True``
        **kwargs : dict, optional
            Extra arguments to `xr.Dataset.to_zarr`: refer to
            xarray's documentation for a list of all possible arguments.
        """
        from ..convert.api import to_file

        return to_file(
            echodata=self,
            engine="zarr",
            save_path=save_path,
            compress=compress,
            overwrite=overwrite,
            parallel=parallel,
            output_storage_options=output_storage_options,
            consolidated=consolidated,
            **kwargs,
        )

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
