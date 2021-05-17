import uuid
from collections import OrderedDict
from html import escape
from pathlib import Path
import warnings
import fsspec

import xarray as xr
from zarr.errors import GroupNotFoundError

from ..utils.repr import HtmlTemplate
from ..utils.io import sanitize_file_path, check_file_existance
from .convention import _get_convention

XARRAY_ENGINE_MAP = {
    ".nc": "netcdf4",
    ".zarr": "zarr",
}


class EchoData:
    """Echo data model class for handling raw converted data,
    including multiple files associated with the same data set.
    """

    def __init__(
            self,
            converted_raw_path=None,
            storage_options=None,
            source_file=None,
            xml_path=None,
            sonar_model=None,
    ):

        # TODO: consider if should open datasets in init
        #  or within each function call when echodata is used. Need to benchmark.

        self.storage_options = storage_options if storage_options is not None else {}
        self.source_file = source_file
        self.xml_path = xml_path
        self.sonar_model = sonar_model
        self.converted_raw_path = None

        self.__setup_groups()
        self.__read_converted(converted_raw_path)

    def __repr__(self) -> str:
        """Make string representation of InferenceData object."""
        existing_groups = [
            f"{group}: ({self.__group_map[group]['name']}) {self.__group_map[group]['description']}"  # noqa
            for group in self.__group_map.keys()
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
                for group in self.__group_map.keys():
                    if isinstance(getattr(self, group), xr.Dataset):
                        xr_data = getattr(self, group)._repr_html_()
                        xr_collections.append(
                            HtmlTemplate.element_template.format(  # noqa
                                group_id=group + str(uuid.uuid4()),
                                group=group,
                                group_name=self.__group_map[group]["name"],
                                group_description=self.__group_map[group]["description"],
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
        self.__group_map = OrderedDict(_get_convention()["groups"])
        for group in self.__group_map.keys():
            setattr(self, group, None)

    def __read_converted(self, converted_raw_path):
        if converted_raw_path is not None:
            self._check_path(converted_raw_path)
            converted_raw_path = self._sanitize_path(converted_raw_path)
            self._load_file(converted_raw_path)
            self.sonar_model = self.top.keywords

        self.converted_raw_path = converted_raw_path

    @classmethod
    def _load_convert(cls, convert_obj):
        new_cls = cls()
        for group in new_cls.__group_map.keys():
            if hasattr(convert_obj, group):
                setattr(new_cls, group, getattr(convert_obj, group))

        setattr(new_cls, "sonar_model", getattr(convert_obj, "sonar_model"))
        setattr(new_cls, "source_file", getattr(convert_obj, "source_file"))
        return new_cls

    def _load_file(self, raw_path):
        """Lazy load Top-level, Beam, Environment, and Vendor groups from raw file."""
        for group, value in self.__group_map.items():
            # EK80 data may have a Beam_power group if both complex and power data exist.
            # ADCP data adds a Beam_complex group
            ds = None
            try:
                ds = self._load_group(
                    raw_path,
                    group=value["ep_group"],
                )
            except (OSError, GroupNotFoundError):
                # Skips group not found errors for EK80 and ADCP
                ...
            if group == "top":
                self.sonar_model = ds.keywords.upper()

            if isinstance(ds, xr.Dataset):
                setattr(self, group, ds)

    def _check_path(self, filepath):
        """ Check if converted_raw_path exists """
        file_exists = check_file_existance(
            filepath,
            self.storage_options
        )
        if not file_exists:
            raise FileNotFoundError(
                f"There is no file named {filepath}"
            )

    def _sanitize_path(self, filepath):
        filepath = sanitize_file_path(
            filepath,
            self.storage_options
        )
        return filepath

    def _check_suffix(self, filepath):
        """ Check if file type is supported. """
        # TODO: handle multiple files through the same set of checks for combining files
        if isinstance(filepath, fsspec.FSMap):
            suffix = Path(filepath.root).suffix
        else:
            suffix = Path(filepath).suffix

        if suffix not in XARRAY_ENGINE_MAP:
            raise ValueError("Input file type not supported!")

        return suffix

    def _load_group(self, filepath, group=None):
        """ Loads each echodata group """
        suffix = self._check_suffix(filepath)
        return xr.open_dataset(
            filepath,
            group=group,
            engine=XARRAY_ENGINE_MAP[suffix]
        )

    def to_netcdf(self, save_path=None, **kwargs):
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

    def to_zarr(self, save_path=None, **kwargs):
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
    def nc_path(self):
        warnings.warn(
            "`nc_path` is deprecated, Use `converted_raw_path` instead.",
            DeprecationWarning,
            2,
        )
        if self.converted_raw_path.endswith('.nc'):
            return self.converted_raw_path
        else:
            path = Path(self.converted_raw_path)
            return str(path.parent / (path.stem + '.nc'))

    @property
    def zarr_path(self):
        warnings.warn(
            "`zarr_path` is deprecated, Use `converted_raw_path` instead.",
            DeprecationWarning,
            2,
        )
        if self.converted_raw_path.endswith('.zarr'):
            return self.converted_raw_path
        else:
            path = Path(self.converted_raw_path)
            return str(path.parent / (path.stem + '.zarr'))

    def raw2nc(
        self, save_path=None, combine_opt=False, overwrite=False, compress=True
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
        self, save_path=None, combine_opt=False, overwrite=False, compress=True
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
