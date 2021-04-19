import os
import warnings
import uuid
from collections import OrderedDict
from datetime import datetime as dt
from html import escape
from pathlib import Path

import fsspec
from fsspec.implementations.local import LocalFileSystem

import xarray as xr
import zarr
from zarr.errors import GroupNotFoundError

from ..utils.repr import HtmlTemplate
from ..utils import io
from .convention import _get_convention
from ..convert import Convert

XARRAY_ENGINE_MAP = {
    ".nc": "netcdf4",
    ".zarr": "zarr",
}

COMPRESSION_SETTINGS = {
    'netcdf4': {'zlib': True, 'complevel': 4},
    'zarr': {'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)}
}

DEFAULT_CHUNK_SIZE = {
    'range_bin': 25000,
    'ping_time': 2500
}


class EchoData:
    """Echo data model class for handling raw converted data,
    including multiple files associated with the same data set.
    """

    def __init__(
            self,
            converted_raw_path=None,
            storage_options=None,
            convert_obj: Convert = None
    ):

        # TODO: consider if should open datasets in init
        #  or within each function call when echodata is used. Need to benchmark.

        self.converted_raw_path = converted_raw_path
        self.storage_options = storage_options if storage_options is not None else {}

        self.__setup_groups()
        if converted_raw_path:
            # TODO: verify if converted_raw_path is valid on either local or remote filesystem
            self._load_file(converted_raw_path)
            self.sonar_model = self.top.keywords

        self.convert_obj = convert_obj  # used to handle raw file conversion

    def __repr__(self) -> str:
        """Make string representation of InferenceData object."""
        existing_groups = [
            f"{self.__group_map[group]['name']}: {self.__group_map[group]['description']}"  # noqa
            for group in self.__group_map.keys()
            if isinstance(getattr(self, group), xr.Dataset)
        ]
        msg = "EchoData: standardized raw data from {file_path}\n  > {options}".format(
            options="\n  > ".join(existing_groups),
            file_path=self.converted_raw_path,
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
                                group_name=self.__group_map[group]["name"],
                                group_description=self.__group_map[group]["description"],
                                xr_data=xr_data,
                            )
                        )
                elements = "".join(xr_collections)
                formatted_html_template = HtmlTemplate.html_template.format(
                    elements, file_path=str(self.converted_raw_path)
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

    def _load_file(self, raw_path):
        """Lazy load Top-level, Beam, Environment, and Vendor groups from raw file."""
        for group, value in self.__group_map.items():
            # EK80 data may have a Beam_power group if both complex and power data exist.
            try:
                ds = self._load_groups(raw_path, group=value["ep_group"])
            except (OSError, GroupNotFoundError):
                if group == "beam_power":
                    ds = None
                    pass
            if group == "top":
                self.sonar_model = ds.keywords.upper()

            if isinstance(ds, xr.Dataset):
                setattr(self, group, ds)

    def _check_path(self):
        """ Check if converted_raw_path exists """
        pass

    @staticmethod
    def _load_groups(filepath, group=None):
        # TODO: handle multiple files through the same set of checks for combining files
        suffix = Path(filepath).suffix
        if suffix not in XARRAY_ENGINE_MAP:
            raise ValueError("Input file type not supported!")

        return xr.open_dataset(filepath, group=group, engine=XARRAY_ENGINE_MAP[suffix])

    @staticmethod
    def _validate_path(
            source_file,
            file_format,
            output_storage_options={},
            save_path=None
    ):
        """Assemble output file names and path.

        Parameters
        ----------
        file_format : str {'.nc', '.zarr'}
        save_path : str
            Either a directory or a file. If none then the save path is the same as the raw file.
        """
        if save_path is None:
            warnings.warn("save_path is not provided")

            current_dir = Path.cwd()
            # Check permission, raise exception if no permission
            io.check_file_permissions(current_dir)
            out_dir = current_dir.joinpath(Path("temp_echopype_output"))
            if not out_dir.exists():
                out_dir.mkdir(parents=True)

            warnings.warn(f"Resulting converted file(s) will be available at {str(out_dir)}")
            out_path = [
                str(out_dir.joinpath(Path(os.path.splitext(Path(f).name)[0] + file_format)))
                for f in source_file
            ]

        else:
            fsmap = fsspec.get_mapper(save_path, **output_storage_options)
            output_fs = fsmap.fs

            # Use the full path such as s3://... if it's not local, otherwise use root
            if isinstance(output_fs, LocalFileSystem):
                root = fsmap.root
            else:
                root = save_path

            fname, ext = os.path.splitext(root)
            if ext == "":  # directory
                out_dir = fname
                out_path = [
                    os.path.join(root, os.path.splitext(os.path.basename(f))[0] + file_format)
                    for f in source_file
                ]
            else:  # file
                out_dir = os.path.dirname(root)
                if len(source_file) > 1:  # get dirname and assemble path
                    out_path = [
                        os.path.join(
                            out_dir, os.path.splitext(os.path.basename(f))[0] + file_format
                        )
                        for f in source_file
                    ]
                else:
                    # force file_format to conform
                    out_path = [
                        os.path.join(
                            out_dir, os.path.splitext(os.path.basename(fname))[0] + file_format
                        )
                    ]

        # Create folder if save_path does not exist already
        fsmap = fsspec.get_mapper(str(out_dir), **output_storage_options)
        fs = fsmap.fs
        if file_format == ".nc" and not isinstance(fs, LocalFileSystem):
            raise ValueError("Only local filesystem allowed for NetCDF output.")
        else:
            try:
                # Check permission, raise exception if no permission
                io.check_file_permissions(fsmap)
                if isinstance(fs, LocalFileSystem):
                    # Only make directory if local file system
                    # otherwise it will just create the object path
                    fs.mkdir(fsmap.root)
            except FileNotFoundError:
                raise ValueError("Specified save_path is not valid.")

        return out_path  # output_path is always a list

    @staticmethod
    def _normalize_path(out_f, convert_type, output_storage_options):
        if convert_type == "zarr":
            return fsspec.get_mapper(out_f, **output_storage_options)
        elif convert_type == "netcdf4":
            return out_f

    def _to_file(
            self,
            engine,
            save_path=None,
            compress=True,
            overwrite=False,
            parallel=False,
            output_storage_options={},
            **kwargs
    ):
        """Save content of EchoData to netCDF or zarr.

        Parameters
        ----------
        engine : str {'netcdf4', 'zarr'}
            type of converted file
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
        # TODO: revise below since only need 1 output file in use case EchoData.to_zarr()/to_netcdf()
        if parallel:
            raise NotImplementedError(
                "Parallel conversion is not yet implemented."
            )
        if engine not in ["netcdf4", "zarr"]:
            raise ValueError("Unknown type to convert file to!")

        # Assemble output file names and path
        format_mapping = dict(map(reversed, XARRAY_ENGINE_MAP.items()))
        output_file = self._validate_path(
            source_file=self.convert_obj.source_file,
            file_format=format_mapping[engine],
            save_path=save_path
        )

        # Get all existing files
        exist_list = []
        fs = fsspec.get_mapper(
            output_file[0], **output_storage_options
        ).fs  # get file system
        for out_f in output_file:
            if fs.exists(out_f):
                exist_list.append(out_f)

        # Sequential or parallel conversion
        for src_f, out_f in zip(self.convert_obj.source_file, output_file):
            if out_f in exist_list and not overwrite:
                print(
                    f"{dt.now().strftime('%H:%M:%S')}  {src_f} has already been converted to {engine}. "
                    f"File saving not executed."
                )
                continue
            else:
                if out_f in exist_list:
                    print(f"{dt.now().strftime('%H:%M:%S')}  overwriting {out_f}")
                else:
                    print(f"{dt.now().strftime('%H:%M:%S')}  saving {out_f}")
                self.__save_groups_to_file(
                    output_path=self._normalize_path(out_f, engine, output_storage_options),
                    engine=engine,
                    compress=compress
                )

        # If only one output file make it a string instead of a list
        if len(output_file) == 1:
            output_file = output_file[0]

        # Link path to saved file with attribute as if from open_converted
        self.converted_raw_path = output_file

    def to_netcdf(self, **kwargs):
        """Save content of EchoData to netCDF.

        Parameters
        ----------
        engine : str {'netcdf4', 'zarr'}
            type of converted file
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
        return self._to_file("netcdf4", **kwargs)

    def to_zarr(self, **kwargs):
        """Save content of EchoData to zarr.

        Parameters
        ----------
        engine : str {'netcdf4', 'zarr'}
            type of converted file
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
        return self._to_file("zarr", **kwargs)

    def __save_groups_to_file(self, output_path, engine, compress=True):
        """Serialize all groups to file.
        """
        # TODO: in terms of chunking, would using rechunker at the end be faster and more convenient?

        # Top-level group
        io.save_file(
            self.top,
            path=output_path,
            mode='w',
            engine=engine
        )

        # Provenance group
        io.save_file(
            self.provenance,
            path=output_path,
            group='Provenance',
            mode='a',
            engine=engine
        )

        # Environment group
        io.save_file(
            self.environment.chunk({'ping_time': DEFAULT_CHUNK_SIZE['ping_time']}),  # TODO: chunking necessary?
            path=output_path,
            mode='a',
            engine=engine,
            group='Environment'
        )

        # Sonar group
        io.save_file(
            self.sonar,
            path=output_path,
            group='Sonar',
            mode='a',
            engine=engine
        )

        # Beam group
        io.save_file(
            self.beam.chunk({'range_bin': DEFAULT_CHUNK_SIZE['range_bin'],
                             'ping_time': DEFAULT_CHUNK_SIZE['ping_time']}),
            path=output_path,
            mode='a',
            engine=engine,
            group='Beam',
            compression_settings=COMPRESSION_SETTINGS[engine] if compress else None
        )
        if self.beam_power is not None:
            io.save_file(
                self.beam_power.chunk({'range_bin': DEFAULT_CHUNK_SIZE['range_bin'],
                                       'ping_time': DEFAULT_CHUNK_SIZE['ping_time']}),
                path=output_path,
                mode='a',
                engine=engine,
                group='Beam_power',
                compression_settings=COMPRESSION_SETTINGS[engine] if compress else None
            )

        # Platform group
        io.save_file(
            self.platform,  # TODO: chunking necessary? location_time and mru_time (EK80) only
            path=output_path,
            mode='a',
            engine=engine,
            group='Platform',
            compression_settings=COMPRESSION_SETTINGS[engine] if compress else None
        )

        # Platform/NMEA group: some sonar model does not produce NMEA data
        if hasattr(self, 'nmea'):
            io.save_file(
                self.nmea,  # TODO: chunking necessary?
                path=output_path,
                mode='a',
                engine=engine,
                group='Platform/NMEA',
                compression_settings=COMPRESSION_SETTINGS[engine] if compress else None
            )

        # Vendor-specific group
        if "ping_time" in self.vendor:
            io.save_file(
                self.vendor,  # TODO: chunking necessary?
                path=output_path,
                mode='a',
                engine=engine,
                group='Vendor',
                compression_settings=COMPRESSION_SETTINGS[engine] if compress else None
            )
        else:
            io.save_file(
                self.vendor.chunk({'ping_time': DEFAULT_CHUNK_SIZE['ping_time']}),  # TODO: chunking necessary?
                path=output_path,
                mode='a',
                engine=engine,
                group='Vendor',
                compression_settings=COMPRESSION_SETTINGS[engine] if compress else None
            )
