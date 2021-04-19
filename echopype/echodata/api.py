import os
import warnings
from datetime import datetime as dt
from pathlib import Path

import fsspec
from fsspec.implementations.local import LocalFileSystem
import zarr

from ..utils import io

from .echodata import EchoData, XARRAY_ENGINE_MAP
from ..convert.convert import Convert

COMPRESSION_SETTINGS = {
    'netcdf4': {'zlib': True, 'complevel': 4},
    'zarr': {'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)},
}

DEFAULT_CHUNK_SIZE = {'range_bin': 25000, 'ping_time': 2500}


def _normalize_path(out_f, convert_type, output_storage_options):
    if convert_type == "zarr":
        return fsspec.get_mapper(out_f, **output_storage_options)
    elif convert_type == "netcdf4":
        return out_f


def _validate_path(
    source_file, file_format, output_storage_options={}, save_path=None
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

        warnings.warn(
            f"Resulting converted file(s) will be available at {str(out_dir)}"
        )
        out_path = [
            str(
                out_dir.joinpath(
                    Path(os.path.splitext(Path(f).name)[0] + file_format)
                )
            )
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
                os.path.join(
                    root,
                    os.path.splitext(os.path.basename(f))[0] + file_format,
                )
                for f in source_file
            ]
        else:  # file
            out_dir = os.path.dirname(root)
            if len(source_file) > 1:  # get dirname and assemble path
                out_path = [
                    os.path.join(
                        out_dir,
                        os.path.splitext(os.path.basename(f))[0] + file_format,
                    )
                    for f in source_file
                ]
            else:
                # force file_format to conform
                out_path = [
                    os.path.join(
                        out_dir,
                        os.path.splitext(os.path.basename(fname))[0]
                        + file_format,
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


def to_file(
    echodata: EchoData,
    engine,
    save_path=None,
    compress=True,
    overwrite=False,
    parallel=False,
    output_storage_options={},
    **kwargs,
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
    if engine not in XARRAY_ENGINE_MAP.values():
        raise ValueError("Unknown type to convert file to!")

    # Assemble output file names and path
    format_mapping = dict(map(reversed, XARRAY_ENGINE_MAP.items()))
    output_file = _validate_path(
        source_file=echodata.source_file,
        file_format=format_mapping[engine],
        save_path=save_path,
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
    for src_f, out_f in zip(echodata.source_file, output_file):
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
            _save_groups_to_file(
                echodata,
                output_path=_normalize_path(
                    out_f, engine, output_storage_options
                ),
                engine=engine,
                compress=compress,
            )

    # If only one output file make it a string instead of a list
    if len(output_file) == 1:
        output_file = output_file[0]

    # Link path to saved file with attribute as if from open_converted
    echodata.converted_raw_path = output_file

    return echodata


def _save_groups_to_file(echodata, output_path, engine, compress=True):
    """Serialize all groups to file."""
    # TODO: in terms of chunking, would using rechunker at the end be faster and more convenient?

    # Top-level group
    io.save_file(echodata.top, path=output_path, mode='w', engine=engine)

    # Provenance group
    io.save_file(
        echodata.provenance,
        path=output_path,
        group='Provenance',
        mode='a',
        engine=engine,
    )

    # Environment group
    io.save_file(
        echodata.environment.chunk(
            {'ping_time': DEFAULT_CHUNK_SIZE['ping_time']}
        ),  # TODO: chunking necessary?
        path=output_path,
        mode='a',
        engine=engine,
        group='Environment',
    )

    # Sonar group
    io.save_file(
        echodata.sonar,
        path=output_path,
        group='Sonar',
        mode='a',
        engine=engine,
    )

    # Beam group
    io.save_file(
        echodata.beam.chunk(
            {
                'range_bin': DEFAULT_CHUNK_SIZE['range_bin'],
                'ping_time': DEFAULT_CHUNK_SIZE['ping_time'],
            }
        ),
        path=output_path,
        mode='a',
        engine=engine,
        group='Beam',
        compression_settings=COMPRESSION_SETTINGS[engine]
        if compress
        else None,
    )
    if echodata.beam_power is not None:
        io.save_file(
            echodata.beam_power.chunk(
                {
                    'range_bin': DEFAULT_CHUNK_SIZE['range_bin'],
                    'ping_time': DEFAULT_CHUNK_SIZE['ping_time'],
                }
            ),
            path=output_path,
            mode='a',
            engine=engine,
            group='Beam_power',
            compression_settings=COMPRESSION_SETTINGS[engine]
            if compress
            else None,
        )

    # Platform group
    io.save_file(
        echodata.platform,  # TODO: chunking necessary? location_time and mru_time (EK80) only
        path=output_path,
        mode='a',
        engine=engine,
        group='Platform',
        compression_settings=COMPRESSION_SETTINGS[engine]
        if compress
        else None,
    )

    # Platform/NMEA group: some sonar model does not produce NMEA data
    if hasattr(echodata, 'nmea'):
        io.save_file(
            echodata.nmea,  # TODO: chunking necessary?
            path=output_path,
            mode='a',
            engine=engine,
            group='Platform/NMEA',
            compression_settings=COMPRESSION_SETTINGS[engine]
            if compress
            else None,
        )

    # Vendor-specific group
    if "ping_time" in echodata.vendor:
        io.save_file(
            echodata.vendor.chunk(
                {'ping_time': DEFAULT_CHUNK_SIZE['ping_time']}
            ),  # TODO: chunking necessary?
            path=output_path,
            mode='a',
            engine=engine,
            group='Vendor',
            compression_settings=COMPRESSION_SETTINGS[engine]
            if compress
            else None,
        )
    else:
        io.save_file(
            echodata.vendor,  # TODO: chunking necessary?
            path=output_path,
            mode='a',
            engine=engine,
            group='Vendor',
            compression_settings=COMPRESSION_SETTINGS[engine]
            if compress
            else None,
        )


def open_raw(file=None, model=None, xml_path=None, convert_params=None, storage_options=None):
    """Create an EchoData object containing parsed data from a single raw data file.

    The EchoData object can be used for adding metadata and ancillary data
    as well as to serialize the parsed data to zarr or netcdf.

    Parameters
    ----------
    file : str or list
        path to raw data file(s)
    model : str
        model of the sonar instrument
    xml_path : str
        path to XML config file used by AZFP
    convert_params : dict
        parameters (metadata) that may not exist in the raw file
        and need to be added to the converted file
    storage_options : dict
        options for cloud storage
    """

    convert_obj = Convert.open_raw(file=file, model=model,
                                   xml_path=xml_path,
                                   convert_params=convert_params,
                                   storage_options=storage_options)
    # Set up echodata object
    echodata = EchoData._load_convert(convert_obj)

    return echodata


def open_converted(converted_raw_path, storage_options=None):
    """Create an EchoData object from a single converted zarr/nc files.
    """
    # TODO: combine multiple files when opening
    return EchoData(converted_raw_path=converted_raw_path, storage_options=storage_options)
