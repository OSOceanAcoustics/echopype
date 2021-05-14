"""
echopype utilities for file handling
"""
import os
import sys
from typing import Union
from pathlib import Path

import fsspec
from fsspec import FSMap
from fsspec.implementations.local import LocalFileSystem


def get_files_from_dir(folder):
    """Retrieves all Netcdf and Zarr files from a given folder"""
    valid_ext = ['.nc', '.zarr']
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1] in valid_ext]


def save_file(ds, path, mode, engine, group=None, compression_settings=None):
    """Saves a dataset to netcdf or zarr depending on the engine
    If ``compression_settings`` are set, compress all variables with those settings"""
    encoding = {var: compression_settings for var in ds.data_vars} if compression_settings is not None else {}
    # Allows saving both NetCDF and Zarr files from an xarray dataset
    if engine == 'netcdf4':
        ds.to_netcdf(path=path, mode=mode, group=group, encoding=encoding)
    elif engine == 'zarr':
        ds.to_zarr(store=path, mode=mode, group=group, encoding=encoding)
    else:
        raise ValueError(f"{engine} is not a supported save format")


def get_file_format(file):
    """Gets the file format (either Netcdf4 or Zarr) from the file extension"""
    if isinstance(file, list):
        file = file[0]
    elif isinstance(file, FSMap):
        file = file.root

    if file.endswith('.nc'):
        return 'netcdf4'
    elif file.endswith('.zarr'):
        return 'zarr'
    else:
        raise ValueError(f"Unsupported file format: {os.path.splitext(file)[1]}")


def sanitize_file_path(
    file_path: Union[str, Path, FSMap],
    storage_options: dict = {},
) -> Union[Path, FSMap]:
    """Determines file path, either Path or FSMap"""
    if isinstance(file_path, Path):
        # Check for extension
        if ":/" in str(file_path):
            raise ValueError(f"{file_path} is not a valid posix path.")

        return file_path
    elif isinstance(file_path, str):
        if "://" in file_path:
            return fsspec.get_mapper(file_path, **storage_options)
        return Path(file_path)
    elif isinstance(file_path, fsspec.FSMap):
        root = file_path.root
        if (Path(root).suffix == '.nc'):
            if not isinstance(
                file_path.fs, LocalFileSystem
            ):
                # For special case of netcdf.
                # netcdf4 engine can only read Path or string
                raise ValueError("Only local netcdf4 is supported.")
            return root
        return file_path
    else:
        raise ValueError(
            f"{type(file_path)} is not supported. Please pass posix path, path string, or FSMap."  # noqa
        )


def check_file_existance(
    file_path: Union[str, Path, FSMap], storage_options: dict = {}
) -> bool:
    """
    Checks if file exists in the specified path

    Parameters
    ----------
    file_path : str or pathlib.Path or fsspec.FSMap
        path to file
    storage_options : dict
        options for cloud storage
    """
    if isinstance(file_path, Path):
        # Check for extension
        if ":/" in str(file_path):
            raise ValueError(f"{file_path} is not a valid posix path.")

        if file_path.exists():
            return True
        else:
            return False
    elif isinstance(file_path, str) or isinstance(file_path, FSMap):
        if isinstance(file_path, FSMap):
            fsmap = file_path
        else:
            fsmap = fsspec.get_mapper(file_path, **storage_options)

        if not fsmap.fs.exists(fsmap.root):
            return False
        else:
            return True
    else:
        raise ValueError(
            f"{type(file_path)} is not supported. Please pass posix path, path string, or FSMap."  # noqa
        )


def check_file_permissions(FILE_DIR):
    try:
        if isinstance(FILE_DIR, FSMap):
            base_dir = os.path.dirname(FILE_DIR.root)
            if not base_dir:
                base_dir = FILE_DIR.root
            TEST_FILE = os.path.join(base_dir, ".permission_test").replace('\\', '/')
            with FILE_DIR.fs.open(TEST_FILE, "w") as f:
                f.write("testing\n")
            FILE_DIR.fs.delete(TEST_FILE)
        elif isinstance(FILE_DIR, Path):
            TEST_FILE = FILE_DIR.joinpath(Path('.permission_test'))
            TEST_FILE.write_text("testing\n")

            # Do python version check since missing_ok is for python 3.9 and up
            if sys.version_info >= (3, 9):
                TEST_FILE.unlink(missing_ok=True)
            else:
                TEST_FILE.unlink()
        else:
            TEST_FILE = os.path.join(FILE_DIR, ".permission_test")
            with open(TEST_FILE, "w") as f:
                f.write("testing\n")
            os.remove(TEST_FILE)
    except Exception:
        raise PermissionError("Writing to specified path is not permitted.")
