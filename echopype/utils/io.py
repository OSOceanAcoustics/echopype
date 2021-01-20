"""
echopype utilities for file handling
"""
import os
from collections.abc import MutableMapping
from fsspec import FSMap


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
    elif isinstance(file, MutableMapping):
        file = file.root

    if file.endswith('.nc'):
        return 'netcdf4'
    elif file.endswith('.zarr'):
        return 'zarr'
    else:
        raise ValueError(f"Unsupported file format: {os.path.splitext(file)[1]}")


def check_file_permissions(FILE_DIR):
    try:
        if isinstance(FILE_DIR, FSMap):
            base_dir = os.path.dirname(FILE_DIR.root)
            TEST_FILE = os.path.join(base_dir, ".permission_test")
            with FILE_DIR.fs.open(TEST_FILE, "w") as f:
                f.write("testing\n")
            FILE_DIR.fs.delete(TEST_FILE)
        else:
            TEST_FILE = os.path.join(FILE_DIR, ".permission_test")
            with open(TEST_FILE, "w") as f:
                f.write("testing\n")
            os.remove(TEST_FILE)
        return True
    except Exception as e:  # pragma: no cover
        print(e)
        return False
