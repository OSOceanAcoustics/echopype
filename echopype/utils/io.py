"""
echopype utilities for file handling
"""
import os
import functools
import xarray as xr
from collections import MutableMapping


# TODO: @ngkavin: if you are using this only for the process objects, this should not be here.
#  My suggestion to factor this out was to have it work for both convert and process.
#  Let's discuss what you are doing differently in both cases and see if can combine.

def validate_proc_path(ed, postfix, save_path=None):
    """Creates a directory if it doesn't exist. Returns a valid save path.
    """
    def _assemble_path():
        file_in = os.path.basename(ed.raw_path[0])
        file_name, file_ext = os.path.splitext(file_in)
        return file_name + postfix + file_ext

    if save_path is None:
        save_dir = os.path.dirname(ed.raw_path[0])
        file_out = _assemble_path()
    else:
        path_ext = os.path.splitext(save_path)[1]
        # If given save_path is file, split into directory and file
        if path_ext != '':
            save_dir, file_out = os.path.split(save_path)
            if save_dir == '':  # save_path is only a filename without directory
                save_dir = os.path.dirname(ed.raw_path)  # use directory from input file
        # If given save_path is a directory, get a filename from input .nc file
        else:
            save_dir = save_path
            file_out = _assemble_path()

    # Create folder if not already exists
    if save_dir == '':
        save_dir = os.getcwd()
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    return os.path.join(save_dir, file_out)


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


def get_file_format(file):
    """Gets the file format (either Netcdf4 or Zarr) from the file extension"""
    file = file.root if isinstance(file, MutableMapping) else file
    if file.endswith('.nc'):
        return 'netcdf4'
    elif file.endswith('.zarr'):
        return 'zarr'
    else:
        raise ValueError(f"Unsupported file format: {os.path.splitext(file)[1]}")
