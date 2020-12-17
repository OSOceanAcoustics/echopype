"""
echopype utilities for file handling
"""
import os
import functools
import xarray as xr


# TODO: @ngkavin: if you are using this only for the process objects, this should not be here.
#  My suggestion to factor this out was to have it work for both convert and process.
#  Let's discuss what you are doing differently in both cases and see if can combine.

def validate_proc_path(ed, postfix, save_path=None):
    """Creates a directory if it doesnt exist. Returns a valid save path.
    """
    def _assemble_path():
        file_in = os.path.basename(ed.raw_path)
        file_name, file_ext = os.path.splitext(file_in)
        return file_name + postfix + file_ext

    if save_path is None:
        save_dir = os.path.dirname(ed.raw_path)
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
    valid_ext = ['.nc', '.zarr']
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1] in valid_ext]


def _check_key_param_consistency(group='Beam'):
    """Decorator to check if key params in the files for the specified group
    to make sure the files can be opened together.
    """
    def wrapper(open_dataset):
        functools.wraps(open_dataset)

        def from_raw(ed):
            if ed.raw_path is None:
                raise ValueError("No raw files to open")
            elif len(ed.raw_path) == 1:
                return ed._open_dataset(ed.raw_path[0], group=group)
            else:
                try:
                    ds = open_dataset(ed)
                    if group == 'Vendor':
                        return ed._open_dataset(ed.raw_path[0], group=group)
                    else:
                        return ds
                except xr.MergeError as e:
                    var = str(e).split("'")[1]
                    raise ValueError(f"Files cannot be opened due to {var} changing across the files")
        return from_raw
    return wrapper
