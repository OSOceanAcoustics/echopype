"""
echopype utilities for file handling
"""
import os


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
