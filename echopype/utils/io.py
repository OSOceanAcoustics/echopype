"""
echopype utilities for file handling
"""
import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union

import fsspec
from fsspec import FSMap
from fsspec.implementations.local import LocalFileSystem

if TYPE_CHECKING:
    from ..core import PathHint
SUPPORTED_ENGINES = {
    "netcdf4": {
        "ext": ".nc",
    },
    "zarr": {
        "ext": ".zarr",
    },
}


def get_files_from_dir(folder):
    """Retrieves all Netcdf and Zarr files from a given folder"""
    valid_ext = [".nc", ".zarr"]
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1] in valid_ext]


def save_file(ds, path, mode, engine, group=None, compression_settings=None):
    """Saves a dataset to netcdf or zarr depending on the engine
    If ``compression_settings`` are set, compress all variables with those settings"""
    encoding = (
        {var: compression_settings for var in ds.data_vars}
        if compression_settings is not None
        else {}
    )
    # Allows saving both NetCDF and Zarr files from an xarray dataset
    if engine == "netcdf4":
        ds.to_netcdf(path=path, mode=mode, group=group, encoding=encoding)
    elif engine == "zarr":
        ds.to_zarr(store=path, mode=mode, group=group, encoding=encoding)
    else:
        raise ValueError(f"{engine} is not a supported save format")


def get_file_format(file):
    """Gets the file format (either Netcdf4 or Zarr) from the file extension"""
    if isinstance(file, list):
        file = file[0]
    elif isinstance(file, FSMap):
        file = file.root

    if file.endswith(".nc"):
        return "netcdf4"
    elif file.endswith(".zarr"):
        return "zarr"
    else:
        raise ValueError(f"Unsupported file format: {os.path.splitext(file)[1]}")


def _get_suffix(filepath: Union[str, Path, FSMap]) -> str:
    """Check if file type is supported."""
    # TODO: handle multiple files through the same set of checks for combining files
    if isinstance(filepath, FSMap):
        suffix = Path(filepath.root).suffix
    else:
        suffix = Path(str(filepath)).suffix

    if suffix not in [".nc", ".zarr"]:
        raise ValueError("Input file type not supported!")

    return suffix


def sanitize_file_path(
    file_path: "PathHint",
    storage_options: Dict[str, str] = {},
    is_dir: bool = False,
) -> Union[Path, FSMap]:
    """
    Cleans and checks the user output file path type to
    a standardized Path or FSMap type.

    Parameters
    ----------
    file_path : str | Path | FSMap
        The source file path
    engine : str {'netcdf4', 'zarr'}
        The engine to be used for file output
    storage_options : dict
        Storage options for file path
    is_dir : bool
        Flag for the function to know
        if file_path is a directory or not.
        If not, suffix will be determined.
    """

    if not is_dir:
        suffix = _get_suffix(file_path)
    else:
        suffix = ""

    if isinstance(file_path, Path):
        # Check for extension
        if ":/" in str(file_path):
            raise ValueError(f"{file_path} is not a valid posix path.")

        if suffix == ".zarr":
            return fsspec.get_mapper(str(file_path))
        return file_path
    elif isinstance(file_path, str):
        if "://" in file_path:
            if suffix == ".nc":
                raise ValueError("Only local netcdf4 is supported.")
            return fsspec.get_mapper(file_path, **storage_options)
        elif suffix == ".zarr":
            return fsspec.get_mapper(file_path)
        else:
            return Path(file_path)
    elif isinstance(file_path, fsspec.FSMap):
        root = file_path.root
        if suffix == ".nc":
            if not isinstance(file_path.fs, LocalFileSystem):
                # For special case of netcdf.
                # netcdf4 engine can only read Path or string
                raise ValueError("Only local netcdf4 is supported.")
            return Path(root)
        return file_path
    else:
        raise ValueError(
            f"{type(file_path)} is not supported. Please pass posix path, path string, or FSMap."  # noqa
        )


def validate_output_path(
    source_file: str,
    engine: str,
    output_storage_options: Dict = {},
    save_path: Union[None, Path, str] = None,
) -> str:
    """
    Assemble output file names and path.

    Parameters
    ----------
    source_file : str
        The source file path
    engine : str {'netcdf4', 'zarr'}
        The engine to be used for file output
    output_storage_options : dict
        Storage options for remote output path
    save_path : str | Path | None
        Either a directory or a file. If none then the save path is 'temp_echopype_output/'
        in the current working directory.
    """
    if engine not in SUPPORTED_ENGINES:
        ValueError(f"Engine {engine} is not supported for file export.")

    file_ext = SUPPORTED_ENGINES[engine]["ext"]

    if save_path is None:
        warnings.warn("save_path is not provided")

        current_dir = Path.cwd()
        # Check permission, raise exception if no permission
        check_file_permissions(current_dir)
        out_dir = current_dir.joinpath(Path("temp_echopype_output"))
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        warnings.warn(
            f"Resulting converted file(s) will be available at {str(out_dir)}"
        )
        out_path = str(out_dir / (Path(source_file).stem + file_ext))
    elif not isinstance(save_path, Path) and not isinstance(save_path, str):
        raise TypeError("save_path must be a string or Path")
    else:
        if isinstance(save_path, str):
            # Clean folder path by stripping '/' at the end
            save_path = save_path.strip("/")

            # Determine whether this is a directory or not
            is_dir = True if Path(save_path).suffix == "" else False
        else:
            is_dir = True if save_path.suffix == "" else False

        # Cleans path
        sanitized_path = sanitize_file_path(
            save_path, storage_options=output_storage_options, is_dir=is_dir
        )

        # Check file permissions
        if is_dir:
            check_file_permissions(sanitized_path)
            out_path = os.path.join(save_path, Path(source_file).stem + file_ext)
        else:
            if isinstance(sanitized_path, Path):
                check_file_permissions(sanitized_path.parent)
                final_path = sanitized_path
            else:
                path_dir = fsspec.get_mapper(
                    os.path.dirname(save_path), **output_storage_options
                )
                check_file_permissions(path_dir)
                final_path = Path(save_path)
            if final_path.suffix != file_ext:
                warnings.warn(
                    "Mismatch between specified engine and save_path found; forcing output format to engine."  # noqa
                )
            out_path = str(
                final_path.parent.joinpath(final_path.stem + file_ext).absolute()
            )
    return out_path


def check_file_existence(
    file_path: "PathHint", storage_options: Dict[str, str] = {}
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
            TEST_FILE = os.path.join(base_dir, ".permission_test").replace("\\", "/")
            with FILE_DIR.fs.open(TEST_FILE, "w") as f:
                f.write("testing\n")
            FILE_DIR.fs.delete(TEST_FILE)
        elif isinstance(FILE_DIR, (Path, str)):
            if isinstance(FILE_DIR, str):
                FILE_DIR = Path(FILE_DIR)

            if not FILE_DIR.exists():
                warnings.warn(
                    f"{str(FILE_DIR)} does not exist. Attempting to create it."
                )
                FILE_DIR.mkdir(exist_ok=True, parents=True)
            TEST_FILE = FILE_DIR.joinpath(Path(".permission_test"))
            TEST_FILE.write_text("testing\n")

            # Do python version check since missing_ok is for python 3.9 and up
            if sys.version_info >= (3, 9):
                TEST_FILE.unlink(missing_ok=True)
            else:
                TEST_FILE.unlink()
    except Exception:
        raise PermissionError("Writing to specified path is not permitted.")
