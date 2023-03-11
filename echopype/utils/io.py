"""
echopype utilities for file handling
"""
import os
import pathlib
import platform
import sys
from pathlib import Path, WindowsPath
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import fsspec
import xarray as xr
from fsspec import FSMap
from fsspec.implementations.local import LocalFileSystem

from ..utils.coding import set_storage_encodings
from ..utils.log import _init_logger

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

logger = _init_logger(__name__)


ECHOPYPE_DIR = Path(os.path.expanduser("~")) / ".echopype"


def init_ep_dir():
    """Initialize hidden directory for echopype"""
    if not ECHOPYPE_DIR.exists():
        ECHOPYPE_DIR.mkdir(exist_ok=True)


def get_files_from_dir(folder):
    """Retrieves all Netcdf and Zarr files from a given folder"""
    valid_ext = [".nc", ".zarr"]
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1] in valid_ext]


def save_file(ds, path, mode, engine, group=None, compression_settings=None, **kwargs):
    """
    Saves a dataset to netcdf or zarr depending on the engine
    If ``compression_settings`` are set, compress all variables with those settings
    """

    # set zarr or netcdf specific encodings for each variable in ds
    encoding = set_storage_encodings(ds, compression_settings, engine)

    # Allows saving both NetCDF and Zarr files from an xarray dataset
    if engine == "netcdf4":
        ds.to_netcdf(path=path, mode=mode, group=group, encoding=encoding, **kwargs)
    elif engine == "zarr":
        ds.to_zarr(store=path, mode=mode, group=group, encoding=encoding, **kwargs)
    else:
        raise ValueError(f"{engine} is not a supported save format")


def get_file_format(file):
    """Gets the file format (either Netcdf4 or Zarr) from the file extension"""
    if isinstance(file, list):
        file = file[0]
    elif isinstance(file, FSMap):
        file = file.root

    if isinstance(file, str) and file.endswith(".nc"):
        return "netcdf4"
    elif isinstance(file, str) and file.endswith(".zarr"):
        return "zarr"
    elif isinstance(file, pathlib.Path) and file.suffix == ".nc":
        return "netcdf4"
    elif isinstance(file, pathlib.Path) and file.suffix == ".zarr":
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
    save_path: Optional[Union[Path, str]] = None,
) -> str:
    """
    Assembles output file names and path.

    The final resulting file will be saved as provided in save path.
    If a directory path is provided then the final file name will use
    the same name as the source file and saved within the directory
    path in `save_path` or echopype's `temp_output` directory.

    Example 1.
    source_file - test.raw
    engine - zarr
    save_path - /path/dir/
    output is /path/dir/test.zarr

    Example 2.
    source_file - test.raw
    engine - zarr
    save_path - None
    output is ~/.echopype/temp_output/test.zarr

    Example 3.
    source_file - test.raw
    engine - zarr
    save_path - /path/dir/myzarr.zarr
    output is /path/dir/myzarr.zarr

    Parameters
    ----------
    source_file : str
        The source file path
    engine : str {'netcdf4', 'zarr'}
        The engine to be used for file output
    output_storage_options : dict
        Storage options for remote output path
    save_path : str | Path | None
        Either a directory or a file path.
        If it's not provided, we will save output file(s)
        in the echopype's `temp_output` directory.

    Returns
    -------
    str
        The final string path of the resulting file.

    Raises
    ------
    ValueError
        If engine is not one of the supported output engine of
        zarr or netcdf
    TypeError
        If `save_path` is not of type Path or str
    """
    if engine not in SUPPORTED_ENGINES:
        ValueError(f"Engine {engine} is not supported for file export.")

    file_ext = SUPPORTED_ENGINES[engine]["ext"]

    if save_path is None:
        logger.warning("A directory or file path is not provided!")

        out_dir = ECHOPYPE_DIR / "temp_output"
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        logger.warning(f"Resulting converted file(s) will be available at {str(out_dir)}")
        out_path = str(out_dir / (Path(source_file).stem + file_ext))
    elif not isinstance(save_path, Path) and not isinstance(save_path, str):
        raise TypeError("save_path must be a string or Path")
    else:
        # convert save_path into a nicely formatted Windows path if we are on
        # a Windows machine and the path is not a cloud storage path. Then convert back to a string.
        if platform.system() == "Windows":
            if isinstance(save_path, str) and ("://" not in save_path):
                save_path = str(WindowsPath(save_path).absolute())

        if isinstance(save_path, str):
            # Clean folder path by stripping '/' at the end
            if save_path.endswith("/") or save_path.endswith("\\"):
                save_path = save_path[:-1]

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
                out_path = str(final_path.parent.joinpath(final_path.stem + file_ext).absolute())
            else:
                path_dir = fsspec.get_mapper(os.path.dirname(save_path), **output_storage_options)
                check_file_permissions(path_dir)
                final_path = Path(save_path)
                out_path = save_path
            if final_path.suffix != file_ext:
                logger.warning(
                    "Mismatch between specified engine and save_path found; forcing output format to engine."  # noqa
                )
    return out_path


def check_file_existence(file_path: "PathHint", storage_options: Dict[str, str] = {}) -> bool:
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
                logger.warning(f"{str(FILE_DIR)} does not exist. Attempting to create it.")
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


def env_indep_joinpath(*args: Tuple[str, ...]) -> str:
    """
    Joins a variable number of paths taking into account the form of
    cloud storage paths.

    Parameters
    ----------
    *args: tuple of str
        A variable number of strings that should be joined in the order
        they are provided

    Returns
    -------
    joined_path: str
        Full path constructed by joining all input strings
    """

    if "://" in args[0]:
        # join paths for cloud storage path
        joined_path = r"/".join(args)
    else:
        # join paths for non-cloud storage path
        joined_path = os.path.join(*args)

    return joined_path


def validate_source_ds_da(
    source_ds_da: Union[xr.Dataset, xr.DataArray, str, Path], storage_options: Optional[dict]
) -> Tuple[Union[xr.Dataset, str, xr.DataArray], Optional[str]]:
    """
    This function ensures that ``source_ds_da`` is of the correct
    type and validates the path of ``source_ds_da``, if it is provided.

    Parameters
    ----------
    source_ds_da: xr.Dataset, xr.DataArray, str or pathlib.Path
        A source that points to a Dataset or DataArray. If the input is a path,
        it specifies the path to a zarr or netcdf file.
    storage_options: dict, optional
        Any additional parameters for the storage backend, corresponding to the
        path provided for ``source_ds_da``

    Returns
    -------
    source_ds_da: xr.Dataset or xr.DataArray or str
        A Dataset or DataArray which will be the same as the input ``source_ds_da`` or
        a validated path to a zarr or netcdf file
    file_type: {"netcdf4", "zarr"}, optional
        The file type of the input path if ``source_ds_da`` is a path, otherwise ``None``
    """

    # initialize file_type
    file_type = None

    # make sure that storage_options is of the appropriate type
    if not isinstance(storage_options, dict):
        raise TypeError("storage_options must be a dict!")

    # check that source_ds_da is of the correct type, if it is a path validate
    # the path and open the Dataset or DataArray using xarray
    if not isinstance(source_ds_da, (xr.Dataset, xr.DataArray, str, Path)):
        raise TypeError("source_ds_da must be a Dataset or DataArray or str or pathlib.Path!")
    elif isinstance(source_ds_da, (str, Path)):
        # determine if we obtained a zarr or netcdf file
        file_type = get_file_format(source_ds_da)

        # validate source_ds_da if it is a path
        source_ds_da = validate_output_path(
            source_file="blank",  # will be unused since source_ds cannot be none
            engine=file_type,
            output_storage_options=storage_options,
            save_path=source_ds_da,
        )

        # check that the path exists
        check_file_existence(file_path=source_ds_da, storage_options=storage_options)

    return source_ds_da, file_type
