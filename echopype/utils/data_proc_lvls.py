"""
Contains common functions used throughout the data
processing levels.
"""
import pathlib
from typing import Optional, Tuple, Union

import xarray as xr

from echopype.utils.io import check_file_existence, get_file_format, validate_output_path


def validate_source_ds(
    source_ds: Union[xr.Dataset, str, pathlib.Path], storage_options: Optional[dict]
) -> Tuple[Union[xr.Dataset, str, xr.DataArray], Optional[str]]:
    """
    This function ensures that ``source_ds`` is of the correct
    type and validates the path of ``source_ds``, if it is provided.

    Parameters
    ----------
    source_ds: xr.Dataset or str or pathlib.Path
        A source that points to a Dataset. If the input is a path, it specifies
        the path to a zarr or netcdf file.
    storage_options: dict, optional
        Any additional parameters for the storage backend, corresponding to the
        path provided for ``source_ds``

    Returns
    -------
    source_ds: xr.Dataset or str
        A Dataset which will be the same as the input ``source_ds`` or a validated
        path to a zarr or netcdf file
    file_type: {"netcdf4", "zarr"}, optional
        The file type of the input path if ``source_ds`` is a path, otherwise ``None``
    """

    # initialize file_type
    file_type = None

    # make sure that storage_options is of the appropriate type
    if not isinstance(storage_options, dict):
        raise TypeError("storage_options must be a dict!")

    # check that source_ds is of the correct type, if it is a path validate
    # the path and open the dataset using xarray
    if not isinstance(source_ds, (xr.Dataset, str, pathlib.Path)):
        raise TypeError("source_ds must be a Dataset or str or pathlib.Path!")
    elif isinstance(source_ds, (str, pathlib.Path)):

        # determine if we obtained a zarr or netcdf file
        file_type = get_file_format(source_ds)

        # validate source_ds if it is a path
        source_ds = validate_output_path(
            source_file="blank",  # will be unused since source_ds cannot be none
            engine=file_type,
            output_storage_options=storage_options,
            save_path=source_ds,
        )

        # check that the path exists
        check_file_existence(file_path=source_ds, storage_options=storage_options)

    return source_ds, file_type
