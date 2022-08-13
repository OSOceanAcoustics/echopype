from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ..core import PathHint

from .echodata import EchoData


def open_converted(
    converted_raw_path: "PathHint",
    storage_options: Dict[str, str] = None,
    **kwargs
    # kwargs: Dict[str, Any] = {'chunks': 'auto'} # TODO: do we need this?
):
    """Create an EchoData object from a single converted netcdf or zarr file.

    Parameters
    ----------
    converted_raw_path : str
        path to converted data file
    storage_options : dict
        options for cloud storage
    kwargs : dict
        optional keyword arguments to be passed
        into xr.open_dataset

    Returns
    -------
    EchoData object
    """
    # TODO: combine multiple files when opening
    return EchoData.from_file(
        converted_raw_path=converted_raw_path,
        storage_options=storage_options,
        open_kwargs=kwargs,
    )
