import secrets
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple

import fsspec
import numpy as np
from fsspec import AbstractFileSystem
from zarr.storage import FSStore

from ...utils.io import ECHOPYPE_DIR, check_file_permissions, validate_output_path

DEFAULT_ZARR_TEMP_DIR = ECHOPYPE_DIR / "temp_output" / "swap_files"


def _create_zarr_store_map(path, storage_options):
    file_path = validate_output_path(
        source_file=secrets.token_hex(16),
        engine="zarr",
        save_path=path,
        output_storage_options=storage_options,
    )
    return fsspec.get_mapper(file_path, **storage_options)


def delete_store(store: "FSStore | str", fs: Optional[AbstractFileSystem] = None) -> None:
    """
    Delete the store and all its contents.

    Parameters
    ----------
    store : FSStore or str
        The store or store path to delete.
    fs : AbstractFileSystem, optional
        The fsspec file system to use

    Returns
    -------
    None
    """
    if isinstance(store, str):
        if fs is None:
            raise ValueError("Must provide fs if store is a path string")
        store_path = store
    else:
        # Get the file system, this should already have the
        # correct storage options
        fs = store.fs

        # Get the string path to the store
        store_path: str = store.dir_path()

    if fs.exists(store_path):
        print(f"Deleting store: {store_path}")
        # Delete the store when it exists
        fs.rm(store_path, recursive=True)


def create_temp_store(dest_path, dest_storage_options=None, retries: int = 10):
    if dest_path is None:
        # Check permission of cwd, raise exception if no permission
        check_file_permissions(ECHOPYPE_DIR)

        # construct temporary directory that will hold the zarr file
        dest_path = DEFAULT_ZARR_TEMP_DIR
        if not dest_path.exists():
            dest_path.mkdir(parents=True)

    temp_zarr_dir = str(dest_path)

    # Set default storage options if None
    if dest_storage_options is None:
        dest_storage_options = {}

    # attempt to find different zarr_file_name
    attempt = 0
    exists = True
    while exists:
        zarr_store = _create_zarr_store_map(
            path=temp_zarr_dir, storage_options=dest_storage_options
        )
        exists = zarr_store.fs.exists(zarr_store.root)
        attempt += 1

        if attempt == retries and exists:
            raise RuntimeError(
                (
                    "Unable to construct an unused zarr file name for swap ",
                    f"after {retries} retries!",
                )
            )
    return zarr_store


def _get_datagram_max_shape(datagram_dict: Dict[Any, List[np.ndarray]]) -> Optional[Tuple[int]]:
    """
    Get the max shape across all channels for a given datagram.
    """
    # Go through each array list and get the max shape
    # then for each channel append max shape to the n number of arrays
    arr_shapes = []
    for arr_list in datagram_dict.values():
        if arr_list is not None and not all(
            (arr is None) or (arr.size == 0) for arr in arr_list
        ):  # only if there's data in the channel
            arr_shapes.append((len(arr_list),) + max(i.shape for i in arr_list))
    if len(arr_shapes) == 0:
        return None
    return max(arr_shapes, key=itemgetter(1))


def calc_final_shapes(
    data_types: List[str], ping_data_dict: Dict[Any, List[np.ndarray]]
) -> Dict[str, Optional[Tuple[int]]]:
    """Calculate the final shapes for each data type.
    The final shape is the max shape across all channels.

    Example output:
    ```
    {
        'power': (9923, 10417),
        'angle': (9923, 10417, 2),
        'complex': None
    }
    ```

    Parameters
    ----------
    data_types : list
        List of data types to calculate final shapes for
    ping_data_dict : dict
        Dictionary of ping data for each data type

    Returns
    -------
    dict
        Dictionary of final shapes for each data type
    """
    # Compute the max dimension shape for each data type
    datagram_max_shapes = []
    for data_type in data_types:
        # Get the max shape across all channels for a given datagram
        max_shape = _get_datagram_max_shape(ping_data_dict[data_type])
        if max_shape:
            if data_type == "angle":
                # Angle data has 2 variables within,
                # so just take the first 2 dimension shapes
                max_shape = max_shape[:2]
            datagram_max_shapes.append(max_shape)

    all_type_max_shape = None
    if len(datagram_max_shapes) > 0:
        # The the max shape across all data types
        # This should be the maximum expansion shape for all data types
        all_type_max_shape = max(datagram_max_shapes, key=itemgetter(1))

    # Compute final shapes for each data type
    data_type_shapes = {}
    for data_type in data_types:
        # Check the number of channels for a given data type
        n_channels = len(ping_data_dict[data_type])
        if n_channels == 0:
            # If no channels, then set the shape to None
            # since this means no data for this data type
            data_type_shapes[data_type] = None
        elif data_type == "angle":
            # Add 2 to the max shape since angle is 2D array
            data_type_shapes[data_type] = all_type_max_shape + (2,)
        else:
            # For all other data types, just add the max shape
            data_type_shapes[data_type] = all_type_max_shape

    return data_type_shapes
