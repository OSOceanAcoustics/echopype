from os import path

from .echodata import EchoData

import fsspec


def open_converted(converted_raw_path, storage_options=None):
    """Create an EchoData object from a single converted zarr/nc files."""
    # TODO: combine multiple files when opening

    def _check_if_file_exists(file_path) -> bool:
        """Helper function to check if file in converted_raw_path exists."""
        file_exists_locally = path.isfile(file_path)
        fsmap = fsspec.get_mapper(file_path)
        file_exists_in_url = fsmap.fs.exists(fsmap.root)
        if file_exists_locally or file_exists_in_url:
            return True
        return False

    if not _check_if_file_exists(converted_raw_path):
        raise FileNotFoundError(f"File {converted_raw_path} not found")
    return EchoData(
        converted_raw_path=converted_raw_path, storage_options=storage_options
    )
