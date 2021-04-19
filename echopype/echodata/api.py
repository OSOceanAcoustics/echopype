from .echodata import EchoData


def open_converted(converted_raw_path, storage_options=None):
    """Create an EchoData object from a single converted zarr/nc files."""
    # TODO: combine multiple files when opening
    return EchoData(
        converted_raw_path=converted_raw_path, storage_options=storage_options
    )
