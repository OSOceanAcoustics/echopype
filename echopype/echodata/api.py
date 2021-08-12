from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ..core import PathHint

from .echodata import EchoData


def open_converted(
    converted_raw_path: "PathHint", storage_options: Dict[str, str] = None
):
    """Create an EchoData object from a single converted zarr/nc files."""
    # TODO: combine multiple files when opening
    return EchoData(
        converted_raw_path=converted_raw_path, storage_options=storage_options
    )
