from pathlib import Path
import xarray as xr
from zarr.errors import GroupNotFoundError


def open_converted(converted_raw_path, storage_options=None):
    return EchoData(converted_raw_path=converted_raw_path, storage_options=storage_options)


class EchoData:
    """Echo data model class for handling raw converted data,
    including multiple files associated with the same data set.
    """

    def __init__(self, converted_raw_path=None, storage_options=None):

        # TODO: consider if should open datasets in init
        #  or within each function call when echodata is used. Need to benchmark.

        self.top = None
        self.beam = None
        self.beam_power = None  # only present for EK80 containing both complex and power data, may remain empty
        self.environment = None
        self.vendor = None

        self.storage_options = storage_options if storage_options is not None else {}

        # TODO: verify if converted_raw_path is valid on either local or remote filesystem
        if converted_raw_path:
            self._load_file(converted_raw_path)
            self.sonar_model = self.top.keywords

        # EK80 data may have a Beam_power group if both complex and power data exist.
        if self.sonar_model == 'EK80':
            try:
                self.beam_power = self._load_groups(converted_raw_path, group='Beam_power')
            except (OSError, GroupNotFoundError):
                pass

    def _load_file(self, raw_path):
        """Lazy load Top-level, Beam, Environment, and Vendor groups from raw file.
        """
        self.top = self._load_groups(raw_path)
        self.beam = self._load_groups(raw_path, group='Beam')
        self.environment = self._load_groups(raw_path, group='Environment')
        self.vendor = self._load_groups(raw_path, group='Vendor')

    @staticmethod
    def _load_groups(filepath, group=None):
        # TODO: handle multiple files through the same set of checks for combining files
        suffix = Path(filepath).suffix
        if suffix == '.nc':
            return xr.open_dataset(filepath, group=group)
        elif suffix == '.zarr':
            return xr.open_zarr(filepath, group=group)
        else:
            raise ValueError('Input file type not supported!')
