from pathlib import Path
import xarray as xr
from zarr.errors import GroupNotFoundError


class EchoDataNew:
    """Echo data model class for handling multiple variables/files
    associated with the same data set.
    """

    def __init__(self, raw_path=None, Sv_path=None):

        self.raw_top = None
        self.raw_beam = None
        self.raw_env = None
        self.raw_vend = None
        self.Sv = None

        # TODO: need to consider if should open datasets in init here
        #  or within each function call when echodata is used. Need to benchmark.
        if raw_path:
            # TODO: need to check if raw_path is valid on either local or remote filesystem
            self._open_raw(raw_path)
            self.sonar_model = self.raw_top.keywords
        if Sv_path:
            self.Sv = self._open_data_file(Sv_path)

        # EK80 data may have a Beam_power group if both complex and power data exist.
        if self.sonar_model == 'EK80':
            try:
                self.raw_beam_power = self._open_data_file(raw_path, group='Beam_power')
            except (OSError, GroupNotFoundError):
                pass

        # TODO: change this to something following best practices.
        #  Need something for user to set Sv from outside by either
        #  passing in Sv directly or supplying a file path
        self.paths = {
            'raw': raw_path,
            'Sv': Sv_path
        }

    def _open_raw(self, raw_path):
        """Lazy load Top-level, Beam, Environment, and Vendor groups from raw file.
        """
        self.raw_top = self._open_data_file(raw_path)
        self.raw_beam = self._open_data_file(raw_path, group='Beam')
        self.raw_env = self._open_data_file(raw_path, group='Environment')
        self.raw_vend = self._open_data_file(raw_path, group='Vendor')

    @staticmethod
    def _open_data_file(filepath, group=None):
        suffix = Path(filepath).suffix
        if suffix == '.nc':
            return xr.open_dataset(filepath, group=group)
        elif suffix == '.zarr':
            return xr.open_zarr(filepath, group=group)
        else:
            raise ValueError('Input file type not supported!')
