from pathlib import Path
import xarray as xr
from ..calibrate.calibrate import CalibrateEK60

CALIBRATOR = {
    'EK60': CalibrateEK60,
    # 'AZFP': CalibrateAZFP
}


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

        if raw_path:
            self._open_raw(raw_path)
            self.sonar_model = self.raw_top.keywords
        if Sv_path:
            self.Sv = self._open_data_file(Sv_path)

        # TODO: change this to something following to best practices
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

    def get_Sv(self, env_params=None, cal_params=None):
        # Set up calibration object
        cal_obj = CALIBRATOR[self.sonar_model](self)
        if env_params is None:
            env_params = {}
        cal_obj.get_env_params(env_params)
        if cal_params is None:
            cal_params = {}
        cal_obj.get_cal_params(cal_params)

        # Perform calibration
        self.Sv = cal_obj.get_Sv()
