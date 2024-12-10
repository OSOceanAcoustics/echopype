import abc

from ..echodata import EchoData
from ..utils.log import _init_logger
from .ecs import ECSParser

logger = _init_logger(__name__)


class CalibrateBase(abc.ABC):
    """Class to handle calibration for all sonar models."""

    def __init__(self, echodata: EchoData, env_params=None, cal_params=None, ecs_file=None):
        self.echodata = echodata
        self.sonar_type = None
        self.ecs_file = ecs_file
        self.ecs_dict = {}

        # Set ECS to overwrite user-provided dict
        if self.ecs_file is not None:
            if env_params is not None or cal_params is not None:
                logger.warning(
                    "The ECS file takes precedence when it is provided. "
                    "Parameter values provided in 'env_params' and 'cal_params' will not be used!"
                )

            # Parse ECS file to a dict
            ecs = ECSParser(self.ecs_file)
            ecs.parse()
            self.ecs_dict = ecs.get_cal_params()  # apply ECS hierarchy
            self.env_params = {}
            self.cal_params = {}

        else:
            if env_params is None:
                self.env_params = {}
            elif isinstance(env_params, dict):
                self.env_params = env_params
            else:
                raise ValueError("'env_params' has to be None or a dict")

            if cal_params is None:
                self.cal_params = {}
            elif isinstance(cal_params, dict):
                self.cal_params = cal_params
            else:
                raise ValueError("'cal_params' has to be None or a dict")

        # range_meter is computed in compute_Sv/TS in child class
        self.range_meter = None

    @abc.abstractmethod
    def compute_echo_range(self, **kwargs):
        """Calculate range (``echo_range``) in units meter.

        Returns
        -------
        range_meter : xr.DataArray
            range in units meter
        """
        pass

    @abc.abstractmethod
    def _cal_power_samples(self, cal_type, **kwargs):
        """Calibrate power data for EK60, EK80, and AZFP.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'TS' for calculating target strength
        """
        pass

    @abc.abstractmethod
    def compute_Sv(self, **kwargs):
        pass

    @abc.abstractmethod
    def compute_TS(self, **kwargs):
        pass

    def _add_params_to_output(self, ds_out):
        """Add all cal and env parameters to output Sv dataset."""
        # Add env_params
        for key, val in self.env_params.items():
            ds_out[key] = val

        # Add cal_params
        for key, val in self.cal_params.items():
            ds_out[key] = val

        return ds_out

    def _check_echodata_backscatter_size(self):
        """
        Extracts total size of `backscatter_r` and `backscatter_i` in a beam group.
        If the size is above 2 GiB, raises a warning showing a recommended workflow
        that will not overwhelm the system memory.
        """
        # Initialize total nbytes
        if self.echodata.sonar_model == "EK80":
            # Select source of backscatter data
            beam = self.echodata[self.ed_beam_group]

            # Go through waveform and encode cases
            if (self.waveform_mode == "BB") or (
                self.waveform_mode == "CW" and self.encode_mode == "complex"
            ):
                total_nbytes = beam["backscatter_r"].nbytes + beam["backscatter_i"].nbytes
            elif self.waveform_mode == "CW" and self.encode_mode == "power":
                total_nbytes = beam["backscatter_r"].nbytes
        else:
            total_nbytes = self.echodata["Sonar/Beam_group1"]["backscatter_r"].nbytes

        # Compute GigaBytes from Bytes
        total_gb = total_nbytes / (1024**3)

        # Raise Warning if above 2.0
        if total_gb > 2.0:
            logger.warning(
                "The Echodata Backscatter Variables are large and can cause memory issues. "
                "Consider modifying compute_Sv workflow: "
                "Prior to `compute_Sv` run `echodata.chunk(CHUNK_DICTIONARY) "
                "and after `compute_Sv` run `ds_Sv.to_zarr(ZARR_STORE, compute=True)`. "
                "This will ensure that the computation is lazily evaluated, "
                "with the results stored directly in a Zarr store on disk, rather then in memory."
            )
