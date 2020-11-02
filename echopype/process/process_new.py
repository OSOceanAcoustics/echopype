"""
Process classes perform computation on EchoData objects.

Some operations are instrument-dependent, such as calibration to obtain Sv.

Some operations are instrument-agnostic, such as obtaining MVBS or detecting bottom.
"""
from ..utils import uwa
from . import process_classes
from .echodata import EchoDataBase


class Process:
    """UI class for using process objects.

    Use case (AZFP):
        ed = EchoData(raw_path='some_path_to_converted_raw_data_files')
        proc = Process(model='AZFP')
        proc.env_params = {'salinity': 35, 'pressure': 200}  # set env params needed for calibration
        proc.save_paths['Sv'] = 'some_path_for_Sv'           # set paths to save Sv data to
        proc.get_Sv(ed, save=True, save_format='zarr')

    """
    # A dictionary of supported echosounder types
    PROCESS_SONAR = {
        'AZFP': process_classes.ProcessAZFP(),
        'EK60': process_classes.ProcessEK60(),
        'EK80': process_classes.ProcessEK80(),
        'EA640': process_classes.ProcessEK80(),
    }

    def __init__(self, model=None):
        self.sonar_model = model   # type of echosounder
        self.process_obj = self.PROCESS_SONAR[model]   # process object to use

        # TODO: we need something to restrict the type of parameters users
        #  can put in to these dictionaries,
        #  for example, for env_params we allow only:
        #     'sea_water_salinity'            [psu]
        #     'sea_water_temperature'         [degC]
        #     'sea_water_pressure'            [dbars] (~depth in meters)
        #     'speed_of_sound_in_sea_water'   [m/s]
        #     'seawater_absorption'           [dB/m]
        self.env_params = {}   # env parameters
        self.cal_params = {}   # cal parameters, eg: sa_correction for EK60
        self.proc_params = {}  # proc parameters, eg: MVBS bin size

        self.save_paths = {}   # paths to save processing results, index by proc type: Sv, TS, etc.

    def _check_model_echodata_match(self, ed):
        """Check if sonar model corresponds with the type of data in EchoData object.
        """
        # Check is self.sonar_model and ed.sonar_model are the same
        # Raise error if they do not match

    def _autofill_save_path(self, save_type):
        """
        Autofill the paths to save the processing results if not already set.
        The default paths will be to the same folder as the raw data files.

        Use case is something like:
            proc._autofill_save_path(save_type='Sv')

        Parameters
        ----------
        save_type
        """

    def align_to_range(self, ed, param_source='file'):
        """
        Align raw backscatter data along `range` in meter
        instead of `range_bin` in the original data files.

        Parameters
        ----------
        ed : EchoDataBase
            EchoData object to operate on
        param_source
        """
        # Check if we already have range calculated from .calibrate()
        # and if so we can just get range from there instead of re-calculating.

        # Check if sonar model matches
        self._check_model_echodata_match(ed)

        # Obtain env parameters
        #  Here we want to obtain the env params stored in the data file, but
        #  overwrite those that are specified by user
        #  We can first do a check to see what parameters we still need to get
        #  from the raw files before retrieving them (I/O is slow,
        #  so let's not do that unless needed).

        # To get access to parameters stored in the raw data, use:
        ed.get_env_from_raw()
        ed.get_vend_from_raw()

        #  If not already specifed by user, calculate sound speed and absorption
        self.env_params['speed_of_sound_in_sea_water'] = self.process_obj.calc_sound_speed()
        self.env_params['seawater_absorption'] = self.process_obj.calc_seawater_absorption()

        # Calculate range
        self.process_obj.calc_range(ed)

        # Swap dim to align raw backscatter to range instead of range_bin
        # Users then have the option to use ed.Sv.to_zarr() or other xarray function
        # to explore the data.

    def calibrate(self, ed, param_source='file', save=True, save_format='zarr'):
        """Calibrate raw data.

        Parameters
        ----------
        ed : EchoDataBase
            EchoData object to operate on
        param_source : str
            'file' or 'user'
        save : bool
        save_format : str
        """
        # Check if sonar model matches
        self._check_model_echodata_match(ed)

        # Obtain env parameters
        #  Here we want to obtain the env params stored in the data file, but
        #  overwrite those that are specified by user
        #  We can first do a check to see what parameters we still need to get
        #  from the raw files before retrieving them (I/O is slow,
        #  so let's not do that unless needed).

        # To get access to parameters stored in the raw data, use:
        ed.get_env_from_raw()
        ed.get_vend_from_raw()

        #  If not already specifed by user, calculate sound speed and absorption
        self.env_params['speed_of_sound_in_sea_water'] = self.process_obj.calc_sound_speed()
        self.env_params['seawater_absorption'] = self.process_obj.calc_seawater_absorption()

        # Obtain cal parameters
        #  Operations are very similar to those from the env parameters,
        #  for AZFP AFAIK some additional parameters are needed from the vendor group
        #  to calculate range

        # Autofill save paths if not already generated
        if save and ('Sv' not in self.save_paths):
            self._autofill_save_path('Sv')

        # Perform calibration
        #  this operation should make an xarray Dataset in ed.Sv
        #  if save=True: save the results as zarr in self.save_paths['Sv'] and update ed.Sv_path
        #  users obviously would have the option to do ed.Sv.to_zarr() to wherever they like
        self.process_obj.get_Sv(
            ed,
            env_params=self.env_params,
            cal_params=self.cal_params,
            save=save,
            save_format=save_format
        )
