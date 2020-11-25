"""
Process classes perform computation on EchoData objects.

Some operations are instrument-dependent, such as calibration to obtain Sv.

Some operations are instrument-agnostic, such as obtaining MVBS or detecting bottom.
"""
import warnings
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

    def __init__(self, model=None, ed=None):
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
        self._env_params = {}   # env parameters
        self._cal_params = {}   # cal parameters, eg: equivalent beam width sa_correction for EK60
        self._proc_params = {}  # proc parameters, eg: MVBS bin size

        if ed is not None:
            self.init_cal_params(ed)
            self.init_env_params(ed)

    @property
    def proc_params(self):
        # TODO: discussion: use dict of dict
        #   self.proc_params['MVBS'] = {k: v}
        # TODO: ngkavin: implement: get_MVBS() / remove_noise() / get_noise_estimates()
        # get_MVBS() related params:
        #  - MVBS_source: 'Sv' or 'Sv_cleaned'
        #  - MVBS_type: 'binned' or 'rolling'
        #               so far we've only had binned averages (what coarsen is doing)
        #               let's add the functionality to use rolling
        #  - MVBS_ping_num or MVBS_time_interval (one of them has to be given)
        #     - MVBS_ping_num:
        #     - MVBS_time_interval: can use .groupby/resample().mean() or .rolling().mean(),
        #                           based on ping_time
        #       ?? x.resample(time='30s').mean()
        #     - MVBS_distance_interval: can use .groupby().mean(),
        #                               based on distance calculated by lat/lon from Platform group,
        #                               let's put this the last to add
        #  - MVBS_range_bin_num or MVBS_range_interval (use left close right open intervals for now)
        #     - MVBS_range_bin_num:
        #     - MVBS_range_interval: can use .groupby.resample().mean() or .rolling().mean(),
        #                            based on the actual range in meter
        #
        # remove_noise() related params:
        #   - noise_est_ping_num
        #   - noise_est_range_bin_num
        #   - operation: before we calculate the minimum value within each ping number-range bin tile
        #                and use map() to do the noise removal operation.
        #                I think we can use xr.align() with the correct `join` parameter (probably 'left')
        #                to perform the same operation.
        #                Method get_noise_estimates() would naturally be part of the remove_noise() operation.
        #
        # TODO: leewujung: prototype this
        # db_diff() params:
        #   - New method that creates 0-1 (yes-no) masks for crude scatterer classification
        #     based on thresholding the difference of Sv or Sv_clean across pairs of frequencies.
        #   - db_diff_threshold: ('freq1', 'freq2', iterable), the iterable could be
        #
        #   - quick implementation:
        #     ```
        #     # 2-sided threshold: -16 < del_Sv_200_38 <= 2
        #     MVBS_fish_lowrank = xr.where(
        #         -16 < (MVBS_lowrank_200kHz - MVBS_lowrank_38kHz).values,
        #         ds_rpca['low_rank'], np.nan)
        #     MVBS_fish_lowrank = xr.where(
        #         (MVBS_lowrank_200kHz - MVBS_lowrank_38kHz).values <= 2, MVBS_fish_lowrank, np.nan)
        #     ```
        return self._proc_params

    @property
    def env_params(self):
        return self._env_params

    @env_params.setter
    def env_params(self, params):
        if params is None:
            return
        valid_params = ['sea_water_salinity', 'sea_water_temperature',
                        'sea_water_pressure', 'speed_of_sound_in_sea_water', 'seawater_absorption']
        self._env_params = self._check_valid_params(params, self._env_params, valid_params)

    @property
    def cal_params(self):
        return self._cal_params

    @cal_params.setter
    def cal_params(self, params):
        if params is None:
            return
        valid_params = ['gain_correction', 'sa_correction', 'sample_interval', 'slope',
                        'equivalent_beam_angle', 'transmit_power', 'transmit_duration_nominal']
        self._cal_params = self._check_valid_params(params, self._cal_params, valid_params)

    def _check_valid_params(self, params, current_params, valid_params):
        tmp_params = current_params.copy()
        tmp_params.update(params)
        # Removes invalid parameterss
        current_params = {k: v for k, v in tmp_params.items() if k in valid_params}
        if tmp_params != current_params:
            invalid = [k for k in params.keys() if k not in valid_params]
            msg = f"{invalid} will not be used because they are not valid parameters."
            warnings.warn(msg)
        return current_params

    def init_env_params(self, ed, params={}):
        if 'sea_water_salinity' not in params:
            params['sea_water_salinity'] = 29.6     # Default salinity in ppt
            print("Initialize using default sea water salinity of 29.6 ppt")
        if 'sea_water_pressure' not in params:
            params['sea_water_pressure'] == 60      # Default pressure in dbars
            print("Initialize using default sea water salinity of 60 dbars")
        if 'sea_water_temperature' not in params:
            if self.sonar_model == 'AZFP':
                with ed._open_dataset(ed.raw_path, group='Environment') as ds_env:
                    params['sea_water_temperature'] = ds_env.temperature
            else:
                params['sea_water_temperature'] = 8.0   # Default temperature in Celsius
        if self.sonar_model in ['EK60', 'EK80', 'EA640']:
            if 'speed_of_sound_in_sea_water' not in params or 'seawater_absorption' not in params:
                ds_env = ed._open_dataset(ed.raw_path, group="Environment")
                if 'speed_of_sound_in_sea_water' not in params:
                    params['speed_of_sound_in_sea_water'] = ds_env.sound_speed_indicative
                if 'seawater_absorption' not in params and self.sonar_model == 'EK60':
                    params['seawater_absorption'] = ds_env.absorption_indicative

        self.env_params = params

        # Recalculate sound speed and absorption coefficient when environment parameters are changed
        if 'speed_of_sound_in_sea_water' not in self.env_params or 'seawater_absorption' not in self.env_params:
            ss, sa = True, True
            if self.sonar_model != 'AZFP':
                ss = False
            self.recalculate_environment(ed, src='user', ss=ss, sa=sa)

    def init_cal_params(self, ed, params={}):
        if self.sonar_model in ['EK60', 'EK80', 'EA640']:
            if 'gain_correction' not in params:
                params['gain_correction'] = ed.raw.get('gain_correction', None)
            if 'equivalent_beam_angle' not in params:
                params['equivalent_beam_angle'] = ed.raw.get('equivalent_beam_angle', None)
            if 'transmit_power' not in params:
                params['transmit_power'] = ed.raw.get('transmit_power', None)
            if 'transmit_duration_nominal' not in params:
                params['transmit_duration_nominal'] = ed.raw.get('transmit_duration_nominal', None)
            if 'sa_correction' not in params:
                params['sa_correction'] = ed.raw.get('sa_correction', None)

            if self.sonar_model in ['EK80', 'EA640']:
                if 'slope' not in params:
                    params['slope'] = ed.raw.get('slope', None)
                if 'sample_interval' not in params:
                    params['sample_interval'] = ed.raw.get('sample_interval', None)

            self.cal_params = params

    def recalculate_environment(self, ed, src='user', ss=True, sa=True):
        """Retrieves the speed of sound and seawater absorption

        Parameters
        ----------
        src : str
            How the parameters are retrieved.
            'user' for calculated from salinity, tempearture, and pressure
            'file' for retrieved from raw data (Not availible for AZFP)
        """
        if ss:
            self._env_params['speed_of_sound_in_sea_water'] = \
                self.process_obj.calc_sound_speed(ed, self.env_params, src)
        if sa:
            fs = 'AZFP' if self.sonar_model == 'AZFP' else 'FG'
            self._env_params['seawater_absorption'] = \
                self.process_obj.calc_seawater_absorption(ed, self.env_params, src, formula_source=fs)

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
        # Users then have the option to use ed.Sv.zarr() or other xarray function
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

    def get_Sv(self, ed, save=False, save_path=None, save_format='zarr'):
        print('%s  calibrating data in %s' % (dt.now().strftime('%H:%M:%S'), ed.raw_path))
        return self.process_obj.get_Sv(ed=ed, env_params=self.env_params, cal_params=self.cal_params,
                                       save=save, save_path=save_path, save_format=save_format)

    def get_TS(self, ed, save=False, save_path=None, save_format='zarr'):
        print('%s  calibrating data in %s' % (dt.now().strftime('%H:%M:%S'), ed.raw_path))
        return self.process_obj.get_TS(ed=ed, env_params=self.env_params, cal_params=self.cal_params,
                                       save=save, save_path=save_path, save_format=save_format)
    def get_MVBS(self, ed=None, save=False, save_format='zarr'):
        if ed is None:
            pass
            # TODO: print out the need for ed as an input argument
        return self.process_obj.get_MVBS(ed=ed, proc_params=self.proc_params, save=save, save_format=save_format)

    def remove_noise(self, ed=None, save=False, save_format='zarr'):
        if ed is None:
            pass
            # TODO: print out the need for ed as an input argument
        return self.process_obj.remove_noise(ed=ed, proc_params=self.proc_params, save=save, save_format=save_format)
