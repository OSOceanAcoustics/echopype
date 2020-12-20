"""
Process classes perform computation on EchoData objects.

Some operations are instrument-dependent, such as calibration to obtain Sv.

Some operations are instrument-agnostic, such as obtaining MVBS or detecting bottom.
"""
import warnings
from datetime import datetime as dt
from .process_azfp import ProcessAZFP
from .process_ek60 import ProcessEK60
from .process_ek80 import ProcessEK80


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
        'AZFP': ProcessAZFP(),
        'EK60': ProcessEK60(),
        'EK80': ProcessEK80(),
        'EA640': ProcessEK80(),
    }

    def __init__(self, model=None, ed=None):
        self.sonar_model = model   # type of echosounder
        self.process_obj = self.PROCESS_SONAR[model]   # process object to use

        self._env_params = {}   # env parameters
        self._cal_params = {}   # cal parameters, eg: equivalent beam width sa_correction for EK60
        self._proc_params = {}  # proc parameters, eg: MVBS bin size

        if ed is not None:
            self.init_cal_params(ed)
            self.init_env_params(ed)
            self.init_proc_params()

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
        self._env_params = self._check_valid_params(params, self._env_params, self.get_valid_params('env'))

    @property
    def cal_params(self):
        return self._cal_params

    @cal_params.setter
    def cal_params(self, params):
        self._cal_params = self._check_valid_params(params, self._cal_params, self.get_valid_params('cal'))

    @proc_params.setter
    def proc_params(self, params):
        self._proc_params = self._check_valid_params(params, self._proc_params, self.get_valid_params('proc'))

    def _check_valid_params(self, params, current_params, valid_params):
        tmp_params = current_params.copy()
        tmp_params.update(params)
        # Removes invalid parameters
        current_params = {k: v for k, v in tmp_params.items() if k in valid_params}
        if tmp_params != current_params:
            invalid = [k for k in params.keys() if k not in valid_params]
            msg = f"{invalid} will not be used because they are not valid parameters."
            warnings.warn(msg)
        return current_params

    def get_valid_params(self, param_type):
        """Provides the parameters that the users can set.

        Parameters
        ----------
        param_type : str
            'env', 'cal', or 'proc' to get the valid environment, calibration, and process
            parameters respectively.

        Returns
        -------
        List of valid parameters of the selected type.
        """
        if param_type == 'env':
            params = ['water_salinity', 'water_temperature',
                      'water_pressure', 'speed_of_sound_in_water', 'seawater_absorption']

        elif param_type == 'cal':
            params = ['gain_correction', 'sample_interval',
                      'equivalent_beam_angle', 'transmit_duration_nominal']
            if self.sonar_model == 'AZFP':
                params += ['EL', 'DS', 'TVR', 'VTX', 'Sv_offset']
            elif self.sonar_model in ['EK60', 'EK80', 'EA640']:
                params += ['transmit_power', 'sa_correction']
                if self.sonar_model in ['EK80', 'EA640']:
                    params += ['slope']

        elif param_type == 'proc':
            params = ['MVBS_source', 'MVBS_type', 'MVBS_ping_num',
                      'MVBS_time_interval', 'MVBS_distance_interval', 'MVBS_range_bin_num',
                      'MVBS_range_interval', 'noise_est_ping_num', 'noise_est_range_bin_num']
        return params

    def init_env_params(self, ed, params={}):
        if 'water_salinity' not in params:
            params['water_salinity'] = 29.6     # Default salinity in ppt
            # print("Initialize using default water salinity of 29.6 ppt")
        if 'water_pressure' not in params:
            params['water_pressure'] = 60      # Default pressure in dbars
            # print("Initialize using default water pressure of 60 dbars")
        ds_env = ed.get_env_from_raw()
        if 'water_temperature' not in params:
            if self.sonar_model == 'AZFP':
                params['water_temperature'] = ds_env.temperature.mean('ping_time')
            else:
                params['water_temperature'] = 8.0   # Default temperature in Celsius
                # print("Initialize using default water temperature of 8 Celsius")
        if self.sonar_model in ['EK60', 'EK80', 'EA640']:
            if 'speed_of_sound_in_water' not in params or 'seawater_absorption' not in params:
                if 'speed_of_sound_in_water' not in params:
                    params['speed_of_sound_in_water'] = ds_env.sound_speed_indicative
                if 'seawater_absorption' not in params and self.sonar_model == 'EK60':
                    params['seawater_absorption'] = ds_env.absorption_indicative

        self.env_params = params

        # Recalculate sound speed and absorption coefficient when environment parameters are changed
        ss = True if 'speed_of_sound_in_water' not in self.env_params else False
        sa = True if 'seawater_absorption' not in self.env_params else False
        if ss or sa:
            self.recalculate_environment(ed, src='user', ss=ss, sa=sa)

    # TODO: make parameters a list
    def init_cal_params(self, ed, params={}):
        valid_params = self.get_valid_params('cal')
        for param in valid_params:
            if param not in params:
                params[param] = ed.raw.get(param, None)

        self.cal_params = params

    def init_proc_params(self, params={}):
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
        # TODO this might not be the best way to initialize the values
        default_values = ['Sv', 'binned', 30,
                          None, None, 1000,
                          None, 30, 1000]
        for k, v in zip(self.get_valid_params('proc'), default_values):
            if k not in params:
                params[k] = v
        self.proc_params = params

    def recalculate_environment(self, ed, src='user', ss=True, sa=True):
        """Retrieves the speed of sound and seawater absorption

        Parameters
        ----------
        src : str
            How the parameters are retrieved.
            'user' for calculated from salinity, temperature, and pressure
            'file' for retrieved from raw data (Not available for AZFP)
        """
        if ss:
            formula_src = 'AZFP' if self.sonar_model == 'AZFP' else 'Mackenzie'
            self._env_params['speed_of_sound_in_water'] = \
                self.process_obj.calc_sound_speed(ed, self.env_params, src, formula_source=formula_src)
        if sa:
            formula_src = 'AZFP' if self.sonar_model == 'AZFP' else 'FG'
            self._env_params['seawater_absorption'] = \
                self.process_obj.calc_seawater_absorption(ed, self.env_params, src, formula_source=formula_src)

    def _check_model_echodata_match(self, ed):
        """Check if sonar model corresponds with the type of data in EchoData object.
        """
        if ed.raw_path is None:
            raise ValueError("EchoData object does not have raw files to calibrate")
        else:
            if ed.sonar_model != self.sonar_model:
                raise ValueError(f"Proccess sonar {self.sonar_model} does not match EchoData sonar {ed.sonar_model}")

    def align_to_range(self, ed, param_source='file'):
        """
        Align raw backscatter data along `range` in meter
        instead of `range_bin` in the original data files.

        Parameters
        ----------
        ed : EchoData
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

        #  If not already specified by user, calculate sound speed and absorption
        self.env_params['speed_of_sound_in_water'] = self.process_obj.calc_sound_speed()
        self.env_params['seawater_absorption'] = self.process_obj.calc_seawater_absorption()

        # Calculate range
        self.process_obj.calc_range(ed)

        # Swap dim to align raw backscatter to range instead of range_bin
        # Users then have the option to use ed.Sv.zarr() or other xarray function
        # to explore the data.

    def calibrate(self, ed, save=True, save_path=None, save_format='zarr'):
        """Calibrate raw data.

        Parameters
        ----------
        ed : EchoData
            EchoData object to operate on
        save : bool
        save_format : str
        """

        # Perform calibration
        #  this operation should make an xarray Dataset in ed.Sv
        self.get_Sv(
            ed,
            save=save,
            save_path=save_path,
            save_format=save_format
        )

    def _check_initialized(self, groups=['env', 'cal', 'proc']):
        """Raises an error if the specified parameters among the environment, calibration, and process
        parameters were not initialized"""
        if 'env' in groups:
            if not self.env_params:
                raise ValueError("Environment not initialized. Call init_env_params() to initialize.")
        if 'cal' in groups:
            if not self.cal_params:
                raise ValueError("Calibration parameters not initialized. Call init_cal_params() to initialize.")
        if 'proc' in groups:
            if not self.proc_params:
                raise ValueError("Process parameters not initialized. Call init_proc_params() to initialize.")

    def get_Sv(self, ed, save=False, save_path=None, save_format='zarr'):
        """Compute volumn backscattering strength (Sv) from raw data.

        Parameters
        ----------
        ed : EchoData
            EchoData object to operate on
        save : bool
        save_format : str

        Returns
        -------
        Dataset of volume backscatter (Sv)
        """
        # Check to see if required calibration and environment parameters were initialized
        self._check_initialized(['env', 'cal'])
        # Check to see if the data in the raw file matches the calibration function to be used
        self._check_model_echodata_match(ed)
        print(f"{dt.now().strftime('%H:%M:%S')}  calibrating data in {ed.raw_path}")
        return self.process_obj.get_Sv(ed=ed, env_params=self.env_params, cal_params=self.cal_params,
                                       save=save, save_path=save_path, save_format=save_format)

    def get_Sp(self, ed, save=False, save_path=None, save_format='zarr'):
        """Compute point backscattering strength (Sp) from raw data.

        Parameters
        ----------
        ed : EchoData
            EchoData object to operate on
        save :å bool
        save_format : str

        Returns
        -------
        Dataset point backscattering strength (Sp)
        """
        # Check to see if required calibration and environment parameters were initialized
        self._check_initialized(['env', 'cal'])
        # Check to see if the data in the raw file matches the calibration function to be used
        self._check_model_echodata_match(ed)
        print(f"{dt.now().strftime('%H:%M:%S')}  calibrating data in {ed.raw_path}")
        return self.process_obj.get_Sp(ed=ed, env_params=self.env_params, cal_params=self.cal_params,
                                       save=save, save_path=save_path, save_format=save_format)

    def get_MVBS(self, ed, save=False, save_format='zarr'):
        """Averages tiles in Sv

        Parameters
        ----------
        ed : EchoData
            EchoData object to operate on
        save : bool
        save_format : str

        Returns
        -------
        Dataset Mean Volume Backscattering Strength (MVBS)
        """
        return self.process_obj.get_MVBS(ed=ed, env_params=self.env_params, cal_params=self.cal_params,
                                         proc_params=self.proc_params, save=save, save_format=save_format)

    def remove_noise(self, ed, save=False, save_format='zarr'):
        """Removes noise from Sv data

        Parameters
        ----------
        ed : EchoData
            EchoData object to operate on
        save : bool
        save_format : str

        Returns
        -------
        Dataset cleaned Sv (Sv_clean)
        """
        return self.process_obj.remove_noise(ed=ed, env_params=self.env_params,
                                             proc_params=self.proc_params, save=save, save_format=save_format)
