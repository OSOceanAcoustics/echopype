"""
Process classes perform computation on EchoData objects.

Some operations are instrument-dependent, such as calibration to obtain Sv.

Some operations are instrument-agnostic, such as obtaining MVBS or detecting bottom.
"""
import warnings
from datetime import datetime as dt
import xarray as xr
from ..utils import io
from .process_azfp import ProcessAZFP
from .process_ek60 import ProcessEK60
from .process_ek80 import ProcessEK80
from .echodata import EchoData


warnings.simplefilter('always', DeprecationWarning)


class ParamDict(dict):
    def __init__(self, valid_params, param_type, values={}):
        self.valid_params = valid_params
        self.param_type = param_type

        self.update(values)

    def __setitem__(self, key, value):
        # Checks if keys are in the list of valid parameters before saving them

        if self.param_type == 'process':
            if key in self.valid_params:
                if key not in self:
                    # Initialize process parameters with empty list
                    super().__setitem__(key, {})
            else:
                warnings.warn(f"{key} will be excluded from proc_params because it is not a valid process",
                              stacklevel=2)
                return
            for param, val in value.items():
                if param in self.valid_params[key]:
                    super().__getitem__(key).__setitem__(param, val)
                else:
                    warnings.warn(f"{param} will be excluded because it is not a valid parameter of {key}",
                                  stacklevel=2)
        else:
            if key in self.valid_params:
                super().update({key: value})
            else:
                warnings.warn(f"{key} will be excluded because it is not a valid {self.param_type} parameter",
                              stacklevel=2)

    def update(self, param_dict):
        for k, v in param_dict.items():
            self[k] = v


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
        # TODO: Used for backwards compatibility. Delete in future versions
        if model.lower().endswith('.nc') or model.lower().endswith('.zarr'):
            warnings.warn("`Process` has changed. See docs for information on how to use "
                          "the new `Process` class. The old workflow will be removed "
                          "in a future version.", DeprecationWarning, 3)
            raw = model
            self._temp_ed = EchoData(raw)
            engine = 'netcdf4' if model.lower().endswith('.nc') else 'zarr'
            with xr.open_dataset(model, engine=engine) as ds_top:
                model = ds_top.keywords
            ed = self._temp_ed

        self.sonar_model = model   # type of echosounder
        self.process_obj = self.PROCESS_SONAR[model]   # process object to use

        self._env_params = ParamDict(self.get_valid_params('env'), 'environment')
        self._cal_params = ParamDict(self.get_valid_params('cal'), 'calibration')
        self._proc_params = ParamDict(self.get_valid_params('proc'), 'process')

        if ed is not None:
            self.init_env_params(ed)
            self.init_cal_params(ed)
            self.init_proc_params()

    # ---------------------------------------
    # TODO: Accessing Sv/TS/MVBS/Sv_clean from process is for backwards compatibility and
    #  will be replaced with EchoData
    # TODO: this is currently NOT working
    @property
    def Sv(self):
        return self._temp_ed.Sv

    @property
    def TS(self):
        warnings.warn("TS has been renamed to Sp and so is deprecated.", DeprecationWarning, 3)
        return self._temp_ed.Sp

    @property
    def Sv_clean(self):
        return self._temp_ed.Sv_clean

    @property
    def MVBS(self):
        return self._temp_ed.MVBS

    @property
    def Sv_path(self):
        self._temp_ed.close()
        return self._temp_ed.Sv_path

    @property
    def TS_path(self):
        self._temp_ed.close()
        warnings.warn("TS_path has been renamed to Sp_path and so is deprecated.", DeprecationWarning, 3)
        return self._temp_ed.Sp_path

    @property
    def Sv_clean_path(self):
        self._temp_ed.close()
        return self._temp_ed.Sv_clean_path

    @property
    def MVBS_path(self):
        self._temp_ed.close()
        return self._temp_ed.MVBS_path
    # -----------------------------------------

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

    @property
    def cal_params(self):
        return self._cal_params

    @env_params.setter
    def env_params(self, params):
        self._env_params = ParamDict(self.get_valid_params('env'), 'environment', params)

    @cal_params.setter
    def cal_params(self, params):
        self._cal_params = ParamDict(self.get_valid_params('cal'), 'calibration', params)

    @proc_params.setter
    def proc_params(self, params):
        self._env_params = ParamDict(self.get_valid_params('proc'), 'process', params)

    def get_valid_params(self, param_type):
        """Provides the parameters that the users can set.

        Parameters
        ----------
        param_type : str
            'env', 'cal', or 'proc' to get the valid environment, calibration, and process
            parameters respectively.

        Returns
        -------
        List ('env', 'cal') or dictionary ('proc') of valid parameters of the selected type.
        """
        if param_type == 'env':
            params = ['water_salinity', 'water_temperature',
                      'water_pressure', 'speed_of_sound_in_water', 'absorption']

        elif param_type == 'cal':
            params = ['gain_correction', 'equivalent_beam_angle']
            if self.sonar_model == 'AZFP':
                params += ['EL', 'DS', 'TVR', 'VTX', 'Sv_offset']
            elif self.sonar_model in ['EK60', 'EK80', 'EA640']:
                params += ['transmit_power', 'sa_correction']
                if self.sonar_model in ['EK80', 'EA640']:
                    params += ['slope']

        elif param_type == 'proc':
            params = {
                'MVBS': ['source', 'type', 'ping_num',
                         'time_interval', 'distance_interval', 'range_bin_num',
                         'range_interval'],
                'noise_est': ['ping_num', 'range_bin_num',
                              'SNR']
            }

        return params

    def init_env_params(self, ed, params=None):
        if params is None:
            params = {}
        if 'water_salinity' not in params:
            params['water_salinity'] = 29.6     # Default salinity in ppt
            # print("Initialize using default water salinity of 29.6 ppt")
        if 'water_pressure' not in params:
            params['water_pressure'] = 60      # Default pressure in dbars
            # print("Initialize using default water pressure of 60 dbars")
        ds_env = ed.get_env_from_raw()
        if 'water_temperature' not in params:
            if self.sonar_model == 'AZFP':
                params['water_temperature'] = ds_env.temperature
            else:
                params['water_temperature'] = 8.0   # Default temperature in Celsius
                # print("Initialize using default water temperature of 8 Celsius")
        if self.sonar_model in ['EK60', 'EK80', 'EA640']:
            if 'speed_of_sound_in_water' not in params:
                params['speed_of_sound_in_water'] = ds_env.sound_speed_indicative
                # Reindex arrays where environment ping_time does not match beam group ping_time
                if len(params['speed_of_sound_in_water'].ping_time) != len(ed.raw.ping_time):
                    params['speed_of_sound_in_water'] = \
                        params['speed_of_sound_in_water'].reindex({'ping_time': ed.raw.ping_time}, method='ffill')
            if 'absorption' not in params and self.sonar_model == 'EK60':
                params['absorption'] = ds_env.absorption_indicative

        self.env_params = params

        # Recalculate sound speed and absorption coefficient when environment parameters are changed
        ss = True if 'speed_of_sound_in_water' not in self.env_params else False
        sa = True if 'absorption' not in self.env_params else False
        if ss or sa:
            self.recalculate_environment(ed, src='user', ss=ss, sa=sa)

    def init_cal_params(self, ed, params=None):
        if params is None:
            params = {}
        valid_params = self.get_valid_params('cal')

        # Parameters that require additional computation
        # For EK80 BB mode, there is no sa correction table so the sa correction is saved as none
        if 'sa_correction' in valid_params:
            params['sa_correction'] = self.process_obj.get_power_cal_params(ed=ed, param='sa_correction')
            valid_params.remove('sa_correction')
        if 'gain_correction' in valid_params and 'gain_correction' not in ed.raw:
            params['gain_correction'] = self.process_obj.get_power_cal_params(ed=ed, param='gain_correction')
            valid_params.remove('gain_correction')

        for param in valid_params:
            if param not in params:
                params[param] = ed.raw.get(param, None)

        self.cal_params = params

    def init_proc_params(self, params=None):

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
        if params is None:
            params = {}
        default_dictionary = {'source': 'Sv',
                              'type': 'binned',
                              'SNR': 0,
                              'ping_num': 10,
                              'range_bin_num': 100}
        for proccess, param_list in self.get_valid_params('proc').items():
            if proccess not in params:
                params[proccess] = {}
            for param in param_list:
                if param not in params and param in default_dictionary:
                    params[proccess][param] = default_dictionary[param]

        self.proc_params.update(params)

    def recalculate_environment(self, ed, src='user', ss=True, sa=True):
        """Retrieves the speed of sound and absorption

        Parameters
        ----------
        src : str
            How the parameters are retrieved.
            'user' for calculated from salinity, temperature, and pressure
            'file' for retrieved from raw data (Not available for AZFP)
        """
        if ss:
            formula_src = 'AZFP' if self.sonar_model == 'AZFP' else 'Mackenzie'
            self.env_params['speed_of_sound_in_water'] = \
                self.process_obj.calc_sound_speed(ed, self.env_params, src, formula_source=formula_src)
        if sa:
            formula_src = 'AZFP' if self.sonar_model == 'AZFP' else 'FG'
            self.env_params['absorption'] = \
                self.process_obj.calc_absorption(ed, self.env_params, src, formula_source=formula_src)

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
        # TODO: actually do the above, currently the range is forced to be calculated

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
        self.env_params['absorption'] = self.process_obj.calc_absorption()

        # Calculate range
        self.process_obj.calc_range(ed)

        # Swap dim to align raw backscatter to range instead of range_bin
        # Users then have the option to use ed.Sv.zarr() or other xarray function
        # to explore the data.

    def calibrate(self, ed=None, save=True, save_path=None, save_format='zarr'):
        """Calibrate raw data.

        Parameters
        ----------
        ed : EchoData
            EchoData object to operate on
        save : bool
        save_format : str
            either 'zarr' or 'netcdf4'
        """
        # TODO: ed temporarily not required for backwards compatibility
        if ed is None:
            if hasattr(self, '_temp_ed'):
                ed = self._temp_ed
                save_format = 'netcdf4'
            else:
                raise TypeError("`get_Sv` missing required EchoData object")
        # Perform calibration
        #  this operation should make an xarray Dataset in ed.Sv
        self.get_Sv(
            ed,
            save=save,
            save_path=save_path,
            save_format=save_format
        )

    # TODO: calibratie_TS added for backwards compatibility and will be deleted in the future
    def calibrate_TS(self, ed=None, save=True, save_path=None, save_format='zarr'):
        """Calibrate raw data.
        """
        warnings.warn("`calibrate_TS` is deprecated. Use `get_Sp` instead.", DeprecationWarning, 3)
        if ed is None:
            if hasattr(self, '_temp_ed'):
                ed = self._temp_ed
                save_format = 'netcdf4'
            else:
                raise TypeError("`get_Sp` missing required EchoData object")
        self.get_Sp(
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

    def get_Sv(self, ed=None, save=False, save_path=None, save_format='zarr'):
        """Compute volume backscattering strength (Sv) from raw data.

        Parameters
        ----------
        ed : EchoData
            EchoData object to operate on
        save : bool
        save_format : str
            either 'zarr' or 'netcdf4'

        Returns
        -------
        Dataset of volume backscatter (Sv)
        """
        # TODO: ed temporarily not required for backwards compatibility
        if ed is None:
            if hasattr(self, '_temp_ed'):
                ed = self._temp_ed
            else:
                raise TypeError("`get_Sp` missing required EchoData object")

        # Check to see if required calibration and environment parameters were initialized
        self._check_initialized(['env', 'cal'])
        # Check to see if the data in the raw file matches the calibration function to be used
        self._check_model_echodata_match(ed)
        # TODO: below print out the "[" and "]" when there is only one file
        print(f"{dt.now().strftime('%H:%M:%S')}  calibrating data in {ed.raw_path}")
        return self.process_obj.get_Sv(ed=ed, env_params=self.env_params, cal_params=self.cal_params,
                                       save=save, save_path=save_path, save_format=save_format)

    def get_Sp(self, ed, save=False, save_path=None, save_format='zarr'):
        """Compute point backscattering strength (Sp) from raw data.

        Parameters
        ----------
        ed : EchoData
            EchoData object to operate on
        save : bool
        save_format : str
            either 'zarr' or 'netcdf4'

        Returns
        -------
        Dataset point backscattering strength (Sp)
        """
        # TODO: ed temporarily not required for backwards compatibility
        if ed is None:
            if hasattr(self, '_temp_ed'):
                ed = self._temp_ed
            else:
                raise TypeError("`get_Sp` missing required EchoData object")

        # Check to see if required calibration and environment parameters were initialized
        self._check_initialized(['env', 'cal'])
        # Check to see if the data in the raw file matches the calibration function to be used
        self._check_model_echodata_match(ed)
        # TODO: below print out the "[" and "]" when there is only one file
        print(f"{dt.now().strftime('%H:%M:%S')}  calibrating data in {ed.raw_path}")
        return self.process_obj.get_Sp(ed=ed, env_params=self.env_params, cal_params=self.cal_params,
                                       save=save, save_path=save_path, save_format=save_format)

    def get_MVBS(self, ed=None, save=False, save_format='zarr'):
        """Averages tiles in Sv

        Parameters
        ----------
        ed : EchoData
            EchoData object to operate on
        save : bool
        save_format : str
            either 'zarr' or 'netcdf4'

        Returns
        -------
        Dataset Mean Volume Backscattering Strength (MVBS)
        """
        # TODO: ed temporarily not required for backwards compatibility
        if ed is None:
            if hasattr(self, '_temp_ed'):
                ed = self._temp_ed
            else:
                raise TypeError("`get_MVBS` missing required EchoData object")

        # TODO: Change to checking env and cal as well when additional ways of averaging are implemented
        if ed.Sv is None and ed.Sv_clean is None:
            raise ValueError("Data has not been calibrated. "
                             "Call `Process.calibrate(EchoData)` to calibrate.")
        self._check_initialized(['proc'])
        self._check_model_echodata_match(ed)
        print(f"{dt.now().strftime('%H:%M:%S')}  calculating MVBS for {self.proc_params['MVBS']['source']}")
        return self.process_obj.get_MVBS(ed=ed, env_params=self.env_params, cal_params=self.cal_params,
                                         proc_params=self.proc_params, save=save, save_format=save_format)

    def remove_noise(self, ed=None, save=False, save_format='zarr'):
        """Removes noise from Sv data

        Parameters
        ----------
        ed : EchoData
            EchoData object to operate on
        save : bool
        save_format : str
            either 'zarr' or 'netcdf4'

        Returns
        -------
        Dataset cleaned Sv (Sv_clean)
        """
        # TODO: ed temporarily not required for backwards compatibility
        if ed is None:
            if hasattr(self, '_temp_ed'):
                ed = self._temp_ed
            else:
                raise TypeError("`remove_noise` missing required EchoData object")
        if ed.Sv is None:
            raise ValueError("Data has not been calibrated. "
                             "Call `Process.calibrate(EchoData)` to calibrate.")
        self._check_initialized(['env', 'cal', 'proc'])
        self._check_model_echodata_match(ed)
        print(f"{dt.now().strftime('%H:%M:%S')}  removing noise in Sv")
        return self.process_obj.remove_noise(ed=ed, env_params=self.env_params, cal_params=self.cal_params,
                                             proc_params=self.proc_params, save=save, save_format=save_format)
