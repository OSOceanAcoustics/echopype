import os
import warnings
import datetime as dt
import echopype
from echopype.calibrate.calibrate_ek import CalibrateEK60, CalibrateEK80
from echopype.calibrate.calibrate_azfp import CalibrateAZFP
from echopype.preprocess import api as preprocess

CALIBRATOR = {
    'EK60': CalibrateEK60,
    'EK80': CalibrateEK80,
    'AZFP': CalibrateAZFP
}


class Process():
    """Class used for providing backwards compatibility with echopype 0.4.1.
    Allows users to create Process objects from which they can process their data by using
    the new Calibrate and Preprocess functions under the hood.
    """
    def __init__(self, file_path="", salinity=27.9, pressure=59, temperature=None):
        warnings.warn("Initializing a Process object is deprecated. "
                      "More information on how to use the new processing "
                      "functions can be found in the echopype documentation.",
                      DeprecationWarning, 3)
        self.echodata = echopype.open_converted(file_path)
        self._file_format = None
        self.file_path = file_path

        # Initialize AZFP environment parameters. Will error if None
        if self.echodata.sonar_model == 'AZFP':
            if temperature is None:
                temperature = self.echodata.environment['temperature']
            self._env_params = {'salinity': salinity,
                                'temperature': temperature,
                                'pressure': pressure}
        else:
            self._env_params = None

        if 'backscatter_i' in self.echodata.beam:
            self.waveform_mode = 'BB'
            self.encode_mode = 'complex'
        else:
            self.waveform_mode = 'CW'
            self.encode_mode = 'power'

        self.calibrator = CALIBRATOR[self.echodata.sonar_model](
            self.echodata,
            env_params=self._env_params,
            cal_params=None,
            waveform_mode=self.waveform_mode,
        )
        # Deprecated data attributes
        self.Sv = None
        self.Sv_path = None
        self.Sv_clean = None
        self.TS = None
        self.TS_path = None

        # Deprecated proc attributes
        self.noise_est_range_bin_size = 5  # meters per tile for noise estimation
        self.noise_est_ping_size = 30  # number of pings per tile for noise estimation
        self.MVBS_range_bin_size = 5  # meters per tile for MVBS
        self.MVBS_ping_size = 30  # number of pings per tile for MVBS

        if self.file_path .upper().endswith('.NC'):
            self._file_format = 'netcdf4'
        if self.file_path .upper().endswith('.ZARR'):
            self._file_format = 'zarr'

    @property
    def salinity(self):
        return self.calibrator.env_params['salinity']

    @salinity.setter
    def salinity(self, val):
        self.calibrator.env_params['salinity'] = val

    @property
    def temperature(self):
        return self.calibrator.env_params['temperature']

    @temperature.setter
    def temperature(self, val):
        self.calibrator.env_params['temperature'] = val

    @property
    def sound_speed(self):
        return self.calibrator.env_params['sound_speed']

    @sound_speed.setter
    def sound_speed(self, val):
        self.calibrator.env_params['sound_speed'] = val

    @property
    def seawater_absorption(self):
        return self.calibrator.env_params['sound_absorption']

    @seawater_absorption.setter
    def seawater_absorption(self, val):
        self.calibrator.env_params['seawater_absorption'] = val

    @property
    def range(self):
        if self.calibrator.range_meter is None:
            self.calibrator.compute_range_meter('Sv')
        return self.calibrator.range_meter

    @range.setter
    def range(self, rr):
        self.calibrator.range_meter = rr

    def validate_path(self, save_path=None, save_postfix='_Sv', file_path=''):
        """Creates a directory if it doesn't exist. Returns a valid save path.
        """
        def _assemble_path():
            file_in = os.path.basename(file_path)
            file_name, file_ext = os.path.splitext(file_in)
            return file_name + save_postfix + file_ext

        file_path = file_path if file_path else self.file_path
        if save_path is None:
            save_dir = os.path.dirname(file_path)
            file_out = _assemble_path()
        else:
            path_ext = os.path.splitext(save_path)[1]
            # If given save_path is file, split into directory and file
            if path_ext != '':
                save_dir, file_out = os.path.split(save_path)
                if save_dir == '':  # save_path is only a filename without directory
                    save_dir = os.path.dirname(file_path)  # use directory from input file
            # If given save_path is a directory, get a filename from input .nc file
            else:
                save_dir = save_path
                file_out = _assemble_path()

        # Create folder if not already exists
        if save_dir == '':
            # TODO: should we use '.' instead of os.getcwd()?
            save_dir = os.getcwd()  # explicit about path to current directory
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        return os.path.join(save_dir, file_out)

    def _save_dataset(self, ds, path, mode="w"):
        """Save dataset to the appropriate formats.
        A utility method to use the correct function to save the dataset,
        based on the input file format.
        Parameters
        ----------
        ds : xr.Dataset
            xarray dataset object
        path : str
            output file
        """
        if self._file_format == 'netcdf4':
            ds.to_netcdf(path, mode=mode)
        elif self._file_format == 'zarr':
            ds.to_zarr(path, mode=mode)

    def calibrate(self, save=False, save_postfix='_Sv', save_path=None, waveform_mode=None, encode_mode=None):
        """Calibrate Sv by using the sonar specific calibrator"""
        # Use averaged temperature for AZFP
        env_params = self._env_params.copy() if self._env_params is not None else None
        if self.echodata.sonar_model == 'AZFP':
            env_params['temperature'] = env_params['temperature'].mean('ping_time').values
        encode_mode = encode_mode if encode_mode is not None else self.encode_mode
        waveform_mode = waveform_mode if waveform_mode is not None else self.waveform_mode

        # Call calibrate from Calibrate class
        self.Sv = echopype.calibrate.compute_Sv(
            echodata=self.echodata,
            waveform_mode=waveform_mode,
            encode_mode=encode_mode,
            env_params=env_params
        )
        if save:
            self.Sv_path = self.validate_path(save_path, save_postfix)
            print('%s  saving calibrated Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
            self._save_dataset(self.Sv, self.Sv_path, mode="w")

    def calibrate_TS(self, save=False, save_postfix='_TS', save_path=None, waveform_mode=None, encode_mode=None):
        """Calibrate Sp by using the sonar specific calibrator"""
        # Use averaged temperature for AZFP
        env_params = self._env_params.copy() if self._env_params is not None else None
        if self.echodata.sonar_model == 'AZFP':
            env_params['temperature'] = env_params['temperature'].mean('ping_time').values
        encode_mode = encode_mode if encode_mode is not None else self.encode_mode
        waveform_mode = waveform_mode if waveform_mode is not None else self.waveform_mode

        # Call calibrate from Calibrate class
        self.TS = echopype.calibrate.compute_Sp(
            echodata=self.echodata,
            waveform_mode=waveform_mode,
            encode_mode=encode_mode,
            env_params=env_params
        )
        self.TS = self.TS.rename(Sp='TS')
        if save:
            self.TS_path = self.validate_path(save_path, save_postfix)
            print('%s  saving calibrated TS to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.TS_path))
            self._save_dataset(self.TS, self.TS_path, mode="w")

    def calibrate_cw(self, mode='Sv', file_path='', save=False, save_path=None, save_postfix=None):
        """EK80 CW calibration"""
        if mode == 'Sv':
            self.calibrate(save=save, save_postfix=save_postfix, save_path=save_path, waveform_mode='CW')
        elif mode == 'TS':
            self.calibrate_TS(save=save, save_postfix=save_postfix, save_path=save_path, waveform_mode='CW')
        else:
            raise ValueError("Unsupported calibration mode")

    def _get_proc_Sv(self, source_path=None, source_postfix='_Sv'):
        """Private method to return calibrated Sv either from memory or _Sv.nc file.
        This method is called by remove_noise(), noise_estimates() and get_MVBS().
        """
        if self.Sv is None:  # calibration not yet performed
            Sv_path = self.validate_path(save_path=source_path,  # wrangle _Sv path
                                         save_postfix=source_postfix)
            if os.path.exists(Sv_path):  # _Sv exists
                self.Sv = self._open_dataset(Sv_path)  # load _Sv file
            else:
                # if path specification given but file do not exist:
                if (source_path is not None) or (source_postfix != '_Sv'):
                    print('%s  no calibrated data found in specified path: %s' %
                          (dt.datetime.now().strftime('%H:%M:%S'), Sv_path))
                else:
                    print('%s  data has not been calibrated. ' % dt.datetime.now().strftime('%H:%M:%S'))
                print('          performing calibration now and operate from Sv in memory.')
                self.calibrate()  # calibrate, have Sv in memory
        return self.Sv

    def noise_estimates(self, source_postfix='_Sv', source_path=None,
                        noise_est_range_bin_size=None, noise_est_ping_size=None):

        if source_path is None:
            if self.Sv is None:
                self.calibrate()
            else:
                proc_data = self.Sv
            source = "memory"
        else:
            proc_data = self._get_proc_Sv(source_path=source_path, source_postfix=source_postfix)
            source = source_path
        print('%s  Sv source used to estimate noise: %s' % (dt.datetime.now().strftime('%H:%M:%S'), source))

        range_bin_size = noise_est_range_bin_size if \
            noise_est_range_bin_size is not None else self.noise_est_range_bin_size
        ping_size = noise_est_ping_size if noise_est_ping_size is not None else self.noise_est_ping_size

        return preprocess.estimate_noise(proc_data, ping_size, range_bin_size)

    def remove_noise(self, source_postfix='_Sv', source_path=None,
                     noise_est_range_bin_size=None, noise_est_ping_size=None,
                     SNR=0, Sv_threshold=None,
                     save=False, save_postfix='_Sv_clean', save_path=None):

        if source_path is None:
            if self.Sv is None:
                self.calibrate()
            else:
                proc_data = self.Sv
            source = "memory"
        else:
            proc_data = self._get_proc_Sv(source_path=source_path, source_postfix=source_postfix)
            source = source_path
        print('%s  Sv source used to remove noise: %s' % (dt.datetime.now().strftime('%H:%M:%S'), source))

        range_bin_size = noise_est_range_bin_size if \
            noise_est_range_bin_size is not None else self.noise_est_range_bin_size
        ping_size = noise_est_ping_size if noise_est_ping_size is not None else self.noise_est_ping_size

        self.Sv_clean = preprocess.remove_noise(
            proc_data,
            ping_num=ping_size,
            range_bin_num=range_bin_size,
            noise_max=Sv_threshold,
            SNR_threshold=SNR
        )
        # Attributes will cause an error while saving if set to None
        del self.Sv_clean.attrs['noise_max']
        if save:
            self.Sv_clean_path = self.validate_path(save_path=save_path, save_postfix=save_postfix)
            print('%s  saving denoised Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_clean_path))
            self._save_dataset(self.Sv_clean, self.Sv_clean_path)

    def get_MVBS(self, source_postfix='_Sv', source_path=None,
                 MVBS_range_bin_size=None, MVBS_ping_size=None,
                 save=False, save_postfix='_MVBS', save_path=None):

        if source_path is not None:
            proc_data = self._get_proc_Sv(source_path=source_path, source_postfix=source_postfix)
            source = source_path
        else:
            if source_postfix == '_Sv_clean':
                if self.Sv_clean is None:
                    self.remove_noise()
                proc_data = self.Sv_clean
            else:
                if self.Sv is None:
                    self.calibrate()
                proc_data = self.Sv
            source = "memory"
        print('%s  Sv source used to calculate MVBS: %s' % (dt.datetime.now().strftime('%H:%M:%S'), source))

        range_bin_size = MVBS_range_bin_size if MVBS_range_bin_size is not None else self.MVBS_range_bin_size
        ping_size = MVBS_ping_size if MVBS_ping_size is not None else self.MVBS_ping_size

        # Range must have ping_time or it will error. Range does not have ping time for AZFP when temps are averaged
        if proc_data['range'].ndim == 2:
            proc_data = proc_data.assign(range=self.range)

        self.MVBS = preprocess.compute_MVBS_index_binning(
            proc_data,
            range_bin_num=range_bin_size,
            ping_num=ping_size
        ).rename({'Sv': 'MVBS'})
        # Save results in object and as a netCDF file
        if save:
            self.MVBS_path = self.validate_path(save_path=save_path, save_postfix=save_postfix)
            print('%s  saving MVBS to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.MVBS_path))
            self._save_dataset(self.MVBS, self.MVBS_path)
        # Close opened resources
        proc_data.close()


class ProcessAZFP():
    def __new__(self, file_path="", salinity=27.9, pressure=59, temperature=None):
        return Process(file_path=file_path, salinity=salinity, pressure=pressure, temperature=temperature)


class ProcessEK60():
    def __new__(self, file_path=""):
        return Process(file_path=file_path)


class ProcessEK80():
    def __new__(self, file_path=""):
        return Process(file_path=file_path)
