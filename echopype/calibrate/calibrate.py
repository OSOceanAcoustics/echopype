import numpy as np
from ..process.echodata_new import EchoDataNew
from ..utils import uwa


class CalibrateBase:
    """Class to handle calibration for all sonar models.
    """

    ENV_PARAMS = (
        'temperature', 'salinity', 'pressure',
        'sound_speed', 'sound_absorption'
    )

    def __init__(self, echodata, range_meter=None):
        """

        Parameters
        ----------
        echodata : EchoDataNew
        """
        self.sonar_model = None
        self._range_meter = range_meter

        self.echodata = echodata

        self.env_params = dict.fromkeys(self.ENV_PARAMS)

    @property
    def range_meter(self):
        return self._range_meter

    @range_meter.setter
    def range_meter(self, val):
        self._range_meter = val

    @range_meter.getter
    def range_meter(self):
        if self._range_meter is None:
            self._range_meter = self.calc_range_meter()
            return self._range_meter
        else:
            return self._range_meter

    def get_env_params(self, **kwargs):
        pass

    def get_cal_params(self, **kwargs):
        pass

    def calc_range_meter(self):
        pass

    def get_Sv(self):
        pass

    # def calc_sound_speed(self):
    #     pass
    #
    # def calc_sound_absorption(self):
    #     pass


class CalibrateEK(CalibrateBase):

    CAL_PARAMS = ('sa_correction', 'gain_correction', 'equivalent_beam_angle')

    def __init__(self, echodata):
        super().__init__(echodata)

        # cal params specific to EK echosounders
        self.cal_params = dict.fromkeys(self.CAL_PARAMS)

    def _get_vend_power_cal_params(self, param):
        """Get cal parameters stored in the Vendor group.

        Parameters
        ----------
        param : str
            name of parameter to retrieve
        """
        # TODO: need to test with EK80 power/angle data
        #  currently this has only been tested with EK60 data
        ds_vend = self.echodata.raw_vend

        if param not in ds_vend:
            return None

        if param not in ['sa_correction', 'gain_correction']:
            raise ValueError(f"Unknown parameter {param}")

        # Drop NaN ping_time for transmit_duration_nominal
        if np.any(np.isnan(self.echodata.raw_beam['transmit_duration_nominal'])):
            # TODO: resolve performance warning:
            #  /Users/wu-jung/miniconda3/envs/echopype_jan2021/lib/python3.8/site-packages/xarray/core/indexing.py:1369:
            #  PerformanceWarning: Slicing is producing a large chunk. To accept the large
            #  chunk and silence this warning, set the option
            #      >>> with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            #      ...     array[indexer]
            #  To avoid creating the large chunks, set the option
            #      >>> with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            #      ...     array[indexer]
            #    return self.array[key]
            self.echodata.raw_beam = self.echodata.raw_beam.dropna(dim='ping_time', how='any',
                                                                   subset=['transmit_duration_nominal'])

        # Find index with correct pulse length
        unique_pulse_length = np.unique(self.echodata.raw_beam['transmit_duration_nominal'], axis=1)
        idx_wanted = np.abs(ds_vend['pulse_length'] - unique_pulse_length).argmin(dim='pulse_length_bin')

        return ds_vend.sa_correction.sel(pulse_length_bin=idx_wanted).drop('pulse_length_bin')

    def _cal_power(self, cal_type):
        """Calibrate narrowband data from EK60 and EK80.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'Sp' for calculating point backscattering strength

        Returns
        -------
        Sv or Sp
        """
        # Derived params
        wavelength = self.env_params['sound_speed'] / self.echodata.raw_beam['frequency']  # wavelength
        range_meter = self.range_meter

        # Transmission loss
        spreading_loss = 20 * np.log10(range_meter.where(range_meter >= 1, other=1))
        absorption_loss = 2 * self.env_params['sound_absorption'] * range_meter

        if cal_type == 'Sv':
            # Calc gain
            CSv = (10 * np.log10(self.echodata.raw_beam['transmit_power'])
                   + 2 * self.cal_params['gain_correction']
                   + self.cal_params['equivalent_beam_angle']
                   + 10 * np.log10(wavelength ** 2
                                   * self.echodata.raw_beam['transmit_duration_nominal']
                                   * self.env_params['sound_speed']
                                   / (32 * np.pi ** 2)))

            # Calibration and echo integration
            out = (self.echodata.raw_beam['backscatter_r']
                   + spreading_loss + absorption_loss
                   - CSv - 2 * self.cal_params['sa_correction'])
            out.name = 'Sv'

        elif cal_type == 'Sp':
            # Calc gain
            CSp = (10 * np.log10(self.echodata.raw_beam['transmit_power'])
                   + 2 * self.cal_params['gain_correction']
                   + 10 * np.log10(wavelength ** 2 / (16 * np.pi ** 2)))

            # Calibration and echo integration
            out = (self.echodata.raw_beam.backscatter_r
                   + spreading_loss * 2 + absorption_loss
                   - CSp)
            out.name = 'Sp'
        else:
            raise ValueError('cal_type not recognized!')

        # Attach calculated range (with units meter) into data set
        out = out.to_dataset()
        out = out.merge(range_meter)

        return out


class CalibrateEK60(CalibrateEK):

    def __init__(self, echodata):
        super().__init__(echodata)
        self.tvg_correction_factor = 2

    def calc_range_meter(self):
        sample_thickness = self.echodata.raw_beam['sample_interval'] * self.env_params['sound_speed'] / 2
        # TODO Check with the AFSC about the half sample difference
        range_meter = (self.echodata.raw_beam.range_bin
                       - self.tvg_correction_factor) * sample_thickness  # [frequency x range_bin]
        range_meter = range_meter.where(range_meter > 0, other=0)
        range_meter = range_meter.transpose('frequency', 'ping_time', 'range_bin')  # conform with backscatter dim order
        range_meter.name = 'range'  # add name to facilitate xr.merge
        return range_meter

    def get_env_params(self, env_params):
        """Set env params using user inputs or values from data file.

        In cases when temperature, salinity, and pressure values are input by the user simultaneously,
        the sound speed and absorption are re-calculated.

        Parameters
        ----------
        env_params : dict
        """
        # Re-calculate environment parameters if user passes in CTD values
        if ('temperature' in env_params) and ('salinity' in env_params) and ('pressure' in env_params):
            self.env_params['sound_speed'] = uwa.calc_sound_speed(env_params['temperature'],
                                                                  env_params['salinity'],
                                                                  env_params['pressure'])
            self.env_params['sound_absorption'] = uwa.calc_absorption(self.echodata.raw_beam['frequency'],
                                                                      env_params['temperature'],
                                                                      env_params['salinity'],
                                                                      env_params['pressure'])
        # Otherwise get sound speed and absorption from user inputs or raw data file
        else:
            self.env_params['sound_speed'] = env_params.get('sound_speed',
                                                            self.echodata.raw_env['sound_speed_indicative'])
            self.env_params['sound_absorption'] = env_params.get('absorption',
                                                                 self.echodata.raw_env['absorption_indicative'])

    def get_cal_params(self, cal_params):
        """Set cal params using user inputs or values from data file.

        Parameters
        ----------
        cal_params : dict
        """
        self.cal_params['sa_correction'] = cal_params.get('sa_correction',
                                                          self._get_vend_power_cal_params('sa_correction'))
        self.cal_params['gain_correction'] = cal_params.get('gain_correction',
                                                            self._get_vend_power_cal_params('gain_correction'))
        self.cal_params['equivalent_beam_angle'] = cal_params.get('equivalent_beam_angle',
                                                                  self.echodata.raw_beam['gain_correction'])

    def get_Sv(self):
        return self._cal_power(cal_type='Sv')

    def get_Sp(self):
        return self._cal_power(cal_type='Sp')


class CalibrateAZFP(CalibrateBase):

    def __init__(self, echodata):
        super().__init__(echodata)

    # def get_Sv(self):
