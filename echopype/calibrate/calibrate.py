import numpy as np
import xarray as xr
from ..utils import uwa

ENV_PARAMS = (
    'temperature', 'salinity', 'pressure',
    'sound_speed', 'sound_absorption'
)

CAL_PARAMS = {
    'EK': ('sa_correction', 'gain_correction', 'equivalent_beam_angle'),
    'AZFP': ('EL', 'DS', 'TVR', 'VTX', 'equivalent_beam_angle', 'Sv_offset')
}


class CalibrateBase:
    """Class to handle calibration for all sonar models.
    """

    def __init__(self, echodata, range_meter=None):
        self.sonar_model = None
        self._range_meter = range_meter

        self.echodata = echodata

        # initialize all env params to None
        self.env_params = dict.fromkeys(ENV_PARAMS)

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

    def get_Sv(self, **kwargs):
        pass

    def get_Sp(self, **kwargs):
        pass


class CalibrateEK(CalibrateBase):

    def __init__(self, echodata):
        super().__init__(echodata)

        # cal params specific to EK echosounders
        self.cal_params = dict.fromkeys(CAL_PARAMS['EK'])

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

        return ds_vend[param].sel(pulse_length_bin=idx_wanted).drop('pulse_length_bin')

    def get_cal_params(self, cal_params):
        """Get cal params using user inputs or values from data file.

        Parameters
        ----------
        cal_params : dict
        """
        # Params from the Vendor group
        params_from_vend = ['sa_correction', 'gain_correction']
        for p in params_from_vend:
            # substitute if None in user input
            self.cal_params[p] = cal_params[p] if p in cal_params else self._get_vend_power_cal_params(p)

        # Other params
        self.cal_params['equivalent_beam_angle'] = (cal_params['equivalent_beam_angle']
                                                    if 'equivalent_beam_angle' in cal_params
                                                    else self.echodata.raw_beam['equivalent_beam_angle'])

    def _cal_power(self, cal_type, use_raw_beam_power=False):
        """Calibrate power data from EK60 and EK80.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'Sp' for calculating point backscattering strength
        use_raw_beam_power : bool
            whether to use raw_beam_power.
            If True use echodata.raw_beam; if False use echodata.raw_beam_power.
            Note raw_beam_power could only exist for EK80 data.

        Returns
        -------
        Sv or Sp
        """
        # Select source of backscatter data
        if use_raw_beam_power:
            raw_beam = self.echodata.raw_beam_power
        else:
            raw_beam = self.echodata.raw_beam

        # Derived params
        wavelength = self.env_params['sound_speed'] / raw_beam['frequency']  # wavelength
        range_meter = self.range_meter

        # Transmission loss
        spreading_loss = 20 * np.log10(range_meter.where(range_meter >= 1, other=1))
        absorption_loss = 2 * self.env_params['sound_absorption'] * range_meter

        if cal_type == 'Sv':
            # Calc gain
            CSv = (10 * np.log10(raw_beam['transmit_power'])
                   + 2 * self.cal_params['gain_correction']
                   + self.cal_params['equivalent_beam_angle']
                   + 10 * np.log10(wavelength ** 2
                                   * raw_beam['transmit_duration_nominal']
                                   * self.env_params['sound_speed']
                                   / (32 * np.pi ** 2)))

            # Calibration and echo integration
            out = (raw_beam['backscatter_r']
                   + spreading_loss + absorption_loss
                   - CSv - 2 * self.cal_params['sa_correction'])
            out.name = 'Sv'

        elif cal_type == 'Sp':
            # Calc gain
            CSp = (10 * np.log10(raw_beam['transmit_power'])
                   + 2 * self.cal_params['gain_correction']
                   + 10 * np.log10(wavelength ** 2 / (16 * np.pi ** 2)))

            # Calibration and echo integration
            out = (raw_beam.backscatter_r
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
        range_meter = range_meter.transpose('frequency', 'ping_time', 'range_bin')  # conform with backscatter dim order
        range_meter.name = 'range'  # add name to facilitate xr.merge
        return range_meter

    def get_env_params(self, env_params):
        """Get env params using user inputs or values from data file.

        EK60 file by default contains only sound speed and absorption.
        In cases when temperature, salinity, and pressure values are supplied
        by the user simultaneously, the sound speed and absorption are re-calculated.

        Parameters
        ----------
        env_params : dict
        """
        # Re-calculate environment parameters if user supply all env variables
        if ('temperature' in env_params) and ('salinity' in env_params) and ('pressure' in env_params):
            for p in ['temperature', 'salinity', 'pressure']:
                self.env_params[p] = env_params[p]
            self.env_params['sound_speed'] = uwa.calc_sound_speed(temperature=self.env_params['temperature'],
                                                                  salinity=self.env_params['salinity'],
                                                                  pressure=self.env_params['pressure'])
            self.env_params['sound_absorption'] = uwa.calc_absorption(frequency=self.echodata.raw_beam['frequency'],
                                                                      temperature=self.env_params['temperature'],
                                                                      salinity=self.env_params['salinity'],
                                                                      pressure=self.env_params['pressure'])
        # Otherwise get sound speed and absorption from user inputs or raw data file
        else:
            self.env_params['sound_speed'] = (env_params['sound_speed']
                                              if 'sound_speed' in env_params
                                              else self.echodata.raw_env['sound_speed_indicative'])
            self.env_params['sound_absorption'] = (env_params['sound_absorption']
                                                   if 'sound_absorption' in env_params
                                                   else self.echodata.raw_env['absorption_indicative'])

    def get_Sv(self):
        return self._cal_power(cal_type='Sv')

    def get_Sp(self):
        return self._cal_power(cal_type='Sp')


class CalibrateEK80(CalibrateEK):

    def __init__(self, echodata):
        super().__init__(echodata)

        # cal params used by both complex and power data calibration
        # TODO: will have to add complex data-specific params, like the freq-dependent gain factor
        self.cal_params = dict.fromkeys(CAL_PARAMS['EK'])

    def get_env_params(self, env_params):
        """Get env params using user inputs or values from data file.

        EK80 file by default contains sound speed, temperature, depth, salinity, and acidity,
        therefore absorption is always calculated unless it is supplied by the user.
        In cases when temperature, salinity, and pressure values are supplied
        by the user simultaneously, both the sound speed and absorption are re-calculated.

        Parameters
        ----------
        env_params : dict
        """
        # Re-calculate environment parameters if user supply all env variables
        if ('temperature' in env_params) and ('salinity' in env_params) and ('pressure' in env_params):
            for p in ['temperature', 'salinity', 'pressure']:
                self.env_params[p] = env_params[p]
            self.env_params['sound_speed'] = uwa.calc_sound_speed(temperature=self.env_params['temperature'],
                                                                  salinity=self.env_params['salinity'],
                                                                  pressure=self.env_params['pressure'])
            self.env_params['sound_absorption'] = uwa.calc_absorption(frequency=self.echodata.raw_beam['frequency'],
                                                                      temperature=self.env_params['temperature'],
                                                                      salinity=self.env_params['salinity'],
                                                                      pressure=self.env_params['pressure'])
        # Otherwise
        #  get temperature, salinity, and pressure from raw data file
        #  get sound speed from user inputs or raw data file
        #  get absorption from user inputs or computing from env params stored in raw data file
        else:
            # pressure is encoded as "depth" in EK80  # TODO: change depth to pressure in EK80 file?
            for p1, p2 in zip(['temperature', 'salinity', 'pressure'],
                              ['temperature', 'salinity', 'depth']):
                self.env_params[p1] = env_params[p1] if p1 in env_params else self.echodata.raw_env[p2]
            self.env_params['sound_speed'] = (env_params['sound_speed']
                                              if 'sound_speed' in env_params
                                              else self.echodata.raw_env['sound_speed_indicative'])
            self.env_params['sound_absorption'] = (
                env_params['sound_absorption']
                if 'sound_absorption' in env_params
                else uwa.calc_absorption(frequency=self.echodata.raw_beam['frequency'],
                                         temperature=self.env_params['temperature'],
                                         salinity=self.env_params['salinity'],
                                         pressure=self.env_params['pressure']))

    def _cal_complex(self, cal_type):
        return 1

    def get_Sv(self, mode=None):
        """Compute volume backscattering strength (Sv).

        Parameters
        ----------
        mode : str
            For EK80 data by default calibration is performed for all available data
            including complex and power samples.
            Use ``complex`` to compute Sv from only complex samples,
            and ``power`` to compute Sv from only power samples,

            For all other sonar systems, calibration for power samples is performed.

        Returns
        -------
        Sv : xr.DataSet
            A DataSet containing volume backscattering strength (``Sv``)
            and the corresponding range (``range``) in units meter.
            For EK80 data, data variable ``Sv`` contains Sv computed from power samples, and
            data variable ``Sv_complex`` contains Sv computed from complex samples.
            They are separately stored due to the dramatically different sample interval along range.
        """
        # Default to computing Sv from both power and complex samples
        flag_complex = True
        flag_power = True

        # Default to use self.echodata.raw_beam for _cal_power
        use_raw_beam_power = False

        # Figure out what cal to do
        if hasattr(self.echodata, 'raw_beam_power'):  # both power and complex samples exist
            use_raw_beam_power = True  # use self.echodata.raw_beam_power for _cal_power
            if mode == 'complex':
                flag_power = False
            elif mode == 'power':
                flag_complex = False
            else:
                raise ValueError('Input mode not recognized!')
        else:  # only power OR complex samples exist
            if 'quadrant' in self.echodata.raw_beam.dims:  # complex samples
                flag_power = False
                if mode == 'power':
                    raise TypeError('EchoData does not contain power samples!')  # user selects the wrong mode
            else:  # power samples
                flag_complex = False
                if mode == 'complex':
                    raise TypeError('EchoData does not contain complex samples!')  # user selects the wrong mode

        # Compute Sv and combine data sets if both Sv and Sv_complex are calculated
        if flag_power:
            ds_Sv = self._cal_power(cal_type='Sv', use_raw_beam_power=use_raw_beam_power)
        else:
            ds_Sv = xr.Dataset()
        if flag_complex:
            ds_Sv_complex = self._cal_complex(cal_type='Sv')
        else:
            ds_Sv_complex = xr.Dataset()
        ds_combine = xr.merge((ds_Sv, ds_Sv_complex))

        return ds_combine
