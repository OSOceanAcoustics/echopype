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

    def get_Sv(self):
        pass

    def get_Sp(self):
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
            self.env_params['sound_speed'] = uwa.calc_sound_speed(self.env_params['temperature'],
                                                                  self.env_params['salinity'],
                                                                  self.env_params['pressure'])
            self.env_params['sound_absorption'] = uwa.calc_absorption(self.echodata.raw_beam['frequency'],
                                                                      self.env_params['temperature'],
                                                                      self.env_params['salinity'],
                                                                      self.env_params['pressure'])
        # Otherwise get sound speed and absorption from user inputs or raw data file
        else:
            self.env_params['sound_speed'] = (env_params['sound_speed']
                                              if 'sound_speed' in env_params
                                              else self.echodata.raw_env['sound_speed_indicative'])
            self.env_params['sound_absorption'] = (env_params['sound_absorption']
                                                   if 'sound_absorption' in env_params
                                                   else self.echodata.raw_env['absorption_indicative'])

    def get_cal_params(self, cal_params):
        """Set cal params using user inputs or values from data file.

        Parameters
        ----------
        cal_params : dict
        """
        # Params from the Vendor group
        params_from_vend = ['sa_correction', 'gain_correction']
        for p in params_from_vend:
            # substitute if None in user input
            self.cal_params[p] = cal_params[p] if p in cal_params else self.echodata.raw_vend[p]

        # Other params
        self.cal_params['equivalent_beam_angle'] = (cal_params['equivalent_beam_angle']
                                                    if 'equivalent_beam_angle' in cal_params
                                                    else self.echodata.raw_beam['equivalent_beam_angle'])

    def get_Sv(self):
        return self._cal_power(cal_type='Sv')

    def get_Sp(self):
        return self._cal_power(cal_type='Sp')


class CalibrateAZFP(CalibrateBase):

    def __init__(self, echodata):
        super().__init__(echodata)

        # cal params specific to AZFP
        self.cal_params = dict.fromkeys(CAL_PARAMS['AZFP'])

    def get_cal_params(self, cal_params):
        """Set cal params using user inputs or values from data file.

        Parameters
        ----------
        cal_params : dict
        """
        # Params from the Vendor group
        params_from_vend = ['EL', 'DS', 'TVR', 'VTX', 'Sv_offset']
        for p in params_from_vend:
            # substitute if None in user input
            self.cal_params[p] = cal_params[p] if cal_params[p] else self.echodata.raw_vend[p]

        # Other params
        self.cal_params['equivalent_beam_angle'] = (cal_params['equivalent_beam_angle']
                                                    if cal_params['equivalent_beam_angle']
                                                    else self.echodata.raw_beam['equivalent_beam_angle'])

    def get_env_params(self, env_params):
        """Set cal params using user inputs or values from data file.

        Parameters
        ----------
        env_params : dict
        """
        # Temperature comes from either user input or data file
        self.env_params['temperature'] = (env_params['temperature']
                                          if env_params['temperature']
                                          else self.echodata.raw_env['temperature'])

        # Salinity and pressure always come from user input
        if (not env_params['salinity']) or (not env_params['pressure']):
            raise ReferenceError('Please supply both salinity and pressure in env_params.')
        else:
            self.env_params['salinity'] = env_params['salinity']
            self.env_params['pressure'] = env_params['pressure']

        # Always calculate sound speed and absorption
        self.env_params['sound_speed'] = uwa.calc_sound_speed(self.env_params['temperature'],
                                                              self.env_params['salinity'],
                                                              self.env_params['pressure'])
        self.env_params['sound_absorption'] = uwa.calc_absorption(self.echodata.raw_beam['frequency'],
                                                                  self.env_params['temperature'],
                                                                  self.env_params['salinity'],
                                                                  self.env_params['pressure'])

    def calc_range_meter(self):
        """Calculate range in meter using AZFP formula.
        """
        # TODO: double check the implementation below against reference manual
        # TODO: make sure the dimensions work out

        range_samples = self.echodata.raw_vend['number_of_samples_per_average_bin']
        dig_rate = self.echodata.raw_vend['digitization_rate']
        lockout_index = self.echodata.raw_vend['lockout_index']
        sound_speed = self.env_params['sound_speed']
        bins_to_avg = 1   # keep this in ref of AZFP matlab code, set to 1 since we want to calculate from raw data

        # Below is from LoadAZFP.m, the output is effectively range_bin+1 when bins_to_avg=1
        range_mod = xr.DataArray(np.arange(1, len(self.echodata.raw_beam.range_bin) - bins_to_avg + 2, bins_to_avg),
                                 coords=[('range_bin', self.echodata.raw_beam.range_bin)])

        # Calculate range using parameters for each freq
        range_meter = (lockout_index / (2 * dig_rate) * sound_speed + sound_speed / 4 *
                       (((2 * range_mod - 1) * range_samples * bins_to_avg - 1) / dig_rate +
                        self.echodata.raw_beam['transmit_duration_nominal']))

        # TODO: tilt is only relevant when calculating "depth" but "range" is general,
        #  this correction should be done outside of the functions.
        #  Probably the best is to show in a notebook and mentioned in doc,
        #  since AZFP mooring cage has a 15 deg tilt.
        # if tilt_corrected:
        #     range_meter = self.echodata.raw_beam.cos_tilt_mag.mean() * range_meter

        return range_meter

    def _cal_power(self, cal_type):
        """Calibrate to get volume backscattering strength (Sv) from AZFP power data.

        The calibration formula used here is documented in eq.(9) on p.85
        of GU-100-AZFP-01-R50 Operator's Manual.
        Note a Sv_offset factor that varies depending on frequency is used
        in the calibration as documented on p.90.
        See calc_Sv_offset() in convert/azfp.py
        """
        range_meter = self.calc_range_meter()

        if cal_type == 'Sv':
            out = (self.cal_params['EL']
                   - 2.5 / self.cal_params['DS']
                   + self.echodata.raw_beam.backscatter_r / (26214 * self.cal_params['DS'])
                   - self.cal_params['TVR']
                   - 20 * np.log10(self.cal_params['VTX'])
                   + 20 * np.log10(range_meter)
                   + 2 * self.env_params['sound_absorption'] * range_meter
                   - 10 * np.log10(0.5 * self.env_params['sound_speed'] *
                                   self.echodata.raw_beam['transmit_duration_nominal'] *
                                   self.cal_params['equivalent_beam_angle']) + self.cal_params['Sv_offset'])
            out.name = 'Sv'

        elif cal_type == 'Sp':
            out = (self.cal_params['EL']
                   - 2.5 / self.cal_params['DS']
                   + self.echodata.raw_beam.backscatter_r / (26214 * self.cal_params['DS'])
                   - self.cal_params['TVR']
                   - 20 * np.log10(self.cal_params['VTX'])
                   + 40 * np.log10(range_meter)
                   + 2 * self.env_params['sound_absorption'] * range_meter)
            out.name = 'Sp'
        else:
            raise ValueError('cal_type not recognized!')

        # Attach calculated range (with units meter) into data set
        out = out.to_dataset()
        out = out.merge(range_meter)

        return out

    def get_Sv(self):
        return self._cal_power(cal_type='Sv')

    def get_Sp(self):
        return self._cal_power(cal_type='Sp')
