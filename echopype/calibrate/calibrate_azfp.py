import numpy as np
import xarray as xr
from .calibrate_ek import CalibrateBase
from .calibrate_ek import CAL_PARAMS
from ..utils import uwa


class CalibrateAZFP(CalibrateBase):

    def __init__(self, echodata):
        super().__init__(echodata)

        # cal params specific to AZFP
        self.cal_params = dict.fromkeys(CAL_PARAMS['AZFP'])

        self.range_meter = self.calc_range_meter()

    def get_cal_params(self, cal_params):
        """Get cal params using user inputs or values from data file.

        Parameters
        ----------
        cal_params : dict
        """
        # Params from the Beam group
        for p in ['EL', 'DS', 'TVR', 'VTX', 'Sv_offset', 'equivalent_beam_angle']:
            # substitute if None in user input
            self.cal_params[p] = cal_params[p] if p in cal_params else self.echodata.raw_beam[p]

    def get_env_params(self, env_params):
        """Get cal params using user inputs or values from data file.

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
        self.env_params['sound_speed'] = uwa.calc_sound_speed(temperature=self.env_params['temperature'],
                                                              salinity=self.env_params['salinity'],
                                                              pressure=self.env_params['pressure'],
                                                              formula_source='AZFP')
        self.env_params['sound_absorption'] = uwa.calc_absorption(frequency=self.echodata.raw_beam['frequency'],
                                                                  temperature=self.env_params['temperature'],
                                                                  salinity=self.env_params['salinity'],
                                                                  pressure=self.env_params['pressure'],
                                                                  formula_source='AZFP')

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
        range_meter.name = 'range'  # add name to facilitate xr.merge

        # TODO: tilt is only relevant when calculating "depth" but "range" is general,
        #  this correction should be done outside of the functions.
        #  Probably the best is to show in a notebook and mentioned in doc,
        #  since AZFP mooring cage has a 15 deg tilt.
        # if tilt_corrected:
        #     range_meter = self.echodata.raw_beam.cos_tilt_mag.mean() * range_meter

        return range_meter

    def _cal_power(self, cal_type, **kwargs):
        """Calibrate to get volume backscattering strength (Sv) from AZFP power data.

        The calibration formulae used here is based on Appendix G in
        the GU-100-AZFP-01-R50 Operator's Manual.
        Note a Sv_offset factor that varies depending on frequency is used
        in the calibration as documented on p.90.
        See calc_Sv_offset() in convert/azfp.py
        """
        spreading_loss = 20 * np.log10(self.range_meter)
        absorption_loss = 2 * self.env_params['sound_absorption'] * self.range_meter
        SL = self.cal_params['TVR'] + 20 * np.log10(self.cal_params['VTX'])  # eq.(2)
        a = self.cal_params['DS']  # scaling factor (slope) in Fig.G-1, units Volts/dB], see p.84
        EL = self.cal_params['EL'] - 2.5 / a + self.echodata.raw_beam.backscatter_r / (26214 * a)  # eq.(5)

        if cal_type == 'Sv':
            # eq.(9)
            out = (EL - SL + spreading_loss + absorption_loss
                   - 10 * np.log10(0.5 * self.env_params['sound_speed'] *
                                   self.echodata.raw_beam['transmit_duration_nominal'] *
                                   self.cal_params['equivalent_beam_angle'])
                   + self.cal_params['Sv_offset'])  # see p.90-91 for this correction to Sv
            out.name = 'Sv'

        elif cal_type == 'Sp':
            # eq.(10)
            out = EL - SL + 2 * spreading_loss + absorption_loss
            out.name = 'Sp'
        else:
            raise ValueError('cal_type not recognized!')

        # Attach calculated range (with units meter) into data set
        out = out.to_dataset()
        out = out.merge(self.range_meter)

        return out

    def compute_Sv(self):
        return self._cal_power(cal_type='Sv')

    def compute_Sp(self):
        return self._cal_power(cal_type='Sp')
