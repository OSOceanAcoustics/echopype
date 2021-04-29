import numpy as np
from .calibrate_ek import CalibrateBase
from .calibrate_base import CAL_PARAMS, ENV_PARAMS
from ..utils import uwa


class CalibrateAZFP(CalibrateBase):

    def __init__(self, echodata, env_params, cal_params, **kwargs):
        super().__init__(echodata)

        # initialize env and cal params
        self.env_params = dict.fromkeys(ENV_PARAMS)
        self.cal_params = dict.fromkeys(CAL_PARAMS['AZFP'])

        # load env and cal parameters
        if env_params is None:
            env_params = {}
        self.get_env_params(env_params)
        if cal_params is None:
            cal_params = {}
        self.get_cal_params(cal_params)

        # self.range_meter computed under self._cal_power()
        # because the implementation is different for Sv and Sp

    def get_cal_params(self, cal_params):
        """Get cal params using user inputs or values from data file.

        Parameters
        ----------
        cal_params : dict
        """
        # Params from the Beam group
        for p in ['EL', 'DS', 'TVR', 'VTX', 'Sv_offset', 'equivalent_beam_angle']:
            # substitute if None in user input
            self.cal_params[p] = cal_params[p] if p in cal_params else self.echodata.beam[p]

    def get_env_params(self, env_params):
        """Get env params using user inputs or values from data file.

        Parameters
        ----------
        env_params : dict
        """
        # Temperature comes from either user input or data file
        self.env_params['temperature'] = (env_params['temperature']
                                          if 'temperature' in env_params
                                          else self.echodata.environment['temperature'])

        # Salinity and pressure always come from user input
        if ('salinity' not in env_params) or ('pressure' not in env_params):
            raise ReferenceError('Please supply both salinity and pressure in env_params.')
        else:
            self.env_params['salinity'] = env_params['salinity']
            self.env_params['pressure'] = env_params['pressure']

        # Always calculate sound speed and absorption
        self.env_params['sound_speed'] = uwa.calc_sound_speed(
            temperature=self.env_params['temperature'],
            salinity=self.env_params['salinity'],
            pressure=self.env_params['pressure'],
            formula_source='AZFP'
        )
        self.env_params['sound_absorption'] = uwa.calc_absorption(
            frequency=self.echodata.beam['frequency'],
            temperature=self.env_params['temperature'],
            salinity=self.env_params['salinity'],
            pressure=self.env_params['pressure'],
            formula_source='AZFP'
        )

    def compute_range_meter(self, cal_type):
        """Calculate range in meter using AZFP formula.

        Note the range calculation differs for Sv and Sp per AZFP matlab code.

        Parameters
        ----------
        cal_type : str
            'Sv' for calculating volume backscattering strength, or
            'Sp' for calculating point backscattering strength
        """
        # Notation below follows p.86 of user manual
        N = self.echodata.vendor['number_of_samples_per_average_bin']  # samples per bin
        f = self.echodata.vendor['digitization_rate']  # digitization rate
        L = self.echodata.vendor['lockout_index']  # number of lockout samples
        sound_speed = self.env_params['sound_speed']
        bins_to_avg = 1   # keep this in ref of AZFP matlab code, set to 1 since we want to calculate from raw data

        # Calculate range using parameters for each freq
        #  This is "the range to the centre of the sampling volume for bin m" from p.86 of user manual
        if cal_type == 'Sv':
            range_offset = 0
        else:
            range_offset = sound_speed * self.echodata.beam['transmit_duration_nominal'] / 4  # from matlab code
        range_meter = (sound_speed * L / (2 * f)
                       + sound_speed / 4 * (((2 * (self.echodata.beam.range_bin + 1) - 1) * N * bins_to_avg - 1) / f
                                            + self.echodata.beam['transmit_duration_nominal'])
                       - range_offset)
        range_meter.name = 'range'  # add name to facilitate xr.merge

        self.range_meter = range_meter

    def _cal_power(self, cal_type, **kwargs):
        """Calibrate to get volume backscattering strength (Sv) from AZFP power data.

        The calibration formulae used here is based on Appendix G in
        the GU-100-AZFP-01-R50 Operator's Manual.
        Note a Sv_offset factor that varies depending on frequency is used
        in the calibration as documented on p.90.
        See calc_Sv_offset() in convert/azfp.py
        """
        # Compute range in meters
        self.compute_range_meter(cal_type=cal_type)  # range computation different for Sv and Sp per AZFP matlab code

        # Compute various params
        spreading_loss = 20 * np.log10(self.range_meter)  # TODO: take care of dividing by zero encountered in log10
        absorption_loss = 2 * self.env_params['sound_absorption'] * self.range_meter
        SL = self.cal_params['TVR'] + 20 * np.log10(self.cal_params['VTX'])  # eq.(2)
        a = self.cal_params['DS']  # scaling factor (slope) in Fig.G-1, units Volts/dB], see p.84
        EL = self.cal_params['EL'] - 2.5 / a + self.echodata.beam.backscatter_r / (26214 * a)  # eq.(5)

        if cal_type == 'Sv':
            # eq.(9)
            out = (EL - SL + spreading_loss + absorption_loss
                   - 10 * np.log10(0.5 * self.env_params['sound_speed'] *
                                   self.echodata.beam['transmit_duration_nominal'] *
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

        # Add env and cal parameters
        out = self._add_params_to_output(out)

        return out

    def compute_Sv(self, **kwargs):
        return self._cal_power(cal_type='Sv')

    def compute_Sp(self, **kwargs):
        return self._cal_power(cal_type='Sp')
