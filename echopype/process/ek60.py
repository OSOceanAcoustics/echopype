"""
echopype data model inherited from based class Process for EK60 data.
"""

import os
import datetime as dt
import numpy as np
import xarray as xr
from .processbase import ProcessBase
from echopype.utils import uwa


class ProcessEK60(ProcessBase):
    """Class for manipulating EK60 echo data already converted to netCDF.
    """
    def __init__(self, file_path=""):
        ProcessBase.__init__(self, file_path)
        self.tvg_correction_factor = 2  # range bin offset factor for calculating time-varying gain in EK60

        # Initialize environment-related parameters
        self._sound_speed = self.calc_sound_speed()
        self._sample_thickness = self.calc_sample_thickness()
        self._range = self.calc_range()
        self._seawater_absorption = self.calc_seawater_absorption()

        # Initialize calibration-related parameters
        with self._open_dataset(self.file_path, group="Beam") as ds_beam:
            self._gain_correction = ds_beam.gain_correction
            self._equivalent_beam_angle = ds_beam.equivalent_beam_angle
            self._sa_correction = ds_beam.sa_correction

    # EK60 calibration parameters
    @property
    def gain_correction(self):
        return self._gain_correction

    @gain_correction.setter
    def gain_correction(self, gc):
        self._gain_correction.values = gc

    @property
    def equivalent_beam_angle(self):
        return self._equivalent_beam_angle

    @equivalent_beam_angle.setter
    def equivalent_beam_angle(self, eba):
        self._equivalent_beam_angle.values = eba

    @property
    def sa_correction(self):
        return self._sa_correction

    @sa_correction.setter
    def sa_correction(self, sac):
        self._sa_correction.values = sac

    # Environmental and derived parameters
    def calc_sound_speed(self, src='file'):
        if src == 'file':
            with self._open_dataset(self.file_path, group="Environment") as ds_env:
                return ds_env.sound_speed_indicative
        elif src == 'user':
            ss = uwa.calc_sound_speed(salinity=self.salinity,
                                      temperature=self.temperature,
                                      pressure=self.pressure)
            return ss * np.ones(self.sound_speed.size)
        else:
            ValueError('Not sure how to update sound speed!')

    def calc_seawater_absorption(self, src='file'):
        """Returns the seawater absorption values from the .nc file.
        """
        if src == 'file':
            with self._open_dataset(self.file_path, group="Environment") as ds_env:
                return ds_env.absorption_indicative
        elif src == 'user':
            with self._open_dataset(self.file_path, group='Beam') as ds_beam:
                freq = ds_beam.frequency.astype(np.int64)  # should already be in unit [Hz]
            return uwa.calc_seawater_absorption(freq,
                                                temperature=self.temperature,
                                                salinity=self.salinity,
                                                pressure=self.pressure,
                                                formula_source='AM')
        else:
            ValueError('Not sure how to update seawater absorption!')

    def calc_sample_thickness(self):
        with self._open_dataset(self.file_path, group="Beam") as ds_beam:
            sth = self.sound_speed * ds_beam.sample_interval / 2  # sample thickness
            return sth

    def calc_range(self):
        """Calculates range in meters using parameters stored in the .nc file.
        """
        with self._open_dataset(self.file_path, group="Beam") as ds_beam:
            range_meter = self.sample_thickness * ds_beam.range_bin - \
                self.tvg_correction_factor * self.sample_thickness  # DataArray [frequency x range_bin]
            range_meter = range_meter.where(range_meter > 0, other=0)
            return range_meter

    def calibrate(self, save=False, save_postfix='_Sv', save_path=None):
        """Perform echo-integration to get volume backscattering strength (Sv) from EK60 power data.

        Parameters
        -----------
        save : bool, optional
            whether to save calibrated Sv output
            default to ``False``
        save_postfix : str
            Filename postfix, default to '_Sv'
        save_path : str
            Full filename to save to, overwriting the RAWFILENAME_Sv.nc default
        """
        # Print raw data nc file
        print('%s  calibrating data in %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.file_path))

        # Open data set for Environment and Beam groups
        ds_beam = self._open_dataset(self.file_path, group="Beam")

        # Derived params
        wavelength = self.sound_speed / ds_beam.frequency  # wavelength

        # Get backscatter_r and range_bin
        backscatter_r = ds_beam['backscatter_r']

        # Calc gain
        CSv = 10 * np.log10((ds_beam.transmit_power * (10 ** (self.gain_correction / 10)) ** 2 *
                             wavelength ** 2 * self.sound_speed * ds_beam.transmit_duration_nominal *
                             10 ** (self.equivalent_beam_angle / 10)) /
                            (32 * np.pi ** 2))

        # Get TVG and absorption
        range_meter = self.range
        TVG = np.real(20 * np.log10(range_meter.where(range_meter >= 1, other=1)))
        ABS = 2 * self.seawater_absorption * range_meter

        # Calibration and echo integration
        Sv = backscatter_r + TVG + ABS - CSv - 2 * self.sa_correction
        Sv.name = 'Sv'
        Sv = Sv.to_dataset()

        # Attach calculated range into data set
        Sv['range'] = (('frequency', 'range_bin'), self.range)

        # Save calibrated data into the calling instance and
        #  to a separate .nc file in the same directory as the data filef.Sv = Sv
        self.Sv = Sv
        if save:
            self.Sv_path = self.validate_path(save_path, save_postfix)
            print('%s  saving calibrated Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
            self._save_dataset(Sv, self.Sv_path, mode="w")

        # Close opened resources
        ds_beam.close()

    def calibrate_TS(self, save=False, save_postfix='_TS', save_path=None):
        """Perform echo-integration to get Target Strength (TS) from EK60 power data.

        Parameters
        -----------
        save : bool, optional
            whether to save calibrated TS output
            default to ``False``
        save_postfix : str, optional
            Filename postfix, default to '_TS'
        save_path : str, optional
            Full filename to save the TS calculation results, overwritting the RAWFILE_TS.nc default
        """

        # Open data set for Environment and Beam groups
        ds_env = self._open_dataset(self.file_path, group="Environment")
        ds_beam = self._open_dataset(self.file_path, group="Beam")
        # Derived params
        wavelength = self.sound_speed / ds_env.frequency  # wavelength

        # Get backscatter_r and range_bin
        backscatter_r = ds_beam['backscatter_r']
        # Calc gain
        CSp = 10 * np.log10((ds_beam.transmit_power * (10 ** (ds_beam.gain_correction / 10)) ** 2 *
                             wavelength ** 2) /
                            (16 * np.pi ** 2))

        # Get TVG and absorption
        range_meter = self.range
        TVG = np.real(40 * np.log10(range_meter.where(range_meter >= 1, other=1)))
        ABS = 2 * self.seawater_absorption * range_meter

        # Calibration and echo integration
        TS = backscatter_r + TVG + ABS - CSp
        TS.name = 'TS'
        TS = TS.to_dataset()

        # Attach calculated range into data set
        TS['range'] = (('frequency', 'range_bin'), self.range)

        # Save calibrated data into the calling instance and
        #  to a separate .nc file in the same directory as the data filef.Sv = Sv
        self.TS = TS
        if save:
            self.TS_path = self.validate_path(save_path, save_postfix)
            print('%s  saving calibrated TS to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.TS_path))
            self._save_dataset(TS, self.TS_path, mode="w")

        # Close opened resources
        ds_env.close()
        ds_beam.close()
