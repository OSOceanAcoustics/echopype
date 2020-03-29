"""
echopype data model inherited from based class EchoData for AZFP data.
"""

import os
import datetime as dt
import numpy as np
import xarray as xr
from .modelbase import ModelBase
from echopype.utils import uwa


class ModelAZFP(ModelBase):
    """Class for manipulating AZFP echo data that is already converted to netCDF."""

    def __init__(self, file_path="", salinity=29.6, pressure=60, temperature=None):
        ModelBase.__init__(self, file_path)
        self._salinity = salinity    # salinity in [psu]
        self._pressure = pressure    # pressure in [dbars] (approximately equal to depth in meters)
        if temperature is None:
            with xr.open_dataset(self.file_path, group='Environment') as ds_env:
                print("Initialize using average temperature recorded by instrument")
                self._temperature = np.nanmean(ds_env.temperature)   # temperature in [Celsius]
        else:
            self._temperature = temperature

        # Initialize environment-related parameters
        self._sound_speed = self.calc_sound_speed()
        self._sample_thickness = self.calc_sample_thickness()
        self._range = self.calc_range()
        self._seawater_absorption = self.calc_seawater_absorption()

        self._tilt_angle = None      # instrument tilt angle in [degrees]

    @property
    def tilt_angle(self):
        """Gets the tilt of the echosounder from the .nc file

        Returns
        -------
        Tilt of echosounder in degrees
        """
        if self._tilt_angle is None:
            with xr.open_dataset(self.file_path, group='Beam') as ds_beam:
                self._tilt_angle = np.rad2deg(np.arccos(ds_beam.cos_tilt_mag.mean().data))
        return self._tilt_angle

    def calc_sound_speed(self, src='user'):
        if src == 'user':
            return uwa.calc_sound_speed(temperature=self.temperature,
                                        salinity=self.salinity,
                                        pressure=self.pressure,
                                        formula_source='AZFP')
        else:
            ValueError('Not sure how to calculate sound speed for AZFP!')

    def calc_seawater_absorption(self, src='user'):
        """Calculates seawater absorption in dB/km using AZFP-supplied formula.

        Returns
        -------
        An xarray DataArray containing the sea absorption with coordinate frequency
        """
        with xr.open_dataset(self.file_path, group='Beam') as ds_beam:
            freq = ds_beam.frequency.astype(np.int64)  # should already be in unit [Hz]
        if src == 'user':
            return uwa.calc_seawater_absorption(freq,
                                                temperature=self.temperature,
                                                salinity=self.salinity,
                                                pressure=self.pressure,
                                                formula_source='AZFP')
        else:
            ValueError('For AZFP seawater absorption needs to be calculated '
                       'based on user-input environmental parameters.')

    def calc_sample_thickness(self):
        """Gets ``sample_thickness`` for AZFP data.

        This will call ``calc_sound_speed`` since sound speed is `not` part of the raw AZFP .01A data file.
        """
        with xr.open_dataset(self.file_path, group="Beam") as ds_beam:
            sth = self.sound_speed * ds_beam.sample_interval / 2
            return sth

    def calc_range(self, tilt_corrected=False):
        """Calculates range in meters using AZFP-supplied formula, instead of from sample_interval directly.

        Parameters
        ----------
        tilt_corrected : bool
            Modifies the range to take into account the tilt of the transducer. Defaults to `False`.

        Returns
        -------
        An xarray DataArray containing the range with coordinate frequency
        """
        ds_beam = xr.open_dataset(self.file_path, group='Beam')
        ds_vend = xr.open_dataset(self.file_path, group='Vendor')

        range_samples = ds_vend.number_of_samples_per_average_bin   # WJ: same as "range_samples_per_bin" used to calculate "sample_interval"
        pulse_length = ds_beam.transmit_duration_nominal   # units: seconds
        bins_to_avg = 1   # set to 1 since we want to calculate from raw data
        sound_speed = self.sound_speed
        dig_rate = ds_vend.digitization_rate
        lockout_index = ds_vend.lockout_index

        # Below is from LoadAZFP.m, the output is effectively range_bin+1 when bins_to_avg=1
        range_mod = xr.DataArray(np.arange(1, len(ds_beam.range_bin) - bins_to_avg + 2, bins_to_avg),
                                 coords=[('range_bin', ds_beam.range_bin)])

        # Calculate range using parameters for each freq
        range_meter = (sound_speed * lockout_index / (2 * dig_rate) + sound_speed / 4 *
                       (((2 * range_mod - 1) * range_samples * bins_to_avg - 1) / dig_rate +
                        pulse_length))

        if tilt_corrected:
            range_meter = ds_beam.cos_tilt_mag.mean() * range_meter

        ds_beam.close()
        ds_vend.close()

        return range_meter

    def calibrate(self, save=False, save_postfix='_Sv', save_path=None):
        """Perform echo-integration to get volume backscattering strength (Sv) from AZFP power data.

        The calibration formula used here is documented in eq.(9) on p.85
        of GU-100-AZFP-01-R50 Operator's Manual.
        Note a Sv_offset factor that varies depending on frequency is used
        in the calibration as documented on p.90.
        See calc_Sv_offset() in convert/azfp.py

        Parameters
        ----------
        save : bool, optional
               whether to save calibrated Sv output
               default to ``True``
        save_postfix : str
            Filename postfix, default to '_Sv'
        save_path : str
            Full filename to save to, overwriting the RAWFILE_Sv.nc default
        """
        # Print raw data nc file
        print('%s  calibrating data in %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.file_path))

        # Open data set for Environment and Beam groups
        ds_beam = xr.open_dataset(self.file_path, group="Beam")

        range_meter = self.range
        Sv = (ds_beam.EL - 2.5 / ds_beam.DS + ds_beam.backscatter_r / (26214 * ds_beam.DS) -
              ds_beam.TVR - 20 * np.log10(ds_beam.VTX) + 20 * np.log10(range_meter) +
              2 * self.seawater_absorption * range_meter -
              10 * np.log10(0.5 * self.sound_speed *
                            ds_beam.transmit_duration_nominal *
                            ds_beam.equivalent_beam_angle) + ds_beam.Sv_offset)

        Sv.name = 'Sv'
        Sv = Sv.to_dataset()

        # Attached calculated range into the dataset
        Sv['range'] = (('frequency', 'range_bin'), self.range)

        # Save calibrated data into the calling instance and
        #  to a separate .nc file in the same directory as the data filef.Sv = Sv
        self.Sv = Sv
        if save:
            self.Sv_path = self.validate_path(save_path, save_postfix)
            print("{} saving calibrated Sv to {}".format(dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
            self.Sv.to_netcdf(path=self.Sv_path, mode="w")

        # Close opened resources
        ds_beam.close()

    def calibrate_TS(self, save=False, save_postfix='_TS', save_path=None):
        """Perform echo-integration to get Target Strength (TS) from AZFP power data.

        The calibration formula used here is documented in eq.(10) on p.85
        of GU-100-AZFP-01-R50 Operator's Manual.

        Parameters
        ----------
        save : bool, optional
            whether to save calibrated TS output
            default to ``False``
        save_postfix : str, optional
            Filename postfix, default to '_TS'
        save_path : str, optional
            Full filename to save the TS calculation results, overwritting the RAWFILE_TS.nc default
        """
        with xr.open_dataset(self.file_path, group="Beam") as ds_beam:
            self.TS = (ds_beam.EL - 2.5 / ds_beam.DS + ds_beam.backscatter_r / (26214 * ds_beam.DS) -
                       ds_beam.TVR - 20 * np.log10(ds_beam.VTX) + 40 * np.log10(self.range) +
                       2 * self.seawater_absorption * self.range)
            self.TS.name = "TS"
            if save:
                self.TS_path = self.validate_path(save_path, save_postfix)
                print("{} saving calibrated TS to {}".format(dt.datetime.now().strftime('%H:%M:%S'), self.TS_path))
                self.TS.to_netcdf(path=self.TS_path, mode="w")
