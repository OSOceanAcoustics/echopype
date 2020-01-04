"""
echopype data model inherited from based class EchoData for EK60 data.
"""

import datetime as dt
import numpy as np
import xarray as xr
from .modelbase import ModelBase


class ModelEK60(ModelBase):
    """Class for manipulating EK60 echo data that is already converted to netCDF."""

    def __init__(self, file_path=""):
        ModelBase.__init__(self, file_path)
        self.tvg_correction_factor = 2  # range bin offset factor for calculating time-varying gain in EK60
        self._salinity = None
        self._temperature = None
        self._pressure = None

    def get_salinity(self):
        return self._salinity

    def get_temperature(self):
        return self._temperature

    def get_pressure(self):
        return self._pressure

    def get_sound_speed(self):
        with xr.open_dataset(self.file_path, group="Environment") as ds_env:
            return ds_env.sound_speed_indicative

    def calc_seawater_absorption(self):
        """Returns the seawater absorption values from the .nc file"""
        with xr.open_dataset(self.file_path, group="Environment") as ds_env:
            return ds_env.absorption_indicative

    def calc_sample_thickness(self):
        ds_beam = xr.open_dataset(self.file_path, group="Beam")
        sth = self.sound_speed * ds_beam.sample_interval / 2  # sample thickness
        ds_beam.close()
        return sth

    def calc_range(self):
        """Calculates range in meters using parameters stored in the .nc file.
        """
        with xr.open_dataset(self.file_path, group="Beam") as ds_beam:
            range_meter = ds_beam.range_bin * self.sample_thickness - \
                        self.tvg_correction_factor * self.sample_thickness  # DataArray [frequency x range_bin]
            range_meter = range_meter.where(range_meter > 0, other=0)
            return range_meter

    def calibrate(self, save=False):
        """Perform echo-integration to get volume backscattering strength (Sv) from EK60 power data.

        Parameters
        -----------
        save : bool, optional
            whether to save calibrated Sv output
            default to ``False``
        """

        # Open data set for Environment and Beam groups
        ds_env = xr.open_dataset(self.file_path, group="Environment")
        ds_beam = xr.open_dataset(self.file_path, group="Beam")
        # Derived params
        wavelength = self.sound_speed / ds_env.frequency  # wavelength

        # Get backscatter_r and range_bin
        backscatter_r = ds_beam['backscatter_r']
        range_bin = ds_beam['range_bin']
        # Calc gain
        CSv = 10 * np.log10((ds_beam.transmit_power * (10 ** (ds_beam.gain_correction / 10)) ** 2 *
                             wavelength ** 2 * self.sound_speed * ds_beam.transmit_duration_nominal *
                             10 ** (ds_beam.equivalent_beam_angle / 10)) /
                            (32 * np.pi ** 2))

        # TODO: move TVG and ABS calculation to the parent class, as also noted
        #  correspondingly in model/azfp
        # Get TVG and absorption
        range_meter = self.range
        TVG = np.real(20 * np.log10(range_meter.where(range_meter >= 1, other=1)))
        ABS = 2 * self.seawater_absorption * range_meter

        # Calibration and echo integration
        Sv = backscatter_r + TVG + ABS - CSv - 2 * ds_beam.sa_correction
        Sv.name = 'Sv'
        Sv = Sv.to_dataset()

        # Attach calculated range into data set
        Sv['range'] = (('frequency', 'range_bin'), self.range.T)

        # Save calibrated data into the calling instance and
        #  to a separate .nc file in the same directory as the data filef.Sv = Sv
        self.Sv = Sv
        if save:
            print('%s  saving calibrated Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
            Sv.to_netcdf(path=self.Sv_path, mode="w")

        # Close opened resources
        ds_env.close()
        ds_beam.close()

    def calibrate_TS(self, save=False):
        """Perform echo-integration to get Target Stregnth (TS / Sp) from EK60 power data.

        Parameters
        -----------
        save : bool, optional
            whether to save calibrated TS output
            default to ``False``
        """

        # Open data set for Environment and Beam groups
        ds_env = xr.open_dataset(self.file_path, group="Environment")
        ds_beam = xr.open_dataset(self.file_path, group="Beam")
        # Derived params
        wavelength = self.sound_speed / ds_env.frequency  # wavelength

        # Get backscatter_r and range_bin
        backscatter_r = ds_beam['backscatter_r']
        # Calc gain
        CSp = 10 * np.log10((ds_beam.transmit_power * (10 ** (ds_beam.gain_correction / 10)) ** 2 *
                             wavelength ** 2) /
                            (16 * np.pi ** 2))

        # TODO: move TVG and ABS calculation to the parent class, as also noted
        #  correspondingly in model/azfp
        # Get TVG and absorption
        range_meter = self.range
        TVG = np.real(40 * np.log10(range_meter.where(range_meter >= 1, other=1)))
        ABS = 2 * self.seawater_absorption * range_meter

        # Calibration and echo integration
        TS = backscatter_r + TVG + ABS - CSp
        TS.name = 'TS'
        TS = TS.to_dataset()

        # Attach calculated range into data set
        TS['range'] = (('frequency', 'range_bin'), self.range.T)

        # Save calibrated data into the calling instance and
        #  to a separate .nc file in the same directory as the data filef.Sv = Sv
        self.TS = TS
        if save:
            print('%s  saving calibrated TS to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.TS_path))
            TS.to_netcdf(path=self.TS_path, mode="w")

        # Close opened resources
        ds_env.close()
        ds_beam.close()
