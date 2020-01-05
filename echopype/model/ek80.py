"""
echopype data model inherited from based class EchoData for EK80 data.
"""

import datetime as dt
import numpy as np
import xarray as xr
from echopype.utils import uwa
from .modelbase import ModelBase


class ModelEK80(ModelBase):
    """Class for manipulating EK60 echo data that is already converted to netCDF."""

    def __init__(self, file_path=""):
        ModelBase.__init__(self, file_path)
        self._acidity = None
        self._salinity = None
        self._temperature = None
        self._pressure = None

    def get_salinity(self):
        if self._salinity is None:
            with xr.open_dataset(self.file_path, group="Environment") as ds_env:
                self._salinity = ds_env.salinity
        return self._salinity

    def get_temperature(self):
        if self._temperature is None:
            with xr.open_dataset(self.file_path, group="Environment") as ds_env:
                self._temperature = ds_env.temperature
        return self._temperature

    def get_pressure(self):
        if self._pressure is None:
            with xr.open_dataset(self.file_path, group="Environment") as ds_env:
                self._pressure = ds_env.depth
        return self._pressure

    def get_sound_speed(self):
        with xr.open_dataset(self.file_path, group="Environment") as ds_env:
            return ds_env.sound_speed_indicative

    def calc_seawater_absorption(self, src='FG'):
        with xr.open_dataset(self.file_path, group='Beam') as ds_beam:
            f0 = ds_beam.frequency_start
            f1 = ds_beam.frequency_end
            f_center = (f0 + f1) / 2
        sea_abs = uwa.calc_seawater_absorption(f_center,
                                               temperature=self.temperature,
                                               salinity=self.salinity,
                                               pressure=self.pressure,
                                               formula_source='FG')
        return sea_abs

    def calc_sample_thickness(self):
        with xr.open_dataset(self.file_path, group="Beam") as ds_beam:
            sth = self.sound_speed * ds_beam.sample_interval / 2  # sample thickness
            return sth

    def calc_range(self):
        """Calculates range in meters using parameters stored in the .nc file.
        """
        with xr.open_dataset(self.file_path, group="Beam") as ds_beam:
            range_meter = ds_beam.range_bin * self.sample_thickness - \
                ds_beam.transmit_duration_nominal * self.sound_speed / 2  # DataArray [frequency x range_bin]
            range_meter = range_meter.where(range_meter > 0, other=0)
            return range_meter

    def calibrate(self, save=False):
        # Hard-coded values supplied by Simrad
        Rwbtrx = 1000
        Ztrd = 75

        ds_beam = xr.open_dataset(self.file_path, group="Beam")
        f0 = ds_beam.frequency_start
        f1 = ds_beam.frequency_end
        f_center = (f0 + f1) / 2
        f_nominal = ds_beam.frequency
        c = self.sound_speed

        backscatter_r = ds_beam.backscatter_r
        backscatter_i = ds_beam.backscatter_i
        # Average accross quadrants and take the absolute value of complex backscatter
        prx = np.sqrt(np.mean(backscatter_r, 1) ** 2 + np.mean(backscatter_i, 1) ** 2)
        prx = prx * prx / 2 * (np.abs(Rwbtrx + Ztrd) / Rwbtrx) ** 2 / np.abs(Ztrd)
        sea_abs = self.seawater_absorption

        la2 = (c / f_center) ** 2
        f = np.moveaxis(np.array((ds_beam.frequency_start, ds_beam.frequency_end)), 0, 2)
        # TODO Gfc should be gain interpolated at the center frequency
        Gfc = np.mean(ds_beam.gain_correction.values)
        ptx = ds_beam.transmit_power
        r = self.range
        r = r.where(r >= 1, other=1)

        pass
