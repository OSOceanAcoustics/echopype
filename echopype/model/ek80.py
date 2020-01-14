"""
echopype data model inherited from based class EchoData for EK80 data.
"""

import datetime as dt
import numpy as np
import xarray as xr
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
                self._salinity = ds_env._salinity
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
    
    def calibrate(self, save=False):
        pass