"""
echopype data model inherited from based class EchoData for AZFP data.
"""

import datetime as dt
import numpy as np
import xarray as xr
import arlpy
from .modelbase import ModelBase


class ModelAZFP(ModelBase):
    """Class for manipulating AZFP echo data that is already converted to netCDF."""

    def __init__(self, file_path=""):
        ModelBase.__init__(self, file_path)
        self.salinity = 29.6       # Salinity in psu
        self.pressure = 60         # Pressure in dbars (~ equal to depth in meters)
        self.bins_to_avg = 1
        self.time_to_avg = 40

    # Retrieve sound_speed. Calculate if not stored
    @property
    def sound_speed(self):
        try:
            return self._sound_speed
        except AttributeError:
            return self.calc_sound_speed()

    # Retrieve range. Calculate if not stored
    @property
    def range(self):
        try:
            return self._range
        except AttributeError:
            return self.calc_range()

    # Retrieve sea_abs. Calculate if not stored
    @property
    def sea_abs(self):
        try:
            return self._sea_abs
        except AttributeError:
            return self.calc_sea_abs()

    @property
    def temperature(self):
        try:
            return self._temperature
        except AttributeError:
            with xr.open_dataset(self.file_path, group='Environment') as ds_env:
                self._temperature = ds_env.temperature
                return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        self._temperature = temperature

    def calc_range(self, tilt_corrected=False):
        """Calculates the range in meters using sound speed and other measured values

        Parameters
        ----------
        tilt_corrected : bool
                         Modifies the range to take into account the tilt of the transducer.
                         Defaults to `False`

        Returns
        -------
        An xarray DataArray containing the range with coordinate frequency
        """
        ds_beam = xr.open_dataset(self.file_path, group='Beam')
        ds_vend = xr.open_dataset(self.file_path, group='Vendor')

        frequency = ds_beam.frequency
        range_samples = ds_vend.number_of_samples_per_average_bin
        pulse_length = ds_beam.transmit_duration_nominal   # units: seconds
        bins_to_avg = 1   # set to 1 since we want to calculate from raw data
        range_bin = ds_beam.range_bin
        sound_speed = self.sound_speed
        dig_rate = ds_vend.digitization_rate
        lockout_index = ds_vend.lockout_index

        m = []
        for jj in range(len(frequency)):
            m.append(np.arange(1, len(range_bin) - bins_to_avg + 2,
                     bins_to_avg))
        m = xr.DataArray(m, coords=[('frequency', frequency), ('range_bin', range_bin)])
        # m = xr.DataArray(m, coords=[('frequency', Data[0]['frequency'])])         # If range varies in frequency
        # Create DataArrays for broadcasting on dimension frequency

        # TODO Handle varying range
        # Calculate range from soundspeed for each frequency
        depth = (sound_speed * lockout_index[0] / (2 * dig_rate[0]) + sound_speed / 4 *
                 (((2 * m - 1) * range_samples[0] * bins_to_avg - 1) / dig_rate[0] +
                 (pulse_length / np.timedelta64(1, 's'))))
        if tilt_corrected:
            depth = ds_beam.cos_tilt_mag.mean() * depth

        ds_beam.close()
        ds_vend.close()

        self._range = depth
        return self._range

    def calc_sound_speed(self):
        """Calculate the sound speed using arlpy. Uses the default salinity and pressure.
        Temperature comes from measurements that varies with the ping.
        A sound speed value is calculated with each temperature value.

        Returns
        -------
        A sound speed for each temperature.
        """
        ss = arlpy.uwa.soundspeed(temperature=self.temperature, salinity=self.salinity, depth=self.pressure)
        self._sound_speed = ss
        return self._sound_speed

    def calc_sea_abs(self):
        """Calculate the sea absorption for each frequency with arlpy.

        Returns
        -------
        An array containing absorption coefficients for each frequency
        """
        with xr.open_dataset(self.file_path, group='Beam') as ds_beam:
            frequency = ds_beam.frequency
        try:
            temp = self.temperature.mean()    # Averages when temperature is a numpy array
        except AttributeError:
            temp = self.temperature
        self._sea_abs = arlpy.uwa.absorption(frequency=frequency, temperature=temp,
                                             salinity=self.salinity, depth=self.pressure)
        mag = -arlpy.utils.mag2db(self._sea_abs)
        return self._sea_abs

    def calibrate(self, save=False):
        """Perform echo-integration to get volume backscattering strength (Sv) from AZFP power data.

        Parameters
        -----------
        save : bool, optional
               whether to save calibrated Sv output
               default to ``True``
        """

        ds_env = xr.open_dataset(self.file_path, group="Environment")
        ds_beam = xr.open_dataset(self.file_path, group="Beam")

        # Average the sound speeds if it is an array as opposed to a single value
        try:
            self.sample_thickness = self.sound_speed.mean() * (ds_beam.sample_interval / np.timedelta64(1, 's')) / 2
        except AttributeError:
            self.sample_thickness = self.sound_speed * (ds_beam.sample_interval / np.timedelta64(1, 's')) / 2

        depth = self.calc_range()
        self.Sv = (ds_beam.EL - 2.5 / ds_beam.DS + ds_beam.backscatter_r / (26214 * ds_beam.DS) -
                   ds_beam.TVR - 20 * np.log10(ds_beam.VTX) + 20 * np.log10(depth) +
                   2 * self.sea_abs * depth -
                   10 * np.log10(0.5 * self.sound_speed *
                                 ds_beam.transmit_duration_nominal.astype('float64') / 1e9 *
                                 ds_beam.equivalent_beam_angle) + ds_beam.Sv_offset)
        if save:
            print("{} saving calibrated Sv to {}".format(dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
            self.Sv.to_dataset(name="Sv").to_netcdf(path=self.Sv_path, mode="w")

        # Close opened resources
        ds_env.close()
        ds_beam.close()

    def calibrate_ts(self, save=False):
        ds_beam = xr.open_dataset(self.file_path, group="Beam")
        depth = self.calc_range()

        self.TS = (ds_beam.EL - 2.5 / ds_beam.DS + ds_beam.backscatter_r / (26214 * ds_beam.DS) -
                   ds_beam.TVR - 20 * np.log10(ds_beam.VTX) + 40 * np.log10(depth) +
                   2 * self.sea_abs * depth)
        if save:
            print("{} saving calibrated TS to {}".format(dt.datetime.now().strftime('%H:%M:%S'), self.TS_path))
            self.TS.to_dataset(name="TS").to_netcdf(path=self.TS_path, mode="w")

        ds_beam.close()

    def get_MVBS(self):
        super().get_MVBS('Sv', self.bins_to_avg, self.time_to_avg)
