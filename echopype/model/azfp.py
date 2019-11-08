"""
echopype data model inherited from based class EchoData for AZFP data.
"""

import datetime as dt
import numpy as np
import xarray as xr
from .modelbase import ModelBase
from echopype.utils import uwa


class ModelAZFP(ModelBase):
    """Class for manipulating AZFP echo data that is already converted to netCDF."""

    def __init__(self, file_path="", salinity=29.6, pressure=60, temperature=None, sound_speed=None):
        ModelBase.__init__(self, file_path)
        self._salinity = salinity           # salinity in [psu]
        self._pressure = pressure           # pressure in [dbars] (approximately equal to depth in meters)
        self._temperature = temperature     # temperature in [Celsius]
        self._sound_speed = sound_speed     # sound speed in [m/s]
        self._tilt_angle = None             # instrument tilt angle in [degrees]

    # TODO: consider moving some of these properties to the parent class,
    #  since it is possible that EK60 users may want to set the environmental
    #  parameters separately from those recorded in the data files.

    @property
    def salinity(self):
        return self._salinity

    @salinity.setter
    def salinity(self, sal):
        self._salinity = sal
        # Update sound speed, sample_thickness, absorption, range
        self.sound_speed = uwa.calc_sound_speed(temperature=self.temperature,
                                                salinity=self.salinity,
                                                pressure=self.pressure)
        self.sample_thickness = self.calc_sample_thickness()
        self.seawater_absorption = self.calc_seawater_absorption()
        self.range = self.calc_range()

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, pres):
        self._pressure = pres
        # Update sound speed, sample_thickness, absorption, range
        self.sound_speed = uwa.calc_sound_speed(temperature=self.temperature,
                                                salinity=self.salinity,
                                                pressure=self.pressure)
        self.sample_thickness = self.calc_sample_thickness()
        self.seawater_absorption = self.calc_seawater_absorption()
        self.range = self.calc_range()

    @property
    def temperature(self):
        if self._temperature is None:
            with xr.open_dataset(self.file_path, group='Environment') as ds_env:
                print("Using average temperature")
                self._temperature = np.nanmean(ds_env.temperature)
        return self._temperature

    @temperature.setter
    def temperature(self, t):
        self._temperature = t
        # Update sound speed, sample_thickness, absorption, range
        self.sound_speed = uwa.calc_sound_speed(temperature=self.temperature,
                                                salinity=self.salinity,
                                                pressure=self.pressure)
        self.sample_thickness = self.calc_sample_thickness()
        self.seawater_absorption = self.calc_seawater_absorption()
        self.range = self.calc_range()

        # TODO: add an option to allow using hourly averaged temperature, this
        #  requires using groupby operation and align the calculation properly
        #  when calculating sound speed (by ping_time).

        # def compute_avg_temp(unpacked_data, hourly_avg_temp):
        #     """Input the data with temperature values and averages all the temperatures
        #
        #     Parameters
        #     ----------
        #     unpacked_data
        #         current unpacked data
        #     hourly_avg_temp
        #         xml parameter
        #
        #     Returns
        #     -------
        #         the average temperature
        #     """
        #     sum = 0
        #     total = 0
        #     for ii in range(len(unpacked_data)):
        #         val = unpacked_data[ii]['temperature']
        #         if not math.isnan(val):
        #             total += 1
        #             sum += val
        #     if total == 0:
        #         print("**** No AZFP temperature found. Using default of {:.2f} "
        #               "degC to calculate sound-speed and range\n"
        #               .format(hourly_avg_temp))
        #         return hourly_avg_temp    # default value
        #     else:
        #         return sum / total

    @property
    def sound_speed(self):
        if self._sound_speed is None:  # if this is empty
            self._sound_speed = uwa.calc_sound_speed(temperature=self.temperature,
                                                     salinity=self.salinity,
                                                     pressure=self.pressure)
        return self._sound_speed

    @sound_speed.setter
    def sound_speed(self, ss):
        self._sound_speed = ss
        # TODO: need to update sample_thickness, absorption, range
        self.sample_thickness = self.calc_sample_thickness()
        self.seawater_absorption = self.calc_seawater_absorption()
        self.range = self.calc_range()

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

    def calc_seawater_absorption(self):
        """Calculates seawater absorption in dB/km using AZFP-supplied formula.

        Returns
        -------
        An xarray DataArray containing the sea absorption with coordinate frequency
        """
        with xr.open_dataset(self.file_path, group='Beam') as ds_beam:
            freq = ds_beam.frequency.astype(np.int64)  # should already be in unit [Hz]
        sea_abs = uwa.calc_seawater_absorption(freq,
                                               temperature=self.temperature,
                                               salinity=self.salinity,
                                               pressure=self.pressure)
        return sea_abs

    def calc_sample_thickness(self):
        """Gets ``sample_thickness`` for AZFP data.

        This will call ``calc_sound_speed`` since sound speed is `not` part of the raw AZFP .01A data file.
        """
        with xr.open_dataset(self.file_path, group="Beam") as ds_beam:
            sth = self.sound_speed * ds_beam.sample_interval / 2
            return sth
            # return sth.mean(dim='ping_time')   # use mean over all ping_time

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
                        (pulse_length / np.timedelta64(1, 's'))))

        if tilt_corrected:
            range_meter = ds_beam.cos_tilt_mag.mean() * range_meter

        ds_beam.close()
        ds_vend.close()

        return range_meter

    def calibrate(self, save=False):
        """Perform echo-integration to get volume backscattering strength (Sv) from AZFP power data.

        Parameters
        ----------
        save : bool, optional
               whether to save calibrated Sv output
               default to ``True``
        """

        # Open data set for Environment and Beam groups
        ds_env = xr.open_dataset(self.file_path, group="Environment")
        ds_beam = xr.open_dataset(self.file_path, group="Beam")

        range_meter = self.range
        self.Sv = (ds_beam.EL - 2.5 / ds_beam.DS + ds_beam.backscatter_r / (26214 * ds_beam.DS) -
                   ds_beam.TVR - 20 * np.log10(ds_beam.VTX) + 20 * np.log10(range_meter) +
                   2 * self.seawater_absorption * range_meter -
                   10 * np.log10(0.5 * self.sound_speed *
                                 ds_beam.transmit_duration_nominal.astype('float64') / 1e9 *
                                 ds_beam.equivalent_beam_angle) + ds_beam.Sv_offset)

        # # TODO: check if sample_thickness calculation should be/is done in a separate method
        # sample_thickness = ds_env.sound_speed_indicative * (ds_beam.sample_interval / np.timedelta64(1, 's')) / 2
        # # range_meter = self.calc_range()
        # self.Sv = (ds_beam.EL - 2.5 / ds_beam.DS + ds_beam.backscatter_r / (26214 * ds_beam.DS) -
        #            ds_beam.TVR - 20 * np.log10(ds_beam.VTX) + 20 * np.log10(range_meter) +
        #            2 * ds_beam.sea_abs * range_meter -
        #            10 * np.log10(0.5 * ds_env.sound_speed_indicative *
        #                          ds_beam.transmit_duration_nominal.astype('float64') / 1e9 *
        #                          ds_beam.equivalent_beam_angle) + ds_beam.Sv_offset)

        # TODO: should do TVG and ABS calculation when doing noise estimates, otherwise it is
        #  difficult to trace errors since they are calculated when performing calibration
        #  --> the current structure is brittle. This is true for EK60 too so the best is to
        #  make self.range and self.seawater_absorption common methods (separately implemented
        #  in EK60 and AZFP class) which can be called to calculate TVG and ABS.
        # Get TVG and absorption
        range_meter = range_meter.where(range_meter > 0, other=0)  # set all negative elements to 0
        TVG = np.real(20 * np.log10(range_meter.where(range_meter != 0, other=1)))
        ABS = 2 * self.seawater_absorption * range_meter

        # Save TVG and ABS for noise estimation use
        self.TVG = TVG
        self.ABS = ABS

        self.Sv.name = "Sv"
        if save:
            print("{} saving calibrated Sv to {}".format(dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
            self.Sv.to_netcdf(path=self.Sv_path, mode="w")

        # Close opened resources
        ds_env.close()
        ds_beam.close()

    def calibrate_TS(self, save=False):
        with xr.open_dataset(self.file_path, group="Beam") as ds_beam:
            self.TS = (ds_beam.EL - 2.5 / ds_beam.DS + ds_beam.backscatter_r / (26214 * ds_beam.DS) -
                       ds_beam.TVR - 20 * np.log10(ds_beam.VTX) + 40 * np.log10(self.range) +
                       2 * self.seawater_absorption * self.range)
            self.TS.name = "TS"
            if save:
                print("{} saving calibrated TS to {}".format(dt.datetime.now().strftime('%H:%M:%S'), self.TS_path))
                self.TS.to_netcdf(path=self.TS_path, mode="w")
