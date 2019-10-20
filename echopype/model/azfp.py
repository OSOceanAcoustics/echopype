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

    def __init__(self, file_path="", salinity=29.6, pressure=60, temperature=None, sound_speed=None):
        ModelBase.__init__(self, file_path)
        self.salinity = salinity           # salinity in [psu]
        self.pressure = pressure           # pressure in [dbars] (approximately equal to depth in meters)
        self.temperature = temperature     # temperature in [Celsius]
        self.sound_speed = sound_speed     # sound speed in [m/s]
        # self._sample_thickness = None
        # self._range = None
        self._seawater_absorption = None

    # TODO: consider moving some of these properties to the parent class,
    #  since it is possible that EK60 users may want to set the environmental
    #  parameters separately from those recorded in the data files.

    @property
    def salinity(self):
        return self._salinity

    @salinity.setter
    def salinity(self, sal):
        self._salinity = sal
        # TODO: need to update sound speed, sample_thickness, absorption, range

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, pres):
        self._pressure = pres
        # TODO: need to update sound speed, sample_thickness, absorption, range

    @property
    def temperature(self):
        if self._temperature is None:
            with xr.open_dataset(self.file_path, group='Environment') as ds_env:
                self._temperature = ds_env.temperature
        return self._temperature

    @temperature.setter
    def temperature(self, t):
        self._temperature = t
        # TODO: need to update sound speed, sample_thickness, absorption, range

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
            self._sound_speed = self.calc_sound_speed()
        return self._sound_speed

    @sound_speed.setter
    def sound_speed(self, ss):
        self._sound_speed = ss
        # TODO: need to update sample_thickness, absorption, range

    @property
    def seawater_absorption(self):
        if self._seawater_absorption is None:  # if this is empty
            self._seawater_absorption = self.calc_seawater_absorption()
        return self._seawater_absorption

    @seawater_absorption.setter
    def seawater_absorption(self, sea_abs):
        self._seawater_absorption = sea_abs

    def calc_sample_thickness(self):
        """Gets ``sample_thickness`` for AZFP data.

        This will call ``calc_sound_speed`` since sound speed is `not` part of the raw AZFP .01A data file.
        """
        with xr.open_dataset(self.file_path, group="Beam") as ds_beam:
            sth = self.sound_speed * ds_beam.sample_interval / 2
            return sth.mean(dim='ping_time')   # use mean over all ping_time

    def calc_sound_speed(self, formula_source='AZFP'):
        """Calculate sound speed in meters per second. Uses the default salinity and pressure.

        Parameters
        ----------
        formula_source : str
            Source of formula used for calculating sound speed.
            Default is to use the formula supplied by AZFP (``formula_source='AZFP'``).
            Another option is to use Mackenzie (1981) supplied by ``arlpy`` (``formula_source='Mackenzie'``).

        Returns
        -------
        A sound speed [m/s] for each temperature.
        """
        if formula_source == 'Mackenzie':  # Mackenzie (1981) supplied by arlpy
            ss = arlpy.uwa.soundspeed(temperature=self.temperature,
                                      salinity=self.salinity,
                                      depth=self.pressure)
        else:  # default to formula supplied by AZFP
            z = self.temperature / 10
            ss = (1449.05 + z * (45.7 + z * ((-5.21) + 0.23 * z)) + (1.333 + z * ((-0.126) + z * 0.009)) *
                  (self.salinity - 35.0) + (self.pressure / 1000) * (16.3 + 0.18 * (self.pressure / 1000)))
        return ss

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

        # Converts sound speed to a single number. Otherwise depth will have dimension ping time
        if len(sound_speed) != 1:
            sound_speed = sound_speed.mean()
            # TODO: print out a message showing percentage of variation of sound speed across pings
            #       as the following:
            #       "Use mean sound speed. Sound speed varied by XX% across pings."

        # Below is from LoadAZFP.m, the output is effectively range_bin+1 when bins_to_avg=1
        range_mod = xr.DataArray(np.arange(1, len(ds_beam.range_bin) - bins_to_avg + 2, bins_to_avg),
                                 coords=[('range_bin', ds_beam.range_bin)])

        # Calculate range using parameters for each freq
        range_meter = (sound_speed * lockout_index / (2 * dig_rate) + sound_speed / 4 *
                       (((2 * range_mod - 1) * range_samples * bins_to_avg - 1) / dig_rate +
                        (pulse_length / np.timedelta64(1, 's'))))

        # # Below from @ngkavin --> @leewujung simplified to the above
        # m = []
        # for jj in range(len(frequency)):
        #     m.append(np.arange(1, len(range_bin) - bins_to_avg + 2,
        #              bins_to_avg))
        # m = xr.DataArray(m, coords=[('frequency', frequency), ('range_bin', range_bin)])
        #
        # # Calculate range from sound speed for each frequency
        # range_meter = (sound_speed * lockout_index[0] / (2 * dig_rate[0]) + sound_speed / 4 *
        #                (((2 * m - 1) * range_samples[0] * bins_to_avg - 1) / dig_rate[0] +
        #                 (pulse_length / np.timedelta64(1, 's'))))

        if tilt_corrected:
            range_meter = ds_beam.cos_tilt_mag.mean() * range_meter

        ds_beam.close()
        ds_vend.close()

        return range_meter

    def calc_seawater_absorption(self, formula_source='AZFP'):
        """Calculate the seawater absorption for all frequencies.

        Parameters
        ----------
        formula_source : str
            Source of formula used for calculating sound speed.
            Default is to use the formula supplied by AZFP (``formula_source='AZFP'``).
            Another option is to use Francois and Garrison (1982) supplied by ``arlpy`` (``formula_source='FG'``).

        Returns
        -------
        An array containing absorption coefficients for each frequency in dB/m
        """
        with xr.open_dataset(self.file_path, group='Beam') as ds_beam:
            freq = ds_beam.frequency  # should already be in unit [Hz]

        print('Using averaged temperature for calculating seawater absorption.')
        if formula_source == 'FG':
            linear_abs = arlpy.uwa.absorption(frequency=freq,
                                              temperature=self.temperature.mean(),
                                              salinity=self.salinity,
                                              depth=self.pressure)
            # Convert linear absorption to dB/km. Convert to dB/m
            sea_abs = -arlpy.utils.mag2db(linear_abs) / 1000

        # TODO: write a test function to compare AZFP formula output with outputs from Matlab code
        #  in the same way as you compare the echo data. The comparison should be done for a vector
        #  of frequencies np.logspace(0,6,500).
        else:  # default to formula provided by AZFP
            temp = self.temperature.mean(dim='ping_time')
            temp_k = temp + 273.0
            f1 = 1320.0 * temp_k * np.exp(-1700 / temp_k)
            f2 = 1.55e7 * temp_k * np.exp(-3052 / temp_k)

            # Coefficients for absorption calculations
            k = 1 + self.pressure / 10.0
            a = 8.95e-8 * (1 + temp * (2.29e-2 - 5.08e-4 * temp))
            b = (self.salinity / 35.0) * 4.88e-7 * (1 + 0.0134 * temp) * (1 - 0.00103 * k + 3.7e-7 * (k * k))
            c = (4.86e-13 * (1 + temp * ((-0.042) + temp * (8.53e-4 - temp * 6.23e-6))) *
                 (1 + k * (-3.84e-4 + k * 7.57e-8)))
            if self.salinity == 0:
                sea_abs = c * freq ** 2
            else:
                sea_abs = ((a * f1 * (freq ** 2)) / ((f1 * f1) + (freq ** 2)) +
                           (b * f2 * (freq ** 2)) / ((f2 * f2) + (freq ** 2)) + c * (freq ** 2))

        return sea_abs

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
        Sv = (ds_beam.EL - 2.5 / ds_beam.DS + ds_beam.backscatter_r / (26214 * ds_beam.DS) -
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

        Sv.name = 'Sv'
        Sv = Sv.to_dataset()

        # Attached calculated range into the dataset
        Sv['range'] = (('frequency', 'range_bin'), self.range)

        # Save calibrated data into the calling instance and
        #  to a separate .nc file in the same directory as the data filef.Sv = Sv
        self.Sv = Sv
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
