"""
echopype data model inherited from based class EchoData for AZFP data.
"""

import datetime as dt
import numpy as np
import xarray as xr
from .modelbase import ModelBase


class ModelAZFP(ModelBase):
    """Class for manipulating AZFP echo data that is already converted to netCDF."""

    def __init__(self, file_path=""):
        ModelBase.__init__(self, file_path)

    def calc_range(self, tilt_corrected=False):
        ds_beam = xr.open_dataset(self.file_path, group='Beam')
        ds_vend = xr.open_dataset(self.file_path, group='Vendor')
        ds_env = xr.open_dataset(self.file_path, group='Environment')

        frequency = ds_beam.frequency
        range_samples = ds_beam.number_of_samples_digitized_per_pings
        pulse_length = ds_beam.transmit_duration_nominal   # units: seconds
        bins_to_avg = 1   # set to 1 since we want to calculate from raw data
        range_bin = ds_beam.range_bin
        sound_speed = ds_env.sound_speed_indicative
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
                 (pulse_length / np.timedelta64(1, 's')))).drop('ping_time')
        if tilt_corrected:
            depth = ds_beam.cos_tilt_mag.mean() * depth

        ds_beam.close()
        ds_vend.close()
        ds_env.close()

        return depth

    def calibrate(self, save=False):
        """Perform echo-integration to get volume backscattering strength (Sv) from AZFP power data.

        Parameters
        -----------
        save : bool, optional
               whether to save calibrated Sv output
               default to ``True``
        """

        # Open data set for Environment and Beam groups
        ds_env = xr.open_dataset(self.file_path, group="Environment")
        ds_beam = xr.open_dataset(self.file_path, group="Beam")

        # Derived params # -FC (should choose between 'depth' and 'calc_range' which are the same)
        sample_thickness = ds_env.sound_speed_indicative * (ds_beam.sample_interval / np.timedelta64(1, 's')) / 2
        depth = self.calc_range()

        # From Ek60.py:
        # Calibration and echo integration 
        #Sv = ds_beam.backscatter_r + TVG + ABS - CSv - 2 * ds_beam.sa_correction
        #Sv.name = 'Sv'
        self.Sv = (ds_beam.EL - 2.5 / ds_beam.DS + ds_beam.backscatter_r / (26214 * ds_beam.DS) -
                   ds_beam.TVR - 20 * np.log10(ds_beam.VTX) + 20 * np.log10(depth) +
                   2 * ds_beam.sea_abs * depth -
                   10 * np.log10(0.5 * ds_env.sound_speed_indicative *
                                 ds_beam.transmit_duration_nominal.astype('float64') / 1e9 *
                                 ds_beam.equivalent_beam_angle) + ds_beam.Sv_offset)

        # -FC from ../model/Ek60.py:
        # ---------
        # Get TVG and absorption
        range_meter = self.calc_range()
        ## range_meter = ds_beam.range_bin * sample_thickness - \
        ##               self.tvg_correction_factor * sample_thickness  # DataArray [frequency x range_bin]
        range_meter = range_meter.where(range_meter > 0, other=0)  # set all negative elements to 0
        TVG = np.real(20 * np.log10(range_meter.where(range_meter != 0, other=1)))
        ABS = 2 * ds_env.absorption_indicative * range_meter

        # Save TVG and ABS for noise estimation use
        self.sample_thickness = sample_thickness
        self.TVG = TVG
        self.ABS = ABS
        # ---------

        if save:
            print("{} saving calibrated Sv to {}".format(dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
            self.Sv.to_dataset(name="Sv").to_netcdf(path=self.Sv_path, mode="w")

        # Close opened resources
        ds_env.close()
        ds_beam.close()
        pass

    def calibrate_ts(self, save=False):
        ds_beam = xr.open_dataset(self.file_path, group="Beam")
        depth = self.calc_range()

        self.TS = (ds_beam.EL - 2.5 / ds_beam.DS + ds_beam.backscatter_r / (26214 * ds_beam.DS) -
                   ds_beam.TVR - 20 * np.log10(ds_beam.VTX) + 40 * np.log10(depth) +
                   2 * ds_beam.sea_abs * depth)
        if save:
            print("{} saving calibrated TS to {}".format(dt.datetime.now().strftime('%H:%M:%S'), self.TS_path))
            self.TS.to_dataset(name="TS").to_netcdf(path=self.TS_path, mode="w")

        ds_beam.close()
