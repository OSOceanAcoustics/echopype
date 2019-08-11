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

    def calibrate(self, save=False):
        """Perform echo-integration to get volume backscattering strength (Sv) from EK60 power data.

        TODO: need to write a separate method for calculating TS as have been done for AZFP data.

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
        sample_thickness = ds_env.sound_speed_indicative * ds_beam.sample_interval / 2  # sample thickness
        wavelength = ds_env.sound_speed_indicative / ds_env.frequency  # wavelength

        # Calc gain
        CSv = 10 * np.log10((ds_beam.transmit_power * (10 ** (ds_beam.gain_correction / 10)) ** 2 *
                             wavelength ** 2 * ds_env.sound_speed_indicative * ds_beam.transmit_duration_nominal *
                             10 ** (ds_beam.equivalent_beam_angle / 10)) /
                            (32 * np.pi ** 2))

        # Get TVG and absorption
        range_meter = ds_beam.range_bin * sample_thickness - \
                      self.tvg_correction_factor * sample_thickness  # DataArray [frequency x range_bin]
        range_meter = range_meter.where(range_meter > 0, other=0)  # set all negative elements to 0
        TVG = np.real(20 * np.log10(range_meter.where(range_meter != 0, other=1)))
        ABS = 2 * ds_env.absorption_indicative * range_meter

        # Save TVG and ABS for noise estimation use
        self.sample_thickness = sample_thickness
        self.TVG = TVG
        self.ABS = ABS

        # Calibration and echo integration
        Sv = ds_beam.backscatter_r + TVG + ABS - CSv - 2 * ds_beam.sa_correction
        Sv.name = 'Sv'

        # Save calibrated data into the calling instance and
        # ... to a separate .nc file in the same directory as the data file
        self.Sv = Sv
        if save:
            print('%s  saving calibrated Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
            Sv.to_netcdf(path=self.Sv_path, mode="w")

        # Close opened resources
        ds_env.close()
        ds_beam.close()
