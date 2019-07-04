"""
echopype data model inherited from based class EchoData for AZFP data.
"""

import datetime as dt
import numpy as np
import xarray as xr
from .echo_data import EchoData


class EchoDataAZFP(EchoData):
    """Base class for echo data."""

    def __init__(self, file_path=""):
        EchoData.__init__(self, file_path)

    def calibrate(self, save=True):
        """Perform echo-integration to get volume backscattering strength (Sv) from AZFP power data.

        Parameters
        -----------
        save : bool, optional
               whether to save calibrated Sv output
               default to ``True``
        """
        ds_env = xr.open_dataset(self.file_path, group="Environment")
        ds_beam = xr.open_dataset(self.file_path, group="Beam")
        self.Sv = (ds_beam.EL - 2.5 / ds_beam.DS + ds_beam.backscatter_r / (26214 * ds_beam.DS) -
                   ds_beam.TVR - 20 * np.log10(ds_beam.VTX) + 20 * np.log10(ds_beam.range) +
                   2 * ds_beam.sea_abs * ds_beam.range -
                   10 * np.log10(0.5 * ds_env.sound_speed_indicative *
                                 ds_beam.transmit_duration_nominal.astype('float64') / 1e9 *
                                 ds_beam.equivalent_beam_angle) + ds_beam.Sv_offset)
        if save:
            print("{} saving calibrated Sv to {}".format(dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
            self.Sv.to_dataset(name="Sv").to_netcdf(path=self.Sv_path, mode="w")

        # Close opened resources
        ds_env.close()
        ds_beam.close()
        pass

    def calibrate_ts(self, save=True):
        ds_beam = xr.open_dataset(self.file_path, group="Beam")

        self.TS = (ds_beam.EL - 2.5 / ds_beam.DS + ds_beam.backscatter_r / (26214 * ds_beam.DS) -
                   ds_beam.TVR - 20 * np.log10(ds_beam.VTX) + 40 * np.log10(ds_beam.range) +
                   2 * ds_beam.sea_abs * ds_beam.range)
        if save:
            print("{} saving calibrated TS to {}".format(dt.datetime.now().strftime('%H:%M:%S'), self.TS_path))
            self.TS.to_dataset(name="TS").to_netcdf(path=self.TS_path, mode="w")

        ds_beam.close()
