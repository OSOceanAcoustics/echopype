import os
import datetime as dt
import numpy as np
import xarray as xr

from echopype.convert.azfp import ConvertAZFP

path = "./echopype/data/17082117.01A"
xml_path = "./echopype/data/17041823.XML"

class EchoData(object):
    """Base class for echo data."""

    def __init__(self, file_path=""):
        self.file_path = file_path  # this passes the input through file name test
        # self.tvg_correction_factor = 2  # range bin offset factor for calculating time-varying gain in EK60
        # self.TVG = None  # time varying gain along range
        # self.ABS = None  # absorption along range
        # self.sample_thickness = None  # sample thickness for each frequency
        # self.noise_est_range_bin_size = 5  # meters per tile for noise estimation
        # # self.noise_est_ping_size = 30  # number of pings per tile for noise estimation
        # self.MVBS_range_bin_size = 5  # meters per tile for MVBS
        # self.MVBS_ping_size = 30  # number of pings per tile for MVBS
        self.TS = None  # Target strength
        self.Sv = None  # calibrated volume backscattering strength
        # self.Sv_clean = None  # denoised volume backscattering strength
        # self.MVBS = None  # mean volume backscattering strength

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, p):
        self._file_path = p

        # Load netCDF groups if file format is correct
        pp = os.path.basename(p)
        _, ext = os.path.splitext(pp)

        if ext == '.01A':
            print('Data file in manufacturer format, please convert to .nc first.')
        elif ext == '.nc':
            self.toplevel = xr.open_dataset(self.file_path)

            # Get .nc filenames for storing processed data if computation is performed
            self.Sv_path = os.path.join(os.path.dirname(self.file_path),
                                        os.path.splitext(os.path.basename(self.file_path))[0] + '_Sv.nc')
            self.Sv_clean_path = os.path.join(os.path.dirname(self.file_path),
                                              os.path.splitext(os.path.basename(self.file_path))[0] + '_Sv_clean.nc')
            self.MVBS_path = os.path.join(os.path.dirname(self.file_path),
                                          os.path.splitext(os.path.basename(self.file_path))[0] + '_MVBS.nc')

            # Raise error if the file format convention does not match
            if self.toplevel.sonar_convention_name != 'SONAR-netCDF4':
                raise ValueError('netCDF file convention not recognized.')
        else:
            raise ValueError('Data file format not recognized.')

    def calibrate(self, save=True):
        ds_env = xr.open_dataset(self.file_path, group="Environment")
        ds_beam = xr.open_dataset(self.file_path, group="Beam")

        self.TS = (ds_beam.EL - 2.5 / ds_beam.DS + ds_beam.backscatter_r / (26214 * ds_beam.DS) -
                   ds_beam.TVR - 20 * np.log10(ds_beam.VTX) + 40 * np.log10(ds_beam.range) +
                   2 * ds_beam.sea_abs * ds_beam.range)
        self.Sv = (ds_beam.EL - 2.5 / ds_beam.DS + ds_beam.backscatter_r / (26214 * ds_beam.DS) -
                   ds_beam.TVR - 20 * np.log10(ds_beam.VTX) + 20 * np.log10(ds_beam.range) +
                   2 * ds_beam.sea_abs * ds_beam.range -
                   10 * np.log10(0.5 * ds_env.sound_speed_indicative *
                                 ds_beam.transmit_duration_nominal.astype('float64') / 1e9 *
                                 ds_beam.equivalent_beam_angle) + ds_beam.Sv_offset)
        if save:
            print('%s  saving calibrated Sv to %s' % (dt.datetime.now().strftime('%H:%M:%S'), self.Sv_path))
            Sv.to_netcdf(path=self.Sv_path, mode="w")

        # Close opened resources
        ds_env.close()
        ds_beam.close()
        pass


# tmp = ConvertAZFP(path, xml_path)
# tmp.raw2nc()
test = EchoData('D:/Documents/Projects/echopype/echopype/data/17082117.nc')
test.calibrate()
pass
