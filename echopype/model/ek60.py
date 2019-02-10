"""
echopype data model that keeps tracks of echo data and
its connection to data files.
"""

import os
import xarray as xr


class EchoData(object):
    """Base class for echo data."""

    def __init__(self, file_path=""):
        self.file_path = file_path  # this passes the input through file name test

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, p):
        self._file_path = p

        # Load netCDF groups if file format is correct
        pp = os.path.basename(p)
        _, ext = os.path.splitext(pp)

        if ext is '.raw':
            print('Data file in manufacturer format, please convert to .nc first.')
        elif ext is '.nc':
            self.toplevel = xr.open_dataset(self.file_path)
            if self.toplevel['sonar_convention_version'] == 'SONAR-netCDF4':
                self.provenance = xr.open_dataset(self.file_path, group="Provenance")
                self.environment = xr.open_dataset(self.file_path, group="Environment")
                self.platform = xr.open_dataset(self.file_path, group="Platform")
                self.sonar = xr.open_dataset(self.file_path, group="Sonar")
                self.beam = xr.open_dataset(self.file_path, group="Beam")
            else:
                print('netCDF file convention not recognized.')
        else:
            print('Data file format not recognized.')

    # @file_path.setter
    # def file_path(self, p):
    #     pp = os.path.basename(p)
    #     _, ext = os.path.splitext(pp)
    #     if ext == '.raw':
    #         raise ValueError('Data file in manufacturer format, please convert to .nc format first.')
    #     elif ext == '.nc':
    #         print('Got an .nc file! can start processing!')
    #         # print('Let us try to set some attributes')
    #     elif ext == '':
    #         print('Do nothing with empty file_path')
    #         self.toplevel = []
    #         self.provenance = []
    #         self.environment = []
    #         self.platform = []
    #         self.sonar = []
    #         self.beam = []
    #     else:
    #         # print('Not sure what file this is. EchoData only accepts .nc file as inputs.)
    #         raise ValueError('Not sure what file this is?? try to find a .nc file??')
    #     self._file_path = p


    # def load(self, data_path):
    #     data_file = os.path.basename(data_path)
    #     data_ext = os.path.splitext(data_file)
    #     """Load echo data from file"""
    #     if data_ext is '.raw':
    #         print('Data file in manufacturer format, please convert first.')
    #         print('To convert data, use the following... ')
    #     elif data_ext is '.nc':
    #         self.nc_path = data_path
    #         self.nc_file = data_file
    #         self.toplevel = xr.open_dataset(self.data_path)
    #         if self.toplevel['sonar_convention_version'] == 'SONAR-netCDF4':
    #             # load groups as xarray objects
    #             self.provenance = xr.open_dataset(self.data_path, group="Provenance")
    #             self.environment = xr.open_dataset(self.data_path, group="Environment")
    #             self.platform = xr.open_dataset(self.data_path, group="Platform")
    #             self.sonar = xr.open_dataset(self.data_path, group="Sonar")
    #             self.beam = xr.open_dataset(self.data_path, group="Beam")
    #         else:
    #             print('netCDF file convention not recognized.')
    #     else:
    #         print('Data file format not recognized.')
    #
    # def set_nc_path(self, nc_path):
    #     path = os.path.dirname(nc_path)
    #     if path != '':
    #         self.nc_path = nc_path
    #     self.nc_file = os.path.basename(nc_path)
    #
    # def get_nc_path(self):
    #     return self.nc_path
    #
    # def set_nc_file(self, nc_file):
    #     self.nc_file = os.path.basename(nc_file)
    #
    # def get_nc_file(self):
    #     return self.nc_file



