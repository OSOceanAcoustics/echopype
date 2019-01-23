"""
echopype data model that keeps tracks of echo data and
its connection to data files.
"""

import os
import xarray as xr


class EchoData(object):
    """Based class for echo data."""

    def __init__(self):
        self.data_path = ""
        self.data_file = ""
        self.nc_path = ""
        self.nc_file = ""
        self.sonar_model = ""

        # attributes for loading data
        self.toplevel = []
        self.provenance = []
        self.environment = []
        self.platform = []
        self.sonar = []
        self.beam = []

    def load(self, data_path):
        data_file = os.path.basename(data_path)
        data_ext = os.path.splitext(data_file)
        """Load echo data from file"""
        if data_ext is '.raw':
            print('Data file in manufacturer format, please convert first.')
            print('To convert data, use the following... ')
        elif data_ext is '.nc':
            self.nc_path = data_path
            self.nc_file = data_file
            self.toplevel = xr.open_dataset(self.data_path)
            if self.toplevel['sonar_convention_version'] == 'SONAR-netCDF4':
                # load groups as xarray objects
                self.provenance = xr.open_dataset(self.data_path, group="Provenance")
                self.environment = xr.open_dataset(self.data_path, group="Environment")
                self.platform = xr.open_dataset(self.data_path, group="Platform")
                self.sonar = xr.open_dataset(self.data_path, group="Sonar")
                self.beam = xr.open_dataset(self.data_path, group="Beam")
            else:
                print('netCDF file convention not recognized.')
        else:
            print('Data file format not recognized.')

    def set_nc_path(self, nc_path):
        path = os.path.dirname(nc_path)
        if path != '':
            self.nc_path = nc_path
        self.nc_file = os.path.basename(nc_path)

    def get_nc_path(self):
        return self.nc_path

    def set_nc_file(self, nc_file):
        self.nc_file = os.path.basename(nc_file)

    def get_nc_file(self):
        return self.nc_file



