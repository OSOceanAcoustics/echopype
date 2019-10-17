from __future__ import absolute_import, division, print_function
import os
import numpy as np
import netCDF4
import xarray as xr


class SetGroupsBase:
    """Base class for setting groups in netCDF file.
    """

    def __init__(self, file_path='test.nc'):
        self.file_path = file_path

    def set_toplevel(self, tl_dict):
        """Set attributes in the Top-level group."""
        with netCDF4.Dataset(self.file_path, "w", format="NETCDF4") as ncfile:
            [ncfile.setncattr(k, v) for k, v in tl_dict.items()]

    def set_provenance(self, src_file_names, prov_dict):
        """Set the Provenance group in the nc file.

        Parameters
        ----------
        src_file_names
            source filenames
        prov_dict
            dictionary containing file conversion parameters
                          prov_dict['conversion_software_name']
                          prov_dict['conversion_software_version']
                          prov_dict['conversion_time']
        """
        # create group
        nc_file = netCDF4.Dataset(self.file_path, "a", format="NETCDF4")
        pr = nc_file.createGroup("Provenance")

        # dimensions
        pr.createDimension("filenames", None)

        # variables
        pr_src_fnames = pr.createVariable(src_file_names, str, "filenames")
        pr_src_fnames.long_name = "Source filenames"

        # set group attributes
        for k, v in prov_dict.items():
            pr.setncattr(k, v)

        # close nc file
        nc_file.close()

    def set_sonar(self, sonar_dict):
        """Set the Sonar group in the nc file.

        Parameters
        ----------
        sonar_dict
            dictionary containing sonar parameters
        """
        # create group
        ncfile = netCDF4.Dataset(self.file_path, "a", format="NETCDF4")
        snr = ncfile.createGroup("Sonar")

        # set group attributes
        for k, v in sonar_dict.items():
            snr.setncattr(k, v)

        # close nc file
        ncfile.close()

    def set_nmea(self, nmea_dict):
        """Set the Platform/NMEA group in the nc file.

        Parameters
        ----------
        nmea_dict
            dictionary containing platform parameters
        """
        # Only save platform group if file_path exists
        if not os.path.exists(self.file_path):
            print('netCDF file does not exist, exiting without saving Platform group...')
        else:
            # Convert np.datetime64 numbers to seconds since 1900-01-01
            # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
            time = (nmea_dict['nmea_time'] - np.datetime64('1900-01-01T00:00:00')) \
                   / np.timedelta64(1, 's')
            ds = xr.Dataset(
                {'NMEA_datagram': (['time'], nmea_dict['nmea_datagram'],
                                   {'long_name': 'NMEA datagram'})
                 },
                coords={'time': (['time'], time,
                                 {'axis': 'T',
                                  'calendar': 'gregorian',
                                  'long_name': 'Timestamps for NMEA datagrams',
                                  'standard_name': 'time',
                                  'units': 'seconds since 1900-01-01'})},
                attrs={'description': 'All NMEA sensor datagrams'})
            # save to file
            ds.to_netcdf(path=self.file_path, mode="a", group="Platform/NMEA")
