from __future__ import absolute_import, division, print_function
import os
import numpy as np
import netCDF4
import zarr
import xarray as xr


class SetGroupsBase:
    """Base class for setting groups in netCDF file.
    """

    def __init__(self, file_path='test.nc', compress=True):
        self.file_path = file_path
        filename, ext = os.path.splitext(file_path)
        self.format = ext
        self.compress = compress

    def set_toplevel(self, tl_dict):
        """Set attributes in the Top-level group."""
        if self.format == '.nc':
            with netCDF4.Dataset(self.file_path, "w", format="NETCDF4") as ncfile:
                [ncfile.setncattr(k, v) for k, v in tl_dict.items()]
        elif self.format == '.zarr':
            zarrfile = zarr.open(self.file_path, mode="w")
            for k, v in tl_dict.items():
                zarrfile.attrs[k] = v
        else:
            raise ValueError("Unsupported file format")

    def set_provenance(self, src_file_names, prov_dict):
        """Set the Provenance group in the nc file.

        Parameters
        ----------
        src_file_names
            list of source filenames
        prov_dict
            dictionary containing file conversion parameters
                          prov_dict['conversion_software_name']
                          prov_dict['conversion_software_version']
                          prov_dict['conversion_time']
        """
        # create group
        files = ", ".join([os.path.basename(file) for file in src_file_names])
        if self.format == '.nc':
            file = netCDF4.Dataset(self.file_path, "a", format="NETCDF4")
            pr = file.createGroup("Provenance")
            # dimensions
            pr.createDimension("filenames", None)
            # variables
            pr_src_fnames = pr.createVariable(files, str, "filenames")
            pr_src_fnames.long_name = "Source filenames"

            # set group attributes
            for k, v in prov_dict.items():
                pr.setncattr(k, v)
            # close nc file
            file.close()

        elif self.format == '.zarr':
            file = zarr.open(self.file_path, 'a')
            pr = file.create_group('Provenance')
            pr_src_fnames = pr.create_dataset('filenames', data=files)
            pr_src_fnames.attrs['long_name'] = "Source filenames"
            for k, v in prov_dict.items():
                pr[k] = v

    def set_sonar(self, sonar_dict):
        """Set the Sonar group in the nc file.

        Parameters
        ----------
        sonar_dict
            dictionary containing sonar parameters
        """
        # create group
        if self.format == '.nc':
            ncfile = netCDF4.Dataset(self.file_path, "a", format="NETCDF4")
            snr = ncfile.createGroup("Sonar")

            # set group attributes
            for k, v in sonar_dict.items():
                snr.setncattr(k, v)

            # close nc file
            ncfile.close()
        elif self.format == '.zarr':
            zarrfile = zarr.open(self.file_path, mode='a')
            snr = zarrfile.create_group('Sonar')

            for k, v in sonar_dict.items():
                snr.attrs[k] = v

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
            if self.format == '.nc':
                ds.to_netcdf(path=self.file_path, mode='a', group='Platform/NMEA')
            elif self.format == '.zarr':
                ds.to_zarr(store=self.file_path, mode='a', group='Platform/NMEA')
