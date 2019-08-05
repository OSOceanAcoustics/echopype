"""
Functions to unpack Simrad EK60 .raw and save to .nc.
"""

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

    def set_env(self, env_dict):
        """Set the Environment group in the netCDF file.

        Parameters
        ----------
        env_dict
            dictionary containing environment group params
                         env_dict['frequency']
                         env_dict['absorption_coeff']
                         env_dict['sound_speed']
        """
        # Only save environment group if file_path exists
        if not os.path.exists(self.file_path):
            print('netCDF file does not exist, exiting without saving Environment group...')
        else:
            absorption = xr.DataArray(env_dict['absorption_coeff'],
                                      coords=[env_dict['frequency']], dims=['frequency'],
                                      attrs={'long_name': "Indicative acoustic absorption",
                                             'units': "dB/m",
                                             'valid_min': 0.0})
            sound_speed = xr.DataArray(env_dict['sound_speed'],
                                       coords=[env_dict['frequency']], dims=['frequency'],
                                       attrs={'long_name': "Indicative sound speed",
                                              'standard_name': "speed_of_sound_in_sea_water",
                                              'units': "m/s",
                                              'valid_min': 0.0})
            ds = xr.Dataset({'absorption_indicative': absorption,
                             'sound_speed_indicative': sound_speed},
                            coords={'frequency': (['frequency'], env_dict['frequency'])})

            ds.frequency.attrs['long_name'] = "Acoustic frequency"
            ds.frequency.attrs['standard_name'] = "sound_frequency"
            ds.frequency.attrs['units'] = "Hz"
            ds.frequency.attrs['valid_min'] = 0.0

            # save to file
            ds.to_netcdf(path=self.file_path, mode="a", group="Environment")

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

    def set_platform(self, platform_dict):
        """Set the Platform group in the nc file.

        Parameters
        ----------
        platform_dict
            dictionary containing platform parameters
        """
        # Only save platform group if file_path exists
        if not os.path.exists(self.file_path):
            print('netCDF file does not exist, exiting without saving Platform group...')
        else:
            # Convert np.datetime64 numbers to seconds since 1900-01-01
            # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
            ping_time = (platform_dict['ping_time'] - np.datetime64('1900-01-01T00:00:00')) \
                        / np.timedelta64(1, 's')
            location_time = (platform_dict['location_time'] - np.datetime64('1900-01-01T00:00:00')) \
                            / np.timedelta64(1, 's')

            ds = xr.Dataset(
                {'pitch': (['ping_time'], platform_dict['pitch'],
                            {'long_name': 'Platform pitch',
                            'standard_name': 'platform_pitch_angle',
                            'units': 'arc_degree',
                            'valid_range': (-90.0, 90.0)}),
                    'roll': (['ping_time'], platform_dict['roll'],
                            {'long_name': 'Platform roll',
                            'standard_name': 'platform_roll_angle',
                            'units': 'arc_degree',
                            'valid_range': (-90.0, 90.0)}),
                    'heave': (['ping_time'], platform_dict['heave'],
                            {'long_name': 'Platform heave',
                            'standard_name': 'platform_heave_angle',
                            'units': 'arc_degree',
                            'valid_range': (-90.0, 90.0)}),
                    'latitude': (['location_time'], platform_dict['lat'],
                            {'long_name': 'Platform latitude',
                            'standard_name': 'latitude',
                            'units': 'degrees_north',
                            'valid_range': (-90.0, 90.0)}),
                    'longitude': (['location_time'], platform_dict['lon'],
                                {'long_name': 'Platform longitude',
                                'standard_name': 'longitude',
                                'units': 'degrees_east',
                                'valid_range': (-180.0, 180.0)}),
                    'water_level': ([], platform_dict['water_level'],
                                    {'long_name': 'z-axis distance from the platform coordinate system '
                                                'origin to the sonar transducer',
                                    'units': 'm'})
                    },
                coords={'ping_time': (['ping_time'], ping_time,
                                        {'axis': 'T',
                                        'calendar': 'gregorian',
                                        'long_name': 'Timestamps for position datagrams',
                                        'standard_name': 'time',
                                        'units': 'seconds since 1900-01-01'}),
                        'location_time': (['location_time'], location_time,
                                        {'axis': 'T',
                                        'calendar': 'gregorian',
                                        'long_name': 'Timestamps for NMEA position datagrams',
                                        'standard_name': 'time',
                                        'units': 'seconds since 1900-01-01'})
                        },
                attrs={'platform_code_ICES': '',
                        'platform_name': platform_dict['platform_name'],
                        'platform_type': platform_dict['platform_type']})
            # save to file
            ds.to_netcdf(path=self.file_path, mode="a", group="Platform")

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
