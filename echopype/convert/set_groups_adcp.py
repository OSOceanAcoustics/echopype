from .set_groups_base import SetGroupsBase
import xarray as xr
import numpy as np
import netCDF4
import os


class SetGroupsADCP(SetGroupsBase):

    def set_env(self, env_dict):
        if not os.path.exists(self.file_path):
            print('netCDF file does not exist, exiting without saving Environment group...')
        else:
            ds = xr.Dataset({'temperature': (['ping_time'], env_dict['temperature']),
                             'pressure': (['ping_time'], env_dict['pressure']),
                             'sound_speed_indicative': (['ping_time'], env_dict['sound_speed'])},
                            coords={'ping_time': (['ping_time'], env_dict['ping_time'],
                                    {'axis': 'T',
                                     'calendar': 'gregorian',
                                     'long_name': 'Timestamp of each ping',
                                     'standard_name': 'time',
                                     'units': 'seconds since 1970-01-01'})})
            # save to file
            ds.to_netcdf(path=self.file_path, mode='a', group='Environment')

    def set_platform(self, platform_dict):
        ds = xr.Dataset({'pitch': (['ping_time'], platform_dict['pitch']),
                         'roll': (['ping_time'], platform_dict['roll']),
                         'heading': (['ping_time'], platform_dict['heading'])},
                        coords={'ping_time': (['ping_time'], platform_dict['ping_time'],
                                              {'axis': 'T',
                                               'calendar': 'gregorian',
                                               'long_name': 'Timestamp of each ping',
                                               'standard_name': 'time',
                                               'units': 'seconds since 1970-01-01'})},
                        attrs={'platform_name': platform_dict['platform_name'],
                               'platform_type': platform_dict['platform_type'],
                               'platform_code_ICES': platform_dict['platform_code_ICES']})

        ds.to_netcdf(path=self.file_path, mode='a', group='platform')

    def set_beam(self, beam_dict):
        temp = np.ones((1, len(beam_dict['ping_time']),
                        len(beam_dict['range_bin'])))
        ds = xr.Dataset({'temp': (['frequency', 'ping_time', 'range_bin'], temp)},
                        coords={'ping_time': (['ping_time'], beam_dict['ping_time'],
                                              {'axis': 'T',
                                               'calendar': 'gregorian',
                                               'long_name': 'Timestamp of each ping',
                                               'standard_name': 'time',
                                               'units': 'seconds since 1970-01-01'}),
                                'range_bin': (['range_bin'], beam_dict['range_bin'],
                                              {'units': 'meters'})},
                        attrs={'frequency': beam_dict['frequency']})

        ds.to_netcdf(path=self.file_path, mode='a', group='Beam')
