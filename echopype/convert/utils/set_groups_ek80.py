from .set_groups_base import SetGroupsBase
import xarray as xr
import os
import numpy as np
import netCDF4
import zarr


class SetGroupsEK80(SetGroupsBase):
    """Class for setting groups in netCDF file for EK60 data.
    """

    def set_env(self, env_dict):
        """Set the Environment group in the EK60 netCDF file.

        Parameters
        ----------
        env_dict
            dictionary containing environment group params
                         env_dict['frequency']
                         env_dict['absorption_coeff']
                         env_dict['sound_speed']
        """
        # Only save environment group if file_path exists
        if self.format == '.nc':
            ncfile = netCDF4.Dataset(self.file_path, "a", format="NETCDF4")
            env = ncfile.createGroup("Environment")

            # set group attributes
            for k, v in env_dict.items():
                env.setncattr(k, v)

            # close nc file
            ncfile.close()
        elif self.format == '.zarr':
            zarrfile = zarr.open(self.file_path, mode='a')
            env = zarrfile.create_group('Environment')

            for k, v in env_dict.items():
                env.attrs[k] = v

    def set_platform(self, platform_dict):
        """Set the Platform group in the EK60 nc file.

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
                attrs={'platform_code_ICES': platform_dict['platform_code_ICES'],
                       'platform_name': platform_dict['platform_name'],
                       'platform_type': platform_dict['platform_type']})
            # save to file
            if self.format == '.nc':
                ds.to_netcdf(path=self.file_path, mode='a', group='Platform')
            elif self.format == '.zarr':
                ds.to_zarr(store=self.file_path, mode='a', group='Platform')

    def set_sonar(self, sonar_dict):
        """Set the Sonar group in the nc file.

        Parameters
        ----------
        sonar_dict
            dictionary containing sonar parameters for each sonar
        """
        # TODO: This probably doesn't work for multiple channels
        # create group
        if self.format == '.nc':
            for ch_id, data in sonar_dict.items():
                ncfile = netCDF4.Dataset(self.file_path, "a", format="NETCDF4")
                snr = ncfile.createGroup("Sonar")

                # set group attributes
                for k, v in data.items():
                    snr.setncattr(k, v)

            # close nc file
            ncfile.close()
        elif self.format == '.zarr':
            zarrfile = zarr.open(self.file_path, mode='a')
            snr = zarrfile.create_group('Sonar')
            for ch_id, data in sonar_dict.items():
                for k, v in data.items():
                    snr.attrs[k] = v

    def set_beam(self, beam_dict):
        """Set the Beam group in the EK60 nc file.

        Parameters
        ----------
        beam_dict
            dictionary containing general beam parameters
        """

        # Only save beam group if file_path exists
        if not os.path.exists(self.file_path):
            print('netCDF file does not exist, exiting without saving Beam group...')
        else:
            # Convert np.datetime64 numbers to seconds since 1900-01-01
            # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
            ping_time = (beam_dict['ping_time'] - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')

            ds = xr.Dataset(
                {'channel_id': (['frequency'], beam_dict['channel_id']),
                 'frequency_start': (['frequency', 'ping_time'], np.array(beam_dict['frequency_start']),
                                     {'long_name': 'Starting frequency of the transducer',
                                      'units': 'Hz'}),
                 'frequency_end': (['frequency', 'ping_time'], np.array(beam_dict['frequency_end']),
                                   {'long_name': 'Ending frequency of the transducer',
                                    'units': 'Hz'}),
                 'beamwidth_receive_alongship': (['frequency'], beam_dict['beam_width']['beamwidth_receive_major'],
                                                 {'long_name': 'Half power one-way receive beam width along '
                                                  'alongship axis of beam',
                                                  'units': 'arc_degree',
                                                  'valid_range': (0.0, 360.0)}),
                 'beamwidth_receive_athwartship': (['frequency'], beam_dict['beam_width']['beamwidth_receive_minor'],
                                                   {'long_name': 'Half power one-way receive beam width along '
                                                    'athwartship axis of beam',
                                                    'units': 'arc_degree',
                                                    'valid_range': (0.0, 360.0)}),
                 'beamwidth_transmit_alongship': (['frequency'], beam_dict['beam_width']['beamwidth_transmit_major'],
                                                  {'long_name': 'Half power one-way transmit beam width along '
                                                   'alongship axis of beam',
                                                   'units': 'arc_degree',
                                                   'valid_range': (0.0, 360.0)}),
                 'beamwidth_transmit_athwartship': (['frequency'], beam_dict['beam_width']['beamwidth_transmit_minor'],
                                                    {'long_name': 'Half power one-way transmit beam width along '
                                                     'athwartship axis of beam',
                                                     'units': 'arc_degree',
                                                     'valid_range': (0.0, 360.0)}),
                 'beam_direction_x': (['frequency'], beam_dict['beam_direction']['beam_direction_x'],
                                      {'long_name': 'x-component of the vector that gives the pointing '
                                                    'direction of the beam, in sonar beam coordinate '
                                                    'system',
                                       'units': '1',
                                       'valid_range': (-1.0, 1.0)}),
                 'beam_direction_y': (['frequency'], beam_dict['beam_direction']['beam_direction_x'],
                                      {'long_name': 'y-component of the vector that gives the pointing '
                                                    'direction of the beam, in sonar beam coordinate '
                                                    'system',
                                       'units': '1',
                                       'valid_range': (-1.0, 1.0)}),
                 'beam_direction_z': (['frequency'], beam_dict['beam_direction']['beam_direction_x'],
                                      {'long_name': 'z-component of the vector that gives the pointing '
                                                    'direction of the beam, in sonar beam coordinate '
                                                    'system',
                                       'units': '1',
                                       'valid_range': (-1.0, 1.0)}),
                 'angle_offset_alongship': (['frequency'], beam_dict['beam_angle']['angle_offset_alongship'],
                                            {'long_name': 'electrical alongship angle of the transducer'}),
                 'angle_offset_athwartship': (['frequency'], beam_dict['beam_angle']['angle_offset_athwartship'],
                                              {'long_name': 'electrical athwartship angle of the transducer'}),
                 'angle_sensitivity_alongship': (['frequency'], beam_dict['beam_angle']['angle_sensitivity_alongship'],
                                                 {'long_name': 'alongship sensitivity of the transducer'}),
                 'angle_sensitivity_athwartship': (['frequency'], beam_dict['beam_angle']['angle_sensitivity_athwartship'],
                                                   {'long_name': 'athwartship sensitivity of the transducer'}),
                 'equivalent_beam_angle': (['frequency'], beam_dict['equivalent_beam_angle'],
                                           {'long_name': 'Equivalent beam angle',
                                            'units': 'sr',
                                            'valid_range': (0.0, 4 * np.pi)}),
                 'gain_correction': (['frequency'], beam_dict['gain_correction'],
                                     {'long_name': 'Gain correction',
                                      'units': 'dB'}),
                #  'non_quantitative_processing': (['frequency'], beam_dict['non_quantitative_processing'],
                #                                  {'flag_meanings': 'no_non_quantitative_processing',
                #                                   'flag_values': '0',
                #                                   'long_name': 'Presence or not of non-quantitative '
                #                                                'processing applied to the backscattering '
                #                                                'data (sonar specific)'}),
                 'sample_interval': (['frequency'], beam_dict['sample_interval'],
                                     {'long_name': 'Interval between recorded raw data samples',
                                      'units': 's',
                                      'valid_min': 0.0}),
                #  'sample_time_offset': (['frequency'], beam_dict['sample_time_offset'],
                #                         {'long_name': 'Time offset that is subtracted from the timestamp '
                #                                       'of each sample',
                #                          'units': 's'}),
                 'transmit_bandwidth': (['frequency'], beam_dict['transmit_signal']['transmit_bandwidth'],
                                        {'long_name': 'Nominal bandwidth of transmitted pulse',
                                         'units': 'Hz',
                                         'valid_min': 0.0}),
                 'transmit_duration_nominal': (['frequency'], beam_dict['transmit_signal']['transmit_duration_nominal'],
                                               {'long_name': 'Nominal bandwidth of transmitted pulse',
                                                'units': 's',
                                                'valid_min': 0.0}),
                 'transmit_power': (['frequency'], beam_dict['transmit_signal']['transmit_power'],
                                    {'long_name': 'Nominal transmit power',
                                     'units': 'W',
                                     'valid_min': 0.0}),
                 'transducer_offset_x': (['frequency'], beam_dict['transducer_position']['transducer_offset_x'],
                                         {'long_name': 'x-axis distance from the platform coordinate system '
                                                       'origin to the sonar transducer',
                                          'units': 'm'}),
                 'transducer_offset_y': (['frequency'], beam_dict['transducer_position']['transducer_offset_y'],
                                         {'long_name': 'y-axis distance from the platform coordinate system '
                                                       'origin to the sonar transducer',
                                          'units': 'm'}),
                 'transducer_offset_z': (['frequency'], beam_dict['transducer_position']['transducer_offset_z'],
                                         {'long_name': 'z-axis distance from the platform coordinate system '
                                                       'origin to the sonar transducer',
                                          'units': 'm'})
                 },
                coords={'frequency': (['frequency'], beam_dict['frequency']),
                        'ping_time': (['ping_time'], ping_time,
                                      {'axis': 'T',
                        #                'calendar': 'gregorian',
                                       'long_name': 'Timestamp of each ping',
                                       'standard_name': 'time'}),
                        #                'units': 'seconds since 1900-01-01'}),
                        },
                attrs={'beam_mode': beam_dict['beam_mode'],
                       'conversion_equation_t': beam_dict['conversion_equation_t']})
            if beam_dict['complex']:
                bs = xr.Dataset(
                    {'backscatter_r': (['frequency', 'quadrant', 'ping_time', 'range_bin'], beam_dict['backscatter_r'],
                                       {'long_name': 'Real part of backscatter power',
                                        'units': 'dB'}),
                     'backscatter_i': (['frequency', 'quadrant', 'ping_time', 'range_bin'], beam_dict['backscatter_i'],
                                       {'long_name': 'Imaginary part of backscatter power',
                                        'units': 'dB'})},
                    coords={'frequency': (['frequency'], beam_dict['frequency']),
                            'ping_time': (['ping_time'], ping_time,
                                          {'axis': 'T',
                            #                'calendar': 'gregorian',
                                           'long_name': 'Timestamp of each ping',
                                           'standard_name': 'time'}),
                            #                'units': 'seconds since 1900-01-01'}),
                            'quadrant': (['quadrant'], np.arange(4)),
                            'range_bin': (['range_bin'], beam_dict['range_bin']),
                            })

            else:
                bs = xr.Dataset(
                    {'backscatter_r': (['frequency', 'ping_time', 'range_bin'], beam_dict['backscatter_r'],
                                       {'long_name': 'Real part of backscatter power',
                                        'units': 'dB'}),
                     'backscatter_i': (['frequency', 'ping_time', 'range_bin'],
                                       np.full_like(beam_dict['backscatter_r'], np.nan),
                                       {'long_name': 'Imaginary part of backscatter power',
                                        'units': 'dB'})},
                    coords={'frequency': (['frequency'], beam_dict['frequency']),
                            'ping_time': (['ping_time'], ping_time,
                                          {'axis': 'T',
                            #                'calendar': 'gregorian',
                                           'long_name': 'Timestamp of each ping',
                                           'standard_name': 'time'}),
                            #                'units': 'seconds since 1900-01-01'}),
                            'range_bin': (['range_bin'], beam_dict['range_bin']),
                            })
            ds = xr.merge([ds, bs])

            # Below are specific to Simrad .raw files
            if 'gpt_software_version' in beam_dict:
                ds['gpt_software_version'] = ('frequency', beam_dict['gpt_software_version'])
            if 'sa_correction' in beam_dict:
                ds['sa_correction'] = ('frequency', beam_dict['sa_correction'])

            # save to file
            if self.format == '.nc':
                ds.to_netcdf(path=self.file_path, mode='a', group='Beam',
                             encoding={'backscatter_r': {'zlib': True, 'complevel': 4},
                                       'backscatter_i': {'zlib': True, 'complevel': 4}})
            elif self.format == '.zarr':
                ds.to_zarr(store=self.file_path, mode='a', group='Beam',
                           encoding={'backscatter_r': {'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)},
                                     'backscatter_i': {'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)}})
