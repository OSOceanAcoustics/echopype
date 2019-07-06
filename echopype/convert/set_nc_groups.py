"""
Functions to unpack Simrad EK60 .raw and save to .nc.
"""

from __future__ import absolute_import, division, print_function
import os
import numpy as np
import netCDF4
import xarray as xr


class SetGroups(object):
    """Class for setting groups in netCDF file.
    """

    def __init__(self, file_path='test.nc'):
        self.file_path = file_path

    def set_toplevel(self, tl_dict):
        """Set attributes in the Top-level group."""
        with netCDF4.Dataset(self.file_path, "w", format="NETCDF4") as ncfile:
            [ncfile.setncattr(k, v) for k, v in tl_dict.items()]

    def set_env(self, env_dict, vendor="EK60"):
        """Set the Environment group in the netCDF file.

        Parameters
        ----------
        env_dict
            dictionary containing environment group params
                         env_dict['frequency']
                         env_dict['absorption_coeff']
                         env_dict['sound_speed']
        vendor
            specifies the type of echosounder

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
            # Additional AZFP-specific parameters 'salinity', 'temperature', and 'pressure'
            if vendor == "AZFP":
                salinity = xr.DataArray(env_dict['salinity'],
                                        coords=[env_dict['frequency']], dims=['frequency'],
                                        attrs={'long_name': "Water salinity",
                                               'standard_name': "salinity_of_sea_water",
                                               'units': "PSU"})
                pressure = xr.DataArray(env_dict['pressure'],
                                        coords=[env_dict['frequency']], dims=['frequency'],
                                        attrs={'long_name': "Water pressure",
                                               'standard_name': "pressure_in_sea_water",
                                               'units': "dBar"})
                ds = xr.Dataset({'absorption_indicative': absorption,
                                 'sound_speed_indicative': sound_speed,
                                 'salinity': salinity,
                                 'pressure': pressure},
                                coords={'frequency': (['frequency'], env_dict['frequency']),
                                        'temperature': env_dict['temperature']},
                                attrs={'pressure': env_dict['pressure'],  # pressure in dBar
                                       'salinity': env_dict['salinity']})  # salinity in PSU
            else:  # EK60 doesn't include additional parameters
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

    def set_platform(self, platform_dict, vendor="EK60"):
        """Set the Platform group in the nc file.

        Parameters
        ----------
        platform_dict
            dictionary containing platform parameters
        vendor
            specifies the type of echosounder
        """
        # Only save platform group if file_path exists
        if not os.path.exists(self.file_path):
            print('netCDF file does not exist, exiting without saving Platform group...')
        else:
            # Create an xarray dataset and save to netCDF
            if vendor == "AZFP":
                # AZFP does not record pitch, roll, and heave
                ds = xr.Dataset(
                    {'water_level': ([], platform_dict['water_level'],
                                     {'long_name': 'z-axis distance from the platform coordinate system '
                                     'origin to the sonar transducer',
                                                   'units': 'm'})},
                    coords={'time': 0},
                    attrs={'platform_code_ICES': '',
                           'platform_name': platform_dict['platform_name'],
                           'platform_type': platform_dict['platform_type']})
            else:
                # Convert np.datetime64 numbers to seconds since 1900-01-01
                # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
                ping_time = (platform_dict['time'] - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')

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
                     'water_level': ([], platform_dict['water_level'],
                                     {'long_name': 'z-axis distance from the platform coordinate system '
                                                   'origin to the sonar transducer',
                                      'units': 'm'})
                     },
                    coords={'ping_time': (['ping_time'], ping_time,
                                          {'axis': 'T',
                                           'calendar': 'gregorian',
                                           'long_name': 'Timestamps for position data',
                                           'standard_name': 'time',
                                           'units': 'seconds since 1900-01-01'})},
                    attrs={'platform_code_ICES': '',
                           'platform_name': platform_dict['platform_name'],
                           'platform_type': platform_dict['platform_type']})
            # save to file
            ds.to_netcdf(path=self.file_path, mode="a", group="Platform")

    def set_beam(self, beam_dict, bm_width, bm_dir, tx_pos, tx_sig, out=None, vendor="EK60"):
        """Set the Beam group in the nc file.

        Parameters
        ----------
        beam_dict
            dictionary containing general beam parameters
        bm_width
            dictionary containing parameters related to beamwidth
        bm_dir
            dictionary containing parameters related to beam direction
        tx_pos
            dictionary containing parameters related to transducer position
        tx_sig
            dictionary containing parameters related to transmit signals
        out
            dataset containing the beam group used for AZFP
        vendor
            specifies type of echosounder
        """

        # Only save beam group if file_path exists
        if not os.path.exists(self.file_path):
            print('netCDF file does not exist, exiting without saving Beam group...')
        else:
            # Create an xarray dataset and save to netCDF
            if vendor == "AZFP":
                ds = out
            else:
                # Convert np.datetime64 numbers to seconds since 1900-01-01
                # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
                ping_time = (beam_dict['ping_time'] - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')

                ds = xr.Dataset(
                    {'backscatter_r': (['frequency', 'ping_time', 'range_bin'], beam_dict['backscatter_r'],
                                       {'long_name': 'Backscatter power',
                                        'units': 'dB'}),
                     'beamwidth_receive_major': (['frequency'], bm_width['beamwidth_receive_major'],
                                                 {'long_name': 'Half power one-way receive beam width along '
                                                               'major (horizontal) axis of beam',
                                                  'units': 'arc_degree',
                                                  'valid_range': (0.0, 360.0)}),
                     'beamwidth_receive_minor': (['frequency'], bm_width['beamwidth_receive_minor'],
                                                 {'long_name': 'Half power one-way receive beam width along '
                                                               'minor (vertical) axis of beam',
                                                  'units': 'arc_degree',
                                                  'valid_range': (0.0, 360.0)}),
                     'beamwidth_transmit_major': (['frequency'], bm_width['beamwidth_transmit_major'],
                                                  {'long_name': 'Half power one-way transmit beam width along '
                                                                'major (horizontal) axis of beam',
                                                   'units': 'arc_degree',
                                                   'valid_range': (0.0, 360.0)}),
                     'beamwidth_transmit_minor': (['frequency'], bm_width['beamwidth_transmit_minor'],
                                                  {'long_name': 'Half power one-way transmit beam width along '
                                                                'minor (vertical) axis of beam',
                                                   'units': 'arc_degree',
                                                   'valid_range': (0.0, 360.0)}),
                     'beam_direction_x': (['frequency'], bm_dir['beam_direction_x'],
                                          {'long_name': 'x-component of the vector that gives the pointing '
                                                        'direction of the beam, in sonar beam coordinate '
                                                        'system',
                                           'units': '1',
                                           'valid_range': (-1.0, 1.0)}),
                     'beam_direction_y': (['frequency'], bm_dir['beam_direction_x'],
                                          {'long_name': 'y-component of the vector that gives the pointing '
                                                        'direction of the beam, in sonar beam coordinate '
                                                        'system',
                                           'units': '1',
                                           'valid_range': (-1.0, 1.0)}),
                     'beam_direction_z': (['frequency'], bm_dir['beam_direction_x'],
                                          {'long_name': 'z-component of the vector that gives the pointing '
                                                        'direction of the beam, in sonar beam coordinate '
                                                        'system',
                                           'units': '1',
                                           'valid_range': (-1.0, 1.0)}),
                     'equivalent_beam_angle': (['frequency'], beam_dict['equivalent_beam_angle'],
                                               {'long_name': 'Equivalent beam angle',
                                                'units': 'sr',
                                                'valid_range': (0.0, 4 * np.pi)}),
                     'gain_correction': (['frequency'], beam_dict['gain_correction'],
                                         {'long_name': 'Gain correction',
                                          'units': 'dB'}),
                     'non_quantitative_processing': (['frequency'], beam_dict['non_quantitative_processing'],
                                                     {'flag_meanings': 'no_non_quantitative_processing',
                                                      'flag_values': '0',
                                                      'long_name': 'Presence or not of non-quantitative '
                                                                   'processing applied to the backscattering '
                                                                   'data (sonar specific)'}),
                     'sample_interval': (['frequency'], beam_dict['sample_interval'],
                                         {'long_name': 'Interval between recorded raw data samples',
                                          'units': 's',
                                          'valid_min': 0.0}),
                     'sample_time_offset': (['frequency'], beam_dict['sample_time_offset'],
                                            {'long_name': 'Time offset that is subtracted from the timestamp '
                                                          'of each sample',
                                             'units': 's'}),
                     'transmit_bandwidth': (['frequency'], tx_sig['transmit_bandwidth'],
                                            {'long_name': 'Nominal bandwidth of transmitted pulse',
                                             'units': 'Hz',
                                             'valid_min': 0.0}),
                     'transmit_duration_nominal': (['frequency'], tx_sig['transmit_duration_nominal'],
                                                   {'long_name': 'Nominal bandwidth of transmitted pulse',
                                                    'units': 's',
                                                    'valid_min': 0.0}),
                     'transmit_power': (['frequency'], tx_sig['transmit_power'],
                                        {'long_name': 'Nominal transmit power',
                                         'units': 'W',
                                         'valid_min': 0.0}),
                     'transducer_offset_x': (['frequency'], tx_pos['transducer_offset_x'],
                                             {'long_name': 'x-axis distance from the platform coordinate system '
                                                           'origin to the sonar transducer',
                                              'units': 'm'}),
                     'transducer_offset_y': (['frequency'], tx_pos['transducer_offset_y'],
                                             {'long_name': 'y-axis distance from the platform coordinate system '
                                                           'origin to the sonar transducer',
                                              'units': 'm'}),
                     'transducer_offset_z': (['frequency'], tx_pos['transducer_offset_z'],
                                             {'long_name': 'z-axis distance from the platform coordinate system '
                                                           'origin to the sonar transducer',
                                              'units': 'm'}),
                     },
                    coords={'frequency': (['frequency'], beam_dict['frequency'],
                                          {'units': 'Hz',
                                           'valid_min': 0.0}),
                            'ping_time': (['ping_time'], ping_time,
                                          {'axis': 'T',
                                           'calendar': 'gregorian',
                                           'long_name': 'Timestamp of each ping',
                                           'standard_name': 'time',
                                           'units': 'seconds since 1900-01-01'}),
                            'range_bin': (['range_bin'], beam_dict['range_bin'])},
                    attrs={'beam_mode': beam_dict['beam_mode'],
                           'conversion_equation_t': beam_dict['conversion_equation_t']})

                # Below are specific to Simrad EK60 .raw files
                if 'channel_id' in beam_dict:
                    ds['channel_id'] = ('frequency', beam_dict['channel_id'])
                if 'gpt_software_version' in beam_dict:
                    ds['gpt_software_version'] = ('frequency', beam_dict['gpt_software_version'])
                if 'sa_correction' in beam_dict:
                    ds['sa_correction'] = ('frequency', beam_dict['sa_correction'])

            # save to file
            ds.to_netcdf(path=self.file_path, mode="a", group="Beam")

    def set_vendor_specific(self, vendor_dict, vendor):
        if vendor == "AZFP":
            ds = xr.Dataset(
                {
                    'profile_flag': (['ping_time'], vendor_dict['profile_flag']),
                    'profile_number': (['ping_time'], vendor_dict['profile_number']),
                    'ping_status': (['ping_time'], vendor_dict['ping_status']),
                    'burst_interval': (['ping_time'], vendor_dict['burst_interval']),
                    'digitization_rate': (['ping_time', 'frequency'], vendor_dict['digitization_rate']),
                    'lock_out_index': (['ping_time', 'frequency'], vendor_dict['lock_out_index']),
                    'number_of_bins_per_channel': (['ping_time', 'frequency'], vendor_dict['num_bins']),
                    'number_of_samples_per_average_bin': (['ping_time', 'frequency'], vendor_dict['range_samples']),
                    'ping_per_profile': (['ping_time'], vendor_dict['ping_per_profile']),
                    'average_pings_flag': (['ping_time'], vendor_dict['average_pings_flag']),
                    'number_of_acquired_pings': (['ping_time'], vendor_dict['number_of_acquired_pings']),
                    'ping_period': (['ping_time'], vendor_dict['ping_period']),
                    'first_ping': (['ping_time'], vendor_dict['first_ping']),
                    'last_ping': (['ping_time'], vendor_dict['last_ping']),
                    'data_type': (['ping_time', 'frequency'], vendor_dict['data_type']),
                    'data_error': (['ping_time'], vendor_dict['data_error']),
                    'phase': (['ping_time'], vendor_dict['phase']),
                    'number_of_channels': (['ping_time'], vendor_dict['number_of_channels']),
                    'spare_channel': (['ping_time'], vendor_dict['spare_channel']),
                    'board_number': (['ping_time', 'frequency'], vendor_dict['board_number']),
                    'sensor_flag': (['ping_time'], vendor_dict['sensor_flag']),
                    'ancillary': (['ping_time', 'ancillary_len'], vendor_dict['ancillary']),
                    'ad_channels': (['ping_time', 'ad_len'], vendor_dict['ad_channels'])
                },
                coords={'frequency': (['frequency'], vendor_dict['frequency']),
                        'ping_time': (['ping_time'], vendor_dict['ping_time']),
                        'ancillary_len': (['ancillary_len'], vendor_dict['ancillary_len']),
                        'ad_len': (['ad_len'], vendor_dict['ad_len'])}
            )

            ds.to_netcdf(path=self.file_path, mode="a", group="Vendor")

        else:
            pass
