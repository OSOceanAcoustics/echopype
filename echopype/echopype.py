from __future__ import absolute_import, division, print_function
import os
import numpy as np
import netCDF4
import xarray as xr


# =============================================================
# Functions for setting up nc file structure and variables
# =============================================================


def set_attrs_toplevel(nc_path, attrs_dict):
    """
    Set attributes in the top-level group
    :param nc_path: netcdf file to set top-level group to
    :param attrs_dict: dictionary containing all attributes
    :return:
    """
    with netCDF4.Dataset(nc_path, "w", format="NETCDF4") as ncfile:
        [ncfile.setncattr(k, v) for k, v in attrs_dict.items()]


def set_group_environment(nc_path, env_dict):
    """
    Set the Environment group in the nc file
    :param nc_path: netcdf file to set the Environment group to
    :param env_dict: dictionary containing environment group params
        env_dict['frequency']
        env_dict['absorption_coeff']
        env_dict['sound_speed']
    :return:
    """
    # Only save environment group if nc_path exists
    if not os.path.exists(nc_path):
        print('netCDF file does not exist, exiting without saving Environment group...')
    else:
        da_absorption = xr.DataArray(env_dict['absorption_coeff'],
                                     coords=[env_dict['frequency']], dims=['frequency'],
                                     attrs={'long_name': "Indicative acoustic absorption",
                                            'units': "dB/m",
                                            'valid_min': 0.0})
        da_sound_speed = xr.DataArray(env_dict['sound_speed'],
                                      coords=[env_dict['frequency']], dims=['frequency'],
                                      attrs={'long_name': "Indicative sound speed",
                                             'standard_name': "speed_of_sound_in_sea_water",
                                             'units': "m/s",
                                             'valid_min': 0.0})
        ds = xr.Dataset({'absorption_indicative': da_absorption,
                         'sound_speed_indicative': da_sound_speed},
                        coords={'frequency': (['frequency'], env_dict['frequency'])})
        ds.frequency.attrs['long_name'] = "Acoustic frequency"
        ds.frequency.attrs['standard_name'] = "sound_frequency"
        ds.frequency.attrs['units'] = "Hz"
        ds.frequency.attrs['valid_min'] = 0.0

        # save to file
        ds.to_netcdf(path=nc_path, mode="a", group="Environment")


def set_group_provenance(nc_path, src_fnames, conversion_dict):
    """
    Set the Provenance group in the nc file
    :param nc_path: netcdf file to set the Provenance group to
    :param src_fnames: source filenames
    :param conversion_dict: dictionary containing file conversion parameters
        conversion_dict['conversion_software_name']
        conversion_dict['conversion_software_version']
        conversion_dict['conversion_time']
    :return:
    """
    # create group
    nc_file = netCDF4.Dataset(nc_path, "a", format="NETCDF4")
    pr = nc_file.createGroup("Provenance")

    # dimensions
    pr.createDimension("filenames", None)

    # variables
    pr_src_fnames = pr.createVariable(src_fnames, str, "filenames")
    pr_src_fnames.long_name = "Source filenames"

    # set group attributes
    for k, v in conversion_dict.items():
        pr.setncattr(k, v)

    # close nc file
    nc_file.close()


def set_group_sonar(nc_path, sonar_dict):
    """
    Set the Sonar group in the nc file
    :param nc_path: netcdf file to set the Sonar group to
    :param sonar_dict: dictionary containing sonar parameters
    :return:
    """
    # create group
    ncfile = netCDF4.Dataset(nc_path, "a", format="NETCDF4")
    snr = ncfile.createGroup("Sonar")

    # set group attributes
    for k, v in sonar_dict.items():
        snr.setncattr(k, v)

    # close nc file
    ncfile.close()


def set_group_beam(nc_path, beam_dict):
    """
    Set the Beam group in the nc file
    :param nc_path: netcdf file to set the Beam group to
    :param beam_dict: dictionary containing beam parameters
    :return:
    """
    # Only save environment group if nc_path exists
    if not os.path.exists(nc_path):
        print('netCDF file does not exist, exiting without saving Beam group...')
    else:
        # Create an xarray dataset and save to netCDF
        ds = xr.Dataset({'backscatter_r': (['frequency', 'ping_time', 'range_bin'], beam_dict['backscatter_r'],
                                           {'long_name': 'Backscatter power',
                                            'units': 'dB'}),
                         'beamwidth_receive_major': (['frequency'], beam_dict['beamwidth_receive_major'],
                                                     {'long_name': 'Half power one-way receive beam width along '
                                                                   'major (horizontal) axis of beam',
                                                      'units': 'arc_degree',
                                                      'valid_range': (0.0, 360.0)}),
                         'beamwidth_receive_minor': (['frequency'], beam_dict['beamwidth_receive_minor'],
                                                     {'long_name': 'Half power one-way receive beam width along '
                                                                   'minor (vertical) axis of beam',
                                                      'units': 'arc_degree',
                                                      'valid_range': (0.0, 360.0)}),
                         'beamwidth_transmit_major': (['frequency'], beam_dict['beamwidth_transmit_major'],
                                                      {'long_name': 'Half power one-way transmit beam width along '
                                                                    'major (horizontal) axis of beam',
                                                       'units': 'arc_degree',
                                                       'valid_range': (0.0, 360.0)}),
                         'beamwidth_transmit_minor': (['frequency'], beam_dict['beamwidth_transmit_minor'],
                                                      {'long_name': 'Half power one-way transmit beam width along '
                                                                    'minor (vertical) axis of beam',
                                                       'units': 'arc_degree',
                                                       'valid_range': (0.0, 360.0)}),
                         'beam_direction_x': (['frequency'], beam_dict['beamwidth_transmit_minor'],
                                              {'long_name': 'x-component of the vector that gives the pointing '
                                                            'direction of the beam, in sonar beam coordinate '
                                                            'system',
                                               'units': '1',
                                               'valid_range': (-1.0, 1.0)}),
                         'beam_direction_y': (['frequency'], beam_dict['beamwidth_transmit_minor'],
                                              {'long_name': 'y-component of the vector that gives the pointing '
                                                            'direction of the beam, in sonar beam coordinate '
                                                            'system',
                                               'units': '1',
                                               'valid_range': (-1.0, 1.0)}),
                         'beam_direction_z': (['frequency'], beam_dict['beamwidth_transmit_minor'],
                                              {'long_name': 'z-component of the vector that gives the pointing '
                                                            'direction of the beam, in sonar beam coordinate '
                                                            'system',
                                               'units': '1',
                                               'valid_range': (-1.0, 1.0)}),
                         'equivalent_beam_angle': (['frequency'], beam_dict['equivalent_beam_angle'],
                                                   {'long_name': 'Equivalent beam angle',
                                                    'units': 'sr',
                                                    'valid_range': (0.0, 4*np.pi)}),
                         'gain_correction': (['frequency'], beam_dict['gain_correction'],
                                             {'long_name': 'Gain correction',
                                              'units': 'dB'}),
                         'non_quantitative_processing': (['frequency'], beam_dict['non_quantitative_processing'],
                                                         {'flag_meanings': 'no_non_quantitative_processing',
                                                          'flag_values': '0',
                                                          'long_name': 'Presence or not of non-quantitative '
                                                                       'processing applied to the backscattering '
                                                                       'data (sonar specific)'}),
                         'sample_interval': (['frequency', 'ping_time'], beam_dict['sample_interval'],
                                             {'long_name': 'Interval between recorded raw data samples',
                                              'units': 's',
                                              'valid_min': 0.0}),
                         'sample_time_offset': (['frequency'], beam_dict['sample_time_offset'],
                                                {'long_name': 'Time offset that is subtracted from the timestamp '
                                                              'of each sample',
                                                'units': 's'}),
                         'transmit_bandwidth': (['frequency', 'ping_time'], beam_dict['transmit_bandwidth'],
                                                {'long_name': 'Nominal bandwidth of transmitted pulse',
                                                 'units': 'Hz',
                                                 'valid_min': 0.0}),
                         'transmit_duration_nominal': (['frequency', 'ping_time'],
                                                       beam_dict['transmit_duration_nominal'],
                                                       {'long_name': 'Nominal bandwidth of transmitted pulse',
                                                        'units': 's',
                                                        'valid_min': 0.0}),
                         'transmit_power': (['frequency', 'ping_time'], beam_dict['transmit_power'],
                                            {'long_name': 'Nominal transmit power',
                                             'units': 'W',
                                             'valid_min': 0.0}),
                         'transducer_offset_x': (['frequency'], beam_dict['transducer_offset_x'],
                                                 {'long_name': 'x-axis distance from the platform coordinate system '
                                                               'origin to the sonar transducer',
                                                  'units': 'm'}),
                         'transducer_offset_y': (['frequency'], beam_dict['transducer_offset_y'],
                                                 {'long_name': 'y-axis distance from the platform coordinate system '
                                                               'origin to the sonar transducer',
                                                  'units': 'm'}),
                         'transducer_offset_z': (['frequency'], beam_dict['transducer_offset_z'],
                                                 {'long_name': 'z-axis distance from the platform coordinate system '
                                                               'origin to the sonar transducer',
                                                  'units': 'm'}),
                         'channel_id': (['frequency'], beam_dict['channel_id']),
                         'gpt_software_version': (['frequency'], beam_dict['gpt_software_version']),
                         'sa_correction': (['frequency'], beam_dict['sa_correction'])
                         },
                        coords={'frequency': (['frequency'], beam_dict['frequency'],
                                              {'units': 'Hz',
                                               'valid_min': 0.0}),
                                'ping_time': (['ping_time'], beam_dict['ping_time'],
                                              {'axis': 'T',
                                               'calendar': 'gregorian',
                                               'long_name': 'Timestamp of each ping',
                                               'standard_name': 'time',
                                               'units': 'seconds since 1900-01-01'}),
                                'range_bin': (['range_bin'], beam_dict['range_bin'])},
                        attrs={'beam_mode': 'vertical',
                               'conversion_equation_t': 'type_3'})
        # save to file
        ds.to_netcdf(path=nc_path, mode="a", group="Beam")


def set_group_platform(nc_path, platform_dict):
    """
    Set the Beam group in the nc file
    :param nc_path: netcdf file to set the Beam group to
    :param platform_dict: dictionary containing beam parameters
    :return:
    """
    # Only save environment group if nc_path exists
    if not os.path.exists(nc_path):
        print('netCDF file does not exist, exiting without saving Platform group...')
    else:
        # Create an xarray dataset and save to netCDF
        ds = xr.Dataset({'pitch': (['time'], platform_dict['pitch'],
                                   {'long_name': 'Platform pitch',
                                    'standard_name': 'platform_pitch_angle',
                                    'units': 'arc_degree',
                                    'valid_range': (-90.0, 90.0)}),
                         'roll': (['time'], platform_dict['roll'],
                                  {'long_name': 'Platform roll',
                                   'standard_name': 'platform_roll_angle',
                                   'units': 'arc_degree',
                                   'valid_range': (-90.0, 90.0)}),
                         'heave': (['time'], platform_dict['heave'],
                                   {'long_name': 'Platform heave',
                                    'standard_name': 'platform_heave_angle',
                                    'units': 'arc_degree',
                                    'valid_range': (-90.0, 90.0)}),
                         'water_level': ([], platform_dict['water_level'],
                                         {'long_name': 'z-axis distance from the platform coordinate system '
                                                       'origin to the sonar transducer',
                                          'units': 'm'})
                         },
                        coords={'time': (['time'], platform_dict['time'],
                                         {'axis': 'T',
                                          'calendar': 'gregorian',
                                          'long_name': 'Timestamps for position data',
                                          'standard_name': 'time',
                                          'units': 'seconds since 1900-01-01'})},
                        attrs={'platform_code_ICES': '',
                               'platform_name': platform_dict['platform_name'],
                               'platform_type': platform_dict['platform_type']})
        # save to file
        ds.to_netcdf(path=nc_path, mode="a", group="Platform")