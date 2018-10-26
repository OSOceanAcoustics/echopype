from __future__ import absolute_import, division, print_function
import os
import numpy as np
import pandas as pd
import netCDF4
import xarray as xr

# TODO: need to figure out how to code the time properly in nc files,
# TODO: the SONAR-netCDF4 convention uses "nanoseconds since .. doesn't seem to be allowed under CF convention"


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
        # Create dimensions
        ping_time_dim = ('ping_time', beam_dict['ping_time'])
        freq_dim = ('frequency', beam_dict['frequency'])
        range_bin_dim = ('range_bin', beam_dict['range_bin'])

        # Create an xarray dataset and save to netCDF
        ds = xr.Dataset({'backscatter_r': (['frequency', 'ping_time', 'range_bin'], beam_dict['backscatter_r'],
                                           {'long_name': 'Backscatter power',
                                            'units': 'dB'}),
                         'beamwidth_receive_major': (['frequency'], beam_dict['beamwidth_receive_major'],
                                                     {'long_name': 'Half power one-way receive beam width along major '
                                                                   '(horizontal) axis of beam',
                                                      'units': 'arc_degree',
                                                      'valid_range': (0.0, 360.0)}),
                         'beamwidth_receive_minor': (['frequency'], beam_dict['beamwidth_receive_minor'],
                                                     {'long_name': 'Half power one-way receive beam width along minor '
                                                                   '(vertical) axis of beam',
                                                      'units': 'arc_degree',
                                                      'valid_range': (0.0, 360.0)}),
                         'beamwidth_transmit_major': (['frequency'], beam_dict['beamwidth_transmit_major'],
                                                      {'long_name': 'Half power one-way transmit beam width along major '
                                                                    '(horizontal) axis of beam',
                                                       'units': 'arc_degree',
                                                       'valid_range': (0.0, 360.0)}),
                         'beamwidth_transmit_minor': (['frequency'], beam_dict['beamwidth_transmit_minor'],
                                                      {'long_name': 'Half power one-way transmit beam width along minor '
                                                                   '(vertical) axis of beam',
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
                                             'valid_min': 0.0})
                         },
                        coords={'frequency': freq_dim,
                                'ping_time': ping_time_dim,
                                'range_bin': range_bin_dim},
                        attrs={'beam_mode': 'vertical',
                               'conversion_equation_t': 'type_3'})

        # save to file
        ds.to_netcdf(path=nc_path, mode="a", group="Environment")