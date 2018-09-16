from __future__ import absolute_import, division, print_function
import os
import numpy as np
import pandas as pd
import netCDF4
import xarray as xr


# Functions for setting up nc file structure and variables

def set_attrs_toplevel(nc_path, attrs_dict):
    """
    Set attributes in the top-level group
    :param nc_path: netcdf file to set top-level group to
    :param attrs_dict: dictionary containing all attributes
    :return:
    """
    with netCDF4.Dataset(nc_path, "w", format="NETCDF4") as ncfile:
        [ncfile.setncattr(k, v) for k, v in attrs_dict.items()]


def set_group_environment(nc_path, param_dict):
    """
    Set the Environment group in the nc file
    :param nc_path: netcdf file to set the Environment group to
    :param param_dict: dictionary containing environmental parameters
         param_dict['freq']
         param_dict['absorption']
         param_dict['sound_speed']
    :return:
    """
    if 'freq' in param_dict:
        freq_coord = np.array(param_dict['freq'], dtype='float32')
    else:
        print('Sonar frequency not provided, set to default 0.')
        freq_coord = np.array(0.0, dtype='float32')

    if 'absorption' in param_dict:
        abs_val = np.array(param_dict['absorption'], dtype='float32')
    else:
        print('Sound absorption not provided, set to default 0.')
        abs_val = np.array(0.0, dtype='float32')

    if 'sound_speed' in param_dict:
        ss_val = np.array(param_dict['sound_speed'], dtype='float32')
    else:
        ss_val = np.array(1500., dtype='float32')

    da_absorption = xr.DataArray(abs_val,
                                 coords=[freq_coord], dims=['frequency'],
                                 attrs={'long_name': "Indicative acoustic absorption",
                                        'units': "dB/m",
                                        'valid_min': 0.0})
    da_sound_speed = xr.DataArray(ss_val,
                                  coords=[freq_coord], dims=['frequency'],
                                  attrs={'long_name': "Indicative sound speed",
                                         'standard_name': "speed_of_sound_in_sea_water",
                                         'units': "m/s",
                                         'valid_min': 0.0})
    ds = xr.Dataset({'absorption_indicative': da_absorption,
                     'sound_speed_indicative': da_sound_speed},
                    coords={'frequency': (['frequency'], freq_coord)})
    ds.frequency.attrs['long_name'] = "Acoustic frequency"
    ds.frequency.attrs['standard_name'] = "sound_frequency"
    ds.frequency.attrs['units'] = "Hz"
    ds.frequency.attrs['valid_min'] = 0.0

    # save to file
    if os.path.exists(nc_path):
        ds.to_netcdf(path=nc_path, mode="a")
    else:
        print('netCDF file does not exist, exiting without saving Environment group...')


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
    ncfile = netCDF4.Dataset(nc_path, "w", format="NETCDF4")
    pr = ncfile.createGroup("Provenance")

    # dimensions
    pr.createDimension("filenames", None)

    # variables
    pr_src_fnames = pr.createVariable(src_fnames, str, "filenames")
    pr_src_fnames.long_name = "Source filenames"

    # set group attributes
    for k, v in conversion_dict.items():
        pr.setncattr(k, v)

    # close nc file
    ncfile.close()


def set_group_sonar(nc_path, sonar_dict):
    """
    Set the Sonar group in the nc file
    :param nc_path: netcdf file to set the Provenance group to
    :param sonar_dict: dictionary containing sonar parameters
        sonar_dict['sonar_manufacturer']
        sonar_dict['sonar_model']
        sonar_dict['sonar_serial_number']
        sonar_dict['sonar_software_name']
        sonar_dict['sonar_software_version']
        sonar_dict['sonar_type']
    :return:
    """
    # create group
    ncfile = netCDF4.Dataset(nc_path, "w", format="NETCDF4")
    snr = ncfile.createGroup("Sonar")

    # set group attributes
    for k, v in sonar_dict.items():
        snr.setncattr(k, v)

    # close nc file
    ncfile.close()
