"""
Class to save unpacked echosounder data to appropriate groups in netcdf or zarr.
"""
import os
import shutil
from datetime import datetime as dt
from collections import defaultdict
import xarray as xr
import numpy as np
import re
import pynmea2
import zarr
import netCDF4
from ..._version import get_versions
ECHOPYPE_VERSION = get_versions()['version']
del get_versions

NETCDF_COMPRESSION_SETTINGS = {'zlib': True, 'complevel': 4}
ZARR_COMPRESSION_SETTINGS = {'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)}


class SetGroupsBase:
    """Base class for saving groups to netcdf or zarr from echosounder data files.
    """
    def __init__(self, convert_obj, input_file, output_path,
                 save_ext='.nc', compress=True, overwrite=True, params=None, extra_files=None):
        self.convert_obj = convert_obj   # a convert object ConvertEK60/ConvertAZFP/etc...
        self.input_file = input_file
        self.output_path = output_path
        self.save_ext = save_ext
        self.compress = compress
        self.overwrite = overwrite
        self.ui_param = params
        self.extra_files = extra_files

    def save(self):
        """Actually save groups to file by calling the set methods.
        """

    def set_toplevel(self, sonar_model):
        """Set the top-level group.
        """
        # Collect variables
        # TODO Use ping time
        timestamp_pattern = re.compile(self.convert_obj.timestamp_pattern)
        raw_date_time = timestamp_pattern.match(os.path.basename(self.input_file))
        filedate = raw_date_time['date']
        filetime = raw_date_time['time']
        date_created = dt.strptime(filedate + '-' + filetime, '%Y%m%d-%H%M%S').isoformat() + 'Z'

        tl_dict = {'conventions': 'CF-1.7, SONAR-netCDF4-1.0, ACDD-1.3',
                   'keywords': sonar_model,
                   'sonar_convention_authority': 'ICES',
                   'sonar_convention_name': 'SONAR-netCDF4',
                   'sonar_convention_version': '1.0',
                   'summary': '',
                   'title': '',
                   'date_created': date_created,
                   'survey_name': self.ui_param['survey_name']}
        # Add any extra user defined values
        for k, v in list(self.ui_param.items())[5:]:
            tl_dict[k] = v

        # Save
        if self.save_ext == '.nc':
            with netCDF4.Dataset(self.output_path, "w", format="NETCDF4") as ncfile:
                [ncfile.setncattr(k, v) for k, v in tl_dict.items()]
        elif self.save_ext == '.zarr':
            zarrfile = zarr.open(self.output_path, mode="w")
            for k, v in tl_dict.items():
                zarrfile.attrs[k] = v
        else:
            raise ValueError("Unsupported file format")

    def set_provenance(self):
        """Set the Provenance group.
        """
        # Collect variables
        prov_dict = {'conversion_software_name': 'echopype',
                     'conversion_software_version': ECHOPYPE_VERSION,
                     'conversion_time': dt.utcnow().isoformat(timespec='seconds') + 'Z',    # use UTC time
                     'src_filenames': self.input_file}
        # Save
        if self.save_ext == '.nc':
            with netCDF4.Dataset(self.output_path, "a", format="NETCDF4") as ncfile:
                prov = ncfile.createGroup("Provenance")
                [prov.setncattr(k, v) for k, v in prov_dict.items()]
        elif self.save_ext == '.zarr':
            zarr_file = zarr.open(self.output_path, mode="a")
            prov = zarr_file.create_group('Provenance')
            for k, v in prov_dict.items():
                prov.attrs[k] = v
        else:
            raise ValueError("Unsupported file format")

    def set_sonar(self, sonar_vals):
        """Set the Sonar group.
        """
        # Collect variables
        sonar_dict = dict(zip(('sonar_manufacturer', 'sonar_model', 'sonar_serial_number',
                               'sonar_software_name', 'sonar_software_version', 'sonar_type'), sonar_vals))

        # Save variables
        if self.save_ext == '.nc':
            with netCDF4.Dataset(self.output_path, "a", format="NETCDF4") as ncfile:
                snr = ncfile.createGroup("Sonar")
                # set group attributes
                [snr.setncattr(k, v) for k, v in sonar_dict.items()]

        elif self.save_ext == '.zarr':
            zarrfile = zarr.open(self.output_path, mode='a')
            snr = zarrfile.create_group('Sonar')

            for k, v in sonar_dict.items():
                snr.attrs[k] = v

    def set_nmea(self):
        """Set the Platform/NMEA group.
        """

        if not os.path.exists(self.output_path):
            print('netCDF file does not exist, exiting without saving Platform group...')
        else:
            # Convert np.datetime64 numbers to seconds since 1900-01-01
            # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
            time = (self.convert_obj.nmea_data.nmea_times -
                    np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')
            ds = xr.Dataset(
                {'NMEA_datagram': (['time'], self.convert_obj.nmea_data.raw_datagrams,
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
            if self.save_ext == '.nc':
                nc_encoding = {'time': NETCDF_COMPRESSION_SETTINGS} if self.compress else {}
                ds.to_netcdf(path=self.output_path, mode='a', group='Platform/NMEA', encoding=nc_encoding)
            elif self.save_ext == '.zarr':
                zarr_encoding = {'time': ZARR_COMPRESSION_SETTINGS} if self.compress else {}
                ds.to_zarr(store=self.output_path, mode='a', group='Platform/NMEA', encoding=zarr_encoding)


class SetGroupsEK60(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from EK60 data files.
    """
    def save(self):
        """Actually save groups to file by calling the set methods.
        """
        # filename must have timestamp that matches self.timestamp_pattern
        sonar_values = ('Simrad', self.convert_obj.config_datagram['sounder_name'],
                        '', '', self.convert_obj.config_datagram['version'], 'echosounder')
        self.set_toplevel('EK60')
        self.set_provenance()           # provenance group
        self.set_sonar(sonar_values)    # sonar group

        if 'ENV' in self.convert_obj.data_type:
            self.set_env()           # environment group
            return
        elif 'GPS' in self.convert_obj.data_type:
            self.set_platform()
            return

        self.set_env()              # environment group
        self.set_beam()             # beam group
        self.set_platform()         # platform group
        self.set_nmea()             # platform/NMEA group

    def set_env(self):
        """Set the Environment group.
        """

        # Only save environment group if file_path exists
        if not os.path.exists(self.output_path):
            print('netCDF file does not exist, exiting without saving Environment group...')
        else:
            # Collect variables
            config = self.convert_obj.config_datagram
            ping_data = self.convert_obj.ping_data_dict
            freq = np.array([config['transceivers'][x]['frequency']
                            for x in config['transceivers'].keys()], dtype='float32')
            # Extract absorption and sound speed depending on if the values are identical for all pings
            abs_tmp = np.unique(ping_data[1]['absorption_coefficient']).size
            ss_tmp = np.unique(ping_data[1]['sound_velocity']).size
            # --- if identical for all pings, save only values from the first ping
            if np.all(np.array([abs_tmp, ss_tmp]) == 1):
                abs_val = np.array([ping_data[x]['absorption_coefficient'][0]
                                    for x in config['transceivers'].keys()], dtype='float32')
                ss_val = np.array([ping_data[x]['sound_velocity'][0]
                                  for x in config['transceivers'].keys()], dtype='float32')
            # --- if NOT identical for all pings, save as array of dimension [frequency x ping_time]
            else:  # TODO: right now set_groups_ek60/set_env doens't deal with this case, need to add
                abs_val = np.array([ping_data[x]['absorption_coefficient']
                                    for x in config['transceivers'].keys()],
                                   dtype='float32')
                ss_val = np.array([ping_data[x]['sound_velocity']
                                  for x in config['transceivers'].keys()],
                                  dtype='float32')
            # Assemble variables into a dataset
            absorption = xr.DataArray(abs_val,
                                      coords=[freq], dims={'frequency'},
                                      attrs={'long_name': "Indicative acoustic absorption",
                                             'units': "dB/m",
                                             'valid_min': 0.0})
            sound_speed = xr.DataArray(ss_val,
                                       coords=[freq], dims={'frequency'},
                                       attrs={'long_name': "Indicative sound speed",
                                              'standard_name': "speed_of_sound_in_sea_water",
                                              'units': "m/s",
                                              'valid_min': 0.0})
            ds = xr.Dataset({'absorption_indicative': absorption,
                             'sound_speed_indicative': sound_speed},
                            coords={'frequency': (['frequency'], freq)})

            ds.frequency.attrs['long_name'] = "Acoustic frequency"
            ds.frequency.attrs['standard_name'] = "sound_frequency"
            ds.frequency.attrs['units'] = "Hz"
            ds.frequency.attrs['valid_min'] = 0.0

            # save to file
            if self.save_ext == '.nc':
                ds.to_netcdf(path=self.output_path, mode='a', group='Environment')
            elif self.save_ext == '.zarr':
                # Only save environment group if not appending to an existing .zarr file
                ds.to_zarr(store=self.output_path, mode='a', group='Environment')

    def set_platform(self):
        """Set the Platform group.
        """

        if not os.path.exists(self.output_path):
            print('netCDF file does not exist, exiting without saving Platform group...')
        else:
            # Collect variables
            # Read lat/long from NMEA datagram
            idx_loc = np.argwhere(np.isin(self.convert_obj.nmea_data.messages, ['GGA', 'GLL', 'RMC'])).squeeze()
            nmea_msg = []
            [nmea_msg.append(pynmea2.parse(self.convert_obj.nmea_data.raw_datagrams[x])) for x in idx_loc]
            lat = np.array([x.latitude for x in nmea_msg]) if nmea_msg else [np.nan]
            lon = np.array([x.longitude for x in nmea_msg]) if nmea_msg else [np.nan]
            location_time = (self.convert_obj.nmea_data.nmea_times[idx_loc] -
                             np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's') if nmea_msg else [np.nan]

            # Convert np.datetime64 numbers to seconds since 1900-01-01
            # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
            ping_time = (self.convert_obj.ping_time -
                         np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')

            # Assemble variables into a dataset
            ds = xr.Dataset(
                {'pitch': (['ping_time'], np.array(self.convert_obj.ping_data_dict[1]['pitch'], dtype='float32'),
                           {'long_name': 'Platform pitch',
                            'standard_name': 'platform_pitch_angle',
                            'units': 'arc_degree',
                            'valid_range': (-90.0, 90.0)}),
                 'roll': (['ping_time'], np.array(self.convert_obj.ping_data_dict[1]['roll'], dtype='float32'),
                          {'long_name': 'Platform roll',
                           'standard_name': 'platform_roll_angle',
                           'units': 'arc_degree',
                           'valid_range': (-90.0, 90.0)}),
                 'heave': (['ping_time'], np.array(self.convert_obj.ping_data_dict[1]['heave'], dtype='float32'),
                           {'long_name': 'Platform heave',
                            'standard_name': 'platform_heave_angle',
                            'units': 'arc_degree',
                            'valid_range': (-90.0, 90.0)}),
                 # Get user defined water level (0 if undefined)
                 'water_level': ([], self.ui_param['water_level'],
                                 {'long_name': 'z-axis distance from the platform coordinate system '
                                               'origin to the sonar transducer',
                                  'units': 'm'})
                 },
                coords={'ping_time': (['ping_time'], ping_time,
                                      {'axis': 'T',
                                       'calendar': 'gregorian',
                                       'long_name': 'Timestamps for position datagrams',
                                       'standard_name': 'time',
                                       'units': 'seconds since 1900-01-01'})
                        },
                attrs={'platform_code_ICES': self.ui_param['platform_code_ICES'],
                       'platform_name': self.ui_param['platform_name'],
                       'platform_type': self.ui_param['platform_type']})
            if len(location_time) > 0:
                ds_loc = xr.Dataset(
                    {'latitude': (['location_time'], lat,
                                  {'long_name': 'Platform latitude',
                                   'standard_name': 'latitude',
                                   'units': 'degrees_north',
                                   'valid_range': (-90.0, 90.0)}),
                     'longitude': (['location_time'], lon,
                                   {'long_name': 'Platform longitude',
                                    'standard_name': 'longitude',
                                    'units': 'degrees_east',
                                    'valid_range': (-180.0, 180.0)})},
                    coords={'location_time': (['location_time'], location_time,
                                              {'axis': 'T',
                                               'calendar': 'gregorian',
                                               'long_name': 'Timestamps for NMEA position datagrams',
                                               'standard_name': 'time',
                                               'units': 'seconds since 1900-01-01'})})
                ds = xr.merge([ds, ds_loc])

            # save dataset to file with specified compression settings for all variables
            if self.save_ext == '.nc':
                nc_encoding = {var: NETCDF_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
                ds.to_netcdf(path=self.output_path, mode='a', group='Platform', encoding=nc_encoding)
            elif self.save_ext == '.zarr':
                zarr_encoding = {var: ZARR_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
                ds.to_zarr(store=self.output_path, mode='w', group='Platform', encoding=zarr_encoding)

    def set_beam(self):
        """Set the Beam group.
        """

        if not os.path.exists(self.output_path):
            print('netCDF file does not exist, exiting without saving Beam group...')
        else:
            # Collect variables
            # Convert np.datetime64 numbers to seconds since 1900-01-01
            # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
            ping_time = (self.convert_obj.ping_time - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')
            config = self.convert_obj.config_datagram

            # Additional coordinate variables added by echopype for storing data as a cube with
            # dimensions [frequency x ping_time x range_bin]
            freq = np.array([config['transceivers'][x]['frequency']
                            for x in config['transceivers'].keys()], dtype='float32')
            range_bin = np.arange(self.convert_obj.power_dict.shape[2])

            # Loop through each transducer for channel-specific variables
            param_numerical = {"beamwidth_receive_major": "beamwidth_alongship",
                               "beamwidth_receive_minor": "beamwidth_athwartship",
                               "beamwidth_transmit_major": "beamwidth_alongship",
                               "beamwidth_transmit_minor": "beamwidth_athwartship",
                               "beam_direction_x": "dir_x",
                               "beam_direction_y": "dir_y",
                               "beam_direction_z": "dir_z",
                               "angle_offset_alongship": "angle_offset_alongship",
                               "angle_offset_athwartship": "angle_offset_athwartship",
                               "angle_sensitivity_alongship": "angle_sensitivity_alongship",
                               "angle_sensitivity_athwartship": "angle_sensitivity_athwartship",
                               "transducer_offset_x": "pos_x",
                               "transducer_offset_y": "pos_y",
                               "transducer_offset_z": "pos_z",
                               "equivalent_beam_angle": "equivalent_beam_angle",
                               "gain_correction": "gain"}
            param_str = {"gpt_software_version": "gpt_software_version",
                         "channel_id": "channel_id",
                         "beam_type": "beam_type"}

            beam_dict = dict()
            for encode_name, origin_name in param_numerical.items():
                beam_dict[encode_name] = np.array(
                    [val[origin_name] for key, val in config['transceivers'].items()]).astype('float32')
            beam_dict['transducer_offset_z'] += [self.convert_obj.ping_data_dict[x]['transducer_depth'][0]
                                                 for x in config['transceivers'].keys()]

            for encode_name, origin_name in param_str.items():
                beam_dict[encode_name] = [val[origin_name]
                                          for key, val in config['transceivers'].items()]

            param_name = ['pulse_length', 'transmit_power', 'bandwidth', 'sample_interval']
            param_name_save = ['transmit_duration_nominal', 'transmit_power', 'transmit_bandwidth', 'sample_interval']
            beam_dict['transmit_signal'] = {k: np.array([self.convert_obj.ping_data_dict[x][v]
                                                        for x in config['transceivers'].keys()])
                                            for k, v in zip(param_name_save, param_name)}

            if len(config['transceivers']) == 1:   # only 1 channel
                idx = np.argwhere(np.isclose(beam_dict['transmit_signal']['transmit_duration_nominal'][:, 0],
                                             config['transceivers'][1]['pulse_length_table'])).squeeze()
                idx = np.expand_dims(np.array(idx), axis=0)
            else:
                idx = [np.argwhere(np.isclose(beam_dict['transmit_signal']['transmit_duration_nominal'][:, 0][key - 1],
                                              val['pulse_length_table'])).squeeze()
                       for key, val in config['transceivers'].items()]
            beam_dict['sa_correction'] = \
                np.array([x['sa_correction_table'][y]
                         for x, y in zip(config['transceivers'].values(), np.array(idx))])
            # Assemble variables into a dataset
            ds = xr.Dataset(
                {'backscatter_r': (['frequency', 'ping_time', 'range_bin'], self.convert_obj.power_dict,
                                   {'long_name': 'Backscatter power',
                                    'units': 'dB'}),
                 'angle_athwartship': (['frequency', 'ping_time', 'range_bin'],
                                       self.convert_obj.angle_dict[:, :, :, 0],
                                       {'long_name': 'electrical athwartship angle'}),
                 'angle_alongship': (['frequency', 'ping_time', 'range_bin'], self.convert_obj.angle_dict[:, :, :, 1],
                                     {'long_name': 'electrical alongship angle'}),
                 'beam_type': ('frequency', beam_dict['beam_type'],
                               {'long_name': 'type of transducer (0-single, 1-split)'}),
                 'beamwidth_receive_alongship': (['frequency'], beam_dict['beamwidth_receive_major'],
                                                 {'long_name': 'Half power one-way receive beam width along '
                                                  'alongship axis of beam',
                                                  'units': 'arc_degree',
                                                  'valid_range': (0.0, 360.0)}),
                 'beamwidth_receive_athwartship': (['frequency'], beam_dict['beamwidth_receive_minor'],
                                                   {'long_name': 'Half power one-way receive beam width along '
                                                    'athwartship axis of beam',
                                                    'units': 'arc_degree',
                                                    'valid_range': (0.0, 360.0)}),
                 'beamwidth_transmit_alongship': (['frequency'], beam_dict['beamwidth_transmit_major'],
                                                  {'long_name': 'Half power one-way transmit beam width along '
                                                   'alongship axis of beam',
                                                   'units': 'arc_degree',
                                                   'valid_range': (0.0, 360.0)}),
                 'beamwidth_transmit_athwartship': (['frequency'], beam_dict['beamwidth_transmit_minor'],
                                                    {'long_name': 'Half power one-way transmit beam width along '
                                                        'athwartship axis of beam',
                                                     'units': 'arc_degree',
                                                     'valid_range': (0.0, 360.0)}),
                 'beam_direction_x': (['frequency'], beam_dict['beam_direction_x'],
                                      {'long_name': 'x-component of the vector that gives the pointing '
                                                    'direction of the beam, in sonar beam coordinate '
                                                    'system',
                                                    'units': '1',
                                                    'valid_range': (-1.0, 1.0)}),
                 'beam_direction_y': (['frequency'], beam_dict['beam_direction_x'],
                                      {'long_name': 'y-component of the vector that gives the pointing '
                                                    'direction of the beam, in sonar beam coordinate '
                                                    'system',
                                       'units': '1',
                                       'valid_range': (-1.0, 1.0)}),
                 'beam_direction_z': (['frequency'], beam_dict['beam_direction_x'],
                                      {'long_name': 'z-component of the vector that gives the pointing '
                                                    'direction of the beam, in sonar beam coordinate '
                                                    'system',
                                       'units': '1',
                                       'valid_range': (-1.0, 1.0)}),
                 'angle_offset_alongship': (['frequency'], beam_dict['angle_offset_alongship'],
                                            {'long_name': 'electrical alongship angle of the transducer'}),
                 'angle_offset_athwartship': (['frequency'], beam_dict['angle_offset_athwartship'],
                                              {'long_name': 'electrical athwartship angle of the transducer'}),
                 'angle_sensitivity_alongship': (['frequency'], beam_dict['angle_sensitivity_alongship'],
                                                 {'long_name': 'alongship sensitivity of the transducer'}),
                 'angle_sensitivity_athwartship': (['frequency'], beam_dict['angle_sensitivity_athwartship'],
                                                   {'long_name': 'athwartship sensitivity of the transducer'}),
                 'equivalent_beam_angle': (['frequency'], beam_dict['equivalent_beam_angle'],
                                           {'long_name': 'Equivalent beam angle',
                                            'units': 'sr',
                                            'valid_range': (0.0, 4 * np.pi)}),
                 'gain_correction': (['frequency'], beam_dict['gain_correction'],
                                     {'long_name': 'Gain correction',
                                      'units': 'dB'}),
                 'non_quantitative_processing': (['frequency'], np.array([0, ] * freq.size, dtype='int32'),
                                                 {'flag_meanings': 'no_non_quantitative_processing',
                                                  'flag_values': '0',
                                                  'long_name': 'Presence or not of non-quantitative '
                                                               'processing applied to the backscattering '
                                                               'data (sonar specific)'}),
                 'sample_interval': (['frequency', 'ping_time'], beam_dict['transmit_signal']['sample_interval'],
                                     {'long_name': 'Interval between recorded raw data samples',
                                      'units': 's',
                                      'valid_min': 0.0}),
                 'sample_time_offset': (['frequency'], np.array([2, ] * freq.size, dtype='int32'),
                                        {'long_name': 'Time offset that is subtracted from the timestamp '
                                                      'of each sample',
                                                      'units': 's'}),
                 'transmit_bandwidth': (['frequency', 'ping_time'], beam_dict['transmit_signal']['transmit_bandwidth'],
                                        {'long_name': 'Nominal bandwidth of transmitted pulse',
                                         'units': 'Hz',
                                         'valid_min': 0.0}),
                 'transmit_duration_nominal': (['frequency', 'ping_time'],
                                               beam_dict['transmit_signal']['transmit_duration_nominal'],
                                               {'long_name': 'Nominal bandwidth of transmitted pulse',
                                                             'units': 's',
                                                'valid_min': 0.0}),
                 'transmit_power': (['frequency', 'ping_time'], beam_dict['transmit_signal']['transmit_power'],
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
                                                       'units': 'm'})},
                coords={'frequency': (['frequency'], freq,
                                      {'units': 'Hz',
                                       'valid_min': 0.0}),
                        'ping_time': (['ping_time'], ping_time,
                                      {'axis': 'T',
                                       'calendar': 'gregorian',
                                       'units': 'seconds since 1900-01-01',
                                       'long_name': 'Timestamp of each ping',
                                       'standard_name': 'time'}),
                        'range_bin': range_bin},
                attrs={'beam_mode': 'vertical',
                       'conversion_equation_t': 'type_3'})

            # Below are specific to Simrad EK60 .raw files
            ds['channel_id'] = ('frequency', beam_dict['channel_id'])
            ds['gpt_software_version'] = ('frequency', beam_dict['gpt_software_version'])
            ds['sa_correction'] = ('frequency', beam_dict['sa_correction'])

            # Save dataset with optional compression for all data variables
            if self.save_ext == '.nc':
                nc_encoding = {var: NETCDF_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
                ds.to_netcdf(path=self.output_path, mode='a', group='Beam', encoding=nc_encoding)
            elif self.save_ext == '.zarr':
                zarr_encoding = {var: ZARR_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
                ds = ds.chunk({'range_bin': 25000})
                ds.to_zarr(store=self.output_path, mode='a', group='Beam', encoding=zarr_encoding)


class SetGroupsEK80(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from EK80 data files.
    """
    def save(self):
        """Actually save groups to file by calling the set methods.
        """

        self.set_toplevel('EK80')
        self.set_provenance()    # provenance group

        # Save environment only
        if 'ENV' in self.convert_obj.data_type:
            self.set_env()           # environment group
            return
        elif 'GPS' in self.convert_obj.data_type:
            self.set_platform()
            return

        self.set_env()           # environment group
        self.set_platform()      # platform group
        self.set_nmea()          # platform/NMEA group
        self.set_vendor()        # vendor group
        bb_ch_ids = self.convert_obj.bb_ch_ids
        cw_ch_ids = self.convert_obj.cw_ch_ids
        # If there is both bb and cw data
        if bb_ch_ids and cw_ch_ids:
            new_path = self._copy_file(self.output_path)
            self.set_beam(bb_ch_ids, bb=True, path=self.output_path)
            self.set_sonar(bb_ch_ids, path=self.output_path)
            self.set_beam(cw_ch_ids, bb=False, path=new_path)
            self.set_sonar(cw_ch_ids, path=new_path)
        # If there is only bb data
        elif bb_ch_ids:
            self.set_beam(bb_ch_ids, bb=True, path=self.output_path)
            self.set_sonar(bb_ch_ids, path=self.output_path)
        # If there is only cw data
        else:
            self.set_beam(cw_ch_ids, bb=False, path=self.output_path)
            self.set_sonar(cw_ch_ids, path=self.output_path)

    def set_env(self):
        """Set the Environment group.
        """
        # Collect variables
        env_dict = {'temperature': self.convert_obj.environment['temperature'],
                    'depth': self.convert_obj.environment['depth'],
                    'acidity': self.convert_obj.environment['acidity'],
                    'salinity': self.convert_obj.environment['salinity'],
                    'sound_speed_indicative': self.convert_obj.environment['sound_speed']}

        # Save to file
        if self.save_ext == '.nc':
            with netCDF4.Dataset(self.output_path, "a", format="NETCDF4") as ncfile:
                env = ncfile.createGroup("Environment")
                # set group attributes
                [env.setncattr(k, v) for k, v in env_dict.items()]

        elif self.save_ext == '.zarr':
            zarrfile = zarr.open(self.output_path, mode='a')
            env = zarrfile.create_group('Environment')

            for k, v in env_dict.items():
                env.attrs[k] = v

    def set_platform(self):
        """Set the Platform group.
        """
        # TODO: let's add ConvertEK80.environment['drop_keel_offset'] to
        #  the Platform group even though it is not in the convention.

        if not os.path.exists(self.output_path):
            print('netCDF file does not exist, exiting without saving Platform group...')
        else:
            # Collect variables
            if self.ui_param['water_level'] is not None:
                water_level = self.ui_param['water_level']
            elif 'water_level_draft' in self.convert_obj.environment:
                water_level = self.convert_obj.environment['water_level_draft']
            else:
                water_level = None
                print('WARNING: The water_level_draft was not in the file. Value '
                      'set to None')

            idx_loc = np.argwhere(np.isin(self.convert_obj.nmea_data.messages, ['GGA', 'GLL', 'RMC'])).squeeze()
            nmea_msg = []
            [nmea_msg.append(pynmea2.parse(self.convert_obj.nmea_data.raw_datagrams[x])) for x in idx_loc]
            # Convert np.datetime64 numbers to seconds since 1900-01-01
            # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
            mru_time = (np.array(self.convert_obj.mru.get('timestamp', [np.nan])) -
                        np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's') if nmea_msg else [np.nan]
            location_time = (self.convert_obj.nmea_data.nmea_times[idx_loc] -
                             np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's') if nmea_msg else [np.nan]

            # Assemble variables into a dataset
            ds = xr.Dataset(
                {'pitch': (['mru_time'], np.array(self.convert_obj.mru.get('pitch', [np.nan])),
                           {'long_name': 'Platform pitch',
                            'standard_name': 'platform_pitch_angle',
                            'units': 'arc_degree',
                            'valid_range': (-90.0, 90.0)}),
                 'roll': (['mru_time'], np.array(self.convert_obj.mru.get('roll', [np.nan])),
                          {'long_name': 'Platform roll',
                           'standard_name': 'platform_roll_angle',
                           'units': 'arc_degree',
                           'valid_range': (-90.0, 90.0)}),
                 'heave': (['mru_time'], np.array(self.convert_obj.mru.get('heave', [np.nan])),
                           {'long_name': 'Platform heave',
                            'standard_name': 'platform_heave_angle',
                            'units': 'arc_degree',
                            'valid_range': (-90.0, 90.0)}),
                 'latitude': (['location_time'], np.array([x.latitude for x in nmea_msg]) if nmea_msg else [np.nan],
                              {'long_name': 'Platform latitude',
                               'standard_name': 'latitude',
                               'units': 'degrees_north',
                               'valid_range': (-90.0, 90.0)}),
                 'longitude': (['location_time'], np.array([x.longitude for x in nmea_msg]) if nmea_msg else [np.nan],
                               {'long_name': 'Platform longitude',
                                'standard_name': 'longitude',
                                'units': 'degrees_east',
                                'valid_range': (-180.0, 180.0)}),
                 'water_level': ([], water_level,
                                 {'long_name': 'z-axis distance from the platform coordinate system '
                                               'origin to the sonar transducer',
                                  'units': 'm'})
                 },
                coords={'mru_time': (['mru_time'], mru_time,
                                     {'axis': 'T',
                                      'calendar': 'gregorian',
                                      'long_name': 'Timestamps for MRU datagrams',
                                      'standard_name': 'time',
                                      'units': 'seconds since 1900-01-01'}),
                        'location_time': (['location_time'], location_time,
                                          {'axis': 'T',
                                           'calendar': 'gregorian',
                                           'long_name': 'Timestamps for NMEA datagrams',
                                           'standard_name': 'time',
                                           'units': 'seconds since 1900-01-01'})
                        },
                attrs={'platform_code_ICES': self.ui_param['platform_code_ICES'],
                       'platform_name': self.ui_param['platform_name'],
                       'platform_type': self.ui_param['platform_type']})

            # Save to file
            if self.save_ext == '.nc':
                nc_encoding = {var: NETCDF_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
                ds.to_netcdf(path=self.output_path, mode='a', group='Platform', encoding=nc_encoding)
            elif self.save_ext == '.zarr':
                zarr_encoding = {var: ZARR_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
                ds.to_zarr(store=self.output_path, mode='a', group='Platform', encoding=zarr_encoding)

    def set_beam(self, ch_ids, bb, path):
        """Set the Beam group.
        """
        if not os.path.exists(self.output_path):
            print('netCDF file does not exist, exiting without saving Beam group...')
        else:
            config = self.convert_obj.config_datagram['configuration']
            beam_dict = dict()
            freq = np.array([config[x]['transducer_frequency'] for x in ch_ids], dtype='float32')
            tx_num = len(ch_ids)
            ping_num = len(self.convert_obj.ping_time)

            # Find largest dimensions of array in order to pad and stack smaller arrays
            # max_samples = 0
            # TODO How to determine if a CW data set is split beam or single beam, and how many splits?
            max_splits = max([n_c for n_c in self.convert_obj.n_complex_dict.values()]) if bb else 4
            if bb:
                shape = (len(ch_ids), ping_num, -1, max_splits)
                backscatter = np.array(self.convert_obj.complex_dict).reshape(shape)
                backscatter = np.moveaxis(backscatter, 3, 1)

            # Loop through each transducer for channel-specific variables
            bm_width = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            bm_dir = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            bm_angle = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            tx_pos = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
            beam_dict['equivalent_beam_angle'] = np.zeros(shape=(tx_num,), dtype='float32')
            beam_dict['gain_correction'] = np.zeros(shape=(tx_num,), dtype='float32')
            beam_dict['gpt_software_version'] = []
            beam_dict['channel_id'] = []
            beam_dict['frequency_start'] = []
            beam_dict['frequency_end'] = []
            beam_dict['slope'] = []
            c_seq = 0
            for k, c in config.items():
                if k not in ch_ids:
                    continue
                bm_width['beamwidth_receive_major'][c_seq] = c.get('beam_width_alongship', np.nan)
                bm_width['beamwidth_receive_minor'][c_seq] = c.get('beam_width_athwartship', np.nan)
                bm_width['beamwidth_transmit_major'][c_seq] = c.get('beam_width_alongship', np.nan)
                bm_width['beamwidth_transmit_minor'][c_seq] = c.get('beam_width_athwartship', np.nan)
                bm_dir['beam_direction_x'][c_seq] = c.get('transducer_alpha_x', np.nan)
                bm_dir['beam_direction_y'][c_seq] = c.get('transducer_alpha_y', np.nan)
                bm_dir['beam_direction_z'][c_seq] = c.get('transducer_alpha_z', np.nan)
                bm_angle['angle_offset_alongship'][c_seq] = c.get('angle_offset_alongship', np.nan)
                bm_angle['angle_offset_athwartship'][c_seq] = c.get('angle_offset_athwartship', np.nan)
                bm_angle['angle_sensitivity_alongship'][c_seq] = c.get('angle_sensitivity_alongship', np.nan)
                bm_angle['angle_sensitivity_athwartship'][c_seq] = c.get('angle_sensitivity_athwartship', np.nan)
                tx_pos['transducer_offset_x'][c_seq] = c.get('transducer_offset_x', np.nan)
                tx_pos['transducer_offset_y'][c_seq] = c.get('transducer_offset_y', np.nan)
                tx_pos['transducer_offset_z'][c_seq] = c.get('transducer_offset_z', np.nan)
                beam_dict['equivalent_beam_angle'][c_seq] = c.get('equivalent_beam_angle', np.nan)
                # TODO: gain is 5 values in test dataset
                beam_dict['gain_correction'][c_seq] = c['gain'][c_seq]
                beam_dict['gpt_software_version'].append(c['transceiver_software_version'])
                beam_dict['channel_id'].append(c['channel_id'])
                beam_dict['slope'].append(self.convert_obj.ping_data_dict[k]['slope'])

                # Pad each channel with nan so that they can be stacked
                if bb:
                    beam_dict['frequency_start'].append(self.convert_obj.ping_data_dict[k]['frequency_start'])
                    beam_dict['frequency_end'].append(self.convert_obj.ping_data_dict[k]['frequency_end'])
                c_seq += 1

            # Stack channels and order axis as: channel, quadrant, ping, range
            if bb:
                beam_dict['frequency_start'] = np.unique(beam_dict['frequency_start']).astype(int)
                beam_dict['frequency_end'] = np.unique(beam_dict['frequency_end']).astype(int)
                beam_dict['frequency_center'] = (beam_dict['frequency_start'] + beam_dict['frequency_end']) / 2

            # Loop through each transducer for variables that may vary at each ping
            # -- this rarely is the case for EK60 so we check first before saving
            pl_tmp = np.unique(self.convert_obj.ping_data_dict[ch_ids[0]]['pulse_duration']).size
            pw_tmp = np.unique(self.convert_obj.ping_data_dict[ch_ids[0]]['transmit_power']).size
            # bw_tmp = np.unique(self.ping_data_dict[1]['bandwidth']).size      # Not in EK80
            si_tmp = np.unique(self.convert_obj.ping_data_dict[ch_ids[0]]['sample_interval']).size
            if np.all(np.array([pl_tmp, pw_tmp, si_tmp]) == 1):
                tx_sig = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
                beam_dict['sample_interval'] = np.zeros(shape=(tx_num,), dtype='float32')
                for t_seq in range(tx_num):
                    tx_sig['transmit_duration_nominal'][t_seq] = \
                        np.float32(self.convert_obj.ping_data_dict[ch_ids[t_seq]]['pulse_duration'][0])
                    tx_sig['transmit_power'][t_seq] = \
                        np.float32(self.convert_obj.ping_data_dict[ch_ids[t_seq]]['transmit_power'][0])
                    # tx_sig['transmit_bandwidth'][t_seq] = \
                    #     np.float32((self.parameters[self.ch_ids[t_seq]]['bandwidth'][0])
                    beam_dict['sample_interval'][t_seq] = \
                        np.float32(self.convert_obj.ping_data_dict[ch_ids[t_seq]]['sample_interval'][0])
            else:
                tx_sig = defaultdict(lambda: np.zeros(shape=(tx_num, ping_num), dtype='float32'))
                beam_dict['sample_interval'] = np.zeros(shape=(tx_num, ping_num), dtype='float32')
                for t_seq in range(tx_num):
                    tx_sig['transmit_duration_nominal'][t_seq, :] = \
                        np.array(self.convert_obj.ping_data_dict[ch_ids[t_seq]]['pulse_duration'], dtype='float32')
                    tx_sig['transmit_power'][t_seq, :] = \
                        np.array(self.convert_obj.ping_data_dict[ch_ids[t_seq]]['transmit_power'], dtype='float32')
                    # tx_sig['transmit_bandwidth'][t_seq, :] = \
                    #     np.array(self.ping_data_dictj[self.ch_ids[t_seq]]['bandwidth'], dtype='float32')
                    beam_dict['sample_interval'][t_seq, :] = \
                        np.array(self.convert_obj.ping_data_dict[ch_ids[t_seq]]['sample_interval'], dtype='float32')

            # Build other parameters
            # beam_dict['non_quantitative_processing'] = np.array([0, ] * freq.size, dtype='int32')
            # -- sample_time_offset is set to 2 for EK60 data, this value is NOT from sample_data['offset']
            # beam_dict['sample_time_offset'] = np.array([2, ] * freq.size, dtype='int32')
            pulse_length = 'pulse_duration_fm' if bb else 'pulse_duration'
            # Gets indices from pulse length table using the transmit_duration_nominal values selected
            idx = [np.argwhere(np.isclose(tx_sig['transmit_duration_nominal'][i],
                                          config[ch][pulse_length])).squeeze()
                   for i, ch in enumerate(ch_ids)]
            # Use the indices to select sa_correction values from the sa correction table
            beam_dict['sa_correction'] = \
                np.array([x['sa_correction'][y]
                         for x, y in zip(config.values(), np.array(idx))])
            # Convert np.datetime64 numbers to seconds since 1900-01-01
            # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
            ping_time = (self.convert_obj.ping_time - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')

            ds = xr.Dataset(
                {'channel_id': (['frequency'], beam_dict['channel_id']),
                 'beamwidth_receive_alongship': (['frequency'], bm_width['beamwidth_receive_major'],
                                                 {'long_name': 'Half power one-way receive beam width along '
                                                  'alongship axis of beam',
                                                  'units': 'arc_degree',
                                                  'valid_range': (0.0, 360.0)}),
                 'beamwidth_receive_athwartship': (['frequency'], bm_width['beamwidth_receive_minor'],
                                                   {'long_name': 'Half power one-way receive beam width along '
                                                    'athwartship axis of beam',
                                                    'units': 'arc_degree',
                                                    'valid_range': (0.0, 360.0)}),
                 'beamwidth_transmit_alongship': (['frequency'], bm_width['beamwidth_transmit_major'],
                                                  {'long_name': 'Half power one-way transmit beam width along '
                                                   'alongship axis of beam',
                                                   'units': 'arc_degree',
                                                   'valid_range': (0.0, 360.0)}),
                 'beamwidth_transmit_athwartship': (['frequency'], bm_width['beamwidth_transmit_minor'],
                                                    {'long_name': 'Half power one-way transmit beam width along '
                                                     'athwartship axis of beam',
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
                 'angle_offset_alongship': (['frequency'], bm_angle['angle_offset_alongship'],
                                            {'long_name': 'electrical alongship angle of the transducer'}),
                 'angle_offset_athwartship': (['frequency'], bm_angle['angle_offset_athwartship'],
                                              {'long_name': 'electrical athwartship angle of the transducer'}),
                 'angle_sensitivity_alongship': (['frequency'], bm_angle['angle_sensitivity_alongship'],
                                                 {'long_name': 'alongship sensitivity of the transducer'}),
                 'angle_sensitivity_athwartship': (['frequency'], bm_angle['angle_sensitivity_athwartship'],
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
                 'slope': (['frequency', 'ping_time'], np.array(beam_dict['slope'])),
                 },
                coords={'frequency': (['frequency'], freq,
                                      {'long_name': 'Transducer frequency', 'units': 'Hz'}),
                        'ping_time': (['ping_time'], ping_time,
                                      {'axis': 'T',
                                       'calendar': 'gregorian',
                                       'long_name': 'Timestamp of each ping',
                                       'standard_name': 'time',
                                       'units': 'seconds since 1900-01-01'}),
                        },
                attrs={'beam_mode': 'vertical',
                       'conversion_equation_t': 'type_3'})
            # Save broadband backscatter if present
            if bb:
                ds_bb = xr.Dataset(
                    {'backscatter_r': (['frequency', 'quadrant', 'ping_time', 'range_bin'], np.real(backscatter),
                                       {'long_name': 'Real part of backscatter power',
                                        'units': 'V'}),
                     'backscatter_i': (['frequency', 'quadrant', 'ping_time', 'range_bin'], np.imag(backscatter),
                                       {'long_name': 'Imaginary part of backscatter power',
                                        'units': 'V'})},
                    coords={'frequency': (['frequency'], freq,
                                          {'long_name': 'Center frequency of the transducer',
                                          'units': 'Hz'}),
                            'frequency_start': (['frequency'], beam_dict['frequency_start'],
                                                {'long_name': 'Starting frequency of the transducer',
                                                 'units': 'Hz'}),
                            'frequency_end': (['frequency'], beam_dict['frequency_end'],
                                              {'long_name': 'Ending frequency of the transducer',
                                               'units': 'Hz'}),
                            'ping_time': (['ping_time'], ping_time,
                                          {'axis': 'T',
                                           'calendar': 'gregorian',
                                           'long_name': 'Timestamp of each ping',
                                           'standard_name': 'time',
                                           'units': 'seconds since 1900-01-01'}),
                            'quadrant': (['quadrant'], np.arange(max_splits)),
                            'range_bin': (['range_bin'], np.arange(backscatter.shape[3]))
                            })
                ds = xr.merge([ds, ds_bb])
            # Save continuous wave backscatter
            else:
                ds_cw = xr.Dataset(
                    {'backscatter_r': (['frequency', 'ping_time', 'range_bin'], self.convert_obj.power_dict,
                                       {'long_name': 'Backscattering power',
                                           'units': 'dB'}),
                     'angle_athwartship': (['frequency', 'ping_time', 'range_bin'],
                                           self.convert_obj.angle_dict[:, :, :, 0],
                                           {'long_name': 'electrical athwartship angle'}),
                     'angle_alongship': (['frequency', 'ping_time', 'range_bin'],
                                         self.convert_obj.angle_dict[:, :, :, 1],
                                         {'long_name': 'electrical alongship angle'})},
                    coords={'frequency': (['frequency'], freq,
                                          {'long_name': 'Transducer frequency', 'units': 'Hz'}),
                            'ping_time': (['ping_time'], ping_time,
                                          {'axis': 'T',
                                           'calendar': 'gregorian',
                                           'long_name': 'Timestamp of each ping',
                                           'standard_name': 'time',
                                           'units': 'seconds since 1900-01-01'}),
                            'range_bin': (['range_bin'], np.arange(self.convert_obj.power_dict.shape[2]))
                            })
                ds = xr.merge([ds, ds_cw])

            # Below are specific to Simrad .raw files
            if 'gpt_software_version' in beam_dict:
                ds['gpt_software_version'] = ('frequency', beam_dict['gpt_software_version'])
            if 'sa_correction' in beam_dict:
                ds['sa_correction'] = ('frequency', beam_dict['sa_correction'])
            # Save to file
            if self.save_ext == '.nc':
                nc_encoding = {var: NETCDF_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
                ds.to_netcdf(path=path, mode='a', group='Beam', encoding=nc_encoding)
            elif self.save_ext == '.zarr':
                zarr_encoding = {var: ZARR_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
                ds = ds.chunk({'range_bin': 25000})
                ds.to_zarr(store=path, mode='a', group='Beam', encoding=zarr_encoding)

    def set_vendor(self):
        """Set the Vendor-specific group.
        """
        if not os.path.exists(self.output_path):
            print('netCDF file does not exist, exiting without saving Vendor group...')
        else:
            coeffs = dict()
            decimation_factors = dict()
            for ch in self.convert_obj.ch_ids:
                # Coefficients for wide band transceiver
                coeffs[f'{ch}_WBT_filter'] = self.convert_obj.fil_coeffs[ch][1]
                # Coefficients for pulse compression
                coeffs[f'{ch}_PC_filter'] = self.convert_obj.fil_coeffs[ch][2]
                decimation_factors[f'{ch}_WBT_decimation'] = self.convert_obj.fil_df[ch][1]
                decimation_factors[f'{ch}_PC_decimation'] = self.convert_obj.fil_df[ch][2]

            if self.save_ext == '.nc':
                ncfile = netCDF4.Dataset(self.output_path, "a", format="NETCDF4")
                vdr = ncfile.createGroup("Vendor")
                # Create compound datatype. (2 f32 values to make a c64 value)
                complex64 = np.dtype([("real", np.float32), ("imag", np.float32)])
                complex64_t = vdr.createCompoundType(complex64, "complex64")
                for k, v in coeffs.items():
                    data = np.empty(len(v), complex64)
                    data['real'] = v.real
                    data['imag'] = v.imag
                    vdr.createDimension(k + '_dim', None)
                    var = vdr.createVariable(k, complex64_t, k + '_dim')
                    var[:] = data
                for k, v in decimation_factors.items():
                    vdr.setncattr(k, v)
                # Save xml string
                vdr.setncattr('xml', self.convert_obj.config_datagram['xml'])
            elif self.save_ext == '.zarr':
                ds = xr.Dataset(coeffs)
                ds.attrs['xml'] = self.convert_obj.config_datagram['xml']
                for k, v in decimation_factors.items():
                    ds.attrs[k] = v
                ds.to_zarr(store=self.output_path, mode='a', group='Vendor')

    def set_sonar(self, ch_ids, path):
        if not os.path.exists(path):
            print('netCDF file does not exist, exiting without saving Sonar group...')
        else:
            config = self.convert_obj.config_datagram['configuration']
            # channels['frequency'] = np.array([self.config_datagram['configuration'][x]['transducer_frequency']
            #                                   for x in self.ch_ids], dtype='float32')

            # Collect unique variables
            frequency = []
            serial_number = []
            model = []
            for ch_id, data in config.items():
                frequency.append(data['transducer_frequency'])
                serial_number.append(data['serial_number'])
                model.append(data['transducer_name'])
            # Create dataset
            ds = xr.Dataset(
                {'serial_number': (['frequency'], serial_number),
                 'sonar_model': (['frequency'], model)},
                coords={'frequency': frequency},
                attrs={'sonar_manufacturer': 'Simrad',
                       'sonar_software_name': config[ch_ids[0]]['application_name'],
                       'sonar_software_version': config[ch_ids[0]]['application_version'],
                       'sonar_type': 'echosounder'})

            # save to file
            if self.save_ext == '.nc':
                ds.to_netcdf(path=path, mode='a', group='Sonar')
            elif self.save_ext == '.zarr':
                ds.to_zarr(store=path, mode='a', group='Sonar')

    def _copy_file(self, file):
        # Copy the current file into a new file with _cw appended to filename
        fname, ext = os.path.splitext(file)
        new_path = fname + '_cw' + ext
        self.extra_files.append(new_path)
        if os.path.exists(new_path):
            print("          overwriting: " + new_path)
            if ext == '.zarr':
                shutil.rmtree(new_path)
            elif ext == '.nc':
                os.remove(new_path)
        if ext == '.zarr':
            shutil.copytree(file, new_path)
        elif ext == '.nc':
            shutil.copyfile(file, new_path)
        return new_path


class SetGroupsAZFP(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from AZFP data files.
    """
    def save(self):
        """Actually save groups to file by calling the set methods.
        """
        ping_time = self.convert_obj._get_ping_time()
        sonar_values = ('ASL Environmental Sciences', 'Acoustic Zooplankton Fish Profiler',
                        int(self.convert_obj.unpacked_data['serial_number']),
                        'Based on AZFP Matlab Toolbox', '1.4', 'echosounder')
        self.set_toplevel("AZFP")
        self.set_env(ping_time)
        self.set_provenance()
        self.set_platform()
        self.set_sonar(sonar_values)
        self.set_beam(ping_time)
        self.set_vendor(ping_time)

    def set_env(self, ping_time):
        """Set the Environment group.
        """
        # Only save environment group if file_path exists
        if not os.path.exists(self.output_path):
            print('netCDF file does not exist, exiting without saving Environment group...')
        else:
            ds = xr.Dataset({'temperature': (['ping_time'], self.convert_obj.unpacked_data['temperature'])},
                            coords={'ping_time': (['ping_time'], ping_time,
                                    {'axis': 'T',
                                     'calendar': 'gregorian',
                                     'long_name': 'Timestamp of each ping',
                                     'standard_name': 'time',
                                     'units': 'seconds since 1970-01-01'})},
                            attrs={'long_name': "Water temperature",
                                   'units': "C"})

            # save to file
            if self.save_ext == '.nc':
                ds.to_netcdf(path=self.output_path, mode='a', group='Environment')
            elif self.save_ext == '.zarr':
                ds.to_zarr(store=self.output_path, mode='a', group='Environment')

    def set_platform(self):
        """Set the Platform group.
        """
        platform_dict = {'platform_name': self.ui_param['platform_name'],
                         'platform_type': self.ui_param['platform_type'],
                         'platform_code_ICES': self.ui_param['platform_code_ICES']}
        # Only save platform group if file_path exists
        if not os.path.exists(self.output_path):
            print("netCDF file does not exist, exiting without saving Platform group...")
        else:
            if self.save_ext == '.nc':
                with netCDF4.Dataset(self.output_path, 'a', format='NETCDF4') as ncfile:
                    plat = ncfile.createGroup('Platform')
                    [plat.setncattr(k, v) for k, v in platform_dict.items()]
            elif self.save_ext == '.zarr':
                zarrfile = zarr.open(self.output_path, mode='a')
                plat = zarrfile.create_group('Platform')
                for k, v in platform_dict.items():
                    plat.attrs[k] = v

    def set_beam(self, ping_time):
        """Set the Beam group.
        """
        unpacked_data = self.convert_obj.unpacked_data
        parameters = self.convert_obj.parameters
        anc = np.array(unpacked_data['ancillary'])   # convert to np array for easy slicing
        dig_rate = unpacked_data['dig_rate']         # dim: freq
        freq = np.array(unpacked_data['frequency']) * 1000    # Frequency in Hz

        # Build variables in the output xarray Dataset
        N = []   # for storing backscatter_r values for each frequency
        Sv_offset = np.zeros(freq.shape)
        for ich in range(len(freq)):
            Sv_offset[ich] = self.convert_obj._calc_Sv_offset(freq[ich], unpacked_data['pulse_length'][ich])
            N.append(np.array([unpacked_data['counts'][p][ich]
                               for p in range(len(unpacked_data['year']))]))

        tdn = unpacked_data['pulse_length'] / 1e6  # Convert microseconds to seconds
        range_samples_xml = np.array(parameters['range_samples'])         # from xml file
        range_samples_per_bin = unpacked_data['range_samples_per_bin']    # from data header

        # Calculate sample interval in seconds
        if len(dig_rate) == len(range_samples_per_bin):
            sample_int = range_samples_per_bin / dig_rate
        else:
            raise ValueError("dig_rate and range_samples not unique across frequencies")

        # Largest number of counts along the range dimension among the different channels
        longest_range_bin = np.max(unpacked_data['num_bins'])
        range_bin = np.arange(longest_range_bin)
        # TODO: replace the following with an explicit check of length of range across channels
        try:
            np.array(N)
        # Exception occurs when N is not rectangular,
        #  so it must be padded with nan values to make it rectangular
        except ValueError:
            N = [np.pad(n, ((0, 0), (0, longest_range_bin - n.shape[1])),
                 mode='constant', constant_values=np.nan)
                 for n in N]

        ds = xr.Dataset({'backscatter_r': (['frequency', 'ping_time', 'range_bin'], N),
                         'equivalent_beam_angle': (['frequency'], parameters['BP']),
                         'gain_correction': (['frequency'], parameters['gain']),
                         'sample_interval': (['frequency'], sample_int,
                                             {'units': 's'}),
                         'transmit_duration_nominal': (['frequency'], tdn,
                                                       {'long_name': 'Nominal bandwidth of transmitted pulse',
                                                        'units': 's',
                                                        'valid_min': 0.0}),
                         'temperature_counts': (['ping_time'], anc[:, 4]),
                         'tilt_x_count': (['ping_time'], anc[:, 0]),
                         'tilt_y_count': (['ping_time'], anc[:, 1]),
                         'tilt_x': (['ping_time'], unpacked_data['tilt_x']),
                         'tilt_y': (['ping_time'], unpacked_data['tilt_y']),
                         'cos_tilt_mag': (['ping_time'], unpacked_data['cos_tilt_mag']),
                         'DS': (['frequency'], parameters['DS']),
                         'EL': (['frequency'], parameters['EL']),
                         'TVR': (['frequency'], parameters['TVR']),
                         'VTX': (['frequency'], parameters['VTX']),
                         'Sv_offset': (['frequency'], Sv_offset),
                         'number_of_samples_digitized_per_pings': (['frequency'], range_samples_xml),
                         'number_of_digitized_samples_averaged_per_pings': (['frequency'],
                                                                            parameters['range_averaging_samples'])},
                        coords={'frequency': (['frequency'], freq,
                                              {'units': 'Hz',
                                               'valid_min': 0.0}),
                                'ping_time': (['ping_time'], ping_time,
                                              {'axis': 'T',
                                               'calendar': 'gregorian',
                                               'long_name': 'Timestamp of each ping',
                                               'standard_name': 'time',
                                               'units': 'seconds since 1970-01-01'}),
                                'range_bin': (['range_bin'], range_bin)},
                        attrs={'beam_mode': '',
                               'conversion_equation_t': 'type_4',
                               'number_of_frequency': parameters['num_freq'],
                               'number_of_pings_per_burst': parameters['pings_per_burst'],
                               'average_burst_pings_flag': parameters['average_burst_pings'],
                               # Temperature coefficients
                               'temperature_ka': parameters['ka'],
                               'temperature_kb': parameters['kb'],
                               'temperature_kc': parameters['kc'],
                               'temperature_A': parameters['A'],
                               'temperature_B': parameters['B'],
                               'temperature_C': parameters['C'],
                               # Tilt coefficients
                               'tilt_X_a': parameters['X_a'],
                               'tilt_X_b': parameters['X_b'],
                               'tilt_X_c': parameters['X_c'],
                               'tilt_X_d': parameters['X_d'],
                               'tilt_Y_a': parameters['Y_a'],
                               'tilt_Y_b': parameters['Y_b'],
                               'tilt_Y_c': parameters['Y_c'],
                               'tilt_Y_d': parameters['Y_d']})

        if self.save_ext == '.nc':
            nc_encoding = {var: NETCDF_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
            ds.to_netcdf(path=self.output_path, mode='a', group='Beam', encoding=nc_encoding)
        elif self.save_ext == '.zarr':
            zarr_encoding = {var: ZARR_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
            ds.to_zarr(store=self.output_path, mode='a', group='Beam', encoding=zarr_encoding)

    def set_vendor(self, ping_time):
        """Set the Vendor-specific group.
        """
        unpacked_data = self.convert_obj.unpacked_data
        freq = np.array(unpacked_data['frequency']) * 1000    # Frequency in Hz

        ds = xr.Dataset({
            'digitization_rate': (['frequency'], unpacked_data['dig_rate']),
            'lockout_index': (['frequency'], unpacked_data['lockout_index']),
            'number_of_bins_per_channel': (['frequency'], unpacked_data['num_bins']),
            'number_of_samples_per_average_bin': (['frequency'], unpacked_data['range_samples_per_bin']),
            'board_number': (['frequency'], unpacked_data['board_num']),
            'data_type': (['frequency'], unpacked_data['data_type']),
            'ping_status': (['ping_time'], unpacked_data['ping_status']),
            'number_of_acquired_pings': (['ping_time'], unpacked_data['num_acq_pings']),
            'first_ping': (['ping_time'], unpacked_data['first_ping']),
            'last_ping': (['ping_time'], unpacked_data['last_ping']),
            'data_error': (['ping_time'], unpacked_data['data_error']),
            'sensor_flag': (['ping_time'], unpacked_data['sensor_flag']),
            'ancillary': (['ping_time', 'ancillary_len'], unpacked_data['ancillary']),
            'ad_channels': (['ping_time', 'ad_len'], unpacked_data['ad']),
            'battery_main': (['ping_time'], unpacked_data['battery_main']),
            'battery_tx': (['ping_time'], unpacked_data['battery_tx'])},
            coords={
                'frequency': (['frequency'], freq,
                              {'units': 'Hz',
                               'valid_min': 0.0}),
                'ping_time': (['ping_time'], ping_time,
                              {'axis': 'T',
                               'calendar': 'gregorian',
                               'long_name': 'Timestamp of each ping',
                               'standard_name': 'time',
                               'units': 'seconds since 1970-01-01'}),
                'ancillary_len': (['ancillary_len'], list(range(len(unpacked_data['ancillary'][0])))),
                'ad_len': (['ad_len'], list(range(len(unpacked_data['ad'][0]))))},
            attrs={
                'profile_flag': unpacked_data['profile_flag'],
                'profile_number': unpacked_data['profile_number'],
                'burst_interval': unpacked_data['burst_int'],
                'ping_per_profile': unpacked_data['ping_per_profile'],
                'average_pings_flag': unpacked_data['avg_pings'],
                'spare_channel': unpacked_data['spare_chan'],
                'ping_period': unpacked_data['ping_period'],
                'phase': unpacked_data['phase'],
                'number_of_channels': unpacked_data['num_chan']}
        )

        if self.save_ext == '.nc':
            ds.to_netcdf(path=self.output_path, mode='a', group='Vendor')
        elif self.save_ext == '.zarr':
            ds.to_zarr(store=self.output_path, mode='a', group='Vendor')
