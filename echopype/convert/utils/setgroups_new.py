"""
Class to save unpacked echosounder data to appropriate groups in netcdf or zarr.
"""
import os
import shutil
from datetime import datetime as dt
import xarray as xr
import numpy as np
import re
import pynmea2
import zarr
import netCDF4
from ..._version import get_versions
ECHOPYPE_VERSION = get_versions()['version']
del get_versions


class SetGroupsBase:
    """Base class for saving groups to netcdf or zarr from echosounder data files.
    """
    def __init__(self, convert_obj, input_file, output_path, save_ext='.nc', compress=True, overwrite=True):
        self.convert_obj = convert_obj   # a convert object ConvertEK60/ConvertAZFP/etc...
        self.input_file = input_file
        self.output_path = output_path
        self.save_ext = save_ext
        self.compress = compress
        self.overwrite = overwrite

    def save(self):
        """Actually save groups to file by calling the set methods.
        """

    def set_toplevel(self, sonar_model):
        """Set the top-level group.
        """
        # Collect variables
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
                   'survey_name': self.convert_obj.ui_param['survey_name']}
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
                for k, v in prov_dict.items():
                    prov.setncattr(k, v)
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
            time = (self.convert_obj.nmea_data.nmea_times - np.datetime64('1900-01-01T00:00:00')) \
                    / np.timedelta64(1, 's')
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
                nc_encoding = {'time': dict(zlib=True, complevel=4)} if self.compress else {}
                ds.to_netcdf(path=self.output_path, mode='a', group='Platform/NMEA', encoding=nc_encoding)
            elif self.save_ext == '.zarr':
                zarr_encoding = {'time': dict(compressor=zarr.Blosc(cname='zstd',
                                 clevel=3, shuffle=2))} if self.compress else {}
                ds.to_zarr(store=self.output_path, mode='a', group='Platform/NMEA', encoding=zarr_encoding)

    @staticmethod
    def _remove(path):
        fname, ext = os.path.splitext(path)
        if ext == '.zarr':
            shutil.rmtree(path)
        else:
            os.remove(path)


class SetGroupsEK60(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from EK60 data files.
    """
    def save(self):
        """Actually save groups to file by calling the set methods.
        """
        # filename must have timestamp that matches self.timestamp_pattern
        sonar_values = ('Simrad', self.convert_obj.config_datagram['sounder_name'],
                        '', '', self.convert_obj.config_datagram['version'], 'echosounder')
        self.set_toplevel("EK60")         # top-level group
        self.set_env()              # environment group
        self.set_provenance()       # provenance group
        self.set_sonar(sonar_values)            # sonar group
        self.set_beam()             # beam group
        self.set_platform()         # platform group
        self.set_nmea()             # platform/NMEA group

    def set_env(self):
        """Set the Environment group.
        """
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

        # Only save environment group if file_path exists
        if not os.path.exists(self.output_path):
            print('netCDF file does not exist, exiting without saving Environment group...')
        else:
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
        # Collect variables

        # Read lat/long from NMEA datagram
        idx_loc = np.argwhere(np.isin(self.convert_obj.nmea_data.messages, ['GGA', 'GLL', 'RMC'])).squeeze()
        nmea_msg = []
        [nmea_msg.append(pynmea2.parse(self.convert_obj.nmea_data.raw_datagrams[x])) for x in idx_loc]
        lat = np.array([x.latitude for x in nmea_msg]) if nmea_msg else [np.nan]
        lon = np.array([x.longitude for x in nmea_msg]) if nmea_msg else [np.nan]
        location_time = (self.convert_obj.nmea_data.nmea_times[idx_loc] -
                         np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's') if nmea_msg else [np.nan]

        if not os.path.exists(self.output_path):
            print('netCDF file does not exist, exiting without saving Platform group...')
        else:
            # Assemble variables into a dataset

            # Convert np.datetime64 numbers to seconds since 1900-01-01
            # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
            ping_time = (self.convert_obj.ping_time -
                         np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')

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
                 'water_level': ([], self.convert_obj.ui_param['water_level'],
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
                attrs={'platform_code_ICES': self.convert_obj.ui_param['platform_code_ICES'],
                       'platform_name': self.convert_obj.ui_param['platform_name'],
                       'platform_type': self.convert_obj.ui_param['platform_type']})
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
                nc_settings = dict(zlib=True, complevel=4)
                nc_encoding = {var: nc_settings for var in ds.data_vars} if self.compress else {}
                ds.to_netcdf(path=self.output_path, mode='a', group='Platform', encoding=nc_encoding)
            elif self.save_ext == '.zarr':
                zarr_settings = dict(compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2))
                zarr_encoding = {var: zarr_settings for var in ds.data_vars} if self.compress else {}
                ds.to_zarr(store=self.output_path, mode='w', group='Platform', encoding=zarr_encoding)

    def set_beam(self):
        """Set the Beam group.
        """
        # Collect variables
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

        # TODO: The following code only uses the data of the first range_bin group
        beam_dict['transmit_signal'] = self.convert_obj.tx_sig[0]  # only this range_bin group

        if len(config['transceivers']) == 1:   # only 1 channel
            idx = np.argwhere(np.isclose(self.convert_obj.tx_sig[0]['transmit_duration_nominal'],
                                         config['transceivers'][1]['pulse_length_table'])).squeeze()
            idx = np.expand_dims(np.array(idx), axis=0)
        else:
            idx = [np.argwhere(np.isclose(self.convert_obj.tx_sig[0]['transmit_duration_nominal'][key - 1],
                                          val['pulse_length_table'])).squeeze()
                   for key, val in config['transceivers'].items()]
        beam_dict['sa_correction'] = \
            np.array([x['sa_correction_table'][y]
                     for x, y in zip(config['transceivers'].values(), np.array(idx))])

        if not os.path.exists(self.output_path):
            print('netCDF file does not exist, exiting without saving Beam group...')
        else:
            # Convert np.datetime64 numbers to seconds since 1900-01-01
            # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
            ping_time = (self.convert_obj.ping_time - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')
            # Assemble variables into a dataset
            ds = xr.Dataset(
                {'backscatter_r': (['frequency', 'ping_time', 'range_bin'], self.convert_obj.power_dict,
                                   {'long_name': 'Backscatter power',
                                    'units': 'dB'}),
                 'angle_athwartship': (['frequency', 'ping_time', 'range_bin'], self.convert_obj.angle_dict[:, :, :, 0],
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
                 'sample_interval': (['frequency'], beam_dict['transmit_signal']['sample_interval'],
                                     {'long_name': 'Interval between recorded raw data samples',
                                      'units': 's',
                                      'valid_min': 0.0}),
                 'sample_time_offset': (['frequency'], np.array([2, ] * freq.size, dtype='int32'),
                                        {'long_name': 'Time offset that is subtracted from the timestamp '
                                                      'of each sample',
                                                      'units': 's'}),
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
                comp = {'zlib': True, 'complevel': 4}
                nc_encoding = {var: comp for var in ds.data_vars} if self.compress else {}
                ds.to_netcdf(path=self.output_path, mode='a', group='Beam', encoding=nc_encoding)
            elif self.save_ext == '.zarr':
                comp = {'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)}
                zarr_encoding = {var: comp for var in ds.data_vars} if self.compress else {}
                ds.to_zarr(store=self.output_path, mode='a', group='Beam', encoding=zarr_encoding)


class SetGroupsEK80(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from EK80 data files.
    """
    def save(self):
        """Actually save groups to file by calling the set methods.
        """

    def set_env(self):
        """Set the Environment group.
        """

    def set_platform(self):
        """Set the Platform group.
        """
        # TODO: let's add ConvertEK80.environment['drop_keel_offset'] to
        #  the Platform group even though it is not in the convention.

    def set_beam(self):
        """Set the Beam group.
        """

    def set_vendor(self):
        """Set the Vendor-specific group.
        """


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
        platform_dict = {'platform_name': self.convert_obj.ui_param['platform_name'],
                         'platform_type': self.convert_obj.ui_param['platform_type'],
                         'platform_code_ICES': self.convert_obj.ui_param['platform_code_ICES']}
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
            comp = {'zlib': True, 'complevel': 4}
            nc_encoding = {var: comp for var in ds.data_vars} if self.compress else {}
            ds.to_netcdf(path=self.output_path, mode='a', group='Beam', encoding=nc_encoding)
        elif self.save_ext == '.zarr':
            comp = {'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)}
            zarr_encoding = {var: comp for var in ds.data_vars} if self.compress else {}
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
