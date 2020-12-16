import xarray as xr
import numpy as np
from .set_groups_base import SetGroupsBase


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
        self.set_platform()         # platform group
        self.set_beam()             # beam group
        self.set_nmea()             # platform/NMEA group
        self.set_vendor()           # vendor group

    def set_env(self):
        """Set the Environment group.
        """
        ch_ids = list(self.convert_obj.config_datagram['transceivers'].keys())
        ds_env = []
        # Loop over channels
        for ch in ch_ids:
            ds_tmp = xr.Dataset({
                'absorption_indicative': (['ping_time'],
                                          self.convert_obj.ping_data_dict['sound_velocity'][ch],
                                          {'long_name': 'Indicative acoustic absorption',
                                           'units': 'dB/m',
                                           'valid_min': 0.0}),
                'sound_speed_indicative': (['ping_time'],
                                           self.convert_obj.ping_data_dict['absorption_coefficient'][ch],
                                           {'long_name': 'Indicative sound speed',
                                            'standard_name': 'speed_of_sound_in_sea_water',
                                            'units': 'm/s',
                                            'valid_min': 0.0})},
                                coords={'ping_time': (['ping_time'], self.convert_obj.ping_time[ch],
                                        {'axis': 'T',
                                         'calendar': 'gregorian',
                                         'long_name': 'Timestamps for NMEA position datagrams',
                                         'standard_name': 'time',
                                         'units': 'seconds since 1900-01-01'})})
            # Attach frequency dimension/coordinate
            ds_tmp = ds_tmp.expand_dims(
                {'frequency': [self.convert_obj.config_datagram['transceivers'][ch]['frequency']]})
            ds_tmp['frequency'] = ds_tmp['frequency'].assign_attrs(
                units='Hz',
                long_name='Transducer frequency',
                valid_min=0.0,
            )
            ds_env.append(ds_tmp)

        # Merge data from all channels
        ds = xr.merge(ds_env)

        # Convert np.datetime64 numbers to seconds since 1900-01-01
        # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
        ds['ping_time'] = (ds['ping_time'] - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')

        # save to file
        if self.save_ext == '.nc':
            ds.to_netcdf(path=self.output_path, mode='a', group='Environment')
        elif self.save_ext == '.zarr':
            ds.to_zarr(store=self.output_path, mode='a', group='Environment')

    def set_platform(self):
        """Set the Platform group.
        """

        # Collect variables
        # Read lat/long from NMEA datagram
        lat, lon, location_time = self._parse_NMEA()

        # Convert np.datetime64 numbers to seconds since 1900-01-01
        # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
        ch_ids = list(self.convert_obj.config_datagram['transceivers'].keys())
        ds_plat = []
        # Loop over channels
        for ch in ch_ids:
            ds_tmp = xr.Dataset(
                {'pitch': (['ping_time'], self.convert_obj.ping_data_dict['pitch'][ch],
                           {'long_name': 'Platform pitch',
                            'standard_name': 'platform_pitch_angle',
                            'units': 'arc_degree',
                            'valid_range': (-90.0, 90.0)}),
                 'roll': (['ping_time'], self.convert_obj.ping_data_dict['roll'][ch],
                          {'long_name': 'Platform roll',
                           'standard_name': 'platform_roll_angle',
                           'units': 'arc_degree',
                           'valid_range': (-90.0, 90.0)}),
                 'heave': (['ping_time'], self.convert_obj.ping_data_dict['heave'][ch],
                           {'long_name': 'Platform heave',
                            'standard_name': 'platform_heave_angle',
                            'units': 'arc_degree',
                            'valid_range': (-90.0, 90.0)}),
                 # Get user defined water level (0 if undefined)
                 'water_level': ([], self.ui_param['water_level'],
                                 {'long_name': 'z-axis distance from the platform coordinate system '
                                               'origin to the sonar transducer',
                                  'units': 'm'})},
                coords={'ping_time': (['ping_time'], self.convert_obj.ping_time[ch],
                                      {'axis': 'T',
                                       'calendar': 'gregorian',
                                       'long_name': 'Timestamps for position datagrams',
                                       'standard_name': 'time',
                                       'units': 'seconds since 1900-01-01'})},
                attrs={'platform_code_ICES': self.ui_param['platform_code_ICES'],
                       'platform_name': self.ui_param['platform_name'],
                       'platform_type': self.ui_param['platform_type']})

            # Attach frequency dimension/coordinate
            ds_tmp = ds_tmp.expand_dims(
                {'frequency': [self.convert_obj.config_datagram['transceivers'][ch]['frequency']]})
            ds_tmp['frequency'] = ds_tmp['frequency'].assign_attrs(
                units='Hz',
                long_name='Transducer frequency',
                valid_min=0.0,
            )
            ds_plat.append(ds_tmp)

        # Merge data from all channels
        ds = xr.merge(ds_plat)

        # Add in NMEA location information if it exists
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
            ds = xr.merge([ds, ds_loc], combine_attrs='override')

        # Convert np.datetime64 numbers to seconds since 1900-01-01
        # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
        ds['ping_time'] = (ds['ping_time'] - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')

        # save dataset to file with specified compression settings for all variables
        if self.save_ext == '.nc':
            nc_encoding = {var: self.NETCDF_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
            ds.to_netcdf(path=self.output_path, mode='a', group='Platform', encoding=nc_encoding)
        elif self.save_ext == '.zarr':
            zarr_encoding = {var: self.ZARR_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
            ds = ds.chunk({'location_time': 100, 'ping_time': 100})
            ds.to_zarr(store=self.output_path, mode='w', group='Platform', encoding=zarr_encoding)

    def set_beam(self):
        """Set the Beam group.
        """
        # Convert np.datetime64 numbers to seconds since 1900-01-01
        # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
        config = self.convert_obj.config_datagram

        # Additional coordinate variables added by echopype for storing data as a cube with
        # dimensions [frequency x ping_time x range_bin]
        freq = np.array([config['transceivers'][x]['frequency']
                        for x in config['transceivers'].keys()], dtype='float32')

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
        # Collect numeric parameters from the configuration
        for encode_name, origin_name in param_numerical.items():
            beam_dict[encode_name] = np.array(
                [val[origin_name] for key, val in config['transceivers'].items()]).astype('float32')
        beam_dict['transducer_offset_z'] += [self.convert_obj.ping_data_dict['transducer_depth'][x][0]
                                             for x in config['transceivers'].keys()]
        # Collect string parameters from the configuration
        for encode_name, origin_name in param_str.items():
            beam_dict[encode_name] = [val[origin_name]
                                      for key, val in config['transceivers'].items()]

        # TODO: Need to remember removing INDEX2POWER factor from the backscatter_r
        #  currently this factor is multiplied to the raw data before backscatter_r is saved.
        #  This is if we are encoding only raw data to the .nc/zarr file.
        #  Need discussion since then the units won't match with convention (though it didn't match already...).
        # Assemble variables into a dataset
        ds = xr.Dataset(
            {'beam_type': ('frequency', beam_dict['beam_type'],
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
             'sample_time_offset': (['frequency'], np.array([2, ] * freq.size, dtype='int32'),
                                    {'long_name': 'Time offset that is subtracted from the timestamp '
                                     'of each sample',
                                     'units': 's'}),
             # Below are specific to Simrad EK60 .raw files
             'channel_id': (['frequency'], beam_dict['channel_id']),
             'gpt_software_version': (['frequency'], beam_dict['gpt_software_version'])},
            coords={'frequency': (['frequency'], freq,
                                  {'units': 'Hz',
                                   'long_name': 'Transducer frequency',
                                   'valid_min': 0.0})},
            attrs={'beam_mode': 'vertical',
                   'conversion_equation_t': 'type_3'})

        ch_ids = list(self.convert_obj.config_datagram['transceivers'].keys())
        ds_backscatter = []
        # Loop over channels
        for ch in ch_ids:
            data_shape = self.convert_obj.ping_data_dict['power'][ch].shape
            ds_tmp = xr.Dataset(
                {'backscatter_r': (['ping_time', 'range_bin'], self.convert_obj.ping_data_dict['power'][ch],
                                   {'long_name': 'Backscatter power',
                                    'units': 'dB'}),
                 'angle_athwartship': (['ping_time', 'range_bin'],
                                       self.convert_obj.ping_data_dict['angle'][ch][:, :, 0],
                                       {'long_name': 'electrical athwartship angle'}),
                 'angle_alongship': (['ping_time', 'range_bin'],
                                     self.convert_obj.ping_data_dict['angle'][ch][:, :, 1],
                                     {'long_name': 'electrical alongship angle'}),
                 'sample_interval': (['ping_time'],
                                     self.convert_obj.ping_data_dict['sample_interval'][ch],
                                     {'long_name': 'Interval between recorded raw data samples',
                                      'units': 's',
                                      'valid_min': 0.0}),
                 'transmit_bandwidth': (['ping_time'],
                                        self.convert_obj.ping_data_dict['bandwidth'][ch],
                                        {'long_name': 'Nominal bandwidth of transmitted pulse',
                                         'units': 'Hz',
                                         'valid_min': 0.0}),
                 'transmit_duration_nominal': (['ping_time'], self.convert_obj.ping_data_dict['pulse_length'][ch],
                                               {'long_name': 'Nominal bandwidth of transmitted pulse',
                                                             'units': 's',
                                                'valid_min': 0.0}),
                 'transmit_power': (['ping_time'],
                                    self.convert_obj.ping_data_dict['transmit_power'][ch],
                                    {'long_name': 'Nominal transmit power',
                                     'units': 'W',
                                     'valid_min': 0.0})},
                coords={'ping_time': (['ping_time'], self.convert_obj.ping_time[ch],
                                      {'axis': 'T',
                                       'calendar': 'gregorian',
                                       'long_name': 'Timestamp of each ping',
                                       'standard_name': 'time',
                                       'units': 'seconds since 1900-01-01'}),
                        'range_bin': (['range_bin'], np.arange(data_shape[1]))})

            # Attach frequency dimension/coordinate
            ds_tmp = ds_tmp.expand_dims(
                {'frequency': [self.convert_obj.config_datagram['transceivers'][ch]['frequency']]})
            ds_tmp['frequency'] = ds_tmp['frequency'].assign_attrs(
                units='Hz',
                long_name='Transducer frequency',
                valid_min=0.0,
            )
            ds_backscatter.append(ds_tmp)

        # Merge data from all channels
        ds = xr.merge([ds, xr.merge(ds_backscatter)], combine_attrs='override')  # override keeps the Dataset attributes

        # Convert np.datetime64 numbers to seconds since 1900-01-01
        # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
        ds['ping_time'] = (ds['ping_time'] - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')

        # Save dataset with optional compression for all data variables
        if self.save_ext == '.nc':
            nc_encoding = {var: self.NETCDF_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
            ds.to_netcdf(path=self.output_path, mode='a', group='Beam', encoding=nc_encoding)
        elif self.save_ext == '.zarr':
            zarr_encoding = {var: self.ZARR_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
            ds = ds.chunk({'range_bin': 25000, 'ping_time': 100})
            ds.to_zarr(store=self.output_path, mode='a', group='Beam', encoding=zarr_encoding)

    def set_vendor(self):
        # Retrieve pulse length and sa correction
        config = self.convert_obj.config_datagram['transceivers']
        freq = [v['frequency'] for v in config.values()]
        pulse_length = np.array([v['pulse_length_table'] for v in config.values()])
        gain = np.array([v['gain_table'] for v in config.values()])
        sa_correction = [v['sa_correction_table'] for v in config.values()]
        # Save pulse length and sa correction
        ds = xr.Dataset({
            'sa_correction': (['frequency', 'pulse_length_bin'], sa_correction),
            'gain': (['frequency', 'pulse_length_bin'], gain),
            'pulse_length': (['frequency', 'pulse_length_bin'], pulse_length)},
            coords={
                'frequency': (['frequency'], freq,
                              {'units': 'Hz',
                               'long_name': 'Transducer frequency',
                               'valid_min': 0.0}),
                'pulse_length_bin': (['pulse_length_bin'], np.arange(pulse_length.shape[1]))})

        if self.save_ext == '.nc':
            ds.to_netcdf(path=self.output_path, mode='a', group='Vendor')
        elif self.save_ext == '.zarr':
            ds.to_zarr(store=self.output_path, mode='a', group='Vendor')
