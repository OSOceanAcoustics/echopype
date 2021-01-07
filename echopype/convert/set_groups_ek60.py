import xarray as xr
import numpy as np
from collections import defaultdict
from .set_groups_base import SetGroupsBase
from ..utils import io


class SetGroupsEK60(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from EK60 data files.
    """
    def save(self):
        """Actually save groups to file by calling the set methods.
        """
        # filename must have timestamp that matches self.timestamp_pattern
        sonar_values = ('Simrad', self.parser_obj.config_datagram['sounder_name'],
                        '', '', self.parser_obj.config_datagram['version'], 'echosounder')

        # Save only up to the platform group if the user selects "GPS"
        if 'NME' in self.parser_obj.data_type:
            self.set_toplevel('EK60', date_created=self.parser_obj.nmea['timestamp'][0])
            self.set_provenance()
            self.set_sonar(sonar_values)
            self.set_platform(NMEA_only=True)
            return

        self.set_toplevel('EK60')
        self.set_provenance()           # provenance group
        self.set_sonar(sonar_values)    # sonar group
        self.set_env()              # environment group
        self.set_platform()         # platform group
        self.set_beam()             # beam group
        self.set_nmea()             # platform/NMEA group
        self.set_vendor()           # vendor group

    def set_env(self):
        """Set the Environment group.
        """
        ch_ids = list(self.parser_obj.config_datagram['transceivers'].keys())
        ds_env = []
        # Loop over channels
        for ch in ch_ids:
            ds_tmp = xr.Dataset({
                'absorption_indicative': (['ping_time'],
                                          self.parser_obj.ping_data_dict['absorption_coefficient'][ch],
                                          {'long_name': 'Indicative acoustic absorption',
                                           'units': 'dB/m',
                                           'valid_min': 0.0}),
                'sound_speed_indicative': (['ping_time'],
                                           self.parser_obj.ping_data_dict['sound_velocity'][ch],
                                           {'long_name': 'Indicative sound speed',
                                            'standard_name': 'speed_of_sound_in_sea_water',
                                            'units': 'm/s',
                                            'valid_min': 0.0})},
                                coords={'ping_time': (['ping_time'], self.parser_obj.ping_time[ch],
                                        {'axis': 'T',
                                         'calendar': 'gregorian',
                                         'long_name': 'Timestamps for NMEA position datagrams',
                                         'standard_name': 'time',
                                         'units': 'seconds since 1900-01-01'})})
            # Attach frequency dimension/coordinate
            ds_tmp = ds_tmp.expand_dims(
                {'frequency': [self.parser_obj.config_datagram['transceivers'][ch]['frequency']]})
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
        ds = ds.assign_coords({'ping_time': (['ping_time'], (ds['ping_time'] -
                                             np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's'),
                                             ds.ping_time.attrs)})

        # Save to file
        io.save_file(ds.chunk({'ping_time': 100}),
                     path=self.output_path, mode='a', engine=self.engine, group='Environment')

    def set_platform(self, NMEA_only=False):
        """Set the Platform group.
        """

        # Collect variables
        # Read lat/long from NMEA datagram
        location_time, msg_type, lat, lon = self._parse_NMEA()

        # NMEA dataset: variables filled with nan if do not exist
        ds = xr.Dataset(
            {
                'latitude': (['location_time'], lat,
                             {'long_name': 'Platform latitude',
                              'standard_name': 'latitude',
                              'units': 'degrees_north',
                              'valid_range': (-90.0, 90.0)}),
                'longitude': (['location_time'], lon,
                              {'long_name': 'Platform longitude',
                               'standard_name': 'longitude',
                               'units': 'degrees_east',
                               'valid_range': (-180.0, 180.0)}),
                'sentence_type': (['location_time'], msg_type)
            },
            coords={'location_time': (['location_time'], location_time,
                                      {'axis': 'T',
                                       'calendar': 'gregorian',
                                       'long_name': 'Timestamps for NMEA position datagrams',
                                       'standard_name': 'time',
                                       'units': 'seconds since 1900-01-01'})})
        ds = ds.chunk({'location_time': 100})

        if not NMEA_only:
            # Convert np.datetime64 numbers to seconds since 1900-01-01
            # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
            ch_ids = list(self.parser_obj.config_datagram['transceivers'].keys())

            # TODO: consider allow users to set water_level like in EK80?
            # if self.ui_param['water_level'] is not None:
            #     water_level = self.ui_param['water_level']
            # else:
            #     water_level = np.nan
            #     print('WARNING: The water_level_draft was not in the file. Value '
            #           'set to None.')

            # Loop over channels and merge all
            ds_plat = []
            for ch in ch_ids:
                ds_tmp = xr.Dataset(
                    {'pitch': (['ping_time'], self.parser_obj.ping_data_dict['pitch'][ch],
                               {'long_name': 'Platform pitch',
                                'standard_name': 'platform_pitch_angle',
                                'units': 'arc_degree',
                                'valid_range': (-90.0, 90.0)}),
                     'roll': (['ping_time'], self.parser_obj.ping_data_dict['roll'][ch],
                              {'long_name': 'Platform roll',
                               'standard_name': 'platform_roll_angle',
                               'units': 'arc_degree',
                               'valid_range': (-90.0, 90.0)}),
                     'heave': (['ping_time'], self.parser_obj.ping_data_dict['heave'][ch],
                               {'long_name': 'Platform heave',
                                'standard_name': 'platform_heave_angle',
                                'units': 'arc_degree',
                                'valid_range': (-90.0, 90.0)}),
                     'water_level': (['ping_time'], self.parser_obj.ping_data_dict['transducer_depth'][ch],
                                     {'long_name': 'z-axis distance from the platform coordinate system '
                                                   'origin to the sonar transducer',
                                      'units': 'm'})},
                    coords={'ping_time': (['ping_time'], self.parser_obj.ping_time[ch],
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
                    {'frequency': [self.parser_obj.config_datagram['transceivers'][ch]['frequency']]})
                ds_tmp['frequency'] = ds_tmp['frequency'].assign_attrs(
                    units='Hz',
                    long_name='Transducer frequency',
                    valid_min=0.0,
                )
                ds_plat.append(ds_tmp)

            # Merge data from all channels
            # TODO: for current test data we see all pitch/roll/heave are the same for all freq channels
            #  consider only saving those from the first channel
            ds_plat = xr.merge(ds_plat)

            # Merge with NMEA data
            ds = xr.merge([ds, ds_plat], combine_attrs='override')

            # Convert np.datetime64 numbers to seconds since 1900-01-01
            # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
            ds = ds.assign_coords({'ping_time': (['ping_time'],
                                                 (ds['ping_time'] -
                                                  np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's'),
                                                 ds.ping_time.attrs)}).chunk({'ping_time': 100})

        # Save to file
        io.save_file(ds, path=self.output_path, mode='a', engine=self.engine,
                     group='Platform', compression_settings=self.compression_settings)

    def set_beam(self):
        """Set the Beam group.
        """
        # Get channel keys and frequency
        ch_ids = list(self.parser_obj.config_datagram['transceivers'].keys())
        freq = np.array([v['frequency'] for v in self.parser_obj.config_datagram['transceivers'].values()])

        # Channel-specific variables
        params = [
            'channel_id',
            'beam_type',
            'beamwidth_alongship',
            'beamwidth_athwartship',
            'dir_x',
            'dir_y',
            'dir_z',
            'angle_offset_alongship',
            'angle_offset_athwartship',
            'angle_sensitivity_alongship',
            'angle_sensitivity_athwartship',
            'pos_x',
            'pos_y',
            'pos_z',
            'equivalent_beam_angle',
            'gpt_software_version',
            'gain',
        ]
        beam_params = defaultdict()
        for param in params:
            beam_params[param] = ([self.parser_obj.config_datagram['transceivers'][ch_seq].get(param, np.nan)
                                   for ch_seq in ch_ids])

        # TODO: Need to discuss if to remove INDEX2POWER factor from the backscatter_r
        #  currently this factor is multiplied to the raw data before backscatter_r is saved.
        #  This is if we are encoding only raw data to the .nc/zarr file.
        #  Need discussion since then the units won't match with convention (though it didn't match already...).
        # Assemble variables into a dataset
        ds = xr.Dataset(
            {
                'channel_id': (['frequency'], beam_params['channel_id']),
                'beam_type': ('frequency', beam_params['beam_type'],
                              {'long_name': 'type of transducer (0-single, 1-split)'}),
                'beamwidth_receive_alongship': (['frequency'], beam_params['beamwidth_alongship'],
                                                {'long_name': 'Half power one-way receive beam width along '
                                                              'alongship axis of beam',
                                                 'units': 'arc_degree',
                                                 'valid_range': (0.0, 360.0)}),
                'beamwidth_receive_athwartship': (['frequency'], beam_params['beamwidth_athwartship'],
                                                  {'long_name': 'Half power one-way receive beam width along '
                                                                'athwartship axis of beam',
                                                   'units': 'arc_degree',
                                                   'valid_range': (0.0, 360.0)}),
                'beamwidth_transmit_alongship': (['frequency'], beam_params['beamwidth_alongship'],
                                                 {'long_name': 'Half power one-way transmit beam width along '
                                                               'alongship axis of beam',
                                                  'units': 'arc_degree',
                                                  'valid_range': (0.0, 360.0)}),
                'beamwidth_transmit_athwartship': (['frequency'], beam_params['beamwidth_athwartship'],
                                                   {'long_name': 'Half power one-way transmit beam width along '
                                                                 'athwartship axis of beam',
                                                    'units': 'arc_degree',
                                                    'valid_range': (0.0, 360.0)}),
                'beam_direction_x': (['frequency'], beam_params['dir_x'],
                                     {'long_name': 'x-component of the vector that gives the pointing '
                                                   'direction of the beam, in sonar beam coordinate '
                                                   'system',
                                      'units': '1',
                                      'valid_range': (-1.0, 1.0)}),
                'beam_direction_y': (['frequency'], beam_params['dir_y'],
                                     {'long_name': 'y-component of the vector that gives the pointing '
                                                   'direction of the beam, in sonar beam coordinate '
                                                   'system',
                                      'units': '1',
                                      'valid_range': (-1.0, 1.0)}),
                'beam_direction_z': (['frequency'], beam_params['dir_z'],
                                     {'long_name': 'z-component of the vector that gives the pointing '
                                                   'direction of the beam, in sonar beam coordinate '
                                                   'system',
                                      'units': '1',
                                      'valid_range': (-1.0, 1.0)}),
                'angle_offset_alongship': (['frequency'], beam_params['angle_offset_alongship'],
                                           {'long_name': 'electrical alongship angle of the transducer'}),
                'angle_offset_athwartship': (['frequency'], beam_params['angle_offset_athwartship'],
                                             {'long_name': 'electrical athwartship angle of the transducer'}),
                'angle_sensitivity_alongship': (['frequency'], beam_params['angle_sensitivity_alongship'],
                                                {'long_name': 'alongship sensitivity of the transducer'}),
                'angle_sensitivity_athwartship': (['frequency'], beam_params['angle_sensitivity_athwartship'],
                                                  {'long_name': 'athwartship sensitivity of the transducer'}),
                'equivalent_beam_angle': (['frequency'], beam_params['equivalent_beam_angle'],
                                          {'long_name': 'Equivalent beam angle',
                                           'units': 'sr',
                                           'valid_range': (0.0, 4 * np.pi)}),
                'transducer_offset_x': (['frequency'], beam_params['pos_x'],
                                        {'long_name': 'x-axis distance from the platform coordinate system '
                                                      'origin to the sonar transducer',
                                         'units': 'm'}),
                'transducer_offset_y': (['frequency'], beam_params['pos_y'],
                                        {'long_name': 'y-axis distance from the platform coordinate system '
                                                      'origin to the sonar transducer',
                                         'units': 'm'}),
                'transducer_offset_z': (['frequency'], beam_params['pos_z'],
                                        {'long_name': 'z-axis distance from the platform coordinate system '
                                                      'origin to the sonar transducer',
                                         'units': 'm'}),
                'gain_correction': (['frequency'], beam_params['gain'],
                                    {'long_name': 'Gain correction',
                                     'units': 'dB'}),
                'gpt_software_version': (['frequency'], beam_params['gpt_software_version'])
                # TODO: need to parse and store 'offset' from configuration datagram
                # 'sample_time_offset': (['frequency'], np.array([2, ] * freq.size, dtype='int32'),
                #                        {'long_name': 'Time offset that is subtracted from the timestamp '
                #                                      'of each sample',
                #                         'units': 's'}),
                # 'non_quantitative_processing': (['frequency'], np.array([0, ] * freq.size, dtype='int32'),
                #                                 {'flag_meanings': 'no_non_quantitative_processing',
                #                                  'flag_values': '0',
                #                                  'long_name': 'Presence or not of non-quantitative '
                #                                               'processing applied to the backscattering '
                #                                               'data (sonar specific)'}),
            },
            coords={
                'frequency': (['frequency'], freq,
                              {'units': 'Hz',
                               'long_name': 'Transducer frequency',
                               'valid_min': 0.0})
            },
            attrs={
                'beam_mode': 'vertical',
                'conversion_equation_t': 'type_3'
            }
        )

        # Construct Dataset with ping-by-ping data from all channels
        ds_backscatter = []
        for ch in ch_ids:
            data_shape = self.parser_obj.ping_data_dict['power'][ch].shape
            ds_tmp = xr.Dataset(
                {
                    'backscatter_r': (['ping_time', 'range_bin'], self.parser_obj.ping_data_dict['power'][ch],
                                      {'long_name': 'Backscatter power',
                                       'units': 'dB'}),
                    'sample_interval': (['ping_time'],
                                        self.parser_obj.ping_data_dict['sample_interval'][ch],
                                        {'long_name': 'Interval between recorded raw data samples',
                                         'units': 's',
                                         'valid_min': 0.0}),
                    'transmit_bandwidth': (['ping_time'],
                                           self.parser_obj.ping_data_dict['bandwidth'][ch],
                                           {'long_name': 'Nominal bandwidth of transmitted pulse',
                                            'units': 'Hz',
                                            'valid_min': 0.0}),
                    'transmit_duration_nominal': (['ping_time'], self.parser_obj.ping_data_dict['pulse_length'][ch],
                                                  {'long_name': 'Nominal bandwidth of transmitted pulse',
                                                   'units': 's',
                                                   'valid_min': 0.0}),
                    'transmit_power': (['ping_time'],
                                       self.parser_obj.ping_data_dict['transmit_power'][ch],
                                       {'long_name': 'Nominal transmit power',
                                        'units': 'W',
                                        'valid_min': 0.0}),
                    'data_type': (['ping_time'], self.parser_obj.ping_data_dict['mode'][ch],
                                  {'long_name': 'recorded data type (1-power only, 2-angle only 3-power and angle)'})
                },
                coords={'ping_time': (['ping_time'], self.parser_obj.ping_time[ch],
                                      {'axis': 'T',
                                       'calendar': 'gregorian',
                                       'long_name': 'Timestamp of each ping',
                                       'standard_name': 'time',
                                       'units': 'seconds since 1900-01-01'}),
                        'range_bin': (['range_bin'], np.arange(data_shape[1]))})

            # TODO: below needs to be changed to use self.convert_obj.ping_data_dict['mode'][ch] == 3
            #  1 = Power only, 2 = Angle only 3 = Power & Angle
            # Set angle data if in split beam mode (beam_type == 1)
            # because single beam mode (beam_type == 0) does not record angle data
            if self.parser_obj.config_datagram['transceivers'][ch]['beam_type'] == 1:
                ds_tmp = ds_tmp.assign(
                    {
                        'angle_athwartship': (['ping_time', 'range_bin'],
                                              self.parser_obj.ping_data_dict['angle'][ch][:, :, 0],
                                              {'long_name': 'electrical athwartship angle'}),
                        'angle_alongship': (['ping_time', 'range_bin'],
                                            self.parser_obj.ping_data_dict['angle'][ch][:, :, 1],
                                            {'long_name': 'electrical alongship angle'}),
                    })

            # Attach frequency dimension/coordinate
            ds_tmp = ds_tmp.expand_dims(
                {'frequency': [self.parser_obj.config_datagram['transceivers'][ch]['frequency']]})
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
        ds = ds.assign_coords({'ping_time': (['ping_time'], (ds['ping_time'] -
                                             np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's'),
                                             ds.ping_time.attrs)})

        # Save to file
        io.save_file(ds.chunk({'range_bin': 25000, 'ping_time': 100}),
                     path=self.output_path, mode='a', engine=self.engine,
                     group='Beam', compression_settings=self.compression_settings)

    def set_vendor(self):
        # Retrieve pulse length and sa correction
        config = self.parser_obj.config_datagram['transceivers']
        freq = [v['frequency'] for v in config.values()]
        pulse_length = np.array([v['pulse_length_table'] for v in config.values()])
        gain = np.array([v['gain_table'] for v in config.values()])
        sa_correction = [v['sa_correction_table'] for v in config.values()]
        # Save pulse length and sa correction
        ds = xr.Dataset(
            {
                'sa_correction': (['frequency', 'pulse_length_bin'], sa_correction),
                'gain_correction': (['frequency', 'pulse_length_bin'], gain),
                'pulse_length': (['frequency', 'pulse_length_bin'], pulse_length)
            },
            coords={
                'frequency': (['frequency'], freq,
                              {'units': 'Hz',
                               'long_name': 'Transducer frequency',
                               'valid_min': 0.0}),
                'pulse_length_bin': (['pulse_length_bin'], np.arange(pulse_length.shape[1]))})

        # Save to file
        io.save_file(ds, path=self.output_path, mode='a', engine=self.engine,
                     group='Vendor', compression_settings=self.compression_settings)
