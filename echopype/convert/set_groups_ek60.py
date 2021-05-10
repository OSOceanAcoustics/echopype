import xarray as xr
import numpy as np
from collections import defaultdict
from .set_groups_base import SetGroupsBase, set_encodings
from .set_groups_base import DEFAULT_CHUNK_SIZE


class SetGroupsEK60(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from EK60 data files.
    """
    def set_env(self) -> xr.Dataset:
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
                                         'long_name': 'Timestamps for NMEA position datagrams',
                                         'standard_name': 'time'})})
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

        return set_encodings(ds)

    def set_sonar(self) -> xr.Dataset:
        """Set the Sonar group.
        """
        # Assemble sonar group dictionary
        sonar_dict = {
            'sonar_manufacturer': 'Simrad',
            'sonar_model': self.parser_obj.config_datagram['sounder_name'],
            'sonar_serial_number': '',
            'sonar_software_name': '',
            'sonar_software_version': self.parser_obj.config_datagram['version'],
            'sonar_type': 'echosounder'
        }
        ds = xr.Dataset()
        ds = ds.assign_attrs(sonar_dict)
        return ds

    def set_platform(self, NMEA_only=False) -> xr.Dataset:
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
                                       'long_name': 'Timestamps for NMEA position datagrams',
                                       'standard_name': 'time'})})
        ds = ds.chunk({'location_time': DEFAULT_CHUNK_SIZE['ping_time']})

        if not NMEA_only:
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
                                           'long_name': 'Timestamps for position datagrams',
                                           'standard_name': 'time'})},
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

            ds = ds.chunk({'ping_time': DEFAULT_CHUNK_SIZE['ping_time']})

        return set_encodings(ds)

    def set_beam(self) -> xr.Dataset:
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
                # TODO: check EK60 data spec:
                #  the beamwidths provided are most likely 2-way beamwidth so below needs to change
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
                                  {'long_name': 'recorded data type (1-power only, 2-angle only 3-power and angle)'}),
                    'count': (['ping_time'], self.parser_obj.ping_data_dict['count'][ch],
                              {'long_name': 'Number of samples '}),
                    'offset': (['ping_time'], self.parser_obj.ping_data_dict['offset'][ch],
                               {'long_name': 'Offset of first sample'}),
                    'transmit_mode': (['ping_time'], self.parser_obj.ping_data_dict['transmit_mode'][ch],
                                      {'long_name': '0 = Active, 1 = Passive, 2 = Test, -1 = Unknown'}),
                },
                coords={'ping_time': (['ping_time'], self.parser_obj.ping_time[ch],
                                      {'axis': 'T',
                                       'long_name': 'Timestamp of each ping',
                                       'standard_name': 'time'}),
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

        return set_encodings(ds)

    def set_vendor(self) -> xr.Dataset:
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
        return ds
