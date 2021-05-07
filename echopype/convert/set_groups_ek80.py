from typing import List
from collections import defaultdict
import xarray as xr
import numpy as np
from .set_groups_base import SetGroupsBase, set_encodings


class SetGroupsEK80(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from EK80 data files.
    """
    def set_env(self, env_only=False) -> xr.Dataset:
        """Set the Environment group.
        """
        # If only saving environment group, there is no ping_time so use timestamp of environment datagram
        if env_only:
            ping_time = self.parser_obj.environment['timestamp']
        else:
            ping_time = list(self.parser_obj.ping_time.values())[0][0]
        # Select the first available ping_time
        ping_time = np.array([ping_time.astype('datetime64[ns]')])

        # Collect variables
        ds = xr.Dataset({'temperature': (['ping_time'], [self.parser_obj.environment['temperature']]),
                         'depth': (['ping_time'], [self.parser_obj.environment['depth']]),
                         'acidity': (['ping_time'], [self.parser_obj.environment['acidity']]),
                         'salinity': (['ping_time'], [self.parser_obj.environment['salinity']]),
                         'sound_speed_indicative': (['ping_time'], [self.parser_obj.environment['sound_speed']])},
                        coords={
                            'ping_time': (['ping_time'], ping_time,
                                          {'axis': 'T',
                                           'long_name': 'Timestamp of each ping',
                                           'standard_name': 'time'})})
        return set_encodings(ds)

    def set_sonar(self) -> xr.Dataset:
        # Collect unique variables
        params = ['transducer_frequency',
                  'serial_number',
                  'transducer_name',
                  'application_name',
                  'application_version']
        var = defaultdict(list)
        for ch_id, data in self.parser_obj.config_datagram['configuration'].items():
            for param in params:
                var[param].append(data[param])

        # Create dataset
        ds = xr.Dataset(
            {
                'serial_number': (['frequency'], var['serial_number']),
                'sonar_model': (['frequency'], var['transducer_name']),
                'sonar_software_name': (['frequency'], var['application_name']),  # identical for all channels
                'sonar_software_version': (['frequency'], var['application_version']),  # identical for all channels
            },
            coords={'frequency': var['transducer_frequency']},
            attrs={'sonar_manufacturer': 'Simrad',
                   'sonar_type': 'echosounder'})
        return ds

    def set_platform(self) -> xr.Dataset:
        """Set the Platform group.
        """

        # Collect variables
        if self.ui_param['water_level'] is not None:
            water_level = self.ui_param['water_level']
        elif 'water_level_draft' in self.parser_obj.environment:
            water_level = self.parser_obj.environment['water_level_draft']
        else:
            water_level = np.nan
            print('WARNING: The water_level_draft was not in the file. '
                  'Value set to NaN.')

        location_time, msg_type, lat, lon = self._parse_NMEA()
        mru_time = self.parser_obj.mru.get('timestamp', None)
        mru_time = np.array(mru_time) if mru_time is not None else [np.nan]

        # Assemble variables into a dataset: variables filled with nan if do not exist
        ds = xr.Dataset(
            {
                'pitch': (['mru_time'], np.array(self.parser_obj.mru.get('pitch', [np.nan])),
                          {'long_name': 'Platform pitch',
                           'standard_name': 'platform_pitch_angle',
                           'units': 'arc_degree',
                           'valid_range': (-90.0, 90.0)}),
                'roll': (['mru_time'], np.array(self.parser_obj.mru.get('roll', [np.nan])),
                         {'long_name': 'Platform roll',
                          'standard_name': 'platform_roll_angle',
                          'units': 'arc_degree',
                          'valid_range': (-90.0, 90.0)}),
                'heave': (['mru_time'], np.array(self.parser_obj.mru.get('heave', [np.nan])),
                          {'long_name': 'Platform heave',
                           'standard_name': 'platform_heave_angle',
                           'units': 'arc_degree',
                           'valid_range': (-90.0, 90.0)}),
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
                'sentence_type': (['location_time'], msg_type),
                'water_level': ([], water_level,
                                {'long_name': 'z-axis distance from the platform coordinate system '
                                              'origin to the sonar transducer',
                                 'units': 'm'})
            },
            coords={'mru_time': (['mru_time'], mru_time,
                                 {'axis': 'T',
                                  'long_name': 'Timestamps for MRU datagrams',
                                  'standard_name': 'time'}),
                    'location_time': (['location_time'], location_time,
                                      {'axis': 'T',
                                       'long_name': 'Timestamps for NMEA datagrams',
                                       'standard_name': 'time'})
                    },
            attrs={'platform_code_ICES': self.ui_param['platform_code_ICES'],
                   'platform_name': self.ui_param['platform_name'],
                   'platform_type': self.ui_param['platform_type'],
                   # TODO: check what this 'drop_keel_offset' is
                   'drop_keel_offset': (self.parser_obj.environment['drop_keel_offset'] if
                                        hasattr(self.parser_obj.environment, 'drop_keel_offset') else np.nan)})
        return set_encodings(ds)

    def _assemble_ds_ping_invariant(self, params, data_type):
        """Assemble dataset for ping-invariant params in the Beam group.

        Parameters
        ----------
        data_type : str
            'complex' or 'power'
        params : dict
            beam parameters that do not change across ping
        """
        ch_ids = self.parser_obj.ch_ids[data_type]
        freq = np.array([self.parser_obj.config_datagram['configuration'][ch]['transducer_frequency']
                         for ch in ch_ids])
        beam_params = defaultdict()
        for param in params:
            beam_params[param] = [self.parser_obj.config_datagram['configuration'][ch].get(param, np.nan)
                                  for ch in ch_ids]
        ds = xr.Dataset(
            {
                'channel_id': (['frequency'], ch_ids),
                'beam_type': (['frequency'], beam_params['transducer_beam_type']),
                'beamwidth_twoway_alongship': (['frequency'], beam_params['beam_width_alongship'],
                                               {'long_name': 'Half power two-way beam width along '
                                                             'alongship axis of beam',
                                                'units': 'arc_degree',
                                                'valid_range': (0.0, 360.0)}),
                'beamwidth_twoway_athwartship': (['frequency'], beam_params['beam_width_athwartship'],
                                                 {'long_name': 'Half power two-way beam width along '
                                                               'athwartship axis of beam',
                                                  'units': 'arc_degree',
                                                  'valid_range': (0.0, 360.0)}),
                'beam_direction_x': (['frequency'], beam_params['transducer_alpha_x'],
                                     {'long_name': 'x-component of the vector that gives the pointing '
                                                   'direction of the beam, in sonar beam coordinate '
                                                   'system',
                                      'units': '1',
                                      'valid_range': (-1.0, 1.0)}),
                'beam_direction_y': (['frequency'], beam_params['transducer_alpha_y'],
                                     {'long_name': 'y-component of the vector that gives the pointing '
                                                   'direction of the beam, in sonar beam coordinate '
                                                   'system',
                                      'units': '1',
                                      'valid_range': (-1.0, 1.0)}),
                'beam_direction_z': (['frequency'], beam_params['transducer_alpha_z'],
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
                'transducer_offset_x': (['frequency'], beam_params['transducer_offset_x'],
                                        {'long_name': 'x-axis distance from the platform coordinate system '
                                                      'origin to the sonar transducer',
                                         'units': 'm'}),
                'transducer_offset_y': (['frequency'], beam_params['transducer_offset_y'],
                                        {'long_name': 'y-axis distance from the platform coordinate system '
                                                      'origin to the sonar transducer',
                                         'units': 'm'}),
                'transducer_offset_z': (['frequency'], beam_params['transducer_offset_z'],
                                        {'long_name': 'z-axis distance from the platform coordinate system '
                                                      'origin to the sonar transducer',
                                         'units': 'm'}),
                'transceiver_software_version': (['frequency'], beam_params['transceiver_software_version'],)
            },
            coords={
                'frequency': (['frequency'], freq,
                              {'units': 'Hz',
                               'long_name': 'Transducer frequency',
                               'valid_min': 0.0}),
            },
            attrs={
                'beam_mode': 'vertical',
                'conversion_equation_t': 'type_3'
            }
        )

        return ds

    def _assemble_ds_complex(self, ch):
        num_transducer_sectors = np.unique(np.array(self.parser_obj.ping_data_dict['n_complex'][ch]))
        if num_transducer_sectors.size > 1:  # this is not supposed to happen
            raise ValueError('Transducer sector number changes in the middle of the file!')
        else:
            num_transducer_sectors = num_transducer_sectors[0]
        data_shape = self.parser_obj.ping_data_dict['complex'][ch].shape
        data_shape = (data_shape[0], int(data_shape[1] / num_transducer_sectors), num_transducer_sectors)
        data = self.parser_obj.ping_data_dict['complex'][ch].reshape(data_shape)

        ds_tmp = xr.Dataset(
            {
                'backscatter_r': (['ping_time', 'range_bin', 'quadrant'],
                                  np.real(data),
                                  {'long_name': 'Real part of backscatter power',
                                   'units': 'V'}),
                'backscatter_i': (['ping_time', 'range_bin', 'quadrant'],
                                  np.imag(data),
                                  {'long_name': 'Imaginary part of backscatter power',
                                   'units': 'V'}),
            },
            coords={
                'ping_time': (['ping_time'], self.parser_obj.ping_time[ch],
                              {'axis': 'T',
                               'long_name': 'Timestamp of each ping',
                               'standard_name': 'time'}),
                'range_bin': (['range_bin'], np.arange(data_shape[1])),
                'quadrant': (['quadrant'], np.arange(num_transducer_sectors)),
            }
        )

        # CW data encoded as complex samples do NOT have frequency_start and frequency_end
        # TODO: use PulseForm instead of checking for the existence of FrequencyStart and FrequencyEnd
        if 'frequency_start' in self.parser_obj.ping_data_dict.keys() and \
                self.parser_obj.ping_data_dict['frequency_start'][ch]:
            ds_f_start_end = xr.Dataset(
                {
                    'frequency_start': (['ping_time'],
                                        np.array(self.parser_obj.ping_data_dict['frequency_start'][ch],
                                                 dtype=int),
                                        {'long_name': 'Starting frequency of the transducer',
                                         'units': 'Hz'}),
                    'frequency_end': (['ping_time'],
                                      np.array(self.parser_obj.ping_data_dict['frequency_end'][ch],
                                               dtype=int),
                                      {'long_name': 'Ending frequency of the transducer',
                                       'units': 'Hz'}),

                },
                coords={
                    'ping_time': (['ping_time'], self.parser_obj.ping_time[ch],
                                  {'axis': 'T',
                                   'long_name': 'Timestamp of each ping',
                                   'standard_name': 'time'}),
                }
            )
            ds_tmp = xr.merge([ds_tmp, ds_f_start_end],
                              combine_attrs='override')  # override keeps the Dataset attributes

        return set_encodings(ds_tmp)

    def _assemble_ds_power(self, ch):
        data_shape = self.parser_obj.ping_data_dict['power'][ch].shape
        ds_tmp = xr.Dataset(
            {
                'backscatter_r': (['ping_time', 'range_bin'],
                                  self.parser_obj.ping_data_dict['power'][ch],
                                  {'long_name': 'Backscattering power',
                                   'units': 'dB'}),
            },
            coords={
                'ping_time': (['ping_time'], self.parser_obj.ping_time[ch],
                              {'axis': 'T',
                               'long_name': 'Timestamp of each ping',
                               'standard_name': 'time'}),
                'range_bin': (['range_bin'], np.arange(data_shape[1])),
            }
        )

        # If angle data exist
        if ch in self.parser_obj.ch_ids['angle']:
            ds_tmp = ds_tmp.assign(
                {
                    'angle_athwartship': (['ping_time', 'range_bin'],
                                          self.parser_obj.ping_data_dict['angle'][ch][:, :, 0],
                                          {'long_name': 'electrical athwartship angle'}),
                    'angle_alongship': (['ping_time', 'range_bin'],
                                        self.parser_obj.ping_data_dict['angle'][ch][:, :, 1],
                                        {'long_name': 'electrical alongship angle'}),
                })

        return set_encodings(ds_tmp)

    def _assemble_ds_common(self, ch, range_bin_size):
        """Variables common to complex and power/angle data.
        """
        # pulse duration may have different names
        if 'pulse_length' in self.parser_obj.ping_data_dict:
            pulse_length = np.array(self.parser_obj.ping_data_dict['pulse_length'][ch], dtype='float32')
        else:
            pulse_length = np.array(self.parser_obj.ping_data_dict['pulse_duration'][ch], dtype='float32')

        ds_common = xr.Dataset(
            {
                'sample_interval': (['ping_time'],
                                    self.parser_obj.ping_data_dict['sample_interval'][ch],
                                    {'long_name': 'Interval between recorded raw data samples',
                                     'units': 's',
                                     'valid_min': 0.0}),
                'transmit_power': (['ping_time'],
                                   self.parser_obj.ping_data_dict['transmit_power'][ch],
                                   {'long_name': 'Nominal transmit power',
                                    'units': 'W',
                                    'valid_min': 0.0}),
                'transmit_duration_nominal': (['ping_time'], pulse_length,
                                              {'long_name': 'Nominal bandwidth of transmitted pulse',
                                               'units': 's',
                                               'valid_min': 0.0}),
                'slope': (['ping_time'], self.parser_obj.ping_data_dict['slope'][ch],),
            },
            coords={
                'ping_time': (['ping_time'], self.parser_obj.ping_time[ch],
                              {'axis': 'T',
                               'long_name': 'Timestamp of each ping',
                               'standard_name': 'time'}),
                'range_bin': (['range_bin'], np.arange(range_bin_size)),
            }
        )
        return set_encodings(ds_common)

    def set_beam(self) -> List[xr.Dataset]:
        """Set the Beam group.
        """

        def merge_save(ds_combine, ds_type, group_name):
            """Merge data from all complex or all power/angle channels
            """
            ds_combine = xr.merge(ds_combine)
            if ds_type == 'complex':
                ds_combine = xr.merge([ds_invariant_complex, ds_combine],
                                      combine_attrs='override')  # override keeps the Dataset attributes
            else:
                ds_combine = xr.merge([ds_invariant_power, ds_combine],
                                      combine_attrs='override')  # override keeps the Dataset attributes
            return set_encodings(ds_combine)
            # # Save to file
            # io.save_file(ds_combine.chunk({'range_bin': DEFAULT_CHUNK_SIZE['range_bin'],
            #                                'ping_time': DEFAULT_CHUNK_SIZE['ping_time']}),
            #              path=self.output_path, mode='a', engine=self.engine,
            #              group=group_name, compression_settings=self.compression_settings)

        # Assemble ping-invariant beam data variables
        params = [
            'transducer_beam_type',
            'beam_width_alongship',
            'beam_width_athwartship',
            'transducer_alpha_x',
            'transducer_alpha_y',
            'transducer_alpha_z',
            'angle_offset_alongship',
            'angle_offset_athwartship',
            'angle_sensitivity_alongship',
            'angle_sensitivity_athwartship',
            'transducer_offset_x',
            'transducer_offset_y',
            'transducer_offset_z',
            'equivalent_beam_angle',
            'transceiver_software_version',
        ]

        # Assemble dataset for ping-invariant params
        if self.parser_obj.ch_ids['complex']:
            ds_invariant_complex = self._assemble_ds_ping_invariant(params, 'complex')
        if self.parser_obj.ch_ids['power']:
            ds_invariant_power = self._assemble_ds_ping_invariant(params, 'power')

        # Assemble dataset for backscatter data and other ping-by-ping data
        ds_complex = []
        ds_power = []
        for ch in self.parser_obj.config_datagram['configuration'].keys():
            if ch in self.parser_obj.ch_ids['complex']:
                ds_data = self._assemble_ds_complex(ch)
            elif ch in self.parser_obj.ch_ids['power']:
                ds_data = self._assemble_ds_power(ch)
            else:  # skip for channels containing no data
                continue
            ds_common = self._assemble_ds_common(ch, ds_data.range_bin.size)
            ds_data = xr.merge([ds_data, ds_common],
                               combine_attrs='override')  # override keeps the Dataset attributes
            # Attach frequency dimension/coordinate
            ds_data = ds_data.expand_dims(
                {'frequency': [self.parser_obj.config_datagram['configuration'][ch]['transducer_frequency']]})
            ds_data['frequency'] = ds_data['frequency'].assign_attrs(
                units='Hz',
                long_name='Transducer frequency',
                valid_min=0.0,
            )
            if ch in self.parser_obj.ch_ids['complex']:
                ds_complex.append(ds_data)
            else:
                ds_power.append(ds_data)

        # Merge and save group:
        #  if both complex and power data exist: complex data in Beam group and power data in Beam_power
        #  if only one type of data exist: data in Beam group
        ds_beam_power = None
        if len(ds_complex) > 0:
            ds_beam = merge_save(ds_complex, 'complex', group_name='Beam')
            if len(ds_power) > 0:
                ds_beam_power = merge_save(ds_power, 'power', group_name='Beam_power')
        else:
            ds_beam = merge_save(ds_power, 'power', group_name='Beam')

        return [ds_beam, ds_beam_power]

    def set_vendor(self) -> xr.Dataset:
        """Set the Vendor-specific group.
        """
        config = self.parser_obj.config_datagram['configuration']

        # Table for sa_correction and gain indexed by pulse_length (exist for all channels)
        table_params = ['transducer_frequency', 'pulse_duration', 'sa_correction', 'gain']
        param_dict = defaultdict(list)
        for k, v in config.items():
            for p in table_params:
                param_dict[p].append(v[p])
        for p in param_dict.keys():
            param_dict[p] = np.array(param_dict[p])

        # Param size check
        if not param_dict['pulse_duration'].shape == param_dict['sa_correction'].shape == param_dict['gain'].shape:
            raise ValueError('Narrowband calibration parameters dimension mismatch!')

        ds_table = xr.Dataset(
            {
                'sa_correction': (['frequency', 'pulse_length_bin'], np.array(param_dict['sa_correction'])),
                'gain_correction': (['frequency', 'pulse_length_bin'], np.array(param_dict['gain'])),
                'pulse_length': (['frequency', 'pulse_length_bin'], np.array(param_dict['pulse_duration'])),
            },
            coords={
                'frequency': (['frequency'], param_dict['transducer_frequency'],
                              {'units': 'Hz',
                               'long_name': 'Transducer frequency',
                               'valid_min': 0.0}),
                'pulse_length_bin': (['pulse_length_bin'], np.arange(param_dict['pulse_duration'].shape[1]))
            }
        )

        # Broadband calibration parameters: use the zero padding approach
        cal_ch_ids = [ch for ch in config.keys() if 'calibration' in config[ch]]  # channels with cal params
        ds_cal = []
        for ch_id in cal_ch_ids:
            # TODO: consider using the full_ch_name below in place of channel id (ch_id)
            # full_ch_name = (f"{config[ch]['transceiver_type']} " +
            #                 f"{config[ch]['serial_number']}-" +
            #                 f"{config[ch]['hw_channel_configuration']} " +
            #                 f"{config[ch]['channel_id_short']}")
            cal_params = ['gain', 'impedance', 'phase', 'beamwidth_alongship', 'beamwidth_athwartship',
                               'angle_offset_alongship', 'angle_offset_athwartship']
            param_dict = {}
            for p in cal_params:
                param_dict[p] = (['cal_frequency'], config[ch_id]['calibration'][p])
            ds_ch = xr.Dataset(
                data_vars=param_dict,
                coords={
                    'cal_frequency': (['cal_frequency'], config[ch_id]['calibration']['frequency'],
                                      {'long_name': 'Frequency of calibration parameter', 'units': 'Hz'})
                })
            ds_ch = ds_ch.expand_dims({'cal_channel_id': [ch_id]})
            ds_ch['cal_channel_id'].attrs['long_name'] = 'ID of channels containing broadband calibration information'
            ds_cal.append(ds_ch)
        ds_cal = xr.merge(ds_cal)

        #  Save decimation factors and filter coefficients
        coeffs = dict()
        decimation_factors = dict()
        for ch in self.parser_obj.ch_ids['power'] + self.parser_obj.ch_ids['complex']:
            # filter coeffs and decimation factor for wide band transceiver (WBT)
            coeffs[f'{ch} WBT filter'] = self.parser_obj.fil_coeffs[ch][1]
            decimation_factors[f'{ch} WBT decimation'] = self.parser_obj.fil_df[ch][1]
            # filter coeffs and decimation factor for pulse compression (PC)
            coeffs[f'{ch} PC filter'] = self.parser_obj.fil_coeffs[ch][2]
            decimation_factors[f'{ch} PC decimation'] = self.parser_obj.fil_df[ch][2]

        # Assemble everything into a Dataset
        ds = xr.merge([ds_table, ds_cal])

        # Save filter coefficients as real and imaginary parts as attributes
        for k, v in coeffs.items():
            ds.attrs[k + '_r'] = np.real(v)
            ds.attrs[k + '_i'] = np.imag(v)

        # Save decimation factors as attributes
        for k, v in decimation_factors.items():
            ds.attrs[k] = v

        # Save the entire config XML in vendor group in case of info loss
        ds.attrs['config_xml'] = self.parser_obj.config_datagram['xml']

        return ds
