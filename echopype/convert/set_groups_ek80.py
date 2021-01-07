import os
import shutil
from collections import defaultdict
import xarray as xr
import numpy as np
from ..utils import io
from .set_groups_base import SetGroupsBase


class SetGroupsEK80(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from EK80 data files.
    """
    def save(self):
        """Actually save groups to file by calling the set methods.
        """

        def set_beam_type_specific_groups(ch_ids, bb, path):
            self.set_beam(ch_ids, bb=bb, path=path)
            self.set_sonar(ch_ids, path=path)
            self.set_vendor(ch_ids, bb=bb, path=path)

        # Save environment only
        if 'ENV' in self.parser_obj.data_type:
            self.set_toplevel(self.sonar_model, date_created=self.parser_obj.environment['timestamp'])
            self.set_provenance()
            self.set_env(env_only=True)
            return
        # Save NMEA/MRU data only
        elif 'NME' in self.parser_obj.data_type:
            self.set_toplevel(self.sonar_model, date_created=self.parser_obj.nmea['timestamp'][0])
            self.set_provenance()
            self.set_platform()
            return

        # Save all groups
        self.set_toplevel(self.sonar_model)
        self.set_provenance()    # provenance group
        self.set_env()           # environment group
        self.set_platform()      # platform group
        self.set_nmea()          # platform/NMEA group
        # If there is both bb and cw data
        if self.parser_obj.ch_ids['complex'] and self.parser_obj.ch_ids['power']:
            new_path = self._copy_file(self.output_path)
            set_beam_type_specific_groups(self.parser_obj.ch_ids['complex'], bb=True, path=self.output_path)
            set_beam_type_specific_groups(self.parser_obj.ch_ids['power'], bb=False, path=new_path)
        # If there is only bb data
        elif self.parser_obj.ch_ids['complex']:
            set_beam_type_specific_groups(self.parser_obj.ch_ids['complex'], bb=True, path=self.output_path)
        # If there is only cw data
        else:
            set_beam_type_specific_groups(self.parser_obj.ch_ids['power'], bb=False, path=self.output_path)

    def set_env(self, env_only=False):
        """Set the Environment group.
        """
        # If only saving environment group, there is no ping_time so use timestamp of environment datagram
        if env_only:
            ping_time = self.parser_obj.environment['timestamp']
        else:
            ping_time = list(self.parser_obj.ping_time.values())[0][0]
        # Select the first available ping_time
        ping_time = np.array([(ping_time.astype('datetime64[ns]') -
                               np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')])

        # Collect variables
        ds = xr.Dataset({'temperature': (['ping_time'], [self.parser_obj.environment['temperature']]),
                         'depth': (['ping_time'], [self.parser_obj.environment['depth']]),
                         'acidity': (['ping_time'], [self.parser_obj.environment['acidity']]),
                         'salinity': (['ping_time'], [self.parser_obj.environment['salinity']]),
                         'sound_speed_indicative': (['ping_time'], [self.parser_obj.environment['sound_speed']])},
                        coords={
                            'ping_time': (['ping_time'], ping_time,
                                          {'axis': 'T',
                                           'calendar': 'gregorian',
                                           'long_name': 'Timestamp of each ping',
                                           'standard_name': 'time',
                                           'units': 'seconds since 1900-01-01'})})
        # ds = ds.assign_coords({'ping_time': (['ping_time'], (ds['ping_time'] -
        #                                      np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's'),
        #                                      ds.ping_time.attrs)})

        # Save to file
        io.save_file(ds, path=self.output_path, mode='a', engine=self.engine,
                     group='Environment', compression_settings=self.compression_settings)

    def set_platform(self):
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
        # Convert MRU np.datetime64 numbers to seconds since 1900-01-01
        # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
        mru_time = self.parser_obj.mru.get('timestamp', None)
        mru_time = (np.array(mru_time) - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's') if \
            mru_time is not None else [np.nan]

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
                   'platform_type': self.ui_param['platform_type'],
                   # TODO: check what this 'drop_keel_offset' is
                   'drop_keel_offset': (self.parser_obj.environment['drop_keel_offset'] if
                                        hasattr(self.parser_obj.environment, 'drop_keel_offset') else np.nan)})

        # save to file
        io.save_file(ds.chunk({'location_time': 100, 'mru_time': 100}),
                     path=self.output_path, mode='a', engine=self.engine,
                     group='Platform', compression_settings=self.compression_settings)

    def set_beam(self, ch_ids, bb, path):
        """Set the Beam group.
        """
        # Assemble Dataset for channel-specific Beam group variables
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

        beam_params = defaultdict()
        for param in params:
            beam_params[param] = ([self.parser_obj.config_datagram['configuration'][ch].get(param, np.nan)
                                   for ch in ch_ids])
        # Get the index of the channels listed in the configuration because it does not change across files
        # unlike the channels given in the ping_data_dict
        # TODO: Consider using a combination of channel_id + channel_id_short,
        #  e.g., "WBT 717612-15 ES120-7C Serial No: 680" to make sure there is no conflict
        #  when there are identical type of transducers (they will have different serial numbers)
        ch_ids = [ch for ch in self.parser_obj.config_datagram['configuration'].keys() if ch in ch_ids]
        freq = np.array([self.parser_obj.config_datagram['configuration'][ch]['transducer_frequency']
                         for ch in ch_ids])

        ds = xr.Dataset(
            {
                'channel_id': (['frequency'], ch_ids),
                'beam_type': (['frequency'], beam_params['transducer_beam_type']),
                'beamwidth_receive_alongship': (['frequency'], beam_params['beam_width_alongship'],
                                                {'long_name': 'Half power one-way receive beam width along '
                                                              'alongship axis of beam',
                                                 'units': 'arc_degree',
                                                 'valid_range': (0.0, 360.0)}),
                'beamwidth_receive_athwartship': (['frequency'], beam_params['beam_width_athwartship'],
                                                  {'long_name': 'Half power one-way receive beam width along '
                                                                'athwartship axis of beam',
                                                   'units': 'arc_degree',
                                                   'valid_range': (0.0, 360.0)}),
                'beamwidth_transmit_alongship': (['frequency'], beam_params['beam_width_alongship'],
                                                 {'long_name': 'Half power one-way transmit beam width along '
                                                               'alongship axis of beam',
                                                  'units': 'arc_degree',
                                                  'valid_range': (0.0, 360.0)}),
                'beamwidth_transmit_athwartship': (['frequency'], beam_params['beam_width_athwartship'],
                                                   {'long_name': 'Half power one-way transmit beam width along '
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

        # TODO: Check convention to see what to do with the variables below:
        #  'non_quantitative_processing': (['frequency'], beam_dict['non_quantitative_processing'],
        #                                  {'flag_meanings': 'no_non_quantitative_processing',
        #                                   'flag_values': '0',
        #                                   'long_name': 'Presence or not of non-quantitative '
        #                                                'processing applied to the backscattering '
        #                                                'data (sonar specific)'}),
        #  'sample_time_offset': (['frequency'], beam_dict['sample_time_offset'],
        #                         {'long_name': 'Time offset that is subtracted from the timestamp '
        #                                       'of each sample',
        #                          'units': 's'}),
        #  'transmit_bandwidth': (['frequency'], tx_sig['transmit_bandwidth'],
        #                         {'long_name': 'Nominal bandwidth of transmitted pulse',
        #                          'units': 'Hz',
        #                          'valid_min': 0.0}),

        # Assemble coordinates and data variables
        ds_backscatter = []
        if bb:  # complex data (BB or CW)
            for ch in ch_ids:
                num_transducer_sectors = np.unique(np.array(self.parser_obj.ping_data_dict['n_complex'][ch]))
                if num_transducer_sectors.size > 1:
                    raise ValueError('Transducer sector number changes in the middle of the file!')
                else:
                    num_transducer_sectors = num_transducer_sectors[0]
                data_shape = self.parser_obj.ping_data_dict['complex'][ch].shape
                data_shape = (data_shape[0], int(data_shape[1] / num_transducer_sectors), num_transducer_sectors)
                data = self.parser_obj.ping_data_dict['complex'][ch].reshape(data_shape)

                # CW data has pulse_duration, BB data has pulse_length
                if 'pulse_length' in self.parser_obj.ping_data_dict:
                    pulse_length = np.array(self.parser_obj.ping_data_dict['pulse_length'][ch], dtype='float32')
                else:
                    pulse_length = np.array(self.parser_obj.ping_data_dict['pulse_duration'][ch], dtype='float32')

                # Assemble ping by ping data
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
                        'sample_interval': (['ping_time'],
                                            self.parser_obj.ping_data_dict['sample_interval'][ch],
                                            {'long_name': 'Interval between recorded raw data samples',
                                             'units': 's',
                                             'valid_min': 0.0}),
                        'transmit_duration_nominal': (['ping_time'],
                                                      pulse_length,
                                                      {'long_name': 'Nominal bandwidth of transmitted pulse',
                                                       'units': 's',
                                                       'valid_min': 0.0}),
                        'transmit_power': (['ping_time'],
                                           self.parser_obj.ping_data_dict['transmit_power'][ch],
                                           {'long_name': 'Nominal transmit power',
                                            'units': 'W',
                                            'valid_min': 0.0}),
                        'slope': (['ping_time'], self.parser_obj.ping_data_dict['slope'][ch],),
                    },
                    coords={
                        'ping_time': (['ping_time'], self.parser_obj.ping_time[ch],
                                      {'axis': 'T',
                                       'calendar': 'gregorian',
                                       'long_name': 'Timestamp of each ping',
                                       'standard_name': 'time',
                                       'units': 'seconds since 1900-01-01'}),
                        'range_bin': (['range_bin'], np.arange(data_shape[1])),
                        'quadrant': (['quadrant'], np.arange(num_transducer_sectors)),
                    }
                )

                # CW data encoded as complex samples do NOT have frequency_start and frequency_end
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
                                           'calendar': 'gregorian',
                                           'long_name': 'Timestamp of each ping',
                                           'standard_name': 'time',
                                           'units': 'seconds since 1900-01-01'}),
                        }
                    )
                    ds_tmp = xr.merge([ds_tmp, ds_f_start_end],
                                      combine_attrs='override')  # override keeps the Dataset attributes

                # Attach frequency dimension/coordinate
                ds_tmp = ds_tmp.expand_dims(
                    {'frequency': [self.parser_obj.config_datagram['configuration'][ch]['transducer_frequency']]})
                ds_tmp['frequency'] = ds_tmp['frequency'].assign_attrs(
                    units='Hz',
                    long_name='Transducer frequency',
                    valid_min=0.0,
                )
                ds_backscatter.append(ds_tmp)

        else:  # power and angle data (CW)
            for ch in ch_ids:
                data_shape = self.parser_obj.ping_data_dict['power'][ch].shape
                ds_tmp = xr.Dataset(
                    {
                        'backscatter_r': (['ping_time', 'range_bin'],
                                          self.parser_obj.ping_data_dict['power'][ch],
                                          {'long_name': 'Backscattering power',
                                           'units': 'dB'}),
                        'sample_interval': (['ping_time'],
                                            self.parser_obj.ping_data_dict['sample_interval'][ch],
                                            {'long_name': 'Interval between recorded raw data samples',
                                             'units': 's',
                                             'valid_min': 0.0}),
                        'transmit_duration_nominal': (['ping_time'],
                                                      self.parser_obj.ping_data_dict['pulse_duration'][ch],
                                                      {'long_name': 'Nominal bandwidth of transmitted pulse',
                                                       'units': 's',
                                                       'valid_min': 0.0}),
                        'transmit_power': (['ping_time'],
                                           self.parser_obj.ping_data_dict['transmit_power'][ch],
                                           {'long_name': 'Nominal transmit power',
                                            'units': 'W',
                                            'valid_min': 0.0}),
                        'slope': (['ping_time'],
                                  self.parser_obj.ping_data_dict['slope'][ch]),
                    },
                    coords={
                        'ping_time': (['ping_time'], self.parser_obj.ping_time[ch],
                                      {'axis': 'T',
                                       'calendar': 'gregorian',
                                       'long_name': 'Timestamp of each ping',
                                       'standard_name': 'time',
                                       'units': 'seconds since 1900-01-01'}),
                        'range_bin': (['range_bin'], np.arange(data_shape[1])),
                    }
                )

                # Set angle data if in split beam mode (beam_type == 1)
                # because single beam mode (beam_type == 0) does not record angle data
                if self.parser_obj.config_datagram['configuration'][ch]['transducer_beam_type'] == 1:
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
                    {'frequency': [self.parser_obj.config_datagram['configuration'][ch]['transducer_frequency']]})
                ds_tmp['frequency'] = ds_tmp['frequency'].assign_attrs(
                    units='Hz',
                    long_name='Transducer frequency',
                    valid_min=0.0,
                )
                ds_backscatter.append(ds_tmp)

        # Merge data from all channels
        ds_merge = xr.merge(ds_backscatter)
        ds = xr.merge([ds, ds_merge], combine_attrs='override')  # override keeps the Dataset attributes

        # Convert np.datetime64 numbers to seconds since 1900-01-01
        #  due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
        ds = ds.assign_coords({'ping_time': (['ping_time'], (ds['ping_time'] -
                                             np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's'),
                                             ds.ping_time.attrs)})

        # Save to file
        io.save_file(ds.chunk({'range_bin': 25000, 'ping_time': 100}),
                     path=path, mode='a', engine=self.engine,
                     group='Beam', compression_settings=self.compression_settings)

    def set_vendor(self, ch_ids, bb, path):
        """Set the Vendor-specific group.
        """
        # Save broadband calibration parameters
        config = self.parser_obj.config_datagram['configuration']
        cal_ch_ids = [ch for ch in ch_ids if 'calibration' in config[ch]]
        # Select the first available ping_time
        ds = xr.Dataset()
        if cal_ch_ids:
            full_ch_names = [f"{config[ch]['transceiver_type']} " +
                             f"{config[ch]['serial_number']}-" +
                             f"{config[ch]['hw_channel_configuration']} " +
                             f"{config[ch]['channel_id_short']}" for ch in cal_ch_ids]
            frequency = [config[ch]['calibration']['frequency'] for ch in cal_ch_ids]
            freq_coord = np.unique(np.hstack(frequency))
            tmp = np.full((len(frequency), len(freq_coord)), np.nan)
            params = ['gain', 'impedance', 'phase', 'beamwidth_alongship', 'beamwidth_athwartship',
                      'angle_offset_alongship', 'angle_offset_athwartship']
            param_dict = {}
            for param in params:
                param_val = tmp.copy()
                for i, ch in enumerate(cal_ch_ids):
                    indices = np.searchsorted(freq_coord, frequency[i])
                    param_val[i][indices] = config[ch]['calibration'][param]
                param_dict[param] = (['channel', 'frequency_spectrum'], param_val)

            ds = xr.Dataset(
                data_vars=param_dict,
                coords={
                    'channel': (['channel'], full_ch_names),
                    'frequency': (['frequency_spectrum'], freq_coord,
                                  {'long_name': 'Broadband frequency spectrum', 'units': 'Hz'})
                })
        if not bb:
            # Save pulse length and sa correction
            freq = [config[ch]['transducer_frequency'] for ch in ch_ids]
            pulse_length = np.array([config[ch]['pulse_duration'] for ch in ch_ids])
            sa_correction = [config[ch]['sa_correction'] for ch in ch_ids]
            gain = [config[ch]['gain'] for ch in ch_ids]
            ds_pulse_length = xr.Dataset({
                'sa_correction': (['frequency', 'pulse_length_bin'], sa_correction),
                'gain_correction': (['frequency', 'pulse_length_bin'], gain),
                'pulse_length': (['frequency', 'pulse_length_bin'], pulse_length)},
                coords={
                    'frequency': (['frequency'], freq,
                                  {'units': 'Hz',
                                   'long_name': 'Transducer frequency',
                                   'valid_min': 0.0}),
                    'pulse_length_bin': (['pulse_length_bin'], np.arange(pulse_length.shape[1]))
            })
            ds = xr.merge([ds, ds_pulse_length])

        #  Save decimation factors and filter coefficients
        coeffs = dict()
        decimation_factors = dict()
        for ch in self.parser_obj.ch_ids['power'] + self.parser_obj.ch_ids['complex']:
            # Coefficients for wide band transceiver
            coeffs[f'{ch}_WBT_filter'] = self.parser_obj.fil_coeffs[ch][1]
            # Coefficients for pulse compression
            coeffs[f'{ch}_PC_filter'] = self.parser_obj.fil_coeffs[ch][2]
            decimation_factors[f'{ch}_WBT_decimation'] = self.parser_obj.fil_df[ch][1]
            decimation_factors[f'{ch}_PC_decimation'] = self.parser_obj.fil_df[ch][2]

        # Assemble variables into dataset
        for k, v in coeffs.items():
            # Save filter coefficients as real and imaginary parts as attributes
            ds.attrs[k + '_r'] = np.real(v)
            ds.attrs[k + '_i'] = np.imag(v)
            # Save decimation factors as attributes
        for k, v in decimation_factors.items():
            ds.attrs[k] = v
        ds.attrs['config_xml'] = self.parser_obj.config_datagram['xml']

        # Save to file
        io.save_file(ds, path=path, mode='a', engine=self.engine,
                     group='Vendor', compression_settings=self.compression_settings)

    def set_sonar(self, ch_ids, path):
        config = self.parser_obj.config_datagram['configuration']
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

        # Save to file
        io.save_file(ds, path=path, mode='a', engine=self.engine,
                     group='Sonar', compression_settings=self.compression_settings)

    # TODO: the overwriting message should not be here,
    #  the assembling filename part also should not be here
    #  symptom:
    #  >>> from echopype import Convert
    #  >>> raw_path_bb_cw = './echopype/test_data/ek80/Summer2018--D20180905-T033113.raw'  # Large file (CW and BB)
    #  >>> tmp = Convert(file=raw_path_bb_cw, model='EK80')
    #  >>> tmp.to_netcdf(save_path='/Users/wu-jung/Downloads', overwrite=True)
    #            overwriting: /Users/wu-jung/Downloads/Summer2018--D20180905-T033113.nc
    #  17:14:00 converting file Summer2018--D20180905-T033113.raw, time of first ping: 2018-Sep-05 03:31:13
    #            overwriting: /Users/wu-jung/Downloads/Summer2018--D20180905-T033113_cw.nc
    def _copy_file(self, file):
        # Copy the current file into a new file with _cw appended to filename
        # TODO: here the _cw (_power) filename should be passed down from the convert object
        #  instead of being made on the fly. This is a bug.
        fname, ext = os.path.splitext(file)
        new_path = fname + '_cw' + ext
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
