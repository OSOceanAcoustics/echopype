import os
import shutil
from collections import defaultdict
import xarray as xr
import numpy as np
import zarr
import netCDF4
from .set_groups_base import SetGroupsBase


class SetGroupsEK80(SetGroupsBase):
    """Class for saving groups to netcdf or zarr from EK80 data files.
    """
    def save(self):
        """Actually save groups to file by calling the set methods.
        """

        self.set_toplevel(self.sonar_model)
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
        bb_ch_ids = self.convert_obj.bb_ch_ids
        cw_ch_ids = self.convert_obj.cw_ch_ids
        # If there is both bb and cw data
        if bb_ch_ids and cw_ch_ids:
            new_path = self._copy_file(self.output_path)
            self.set_beam(bb_ch_ids, bb=True, path=self.output_path)
            self.set_sonar(bb_ch_ids, path=self.output_path)
            self.set_vendor(bb_ch_ids, bb=True, path=self.output_path)
            self.set_beam(cw_ch_ids, bb=False, path=new_path)
            self.set_sonar(cw_ch_ids, path=new_path)
            self.set_vendor(cw_ch_ids, bb=False, path=new_path)
        # If there is only bb data
        elif bb_ch_ids:
            self.set_beam(bb_ch_ids, bb=True, path=self.output_path)
            self.set_sonar(bb_ch_ids, path=self.output_path)
            self.set_vendor(bb_ch_ids, bb=True, path=self.output_path)
        # If there is only cw data
        else:
            self.set_beam(cw_ch_ids, bb=False, path=self.output_path)
            self.set_sonar(cw_ch_ids, path=self.output_path)
            self.set_vendor(cw_ch_ids, bb=False, path=self.output_path)

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

        # Collect variables
        if self.ui_param['water_level'] is not None:
            water_level = self.ui_param['water_level']
        elif 'water_level_draft' in self.convert_obj.environment:
            water_level = self.convert_obj.environment['water_level_draft']
        else:
            water_level = None
            print('WARNING: The water_level_draft was not in the file. Value '
                  'set to None')

        lat, lon, location_time, nmea_msg = self._parse_NMEA()
        # Convert np.datetime64 numbers to seconds since 1900-01-01
        # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
        # TODO: @ngkavin: why is nmea_msg used here?
        mru_time = (np.array(self.convert_obj.mru.get('timestamp', [np.nan])) -
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
                   'drop_keel_offset': (self.convert_obj.environment['drop_keel_offset'] if
                                        hasattr(self.convert_obj.environment, 'drop_keel_offset') else np.nan)})

        # Save to file
        if self.save_ext == '.nc':
            nc_encoding = {var: self.NETCDF_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
            ds.to_netcdf(path=self.output_path, mode='a', group='Platform', encoding=nc_encoding)
        elif self.save_ext == '.zarr':
            zarr_encoding = {var: self.ZARR_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
            ds = ds.chunk({'location_time': 100, 'mru_time': 100})
            ds.to_zarr(store=self.output_path, mode='a', group='Platform', encoding=zarr_encoding)

    def set_beam(self, ch_ids, bb, path):
        """Set the Beam group.
        """
        config = self.convert_obj.config_datagram['configuration']
        beam_dict = dict()
        freq = np.array([config[x]['transducer_frequency'] for x in ch_ids], dtype=int)
        tx_num = len(ch_ids)
        ping_num = len(self.convert_obj.ping_time)

        # Find largest dimensions of array in order to pad and stack smaller arrays
        # max_samples = 0
        # TODO How to determine if a CW data set is split beam or single beam, and how many splits?
        max_splits = max([n_c for n_c in self.convert_obj.n_complex_dict.values()]) if bb else 4
        if bb:
            shape = (len(ch_ids), ping_num, -1, max_splits)
            backscatter = np.array(self.convert_obj.ping_data_dict['complex']).reshape(shape)
            backscatter = np.moveaxis(backscatter, 3, 1)

        # Loop through each transducer for channel-specific variables
        beam_dict['gain_correction'] = np.zeros(shape=(tx_num,), dtype='float32')
        beam_dict['gpt_software_version'] = []  # TODO: @ngkavin: change this to wbt_software_version
        beam_dict['channel_id'] = []

        params = ['beam_width_alongship', 'beam_width_athwartship', 'beam_width_alongship',
                  'beam_width_athwartship', 'transducer_alpha_x', 'transducer_alpha_y',
                  'transducer_alpha_z', 'angle_offset_alongship', 'angle_offset_athwartship',
                  'angle_sensitivity_alongship', 'angle_sensitivity_athwartship', 'transducer_offset_x',
                  'transducer_offset_y', 'transducer_offset_z', 'equivalent_beam_angle']

        beam_par = defaultdict(lambda: np.zeros(shape=(tx_num,), dtype='float32'))
        c_seq = 0
        for k, c in config.items():
            if k not in ch_ids:
                continue
            for param in params:
                beam_par[param][c_seq] = c.get(param, np.nan)

            beam_dict['gain_correction'][c_seq] = c['gain'][c_seq]
            beam_dict['gpt_software_version'].append(c['transceiver_software_version'])
            beam_dict['channel_id'].append(c['channel_id'])
            c_seq += 1

        # Stack channels and order axis as: channel, quadrant, ping, range
        # TODO: @ngkavin: change below to if-else testing the power data and frequency start/end combination
        if bb:
            try:
                freq_start = np.array([self.convert_obj.ping_data_dict['frequency_start'][x][0]
                                      for x in ch_ids], dtype=int)
                freq_end = np.array([self.convert_obj.ping_data_dict['frequency_end'][x][0]
                                    for x in ch_ids], dtype=int)
            # Exception occurs when instrument records complex power data without
            # supplying the frequency start and end
            except IndexError:
                freq_start = np.array([config[x].get('transducer_frequency_minimum', np.nan)
                                       for x in ch_ids], dtype=int)
                freq_end = np.array([config[x].get('transducer_frequency_maximum', np.nan)
                                     for x in ch_ids], dtype=int)

        # Loop through each transducer for variables that may vary at each ping
        # -- this rarely is the case for EK60 so we check first before saving
        ch_idx = [self.convert_obj.ch_ids.index(ch) for ch in ch_ids]
        tdn = np.array(list(self.convert_obj.ping_data_dict['pulse_duration'].values()))[ch_idx]
        tx_power = np.array(list(self.convert_obj.ping_data_dict['transmit_power'].values()))[ch_idx]
        smpl_int = np.array(list(self.convert_obj.ping_data_dict['sample_interval'].values()))[ch_idx]
        slope = np.array(list(self.convert_obj.ping_data_dict['slope'].values()))[ch_idx]

        # Build other parameters
        # beam_dict['non_quantitative_processing'] = np.array([0, ] * freq.size, dtype='int32')
        # -- sample_time_offset is set to 2 for EK60 data, this value is NOT from sample_data['offset']
        # beam_dict['sample_time_offset'] = np.array([2, ] * freq.size, dtype='int32')

        # Convert np.datetime64 numbers to seconds since 1900-01-01
        # due to xarray.to_netcdf() error on encoding np.datetime64 objects directly
        ping_time = (self.convert_obj.ping_time - np.datetime64('1900-01-01T00:00:00')) / np.timedelta64(1, 's')

        ds = xr.Dataset(
            {'channel_id': (['frequency'], beam_dict['channel_id']),
             'beamwidth_receive_alongship': (['frequency'], beam_par['beamwidth_receive_major'],
                                             {'long_name': 'Half power one-way receive beam width along '
                                              'alongship axis of beam',
                                              'units': 'arc_degree',
                                              'valid_range': (0.0, 360.0)}),
             'beamwidth_receive_athwartship': (['frequency'], beam_par['beamwidth_receive_minor'],
                                               {'long_name': 'Half power one-way receive beam width along '
                                                'athwartship axis of beam',
                                                'units': 'arc_degree',
                                                'valid_range': (0.0, 360.0)}),
             'beamwidth_transmit_alongship': (['frequency'], beam_par['beamwidth_transmit_major'],
                                              {'long_name': 'Half power one-way transmit beam width along '
                                               'alongship axis of beam',
                                               'units': 'arc_degree',
                                               'valid_range': (0.0, 360.0)}),
             'beamwidth_transmit_athwartship': (['frequency'], beam_par['beamwidth_transmit_minor'],
                                                {'long_name': 'Half power one-way transmit beam width along '
                                                 'athwartship axis of beam',
                                                 'units': 'arc_degree',
                                                 'valid_range': (0.0, 360.0)}),
             'beam_direction_x': (['frequency'], beam_par['beam_direction_x'],
                                  {'long_name': 'x-component of the vector that gives the pointing '
                                                'direction of the beam, in sonar beam coordinate '
                                                'system',
                                   'units': '1',
                                   'valid_range': (-1.0, 1.0)}),
             'beam_direction_y': (['frequency'], beam_par['beam_direction_x'],
                                  {'long_name': 'y-component of the vector that gives the pointing '
                                                'direction of the beam, in sonar beam coordinate '
                                                'system',
                                   'units': '1',
                                   'valid_range': (-1.0, 1.0)}),
             'beam_direction_z': (['frequency'], beam_par['beam_direction_x'],
                                  {'long_name': 'z-component of the vector that gives the pointing '
                                                'direction of the beam, in sonar beam coordinate '
                                                'system',
                                   'units': '1',
                                   'valid_range': (-1.0, 1.0)}),
             'angle_offset_alongship': (['frequency'], beam_par['angle_offset_alongship'],
                                        {'long_name': 'electrical alongship angle of the transducer'}),
             'angle_offset_athwartship': (['frequency'], beam_par['angle_offset_athwartship'],
                                          {'long_name': 'electrical athwartship angle of the transducer'}),
             'angle_sensitivity_alongship': (['frequency'], beam_par['angle_sensitivity_alongship'],
                                             {'long_name': 'alongship sensitivity of the transducer'}),
             'angle_sensitivity_athwartship': (['frequency'], beam_par['angle_sensitivity_athwartship'],
                                               {'long_name': 'athwartship sensitivity of the transducer'}),
             'equivalent_beam_angle': (['frequency'], beam_par['equivalent_beam_angle'],
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
             'sample_interval': (['frequency', 'ping_time'], smpl_int,
                                 {'long_name': 'Interval between recorded raw data samples',
                                  'units': 's',
                                  'valid_min': 0.0}),
            #  'sample_time_offset': (['frequency'], beam_dict['sample_time_offset'],
            #                         {'long_name': 'Time offset that is subtracted from the timestamp '
            #                                       'of each sample',
            #                          'units': 's'}),
            #  'transmit_bandwidth': (['frequency'], tx_sig['transmit_bandwidth'],
            #                         {'long_name': 'Nominal bandwidth of transmitted pulse',
            #                          'units': 'Hz',
            #                          'valid_min': 0.0}),
             'transmit_duration_nominal': (['frequency', 'ping_time'], tdn,
                                           {'long_name': 'Nominal bandwidth of transmitted pulse',
                                            'units': 's',
                                            'valid_min': 0.0}),
             'transmit_power': (['frequency', 'ping_time'], tx_power,
                                {'long_name': 'Nominal transmit power',
                                 'units': 'W',
                                 'valid_min': 0.0}),
             'transducer_offset_x': (['frequency'], beam_par['transducer_offset_x'],
                                     {'long_name': 'x-axis distance from the platform coordinate system '
                                                   'origin to the sonar transducer',
                                      'units': 'm'}),
             'transducer_offset_y': (['frequency'], beam_par['transducer_offset_y'],
                                     {'long_name': 'y-axis distance from the platform coordinate system '
                                                   'origin to the sonar transducer',
                                      'units': 'm'}),
             'transducer_offset_z': (['frequency'], beam_par['transducer_offset_z'],
                                     {'long_name': 'z-axis distance from the platform coordinate system '
                                                   'origin to the sonar transducer',
                                      'units': 'm'}),
             'slope': (['frequency', 'ping_time'], slope)},
            coords={'frequency': (['frequency'], freq,
                                  {'units': 'Hz',
                                   'long_name': 'Transducer frequency',
                                   'valid_min': 0.0}),
                    'ping_time': (['ping_time'], ping_time,
                                  {'axis': 'T',
                                   'calendar': 'gregorian',
                                   'long_name': 'Timestamp of each ping',
                                   'standard_name': 'time',
                                   'units': 'seconds since 1900-01-01'})},
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
                                    'units': 'V'}),
                 'frequency_start': (['frequency'], freq_start,
                                     {'long_name': 'Starting frequency of the transducer',
                                      'units': 'Hz'}),
                 'frequency_end': (['frequency'], freq_end,
                                   {'long_name': 'Ending frequency of the transducer',
                                    'units': 'Hz'})},
                coords={'frequency': (['frequency'], freq,
                                      {'units': 'Hz',
                                       'long_name': 'Transducer frequency',
                                       'valid_min': 0.0}),
                        'ping_time': (['ping_time'], ping_time,
                                      {'axis': 'T',
                                       'calendar': 'gregorian',
                                       'long_name': 'Timestamp of each ping',
                                       'standard_name': 'time',
                                       'units': 'seconds since 1900-01-01'}),
                        'quadrant': (['quadrant'], np.arange(max_splits)),
                        'range_bin': (['range_bin'], np.arange(backscatter.shape[3]))
                        })
            ds = xr.merge([ds, ds_bb], combine_attrs='override')
        # Save continuous wave backscatter
        else:
            # TODO: @ngkavin:
            #  Move sa_correction to the Vendor group as a data variable indexed by pulse length,
            #  and performs the pulse length matching "on the fly" in process.calibrate.
            #  See EK80applyGain for matching tolerance (tol=1e-6).
            ds_cw = xr.Dataset(
                {'backscatter_r': (['frequency', 'ping_time', 'range_bin'],
                                   self.convert_obj.ping_data_dict['power'],
                                   {'long_name': 'Backscattering power',
                                       'units': 'dB'}),
                 'angle_athwartship': (['frequency', 'ping_time', 'range_bin'],
                                       self.convert_obj.ping_data_dict['angle'][:, :, :, 0],
                                       {'long_name': 'electrical athwartship angle'}),
                 'angle_alongship': (['frequency', 'ping_time', 'range_bin'],
                                     self.convert_obj.ping_data_dict['angle'][:, :, :, 1],
                                     {'long_name': 'electrical alongship angle'})},
                coords={'frequency': (['frequency'], freq,
                                      {'units': 'Hz',
                                       'long_name': 'Transducer frequency',
                                       'valid_min': 0.0}),
                        'ping_time': (['ping_time'], ping_time,
                                      {'axis': 'T',
                                       'calendar': 'gregorian',
                                       'long_name': 'Timestamp of each ping',
                                       'standard_name': 'time',
                                       'units': 'seconds since 1900-01-01'}),
                        'range_bin': (['range_bin'], np.arange(self.convert_obj.ping_data_dict['power'].shape[2]))
                        })
            ds = xr.merge([ds, ds_cw], combine_attrs='override')

        # Below are specific to Simrad .raw files
        if 'gpt_software_version' in beam_dict:
            ds['gpt_software_version'] = ('frequency', beam_dict['gpt_software_version'])
        # Save to file
        if self.save_ext == '.nc':
            nc_encoding = {var: self.NETCDF_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
            ds.to_netcdf(path=path, mode='a', group='Beam', encoding=nc_encoding)
        elif self.save_ext == '.zarr':
            zarr_encoding = {var: self.ZARR_COMPRESSION_SETTINGS for var in ds.data_vars} if self.compress else {}
            ds = ds.chunk({'range_bin': 25000, 'ping_time': 100})
            ds.to_zarr(store=path, mode='a', group='Beam', encoding=zarr_encoding)

    def set_vendor(self, ch_ids, bb, path):
        """Set the Vendor-specific group.
        """
        # Save broadband calibration parameters
        config = self.convert_obj.config_datagram['configuration']
        if bb:
            cal_ch_ids = []
            cal_ch_ids = [ch for ch in ch_ids if 'calibration' in config[ch]]
            full_ch_names = [f"{config[ch]['transceiver_type']} " +
                             f"{config[ch]['serial_number']}-" +
                             f"{config[ch]['hw_channel_configuration']} " +
                             f"{config[ch]['channel_id_short']}" for ch in cal_ch_ids]
            frequency = [config[ch]['calibration']['frequency'] for ch in cal_ch_ids]
            freq_coord = np.unique(np.hstack(frequency))
            tmp = np.full((len(frequency), len(freq_coord)), np.nan)
            params = ['gain', 'impedence', 'phase', 'beamwidth_alongship', 'beamwidth_athwartship',
                      'angle_offset_alongship', 'angle_offset_athwartship']
            param_dict = {}
            for param in params:
                param_val = tmp.copy()
                for i, ch in enumerate(cal_ch_ids):
                    indices = np.searchsorted(freq_coord, frequency[i])
                    param_val[i][indices] = config[ch]['calibration'][param]
                param_dict[param] = (['channel', 'frequency'], param_val)

            ds = xr.Dataset(
                data_vars=param_dict,
                coords={
                    'channel': (['channel'], full_ch_names),
                    'frequency': (['frequency'], freq_coord,
                                  {'long_name': 'Transducer frequency', 'units': 'Hz'})
                })
        else:
            # Save pulse length and sa correction
            freq = [config[ch]['transducer_frequency'] for ch in ch_ids]
            pulse_length = np.array([config[ch]['pulse_duration'] for ch in ch_ids])
            sa_correction = [config[ch]['sa_correction'] for ch in ch_ids]
            ds = xr.Dataset({
                'sa_correction': (['frequency', 'pulse_length_bin'], sa_correction),
                'pulse_length': (['frequency', 'pulse_length_bin'], pulse_length)},
                coords={
                    'frequency': (['frequency'], freq,
                                  {'units': 'Hz',
                                   'long_name': 'Transducer frequency',
                                   'valid_min': 0.0}),
                    'pulse_length_bin': (['pulse_length_bin'], np.arange(pulse_length.shape[1]))
            })

        #  Save decimation factors and filter coefficients
        coeffs = dict()
        decimation_factors = dict()
        for ch in self.convert_obj.ch_ids:
            # Coefficients for wide band transceiver
            coeffs[f'{ch}_WBT_filter'] = self.convert_obj.fil_coeffs[ch][1]
            # Coefficients for pulse compression
            coeffs[f'{ch}_PC_filter'] = self.convert_obj.fil_coeffs[ch][2]
            decimation_factors[f'{ch}_WBT_decimation'] = self.convert_obj.fil_df[ch][1]
            decimation_factors[f'{ch}_PC_decimation'] = self.convert_obj.fil_df[ch][2]

        # Assemble variables into dataset
        for k, v in coeffs.items():
            # Save filter coefficients as real and imaginary parts as attributes
            ds.attrs[k + '_r'] = np.real(v)
            ds.attrs[k + '_i'] = np.imag(v)
            # Save decimation factors as attributes
        for k, v in decimation_factors.items():
            ds.attrs[k] = v
        ds.attrs['config_xml'] = self.convert_obj.config_datagram['xml']
        # save to file
        if self.save_ext == '.nc':
            ds.to_netcdf(path=path, mode='a', group='Vendor')
        elif self.save_ext == '.zarr':
            ds.to_zarr(store=path, mode='a', group='Vendor')

    def set_sonar(self, ch_ids, path):
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
