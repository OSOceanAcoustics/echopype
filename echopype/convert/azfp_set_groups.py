from .set_nc_groups import SetGroups
import xarray as xr


class SetAZFPGroups(SetGroups):

    def set_beam(self, beam_dict):
        ds = xr.Dataset({'backscatter_r': (['frequency', 'ping_time', 'range_bin'], beam_dict['backscatter_r']),
                         'equivalent_beam_angle': (['frequency'], beam_dict['EBA']),
                         'gain_correction': (['frequency'], beam_dict['gain_correction']),
                         'sample_interval': (['frequency', 'ping_time'], beam_dict['sample_interval'],
                                             {'units': 'seconds'}),
                         'transmit_duration_nominal': (['frequency'], beam_dict['transmit_duration_nominal'],
                                                       {'units': 'seconds'}),
                         'range': (['frequency', 'range_bin'], beam_dict['range']),
                         'tilt_corr_range': (['frequency', 'range_bin'], beam_dict['tilt_corr_range']),
                         'temperature_counts': (['ping_time'], beam_dict['temperature_counts']),
                         'tilt_x_count': (['ping_time'], beam_dict['tilt_x_count']),
                         'tilt_y_count': (['ping_time'], beam_dict['tilt_y_count']),
                         'tilt_x': (['ping_time'], beam_dict['tilt_x']),
                         'tilt_y': (['ping_time'], beam_dict['tilt_y']),
                         'cos_tilt_mag': (['ping_time'], beam_dict['cos_tilt_mag']),
                         'DS': (['frequency'], beam_dict['DS']),
                         'EL': (['frequency'], beam_dict['EL']),
                         'TVR': (['frequency'], beam_dict['TVR']),
                         'VTX': (['frequency'], beam_dict['VTX']),
                         'Sv_offset': (['frequency'], beam_dict['Sv_offset']),
                         'number_of_samples_digitized_per_pings': (['frequency'], beam_dict['range_samples']),
                         'number_of_digitized_samples_averaged_per_pings': (['frequency'],
                                                                            beam_dict['range_averaging_samples']),
                         'sea_abs': (['frequency'], beam_dict['sea_abs'])},
                        coords={'frequency': (['frequency'], beam_dict['frequency'],
                                              {'units': 'Hz',
                                               'valid_min': 0.0}),
                                'ping_time': (['ping_time'], beam_dict['ping_time'],
                                              {'axis': 'T',
                                               'calendar': 'gregorian',
                                               'long_name': 'Timestamp of each ping',
                                               'standard_name': 'time',
                                               'units': 'seconds since 1970-01-01'}),
                                'range_bin': (['range_bin'], beam_dict['range_bin'])},
                        attrs={'beam_mode': '',
                               'conversion_equation_t': 'type_4',
                               'number_of_frequency': beam_dict['number_of_frequency'],
                               'number_of_pings_per_burst': beam_dict['number_of_pings_per_burst'],
                               'average_burst_pings_flag': beam_dict['average_burst_pings_flag'],
                               # Temperature coefficients
                               'temperature_ka': beam_dict['temperature_ka'],
                               'temperature_kb': beam_dict['temperature_kb'],
                               'temperature_kc': beam_dict['temperature_kc'],
                               'temperature_A': beam_dict['temperature_A'],
                               'temperature_B': beam_dict['temperature_B'],
                               'temperature_C': beam_dict['temperature_C'],
                               # Tilt coefficients
                               'tilt_X_a': beam_dict['tilt_X_a'],
                               'tilt_X_b': beam_dict['tilt_X_b'],
                               'tilt_X_c': beam_dict['tilt_X_c'],
                               'tilt_X_d': beam_dict['tilt_X_d'],
                               'tilt_Y_a': beam_dict['tilt_Y_a'],
                               'tilt_Y_b': beam_dict['tilt_Y_b'],
                               'tilt_Y_c': beam_dict['tilt_Y_c'],
                               'tilt_Y_d': beam_dict['tilt_Y_d']})

        ds.to_netcdf(path=self.file_path, mode="a", group="Beam")
        pass

    def set_vendor_specific(self, vendor_dict):
        ds = xr.Dataset(
            {
                'profile_flag': (['ping_time'], vendor_dict['profile_flag']),
                'profile_number': (['ping_time'], vendor_dict['profile_number']),
                'ping_status': (['ping_time'], vendor_dict['ping_status']),
                'burst_interval': (['ping_time'], vendor_dict['burst_interval']),
                'digitization_rate': (['ping_time', 'frequency'], vendor_dict['digitization_rate']),
                'lock_out_index': (['ping_time', 'frequency'], vendor_dict['lock_out_index']),
                'number_of_bins_per_channel': (['ping_time', 'frequency'], vendor_dict['num_bins']),
                'number_of_samples_per_average_bin': (['ping_time', 'frequency'], vendor_dict['range_samples']),
                'ping_per_profile': (['ping_time'], vendor_dict['ping_per_profile']),
                'average_pings_flag': (['ping_time'], vendor_dict['average_pings_flag']),
                'number_of_acquired_pings': (['ping_time'], vendor_dict['number_of_acquired_pings']),
                'ping_period': (['ping_time'], vendor_dict['ping_period']),
                'first_ping': (['ping_time'], vendor_dict['first_ping']),
                'last_ping': (['ping_time'], vendor_dict['last_ping']),
                'data_type': (['ping_time', 'frequency'], vendor_dict['data_type']),
                'data_error': (['ping_time'], vendor_dict['data_error']),
                'phase': (['ping_time'], vendor_dict['phase']),
                'number_of_channels': (['ping_time'], vendor_dict['number_of_channels']),
                'spare_channel': (['ping_time'], vendor_dict['spare_channel']),
                'board_number': (['ping_time', 'frequency'], vendor_dict['board_number']),
                'sensor_flag': (['ping_time'], vendor_dict['sensor_flag']),
                'ancillary': (['ping_time', 'ancillary_len'], vendor_dict['ancillary']),
                'ad_channels': (['ping_time', 'ad_len'], vendor_dict['ad_channels'])
            },
            coords={'frequency': (['frequency'], vendor_dict['frequency']),
                    'ping_time': (['ping_time'], vendor_dict['ping_time']),
                    'ancillary_len': (['ancillary_len'], vendor_dict['ancillary_len']),
                    'ad_len': (['ad_len'], vendor_dict['ad_len'])}
        )

        ds.to_netcdf(path=self.file_path, mode="a", group="Vendor")
