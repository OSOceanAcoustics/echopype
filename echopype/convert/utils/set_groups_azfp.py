from .set_groups_base import SetGroupsBase
import xarray as xr
import netCDF4
import zarr
import os


class SetGroupsAZFP(SetGroupsBase):
    """Class for setting groups in netCDF file for AZFP data.
    """

    def set_env(self, env_dict):
        """Set the Environment group in the AZFP netCDF file.
        AZFP includes additional variables 'salinity' and 'pressure'

        Parameters
        ----------
        env_dict
            dictionary containing environment group params
                         env_dict['frequency']
                         env_dict['absorption_coeff']
                         env_dict['sound_speed']
        """
        # Only save environment group if file_path exists
        if not os.path.exists(self.file_path):
            print('netCDF file does not exist, exiting without saving Environment group...')
        else:
            ds = xr.Dataset({'temperature': (['ping_time'], env_dict['temperature'])},
                            coords={'ping_time': (['ping_time'], env_dict['ping_time'],
                                    {'axis': 'T',
                                     'calendar': 'gregorian',
                                     'long_name': 'Timestamp of each ping',
                                     'standard_name': 'time',
                                     'units': 'seconds since 1970-01-01'})},
                            attrs={'long_name': "Water temperature",
                                   'units': "C"})

            # save to file
            if self.format == '.nc':
                ds.to_netcdf(path=self.file_path, mode='a', group='Environment')
            elif self.format == '.zarr':
                if not self.append_zarr:
                    ds.to_zarr(store=self.file_path, mode='a', group='Environment')
                else:
                    ds.to_zarr(store=self.file_path, mode='a', group='Environment', append_dim='ping_time')

    def set_platform(self, platform_dict):
        """Set the Platform group in the AZFP nc file. AZFP does not record pitch, roll, and heave.

        Parameters
        ----------
        platform_dict
            dictionary containing platform parameters
        """
        if not os.path.exists(self.file_path):
            print("netCDF file does not exist, exiting without saving Platform group...")
        elif self.format == '.nc':
            ncfile = netCDF4.Dataset(self.file_path, 'a', format='NETCDF4')
            plat = ncfile.createGroup('Platform')
            with netCDF4.Dataset(self.file_path, 'a', format='NETCDF4') as ncfile:
                [plat.setncattr(k, v) for k, v in platform_dict.items()]
        elif self.format == '.zarr' and not self.append_zarr:    # Do not save platform if appending
            zarrfile = zarr.open(self.file_path, mode='a')
            plat = zarrfile.create_group('Platform')
            for k, v in platform_dict.items():
                plat.attrs[k] = v

    def set_beam(self, beam_dict):
        """Set the Beam group in the AZFP nc file.

        Parameters
        ----------
        beam_dict
            dictionary containing general beam parameters
        """

        ds = xr.Dataset({'backscatter_r': (['frequency', 'ping_time', 'range_bin'], beam_dict['backscatter_r']),
                         'equivalent_beam_angle': (['frequency'], beam_dict['EBA']),
                         'gain_correction': (['frequency'], beam_dict['gain_correction']),
                         'sample_interval': (['frequency'], beam_dict['sample_interval'],
                                             {'units': 's'}),
                         'transmit_duration_nominal': (['frequency'], beam_dict['transmit_duration_nominal'],
                                                       {'long_name': 'Nominal bandwidth of transmitted pulse',
                                                        'units': 's',
                                                        'valid_min': 0.0}),
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
                                                                            beam_dict['range_averaging_samples'])},
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
        n_settings = {}
        z_settings = {}
        if self.compress:
            n_settings = {'backscatter_r': {'zlib': True, 'complevel': 4}}
            z_settings = {'backscatter_r': {'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)}}

        if self.format == '.nc':
            ds.to_netcdf(path=self.file_path, mode='a', group='Beam', encoding=n_settings)
        elif self.format == '.zarr':
            if not self.append_zarr:
                ds.to_zarr(store=self.file_path, mode='a', group='Beam', encoding=z_settings)
            else:
                ds.to_zarr(store=self.file_path, mode='a', group='Beam', append_dim='ping_time')

    def set_vendor_specific(self, vendor_dict):
        """Set the Vendor-specific group in the AZFP nc file.

        Parameters
        ----------
        vendor_dict
            dictionary containing vendor-specific parameters
        """

        ds = xr.Dataset({
            'digitization_rate': (['frequency'], vendor_dict['digitization_rate']),
            'lockout_index': (['frequency'], vendor_dict['lockout_index']),
            'number_of_bins_per_channel': (['frequency'], vendor_dict['num_bins']),
            'number_of_samples_per_average_bin': (['frequency'], vendor_dict['range_samples_per_bin']),
            'board_number': (['frequency'], vendor_dict['board_number']),
            'data_type': (['frequency'], vendor_dict['data_type']),
            'ping_status': (['ping_time'], vendor_dict['ping_status']),
            'number_of_acquired_pings': (['ping_time'], vendor_dict['number_of_acquired_pings']),
            'first_ping': (['ping_time'], vendor_dict['first_ping']),
            'last_ping': (['ping_time'], vendor_dict['last_ping']),
            'data_error': (['ping_time'], vendor_dict['data_error']),
            'sensor_flag': (['ping_time'], vendor_dict['sensor_flag']),
            'ancillary': (['ping_time', 'ancillary_len'], vendor_dict['ancillary']),
            'ad_channels': (['ping_time', 'ad_len'], vendor_dict['ad_channels']),
            'battery_main': (['ping_time'], vendor_dict['battery_main']),
            'battery_tx': (['ping_time'], vendor_dict['battery_tx'])},
            coords={
                'frequency': (['frequency'], vendor_dict['frequency'],
                              {'units': 'Hz',
                               'valid_min': 0.0}),
                'ping_time': (['ping_time'], vendor_dict['ping_time'],
                              {'axis': 'T',
                               'calendar': 'gregorian',
                               'long_name': 'Timestamp of each ping',
                               'standard_name': 'time',
                               'units': 'seconds since 1970-01-01'}),
                'ancillary_len': (['ancillary_len'], vendor_dict['ancillary_len']),
                'ad_len': (['ad_len'], vendor_dict['ad_len'])},
            attrs={
                'profile_flag': vendor_dict['profile_flag'],
                'profile_number': vendor_dict['profile_number'],
                'burst_interval': vendor_dict['burst_interval'],
                'ping_per_profile': vendor_dict['ping_per_profile'],
                'average_pings_flag': vendor_dict['average_pings_flag'],
                'spare_channel': vendor_dict['spare_channel'],
                'ping_period': vendor_dict['ping_period'],
                'phase': vendor_dict['phase'],
                'number_of_channels': vendor_dict['number_of_channels']}
        )

        if self.format == '.nc':
            ds.to_netcdf(path=self.file_path, mode='a', group='Vendor')
        elif self.format == '.zarr':
            if not self.append_zarr:
                ds.to_zarr(store=self.file_path, mode='a', group='Vendor')
            else:
                ds.to_zarr(store=self.file_path, mode='a', group='Vendor', append_dim='ping_time')
